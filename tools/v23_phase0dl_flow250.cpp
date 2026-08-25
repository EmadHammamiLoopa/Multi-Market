#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int64_t GRID_US = 250000;
constexpr int64_t EXPECTED_EMITTED = 345600;

using Bids = std::map<double, double, std::greater<double>>;
using Asks = std::map<double, double>;

struct Row {
  int64_t exchange_ts = 0;
  int64_t local_ts = 0;
  bool snapshot = false;
  bool bid = false;
  double price = 0.0;
  double amount = 0.0;
};

struct Book {
  Bids bids;
  Asks asks;
  bool ready = false;

  void clear() {
    bids.clear();
    asks.clear();
    ready = false;
  }

  void apply(const Row& r) {
    if (r.bid) {
      if (r.amount == 0.0) bids.erase(r.price);
      else bids[r.price] = r.amount;
    } else {
      if (r.amount == 0.0) asks.erase(r.price);
      else asks[r.price] = r.amount;
    }
  }

  bool structurally_valid() const {
    return !bids.empty() && !asks.empty() &&
           std::isfinite(bids.begin()->first) &&
           std::isfinite(asks.begin()->first) &&
           bids.begin()->first > 0.0 &&
           asks.begin()->first > bids.begin()->first;
  }

  bool valid() const { return ready && structurally_valid(); }
};

class GzipLineReader {
 public:
  explicit GzipLineReader(const char* path)
      : file_(gzopen(path, "rb")), buffer_(1 << 20) {
    if (!file_) throw std::runtime_error("cannot open gzip input");
  }

  ~GzipLineReader() {
    if (file_) gzclose(file_);
  }

  bool getline(std::string& out) {
    out.clear();
    for (;;) {
      char* p = gzgets(file_, buffer_.data(), static_cast<int>(buffer_.size()));
      if (!p) return !out.empty();
      const size_t n = std::strlen(p);
      out.append(p, n);
      if (n && p[n - 1] == '\n') {
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
          out.pop_back();
        }
        return true;
      }
      if (gzeof(file_)) return !out.empty();
    }
  }

 private:
  gzFile file_ = nullptr;
  std::vector<char> buffer_;
};

bool parse_row(const std::string& line, Row& row) {
  std::array<std::string, 8> fields;
  size_t start = 0;
  size_t field = 0;
  for (size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ',') {
      if (field >= fields.size()) return false;
      fields[field++] = line.substr(start, i - start);
      start = i + 1;
    }
  }
  if (field != fields.size()) return false;

  try {
    row.exchange_ts = std::stoll(fields[2]);
    row.local_ts = std::stoll(fields[3]);
    row.snapshot = fields[4] == "true";
    if (fields[5] == "bid") row.bid = true;
    else if (fields[5] == "ask") row.bid = false;
    else return false;
    row.price = std::stod(fields[6]);
    row.amount = std::stod(fields[7]);
  } catch (...) {
    return false;
  }

  return std::isfinite(row.price) && std::isfinite(row.amount) &&
         row.price > 0.0 && row.amount >= 0.0;
}

template <class Map>
std::vector<std::pair<double, double>> top_levels(const Map& map, int n) {
  std::vector<std::pair<double, double>> out;
  out.reserve(n);
  for (const auto& item : map) {
    if (static_cast<int>(out.size()) == n) break;
    out.push_back(item);
  }
  return out;
}

double level_ofi(const std::vector<std::pair<double, double>>& before,
                 const std::vector<std::pair<double, double>>& after,
                 int index,
                 bool bid) {
  const bool has_before = index < static_cast<int>(before.size());
  const bool has_after = index < static_cast<int>(after.size());
  if (!has_before && !has_after) return 0.0;

  const double p0 = has_before ? before[index].first : 0.0;
  const double q0 = has_before ? before[index].second : 0.0;
  const double p1 = has_after ? after[index].first : 0.0;
  const double q1 = has_after ? after[index].second : 0.0;

  if (!has_before) return bid ? q1 : -q1;
  if (!has_after) return bid ? -q0 : q0;

  if (bid) {
    if (p1 > p0) return q1;
    if (p1 == p0) return q1 - q0;
    return -q0;
  }

  if (p1 < p0) return -q1;
  if (p1 == p0) return q0 - q1;
  return q0;
}

double ofi(const Book& before, const Book& after, int n) {
  const auto b0 = top_levels(before.bids, n);
  const auto a0 = top_levels(before.asks, n);
  const auto b1 = top_levels(after.bids, n);
  const auto a1 = top_levels(after.asks, n);

  double value = 0.0;
  for (int i = 0; i < n; ++i) {
    value += level_ofi(b0, b1, i, true);
    value += level_ofi(a0, a1, i, false);
  }
  return value;
}

template <class Map>
void replenish_deplete_top5(const Map& before, const Map& after,
                            double& replenish, double& deplete) {
  std::set<double> prices;
  int k = 0;
  for (const auto& item : before) {
    if (k++ == 5) break;
    prices.insert(item.first);
  }
  k = 0;
  for (const auto& item : after) {
    if (k++ == 5) break;
    prices.insert(item.first);
  }

  for (double price : prices) {
    double q0 = 0.0;
    double q1 = 0.0;
    auto i0 = before.find(price);
    if (i0 != before.end()) q0 = i0->second;
    auto i1 = after.find(price);
    if (i1 != after.end()) q1 = i1->second;
    const double delta = q1 - q0;
    if (delta > 0.0) replenish += delta;
    else if (delta < 0.0) deplete += -delta;
  }
}

struct FlowBin {
  double ofi_l1 = 0.0;
  double mlofi_l5 = 0.0;
  double mlofi_l10 = 0.0;
  double bid_replenish_l5 = 0.0;
  double ask_replenish_l5 = 0.0;
  double bid_deplete_l5 = 0.0;
  double ask_deplete_l5 = 0.0;

  void clear() {
    ofi_l1 = 0.0;
    mlofi_l5 = 0.0;
    mlofi_l10 = 0.0;
    bid_replenish_l5 = 0.0;
    ask_replenish_l5 = 0.0;
    bid_deplete_l5 = 0.0;
    ask_deplete_l5 = 0.0;
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: v23_phase0dl_flow250 INPUT.csv.gz OUTPUT.csv DAY_START_US DAY_END_US\n";
    return 2;
  }

  const int64_t day_start = std::stoll(argv[3]);
  const int64_t day_end = std::stoll(argv[4]);
  if (day_end <= day_start || day_end - day_start != 86'400'000'000LL) {
    std::cerr << "invalid day bounds\n";
    return 2;
  }

  GzipLineReader input(argv[1]);
  std::ofstream output(argv[2]);
  if (!output) throw std::runtime_error("cannot open output");
  output << "local_timestamp_us,ofi_l1_250ms,mlofi_l5_250ms,mlofi_l10_250ms,"
            "bid_replenish_l5_250ms,ask_replenish_l5_250ms,"
            "bid_deplete_l5_250ms,ask_deplete_l5_250ms,flow_valid\n";

  std::string line;
  if (!input.getline(line)) return 2;
  const std::string expected_header =
      "exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount";
  if (line != expected_header) {
    std::cerr << "unexpected header\n";
    return 2;
  }

  Book book;
  std::vector<Row> group;
  group.reserve(4096);
  FlowBin flow;

  bool group_snapshot = false;
  bool continuity_valid = false;
  int64_t next_grid = day_start;
  int64_t group_ts = std::numeric_limits<int64_t>::min();
  int64_t previous_local = std::numeric_limits<int64_t>::min();

  uint64_t parsed_rows = 0;
  uint64_t bad_rows = 0;
  uint64_t groups = 0;
  uint64_t snapshots = 0;
  uint64_t integrity_latches = 0;
  uint64_t emitted = 0;

  auto emit = [&]() {
    output << next_grid << ','
           << flow.ofi_l1 << ','
           << flow.mlofi_l5 << ','
           << flow.mlofi_l10 << ','
           << flow.bid_replenish_l5 << ','
           << flow.ask_replenish_l5 << ','
           << flow.bid_deplete_l5 << ','
           << flow.ask_deplete_l5 << ','
           << (continuity_valid && book.valid()) << '\n';
    flow.clear();
    next_grid += GRID_US;
    ++emitted;
  };

  auto flush_group = [&]() {
    if (group.empty()) return;

    while (next_grid < group_ts && next_grid < day_end) emit();

    const Book before = book;

    if (group_snapshot) {
      // Snapshot starts a new continuity segment. Any flow accumulated earlier
      // inside the same 250 ms bin must not leak across the reset boundary.
      flow.clear();
      book.clear();
      continuity_valid = false;
      ++snapshots;
    }

    for (const Row& row : group) book.apply(row);

    if (group_snapshot) {
      book.ready = book.structurally_valid();
      continuity_valid = book.ready;
      if (!book.ready) ++integrity_latches;
    } else if (book.ready && !book.structurally_valid()) {
      book.ready = false;
      continuity_valid = false;
      flow.clear();
      ++integrity_latches;
    }

    if (before.valid() && book.valid() && !group_snapshot && continuity_valid) {
      flow.ofi_l1 += ofi(before, book, 1);
      flow.mlofi_l5 += ofi(before, book, 5);
      flow.mlofi_l10 += ofi(before, book, 10);
      replenish_deplete_top5(before.bids, book.bids,
                             flow.bid_replenish_l5, flow.bid_deplete_l5);
      replenish_deplete_top5(before.asks, book.asks,
                             flow.ask_replenish_l5, flow.ask_deplete_l5);
    }

    ++groups;

    while (next_grid <= group_ts && next_grid < day_end) emit();

    group.clear();
    group_snapshot = false;
  };

  while (input.getline(line)) {
    ++parsed_rows;
    Row row;
    if (!parse_row(line, row)) {
      ++bad_rows;
      continue;
    }
    if (row.local_ts < day_start || row.local_ts >= day_end) {
      ++bad_rows;
      continue;
    }
    if (row.local_ts < previous_local) {
      std::cerr << "local timestamp regression at row " << parsed_rows << '\n';
      return 3;
    }
    previous_local = row.local_ts;

    if (group_ts == std::numeric_limits<int64_t>::min()) group_ts = row.local_ts;
    if (row.local_ts != group_ts) {
      flush_group();
      group_ts = row.local_ts;
    }

    group_snapshot = group_snapshot || row.snapshot;
    group.push_back(row);
  }

  flush_group();
  while (next_grid < day_end) emit();

  std::cerr << "parsed_rows=" << parsed_rows
            << " bad_rows=" << bad_rows
            << " groups=" << groups
            << " snapshots=" << snapshots
            << " integrity_latches=" << integrity_latches
            << " emitted=" << emitted << '\n';

  return bad_rows == 0 && emitted == EXPECTED_EMITTED ? 0 : 4;
}
