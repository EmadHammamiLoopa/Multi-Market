#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t GRID_US = 250000;

struct Row {
  int64_t exchange_ts = 0;
  int64_t local_ts = 0;
  bool snapshot = false;
  bool bid = false;
  double price = 0.0;
  double amount = 0.0;
};

struct Book {
  std::map<double, double, std::greater<double>> bids;
  std::map<double, double> asks;
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
    if (bids.empty() || asks.empty()) return false;
    const double b = bids.begin()->first;
    const double a = asks.begin()->first;
    return std::isfinite(b) && std::isfinite(a) && b > 0.0 && a > b;
  }

  bool valid() const { return ready && structurally_valid(); }
};

bool parse_bool(const char* s, size_t n) {
  return n == 4 && (s[0] == 't' || s[0] == 'T');
}

bool parse_row(const std::string& line, Row& out) {
  std::array<std::pair<const char*, size_t>, 8> f{};
  size_t field = 0, start = 0;
  for (size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ',') {
      if (field >= f.size()) return false;
      f[field++] = {line.data() + start, i - start};
      start = i + 1;
    }
  }
  if (field != 8) return false;
  try {
    out.exchange_ts = std::stoll(std::string(f[2].first, f[2].second));
    out.local_ts = std::stoll(std::string(f[3].first, f[3].second));
    out.snapshot = parse_bool(f[4].first, f[4].second);
    const std::string side(f[5].first, f[5].second);
    if (side == "bid") out.bid = true;
    else if (side == "ask") out.bid = false;
    else return false;
    out.price = std::stod(std::string(f[6].first, f[6].second));
    out.amount = std::stod(std::string(f[7].first, f[7].second));
  } catch (...) {
    return false;
  }
  return std::isfinite(out.price) && std::isfinite(out.amount) && out.price > 0.0 && out.amount >= 0.0;
}

class GzLineReader {
 public:
  explicit GzLineReader(const std::string& path) {
    f_ = gzopen(path.c_str(), "rb");
    if (!f_) throw std::runtime_error("cannot open gzip input: " + path);
    buf_.resize(1 << 20);
  }
  ~GzLineReader() { if (f_) gzclose(f_); }

  bool getline(std::string& out) {
    out.clear();
    while (true) {
      char* p = gzgets(f_, buf_.data(), static_cast<int>(buf_.size()));
      if (!p) return !out.empty();
      size_t n = std::char_traits<char>::length(p);
      out.append(p, n);
      if (n && p[n - 1] == '\n') {
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
        return true;
      }
      if (gzeof(f_)) return !out.empty();
    }
  }

 private:
  gzFile f_ = nullptr;
  std::vector<char> buf_;
};

double imbalance(double b, double a) {
  const double t = b + a;
  return t > 0.0 ? (b - a) / t : 0.0;
}

struct Metrics {
  double best_bid = 0, best_ask = 0, bid_q1 = 0, ask_q1 = 0;
  double mid = 0, spread_bps = 0, microprice = 0, micro_bps = 0;
  double bid5 = 0, ask5 = 0, bid10 = 0, ask10 = 0;
  double obi1 = 0, obi5 = 0, obi10 = 0;
};

Metrics metrics(const Book& b) {
  Metrics m;
  auto bi = b.bids.begin();
  auto ai = b.asks.begin();
  m.best_bid = bi->first; m.bid_q1 = bi->second;
  m.best_ask = ai->first; m.ask_q1 = ai->second;
  m.mid = (m.best_bid + m.best_ask) / 2.0;
  m.spread_bps = (m.best_ask - m.best_bid) / m.mid * 10000.0;
  const double d = m.bid_q1 + m.ask_q1;
  m.microprice = d > 0.0 ? (m.best_ask * m.bid_q1 + m.best_bid * m.ask_q1) / d : m.mid;
  m.micro_bps = (m.microprice - m.mid) / m.mid * 10000.0;
  int k = 0;
  for (auto it = b.bids.begin(); it != b.bids.end() && k < 10; ++it, ++k) {
    if (k < 5) m.bid5 += it->second;
    m.bid10 += it->second;
  }
  k = 0;
  for (auto it = b.asks.begin(); it != b.asks.end() && k < 10; ++it, ++k) {
    if (k < 5) m.ask5 += it->second;
    m.ask10 += it->second;
  }
  m.obi1 = imbalance(m.bid_q1, m.ask_q1);
  m.obi5 = imbalance(m.bid5, m.ask5);
  m.obi10 = imbalance(m.bid10, m.ask10);
  return m;
}

void write_header(std::ofstream& o) {
  o << "local_timestamp_us,exchange_timestamp_us,best_bid,best_ask,bid_qty_l1,ask_qty_l1,mid,spread_bps,microprice,microprice_minus_mid_bps,bid_depth_l5,ask_depth_l5,bid_depth_l10,ask_depth_l10,obi_l1,obi_l5,obi_l10,book_valid\n";
}

void emit(std::ofstream& o, int64_t t, int64_t exch, const Book& b) {
  o << t << ',' << exch;
  if (!b.valid()) {
    for (int i = 0; i < 15; ++i) o << ",nan";
    o << ",0\n";
    return;
  }
  const Metrics m = metrics(b);
  o << std::setprecision(12)
    << ',' << m.best_bid << ',' << m.best_ask << ',' << m.bid_q1 << ',' << m.ask_q1
    << ',' << m.mid << ',' << m.spread_bps << ',' << m.microprice << ',' << m.micro_bps
    << ',' << m.bid5 << ',' << m.ask5 << ',' << m.bid10 << ',' << m.ask10
    << ',' << m.obi1 << ',' << m.obi5 << ',' << m.obi10 << ",1\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: v23_phase0dl_depth250 INPUT.csv.gz OUTPUT.csv DAY_START_US DAY_END_US\n";
    return 2;
  }
  const std::string input = argv[1];
  const std::string output = argv[2];
  const int64_t day_start = std::stoll(argv[3]);
  const int64_t day_end = std::stoll(argv[4]);
  if (day_end <= day_start || (day_end - day_start) != 86400000000LL) {
    std::cerr << "invalid day bounds\n";
    return 2;
  }

  GzLineReader r(input);
  std::ofstream out(output);
  if (!out) throw std::runtime_error("cannot open output: " + output);
  write_header(out);

  std::string line;
  if (!r.getline(line)) throw std::runtime_error("missing header");
  const std::string expected = "exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount";
  if (line != expected) throw std::runtime_error("unexpected header: " + line);

  Book book;
  int64_t current_group = std::numeric_limits<int64_t>::min();
  int64_t last_exchange_ts = 0;
  bool group_snapshot = false;
  std::vector<Row> group;
  group.reserve(4096);
  int64_t next_grid = day_start;
  uint64_t parsed = 0, bad = 0, groups = 0, snapshots = 0, integrity_latches = 0, emitted = 0;

  auto flush_group = [&]() {
    if (group.empty()) return;
    while (next_grid < current_group && next_grid < day_end) {
      emit(out, next_grid, last_exchange_ts, book); ++emitted; next_grid += GRID_US;
    }
    if (group_snapshot) { book.clear(); ++snapshots; }
    for (const Row& x : group) {
      book.apply(x);
      last_exchange_ts = std::max(last_exchange_ts, x.exchange_ts);
    }
    if (group_snapshot) {
      book.ready = book.structurally_valid();
      if (!book.ready) ++integrity_latches;
    } else if (book.ready && !book.structurally_valid()) {
      book.ready = false;
      ++integrity_latches;
    }
    ++groups;
    while (next_grid <= current_group && next_grid < day_end) {
      emit(out, next_grid, last_exchange_ts, book); ++emitted; next_grid += GRID_US;
    }
    group.clear(); group_snapshot = false;
  };

  int64_t prev_local = std::numeric_limits<int64_t>::min();
  while (r.getline(line)) {
    ++parsed;
    Row x;
    if (!parse_row(line, x)) { ++bad; continue; }
    if (x.local_ts < day_start || x.local_ts >= day_end) { ++bad; continue; }
    if (x.local_ts < prev_local) {
      std::cerr << "local timestamp regression at row " << parsed << "\n";
      return 3;
    }
    prev_local = x.local_ts;
    if (current_group == std::numeric_limits<int64_t>::min()) current_group = x.local_ts;
    if (x.local_ts != current_group) {
      flush_group();
      current_group = x.local_ts;
    }
    group_snapshot = group_snapshot || x.snapshot;
    group.push_back(x);
  }
  flush_group();
  while (next_grid < day_end) { emit(out, next_grid, last_exchange_ts, book); ++emitted; next_grid += GRID_US; }

  std::cerr << "parsed_rows=" << parsed << " bad_rows=" << bad << " groups=" << groups
            << " snapshots=" << snapshots << " integrity_latches=" << integrity_latches
            << " emitted=" << emitted << "\n";
  if (bad != 0 || emitted != 345600) return 4;
  return 0;
}
