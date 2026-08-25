#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t GRID_US = 250000;
constexpr int64_t DAY_US = 86400000000LL;
constexpr int EXPECTED_ROWS = 345600;
constexpr double EPS = 1e-9;

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  size_t start = 0;
  for (size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == ',') {
      out.emplace_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  return out;
}

double d(const std::string& s) { return std::stod(s); }
int64_t i64(const std::string& s) { return std::stoll(s); }
uint64_t u64(const std::string& s) { return std::stoull(s); }

struct BookRow {
  int64_t ts = 0;
  double bid = NAN, ask = NAN, bidq1 = NAN, askq1 = NAN, mid = NAN;
  double spread = NAN, micro = NAN, micro_bps = NAN;
  double bid5 = NAN, ask5 = NAN, bid10 = NAN, ask10 = NAN;
  double obi1 = NAN, obi5 = NAN, obi10 = NAN;
  bool valid = false;
};

BookRow parse_book(const std::string& line) {
  const auto f = split(line);
  if (f.size() != 18) throw std::runtime_error("BOOK250 width mismatch");
  BookRow r;
  r.ts = i64(f[0]);
  r.bid = d(f[2]); r.ask = d(f[3]); r.bidq1 = d(f[4]); r.askq1 = d(f[5]); r.mid = d(f[6]);
  r.spread = d(f[7]); r.micro = d(f[8]); r.micro_bps = d(f[9]);
  r.bid5 = d(f[10]); r.ask5 = d(f[11]); r.bid10 = d(f[12]); r.ask10 = d(f[13]);
  r.obi1 = d(f[14]); r.obi5 = d(f[15]); r.obi10 = d(f[16]); r.valid = i64(f[17]) == 1;
  return r;
}

struct FlowRow {
  int64_t ts = 0;
  double ofi1 = 0, m5 = 0, m10 = 0, br = 0, ar = 0, bd = 0, ad = 0;
  bool valid = false;
};

FlowRow parse_flow(const std::string& line) {
  const auto f = split(line);
  if (f.size() != 9) throw std::runtime_error("FLOW250 width mismatch");
  FlowRow r;
  r.ts = i64(f[0]); r.ofi1 = d(f[1]); r.m5 = d(f[2]); r.m10 = d(f[3]);
  r.br = d(f[4]); r.ar = d(f[5]); r.bd = d(f[6]); r.ad = d(f[7]); r.valid = i64(f[8]) == 1;
  return r;
}

struct TradeRow {
  int64_t ts = 0;
  double bq = 0, sq = 0, uq = 0;
  uint64_t bc = 0, sc = 0, uc = 0;
};

TradeRow parse_trade(const std::string& line) {
  const auto f = split(line);
  if (f.size() != 7) throw std::runtime_error("TRADE250 width mismatch");
  TradeRow r;
  r.ts = i64(f[0]); r.bq = d(f[1]); r.sq = d(f[2]); r.uq = d(f[3]);
  r.bc = u64(f[4]); r.sc = u64(f[5]); r.uc = u64(f[6]);
  return r;
}

double imbalance(double buy, double sell) {
  const double z = buy + sell;
  return z > 0.0 ? (buy - sell) / z : 0.0;
}

template <class T>
bool finite_all(const T& v) {
  for (double x : v) if (!std::isfinite(x)) return false;
  return true;
}

void write_nan(std::ostream& out, int n) {
  for (int i = 0; i < n; ++i) out << ",nan";
}

struct FlowAgg {
  double ofi1=0,m5=0,m10=0,br=0,ar=0,bd=0,ad=0;
  bool valid=true;
};

FlowAgg flow_last(const std::deque<FlowRow>& q, size_t n) {
  FlowAgg a;
  if (q.size() < n) { a.valid=false; return a; }
  for (size_t k=q.size()-n;k<q.size();++k) {
    const auto& r=q[k];
    a.valid = a.valid && r.valid;
    a.ofi1 += r.ofi1; a.m5 += r.m5; a.m10 += r.m10;
    a.br += r.br; a.ar += r.ar; a.bd += r.bd; a.ad += r.ad;
  }
  return a;
}

struct TradeAgg {
  double bq=0,sq=0; uint64_t bc=0,sc=0; bool valid=true;
};

TradeAgg trade_last(const std::deque<TradeRow>& q, size_t n) {
  TradeAgg a;
  if (q.size() < n) { a.valid=false; return a; }
  for (size_t k=q.size()-n;k<q.size();++k) {
    const auto& r=q[k]; a.bq += r.bq; a.sq += r.sq; a.bc += r.bc; a.sc += r.sc;
  }
  return a;
}

std::vector<bool> load_snapshot_bins(const std::string& path, int64_t day_start, int64_t day_end,
                                     uint64_t& snapshot_groups) {
  std::ifstream in(path); if(!in) throw std::runtime_error("cannot open snapshot index");
  std::string line; if(!std::getline(in,line) || line!="local_timestamp_us")
    throw std::runtime_error("unexpected snapshot header");
  std::vector<bool> bins(EXPECTED_ROWS,false);
  int64_t prev=std::numeric_limits<int64_t>::min(); snapshot_groups=0;
  while(std::getline(in,line)) {
    if(line.empty()) continue;
    int64_t ts=i64(line);
    if(ts<day_start||ts>=day_end||ts<prev) throw std::runtime_error("invalid snapshot timestamp");
    prev=ts; ++snapshot_groups;
    const int64_t off=ts-day_start;
    const int64_t idx=(off+GRID_US-1)/GRID_US; // first grid t >= snapshot timestamp
    if(idx>=0 && idx<EXPECTED_ROWS) bins[static_cast<size_t>(idx)]=true;
  }
  return bins;
}

void header(std::ostream& o) {
  o << "local_timestamp_us,best_bid,best_ask,mid,book_valid,l0_valid,l1_valid,l2_valid,"
    << "spread_bps,microprice_minus_mid_bps,obi_l1,obi_l5,obi_l10,"
    << "log_bid_qty_l1,log_ask_qty_l1,log_bid_depth_l5,log_ask_depth_l5,log_bid_depth_l10,log_ask_depth_l10,"
    << "ofi_l1_250ms,ofi_l1_1s,ofi_l1_3s,mlofi_l5_250ms,mlofi_l5_1s,mlofi_l5_3s,"
    << "mlofi_l10_250ms,mlofi_l10_1s,mlofi_l10_3s,"
    << "trade_qty_imbalance_250ms,trade_qty_imbalance_1s,trade_qty_imbalance_3s,"
    << "trade_count_imbalance_250ms,trade_count_imbalance_1s,trade_count_imbalance_3s,"
    << "d_obi_l1_250ms,d_obi_l1_1s,d_obi_l5_250ms,d_obi_l5_1s,d_obi_l10_250ms,d_obi_l10_1s,"
    << "d_spread_bps_250ms,d_spread_bps_1s,d_microprice_minus_mid_bps_250ms,d_microprice_minus_mid_bps_1s,"
    << "bid_replenish_l5_1s,ask_replenish_l5_1s,bid_deplete_l5_1s,ask_deplete_l5_1s,"
    << "trade_qty_imbalance_1s_x_obi_l5,trade_qty_imbalance_1s_x_microprice_minus_mid_bps,mlofi_l5_1s_x_spread_bps\n";
}

} // namespace

int main(int argc,char** argv) {
  if(argc!=9) {
    std::cerr << "usage: features250 BOOK FLOW TRADE SNAPSHOTS OUTPUT DAY_START_US DAY_END_US SYMBOL\n";
    return 2;
  }
  try {
    const int64_t ds=i64(argv[6]), de=i64(argv[7]);
    if(de-ds!=DAY_US) throw std::runtime_error("invalid day bounds");
    std::ifstream book(argv[1]), flow(argv[2]), trade(argv[3]);
    std::ofstream out(argv[5]);
    if(!book||!flow||!trade||!out) throw std::runtime_error("cannot open input/output");
    std::string hb,hf,ht;
    if(!std::getline(book,hb)||!std::getline(flow,hf)||!std::getline(trade,ht)) throw std::runtime_error("missing header");
    if(hb!="local_timestamp_us,exchange_timestamp_us,best_bid,best_ask,bid_qty_l1,ask_qty_l1,mid,spread_bps,microprice,microprice_minus_mid_bps,bid_depth_l5,ask_depth_l5,bid_depth_l10,ask_depth_l10,obi_l1,obi_l5,obi_l10,book_valid") throw std::runtime_error("unexpected BOOK250 header");
    if(hf!="local_timestamp_us,ofi_l1_250ms,mlofi_l5_250ms,mlofi_l10_250ms,bid_replenish_l5_250ms,ask_replenish_l5_250ms,bid_deplete_l5_250ms,ask_deplete_l5_250ms,flow_valid") throw std::runtime_error("unexpected FLOW250 header");
    if(ht!="local_timestamp_us,buy_qty_250ms,sell_qty_250ms,unknown_qty_250ms,buy_count_250ms,sell_count_250ms,unknown_count_250ms") throw std::runtime_error("unexpected TRADE250 header");

    uint64_t snapshot_groups=0;
    const auto snapshot_bin=load_snapshot_bins(argv[4],ds,de,snapshot_groups);
    header(out); out<<std::setprecision(12);

    std::deque<BookRow> bq;
    std::deque<FlowRow> fq;
    std::deque<TradeRow> tq;
    uint64_t l0_count=0,l1_count=0,l2_count=0,book_valid_count=0;
    uint64_t snapshot_masked_bins=0,unknown_count=0; double unknown_qty=0.0;
    uint64_t violations=0;

    std::string lb,lf,lt;
    for(int idx=0;idx<EXPECTED_ROWS;++idx) {
      if(!std::getline(book,lb)||!std::getline(flow,lf)||!std::getline(trade,lt)) throw std::runtime_error("premature EOF");
      BookRow b=parse_book(lb); FlowRow f=parse_flow(lf); TradeRow t=parse_trade(lt);
      const int64_t expected=ds+static_cast<int64_t>(idx)*GRID_US;
      if(b.ts!=expected||f.ts!=expected||t.ts!=expected) throw std::runtime_error("timestamp/grid mismatch at row "+std::to_string(idx));

      if(snapshot_bin[static_cast<size_t>(idx)]) { f.valid=false; ++snapshot_masked_bins; }
      unknown_count += t.uc; unknown_qty += t.uq;

      bool raw_ok=true;
      if(!finite_all(std::array<double,7>{f.ofi1,f.m5,f.m10,f.br,f.ar,f.bd,f.ad})) raw_ok=false;
      if(f.br < -EPS || f.ar < -EPS || f.bd < -EPS || f.ad < -EPS) raw_ok=false;
      if(!finite_all(std::array<double,3>{t.bq,t.sq,t.uq}) || t.bq < -EPS || t.sq < -EPS || t.uq < -EPS) raw_ok=false;
      if(!raw_ok) ++violations;

      bool book_ok=b.valid;
      if(b.valid) {
        ++book_valid_count;
        if(!finite_all(std::array<double,15>{b.bid,b.ask,b.bidq1,b.askq1,b.mid,b.spread,b.micro,b.micro_bps,b.bid5,b.ask5,b.bid10,b.ask10,b.obi1,b.obi5,b.obi10})) book_ok=false;
        if(!(b.bid>0.0 && b.ask>b.bid && b.mid>0.0 && b.spread>0.0)) book_ok=false;
        if(b.bidq1 < -EPS || b.askq1 < -EPS || b.bid5 < -EPS || b.ask5 < -EPS || b.bid10 < -EPS || b.ask10 < -EPS) book_ok=false;
        if(b.obi1 < -1.0-EPS || b.obi1 > 1.0+EPS || b.obi5 < -1.0-EPS || b.obi5 > 1.0+EPS || b.obi10 < -1.0-EPS || b.obi10 > 1.0+EPS) book_ok=false;
        if(b.micro < b.bid-1e-6 || b.micro > b.ask+1e-6) book_ok=false;
        if(!book_ok) ++violations;
      }

      std::array<double,11> l0{};
      if(book_ok) {
        l0={b.spread,b.micro_bps,b.obi1,b.obi5,b.obi10,
            std::log1p(b.bidq1),std::log1p(b.askq1),std::log1p(b.bid5),std::log1p(b.ask5),std::log1p(b.bid10),std::log1p(b.ask10)};
      }
      bool l0_valid=book_ok && finite_all(l0);

      bq.push_back(b); if(bq.size()>5) bq.pop_front();
      fq.push_back(f); if(fq.size()>13) fq.pop_front();
      tq.push_back(t); if(tq.size()>13) tq.pop_front();

      const FlowAgg f1=flow_last(fq,4), f3=flow_last(fq,12);
      const TradeAgg tr1=trade_last(tq,4), tr3=trade_last(tq,12);
      const double q250=imbalance(t.bq,t.sq), q1=imbalance(tr1.bq,tr1.sq), q3=imbalance(tr3.bq,tr3.sq);
      const double c250=imbalance(static_cast<double>(t.bc),static_cast<double>(t.sc));
      const double c1=imbalance(static_cast<double>(tr1.bc),static_cast<double>(tr1.sc));
      const double c3=imbalance(static_cast<double>(tr3.bc),static_cast<double>(tr3.sc));
      std::array<double,15> l1={f.ofi1,f1.ofi1,f3.ofi1,f.m5,f1.m5,f3.m5,f.m10,f1.m10,f3.m10,q250,q1,q3,c250,c1,c3};
      bool imbal_ok=true;
      for(double x:{q250,q1,q3,c250,c1,c3}) if(x < -1.0-EPS || x > 1.0+EPS) imbal_ok=false;
      // idx>=12 excludes the partial day-start bin and guarantees 12 complete 250 ms bins.
      bool l1_valid=l0_valid && idx>=12 && f3.valid && tr3.valid && finite_all(l1) && imbal_ok;

      std::array<double,17> l2{};
      bool lag_ok=false;
      if(bq.size()>=5) {
        const BookRow& lag1=bq[bq.size()-2];
        const BookRow& lag4=bq[bq.size()-5];
        lag_ok=lag1.valid&&lag4.valid;
        l2={b.obi1-lag1.obi1,b.obi1-lag4.obi1,b.obi5-lag1.obi5,b.obi5-lag4.obi5,
            b.obi10-lag1.obi10,b.obi10-lag4.obi10,b.spread-lag1.spread,b.spread-lag4.spread,
            b.micro_bps-lag1.micro_bps,b.micro_bps-lag4.micro_bps,
            f1.br,f1.ar,f1.bd,f1.ad,q1*b.obi5,q1*b.micro_bps,f1.m5*b.spread};
      }
      bool l2_valid=l1_valid && lag_ok && f1.valid && finite_all(l2);
      if(l0_valid) ++l0_count; if(l1_valid) ++l1_count; if(l2_valid) ++l2_count;
      if(l2_valid && !l1_valid) ++violations; if(l1_valid && !l0_valid) ++violations;

      out<<expected;
      if(book_ok) out<<','<<b.bid<<','<<b.ask<<','<<b.mid<<",1";
      else out<<",nan,nan,nan,0";
      out<<','<<l0_valid<<','<<l1_valid<<','<<l2_valid;
      if(l0_valid) for(double x:l0) out<<','<<x; else write_nan(out,11);
      if(l1_valid) for(double x:l1) out<<','<<x; else write_nan(out,15);
      if(l2_valid) for(double x:l2) out<<','<<x; else write_nan(out,17);
      out<<'\n';
    }
    if(std::getline(book,lb)||std::getline(flow,lf)||std::getline(trade,lt)) throw std::runtime_error("extra rows after expected grid");
    if(l0_count==0||l1_count==0||l2_count==0) ++violations;

    std::cerr<<"rows="<<EXPECTED_ROWS<<" book_valid="<<book_valid_count<<" l0_valid="<<l0_count
             <<" l1_valid="<<l1_count<<" l2_valid="<<l2_count<<" snapshot_groups="<<snapshot_groups
             <<" snapshot_masked_bins="<<snapshot_masked_bins<<" unknown_trades="<<unknown_count
             <<" unknown_qty="<<unknown_qty<<" violations="<<violations<<"\n";
    return violations==0?0:4;
  } catch(const std::exception& e) {
    std::cerr<<"FEATURE_ASSEMBLY_ERROR: "<<e.what()<<"\n";
    return 3;
  }
}
