#include <zlib.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

class GzReader {
 public:
  explicit GzReader(const char* path): f_(gzopen(path,"rb")), buf_(1<<20) {
    if (!f_) throw std::runtime_error("cannot open gzip input");
  }
  ~GzReader(){ if(f_) gzclose(f_); }
  bool getline(std::string& out){
    out.clear();
    for(;;){
      char* p=gzgets(f_,buf_.data(),static_cast<int>(buf_.size()));
      if(!p) return !out.empty();
      size_t n=std::strlen(p); out.append(p,n);
      if(n && p[n-1]=='\n'){
        while(!out.empty()&&(out.back()=='\n'||out.back()=='\r')) out.pop_back();
        return true;
      }
      if(gzeof(f_)) return !out.empty();
    }
  }
 private:
  gzFile f_=nullptr; std::vector<char> buf_;
};

bool parse_snapshot(const std::string& line, int64_t& local_ts, bool& snapshot){
  std::array<std::pair<size_t,size_t>,8> f{}; size_t start=0,k=0;
  for(size_t i=0;i<=line.size();++i){
    if(i==line.size()||line[i]==','){
      if(k>=f.size()) return false;
      f[k++]={start,i-start}; start=i+1;
    }
  }
  if(k!=8) return false;
  try{ local_ts=std::stoll(line.substr(f[3].first,f[3].second)); }
  catch(...){ return false; }
  const std::string flag=line.substr(f[4].first,f[4].second);
  if(flag=="true") snapshot=true;
  else if(flag=="false") snapshot=false;
  else return false;
  return true;
}

int main(int argc,char** argv){
  if(argc!=5){ std::cerr<<"usage: snapshot_scan INPUT OUTPUT DAY_START_US DAY_END_US\n"; return 2; }
  const int64_t ds=std::stoll(argv[3]), de=std::stoll(argv[4]);
  if(de-ds!=86400000000LL) return 2;
  GzReader in(argv[1]); std::ofstream out(argv[2]); if(!out) return 2;
  std::string line; if(!in.getline(line)) return 2;
  if(line!="exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount") return 2;
  out<<"local_timestamp_us\n";
  uint64_t rows=0,bad=0,snapshot_rows=0,snapshot_groups=0;
  int64_t prev=std::numeric_limits<int64_t>::min();
  int64_t last_written=std::numeric_limits<int64_t>::min();
  while(in.getline(line)){
    ++rows; int64_t ts=0; bool snap=false;
    if(!parse_snapshot(line,ts,snap)){ ++bad; continue; }
    if(ts<ds||ts>=de){ ++bad; continue; }
    if(ts<prev){ std::cerr<<"local timestamp regression at row "<<rows<<"\n"; return 3; }
    prev=ts;
    if(snap){
      ++snapshot_rows;
      if(ts!=last_written){ out<<ts<<'\n'; last_written=ts; ++snapshot_groups; }
    }
  }
  std::cerr<<"parsed_rows="<<rows<<" bad_rows="<<bad<<" snapshot_rows="<<snapshot_rows
           <<" snapshot_groups="<<snapshot_groups<<"\n";
  return bad==0?0:4;
}
