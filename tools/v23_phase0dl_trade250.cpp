#include <zlib.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t GRID_US=250000, DAY_US=86400000000LL, EXPECTED=345600;
class G{gzFile f=nullptr;std::vector<char>b;public:explicit G(const char*p):f(gzopen(p,"rb")),b(1<<20){if(!f)throw std::runtime_error("open");}~G(){if(f)gzclose(f);}bool get(std::string&o){o.clear();for(;;){char*p=gzgets(f,b.data(),(int)b.size());if(!p)return !o.empty();size_t n=std::strlen(p);o.append(p,n);if(n&&p[n-1]=='\n'){while(!o.empty()&&(o.back()=='\n'||o.back()=='\r'))o.pop_back();return true;}if(gzeof(f))return !o.empty();}}};
struct T{int64_t l=0;std::string side;double q=0;};
bool parse(const std::string&z,T&t){std::array<std::string,8>f;size_t s=0,k=0;for(size_t i=0;i<=z.size();++i)if(i==z.size()||z[i]==','){if(k>=8)return false;f[k++]=z.substr(s,i-s);s=i+1;}if(k!=8)return false;try{t.l=std::stoll(f[3]);t.side=f[5];t.q=std::stod(f[7]);}catch(...){return false;}return (t.side=="buy"||t.side=="sell"||t.side=="unknown")&&std::isfinite(t.q)&&t.q>=0;}
struct B{double bq=0,sq=0,uq=0;uint64_t bc=0,sc=0,uc=0;void clear(){bq=sq=uq=0;bc=sc=uc=0;}};
}
int main(int c,char**v){if(c!=5)return 2;const int64_t ds=std::stoll(v[3]),de=std::stoll(v[4]);if(de-ds!=DAY_US)return 2;G in(v[1]);std::ofstream out(v[2]);if(!out)return 2;out<<"local_timestamp_us,buy_qty_250ms,sell_qty_250ms,unknown_qty_250ms,buy_count_250ms,sell_count_250ms,unknown_count_250ms\n";std::string z;if(!in.get(z))return 2;if(z!="exchange,symbol,timestamp,local_timestamp,id,side,price,amount")return 2;B bin;int64_t next=ds;uint64_t rows=0,bad=0,em=0;auto emit=[&](){out<<next<<','<<bin.bq<<','<<bin.sq<<','<<bin.uq<<','<<bin.bc<<','<<bin.sc<<','<<bin.uc<<'\n';bin.clear();next+=GRID_US;++em;};while(in.get(z)){++rows;T t;if(!parse(z,t)){++bad;continue;}if(t.l<ds||t.l>=de){++bad;continue;}while(next<t.l&&next<de)emit();if(t.side=="buy"){bin.bq+=t.q;++bin.bc;}else if(t.side=="sell"){bin.sq+=t.q;++bin.sc;}else{bin.uq+=t.q;++bin.uc;}if(t.l==next&&next<de)emit();}while(next<de)emit();std::cerr<<"parsed_rows="<<rows<<" bad_rows="<<bad<<" emitted="<<em<<"\n";return bad==0&&em==EXPECTED?0:4;}
