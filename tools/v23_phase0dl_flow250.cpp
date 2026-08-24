#include <zlib.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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
using Bids=std::map<double,double,std::greater<double>>; using Asks=std::map<double,double>;
struct R{long long e=0,l=0;bool s=0,b=0;double p=0,q=0;};
struct B{Bids b;Asks a;bool ok=0;void clear(){b.clear();a.clear();ok=0;}void put(const R&r){if(r.b){if(r.q==0)b.erase(r.p);else b[r.p]=r.q;}else{if(r.q==0)a.erase(r.p);else a[r.p]=r.q;}}bool good()const{return ok&&!b.empty()&&!a.empty()&&a.begin()->first>b.begin()->first;}};
struct G{gzFile f;std::vector<char>x;G(const char*p):f(gzopen(p,"rb")),x(1<<20){if(!f)throw std::runtime_error("open");}~G(){gzclose(f);}bool get(std::string&o){o.clear();for(;;){char*p=gzgets(f,x.data(),x.size());if(!p)return !o.empty();size_t n=strlen(p);o.append(p,n);if(n&&p[n-1]=='\n'){while(!o.empty()&&(o.back()=='\n'||o.back()=='\r'))o.pop_back();return 1;}if(gzeof(f))return !o.empty();}}};
bool row(const std::string&z,R&r){std::array<std::string,8>f;size_t s=0,k=0;for(size_t i=0;i<=z.size();++i)if(i==z.size()||z[i]==','){if(k==8)return 0;f[k++]=z.substr(s,i-s);s=i+1;}if(k!=8)return 0;try{r.e=stoll(f[2]);r.l=stoll(f[3]);r.s=f[4]=="true";r.b=f[5]=="bid";if(f[5]!="bid"&&f[5]!="ask")return 0;r.p=stod(f[6]);r.q=stod(f[7]);}catch(...){return 0;}return std::isfinite(r.p)&&std::isfinite(r.q)&&r.p>0&&r.q>=0;}
template<class M>std::vector<std::pair<double,double>> top(const M&m,int n){std::vector<std::pair<double,double>>v;for(auto&i:m){if((int)v.size()==n)break;v.push_back(i);}return v;}
double lv(const std::vector<std::pair<double,double>>&x,const std::vector<std::pair<double,double>>&y,int i,bool bid){bool a=i<(int)x.size(),b=i<(int)y.size();if(!a&&!b)return 0;double p=a?x[i].first:0,q=a?x[i].second:0,P=b?y[i].first:0,Q=b?y[i].second:0;if(!a)return bid?Q:-Q;if(!b)return bid?-q:q;if(bid){if(P>p)return Q;if(P==p)return Q-q;return-q;}if(P<p)return-Q;if(P==p)return q-Q;return q;}
double ofi(const B&x,const B&y,int n){auto xb=top(x.b,n),xa=top(x.a,n),yb=top(y.b,n),ya=top(y.a,n);double v=0;for(int i=0;i<n;i++)v+=lv(xb,yb,i,1)+lv(xa,ya,i,0);return v;}
template<class M>void rd(const M&x,const M&y,double&r,double&d){std::set<double>s;int k=0;for(auto&i:x){if(k++==5)break;s.insert(i.first);}k=0;for(auto&i:y){if(k++==5)break;s.insert(i.first);}for(double p:s){double a=0,b=0;auto i=x.find(p);if(i!=x.end())a=i->second;auto j=y.find(p);if(j!=y.end())b=j->second;double z=b-a;if(z>0)r+=z;else d-=z;}}
struct F{double o=0,m5=0,m10=0,br=0,ar=0,bd=0,ad=0;void zero(){o=m5=m10=br=ar=bd=ad=0;}};
int main(int c,char**v){if(c!=5)return 2;long long ds=stoll(v[3]),de=stoll(v[4]),next=ds,gt=LLONG_MIN,prev=LLONG_MIN;G in(v[1]);std::ofstream out(v[2]);out<<"local_timestamp_us,ofi_l1_250ms,mlofi_l5_250ms,mlofi_l10_250ms,bid_replenish_l5_250ms,ask_replenish_l5_250ms,bid_deplete_l5_250ms,ask_deplete_l5_250ms,flow_valid\n";std::string z;if(!in.get(z))return 2;B book;std::vector<R>g;bool snap=0,cont=0;F f;unsigned long long rows=0,bad=0,groups=0,snaps=0,latch=0,em=0;auto emit=[&](){out<<next<<','<<f.o<<','<<f.m5<<','<<f.m10<<','<<f.br<<','<<f.ar<<','<<f.bd<<','<<f.ad<<','<<(cont&&book.good())<<'\n';f.zero();next+=250000;em++;};auto flush=[&](){if(g.empty())return;while(next<gt&&next<de)emit();B before=book;if(snap){book.clear();snaps++;}for(auto&r:g)book.put(r);if(snap){book.ok=!book.b.empty()&&!book.a.empty()&&book.a.begin()->first>book.b.begin()->first;cont=book.ok;}else if(book.ok&&(!book.good())){book.ok=0;cont=0;latch++;}if(before.good()&&book.good()&&!snap){f.o+=ofi(before,book,1);f.m5+=ofi(before,book,5);f.m10+=ofi(before,book,10);rd(before.b,book.b,f.br,f.bd);rd(before.a,book.a,f.ar,f.ad);}groups++;while(next<=gt&&next<de)emit();g.clear();snap=0;};while(in.get(z)){rows++;R r;if(!row(z,r)){bad++;continue;}if(r.l<ds||r.l>=de){bad++;continue;}if(r.l<prev)return 3;prev=r.l;if(gt==LLONG_MIN)gt=r.l;if(r.l!=gt){flush();gt=r.l;}snap=snap||r.s;g.push_back(r);}flush();while(next<de)emit();std::cerr<<"parsed_rows="<<rows<<" bad_rows="<<bad<<" groups="<<groups<<" snapshots="<<snaps<<" integrity_latches="<<latch<<" emitted="<<em<<"\n";return bad==0&&em==345600?0:4;}
