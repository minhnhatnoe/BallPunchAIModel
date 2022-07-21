#define _GLIBCXX_DEBUG
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef long double ld;
#define endl "\n"

signed main(){
    cin.tie(0)->sync_with_stdio(0);
    freopen("data.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    pair<string, ld> tmp;
    vector<pair<string, ld>> a;
    while (cin >> tmp.first >> tmp.second){
        a.push_back(move(tmp));
    }

}