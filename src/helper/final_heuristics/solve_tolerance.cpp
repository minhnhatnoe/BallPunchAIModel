#define _GLIBCXX_DEBUG
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef long double ld;
#define endl "\n"

int main(int argc, char **argv)
{
    cin.tie(0)->sync_with_stdio(0);
    freopen(argv[1], "r", stdin);
    freopen(argv[2], "w", stdout);
    pair<string, ld> tmp;
    vector<pair<string, ld>> v;
    while (cin >> tmp.first >> tmp.second)
    {
        if (tmp.second == 0)
            break;
        v.push_back(move(tmp));
    }
    vector<pair<pair<string, ld>, ll>> cur;
    for (int i = 0; i < v.size() + 1; i++)
    {
        if ((i < v.size() && i > 0 && v[i].second <= 0 && v[i - 1].second > 0) || i == v.size())
        {
            if (cur.size() == 3)
            {
                for (auto frame : cur)
                {
                    v[frame.second].second = 1;
                }
            }
            else if (cur.size() == 4)
            {
                if ((cur[0].first.second <= 0 && cur[3].first.second <= 0) || (cur[0].first.second > 0 && cur[3].first.second > 0))
                {
                    ll pos = cur[0].second;
                    if (cur[3].first.second > cur[0].first.second)
                        pos = cur[3].second;
                    v[pos].second = 1;
                    v[cur[1].second].second = 1;
                    v[cur[2].second].second = 1;
                    v[cur[0].second + cur[3].second - pos].second = 0;
                }
                else
                {
                    for (auto frame : cur)
                    {
                        if (frame.first.second <= 0)
                            v[frame.second].second = 0;
                        else
                            v[frame.second].second = 1;
                    }
                }
            }
            else
            {
                ll r = cur.size() - 1, l = 0;
                if (cur[r].first.second <= 0)
                    r--;
                if (cur[l].first.second <= 0)
                    l++;
                if (r - l + 1 > 6)
                {
                    while (r - l + 1 > 6)
                    {
                        ll minn = min((r - l + 1) / 2, 3ll), maxn = min((r - l + 1) / 2, 6ll);
                        ll dif = rand() % (maxn - minn + 1) + minn;
                        v[cur[l + dif].second].second = 0;
                        l = l + dif + 1;
                    }
                    i = i - cur.size() - 1;
                }
                else
                {
                    ll r = cur.size() - 1, l = 0;
                    if (cur[r].first.second <= 0)
                        r--;
                    if (cur[l].first.second <= 0)
                        l++;
                    while (r - l + 1 > 3)
                    {
                        if (cur[r].first.second < cur[l].first.second)
                            r--;
                        else
                            l++;
                    }
                    for (int i = 0; i < cur.size(); i++)
                    {
                        if (i >= l && i <= r)
                            v[cur[i].second].second = 1;
                        else
                            v[cur[i].second].second = 0;
                    }
                }
            }
            cur.clear();
        }
        else if (i < v.size() && (v[i].second > 0 || (i < v.size() - 1 && v[i].second <= 0 && v[i + 1].second > 0)))
            cur.push_back({v[i], i});
        else if (i < v.size() && v[i].second <= 0 && cur.size() == 0)
            v[i].second = 0;
    }
    cout << "Frame,Label\n";
    for (auto i : v)
    {
        cout << i.first << "," << i.second << "\n";
    }
}