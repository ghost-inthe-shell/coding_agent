#include <cstdio>

using namespace std;
const int N = 100010;
int arr[N];
int n;

int main() {
    int q;
    scanf("%d%d", &n, &q);
    for (int i = 0; i < n; ++i) {
        scanf("%d", &arr[i]);
    }
    while (q--) {
        int k;
        scanf("%d", &k);
        if (arr[0] > k || arr[n - 1] < k) {
            printf("-1 -1\n");
            continue;
        }
        int l = 0, r = n;
        while (l + 1 < r) {
            int mid = l + r >> 1;
            if (arr[mid] >= k) {
                r = mid;
            } else {
                l = mid;
            }
        }
        int terminal = l;
        l = -1, r = n - 1;
        while (l + 1 < r) {
            int mid = l + r >> 1;
            if (arr[mid] >= k) {
                r = mid;
            } else {
                l = mid;
            }
        }
        int start = r;
        if (arr[start] != k || arr[terminal] != k) {
            printf("-1 -1\n");
        } else {
            printf("%d %d\n", start, terminal);
        }
    }

    return 0;
}
