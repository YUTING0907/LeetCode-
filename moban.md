# 1.二分法
```
// 模板：寻找目标值（左闭右开区间）
int binarySearch(int[] nums, int target) {
    int left = 0, right = nums.length; 
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid;
    }
    return -1;
}
```

# 2. 滑动窗口（最长/最小子串）
```
// 模板：动态维护窗口
int slidingWindow(String s, String t) {
    Map<Character, Integer> need = new HashMap<>(), window = new HashMap<>();
    for (char c : t.toCharArray()) need.put(c, need.getOrDefault(c, 0) + 1);
    int left = 0, right = 0, valid = 0;
    int res = Integer.MAX_VALUE;

    while (right < s.length()) {
        char c = s.charAt(right++);
        if (need.containsKey(c)) {
            window.put(c, window.getOrDefault(c, 0) + 1);
            if (window.get(c).equals(need.get(c))) valid++;
        }

        while (valid == need.size()) { // 满足条件收缩
            res = Math.min(res, right - left);
            char d = s.charAt(left++);
            if (need.containsKey(d)) {
                if (window.get(d).equals(need.get(d))) valid--;
                window.put(d, window.get(d) - 1);
            }
        }
    }
    return res == Integer.MAX_VALUE ? -1 : res;
}
```
# 3. 链表反转
```
// 迭代反转链表
ListNode reverse(ListNode head) {
    ListNode prev = null, cur = head;
    while (cur != null) {
        ListNode nxt = cur.next;
        cur.next = prev;
        prev = cur;
        cur = nxt;
    }
    return prev;
}
```
# 4. BFS（层序遍历）
```
// BFS 模板
List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> res = new ArrayList<>();
    if (root == null) return res;
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);

    while (!q.isEmpty()) {
        int size = q.size();
        List<Integer> level = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            TreeNode node = q.poll();
            level.add(node.val);
            if (node.left != null) q.offer(node.left);
            if (node.right != null) q.offer(node.right);
        }
        res.add(level);
    }
    return res;
}
```
# 5. DFS（递归遍历）
```
// DFS 遍历二叉树
void dfs(TreeNode root) {
    if (root == null) return;
    dfs(root.left);
    dfs(root.right);
}
```
# 6. 图的 DFS / BFS
```
// 图 DFS
void dfsGraph(int u, boolean[] visited, List<Integer>[] g) {
    visited[u] = true;
    for (int v : g[u]) {
        if (!visited[v]) dfsGraph(v, visited, g);
    }
}

// 图 BFS
void bfsGraph(int start, List<Integer>[] g) {
    Queue<Integer> q = new LinkedList<>();
    boolean[] visited = new boolean[g.length];
    q.offer(start); visited[start] = true;
    while (!q.isEmpty()) {
        int u = q.poll();
        for (int v : g[u]) {
            if (!visited[v]) {
                visited[v] = true;
                q.offer(v);
            }
        }
    }
}
```
# 7. 单调栈
```
// 模板：找下一个更大元素
int[] nextGreater(int[] nums) {
    int n = nums.length;
    int[] res = new int[n];
    Arrays.fill(res, -1);
    Deque<Integer> st = new ArrayDeque<>(); // 存索引
    for (int i = 0; i < n; i++) {
        while (!st.isEmpty() && nums[i] > nums[st.peek()]) {
            res[st.pop()] = nums[i];
        }
        st.push(i);
    }
    return res;
}
```
# 8. 动态规划（背包）
```
// 0-1 背包
int knapSack(int W, int[] w, int[] v) {
    int n = w.length;
    int[] dp = new int[W + 1];
    for (int i = 0; i < n; i++) {
        for (int j = W; j >= w[i]; j--) {
            dp[j] = Math.max(dp[j], dp[j - w[i]] + v[i]);
        }
    }
    return dp[W];
}
```
# 9. 动态规划（序列型）
```
// 最长递增子序列 LIS
int lengthOfLIS(int[] nums) {
    int[] dp = new int[nums.length];
    int len = 0;
    for (int x : nums) {
        int i = Arrays.binarySearch(dp, 0, len, x);
        if (i < 0) i = -(i + 1);
        dp[i] = x;
        if (i == len) len++;
    }
    return len;
}
```
# 10. LCA 最近公共祖先
```
TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) return root;
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    if (left != null && right != null) return root;
    return left != null ? left : right;
}
```
