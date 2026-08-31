# TTL Cache

`TTLCache` 保存带有效期的键值：

- `put(key, value, ttl_seconds)` 写入值；`ttl_seconds` 必须大于 0。
- 再次写入同一个键会同时替换值并重新计算有效期。
- `get(key)` 返回仍有效的值，不存在或已经到期时抛出 `KeyError`。
- 到期时间为 `t` 的条目只在当前时间严格小于 `t` 时有效；当前时间恰好为 `t` 时已经到期。
- `len(cache)` 只统计仍有效的条目。

构造函数接收可选的 `clock`，测试可以使用可控时钟，不需要真实等待。

运行测试：

```bash
python3 -m unittest -v
```
