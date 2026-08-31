# Tasker

一个只使用 Python 标准库的任务清单示例。

```bash
python3 -m tasker.cli --db tasks.json add "write tests"
python3 -m tasker.cli --db tasks.json list
```

运行测试：

```bash
python3 -m unittest discover -s tests -v
```
