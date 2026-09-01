# Project instructions

- 只使用 Python 标准库，不增加依赖。
- 保留现有的 `SlugError` 和 `make_slug(value: str) -> str` 公开接口。
- 输入不是字符串时抛出 `TypeError`。
- 使用 Unicode `casefold()` 转为小写形式。
- 每一段连续的非字母数字字符（包括空格和下划线）转换为一个连字符。
- 去掉开头和结尾的连字符；如果结果为空，抛出 `SlugError`。
- Unicode 字母和数字由 `str.isalnum()` 判断并保留，不要把实现限制为 ASCII。
- 不要修改现有测试文件。完成后运行 `python3 -m unittest -v`。
