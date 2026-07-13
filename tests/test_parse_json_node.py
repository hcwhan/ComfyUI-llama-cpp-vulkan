"""src/nodes/util/node_parse_json.py 的单元测试: 五种输出类型的转换边缘语义."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from src.nodes.util.node_parse_json import parse_json_node  # noqa: E402


class TestParseJsonNodeConversions(unittest.TestCase):
    def setUp(self):
        self.node = parse_json_node()

    def _run(self, input_str, key):
        any_val, string, integer, number, boolean = self.node.process(input_str, key)
        return any_val, string, integer, number, boolean

    def test_big_int_keeps_precision(self):
        # 回归: 超过 2^53 的大整数(如雪花 ID)不得经 float 中转丢精度
        snowflake = 9007199254740993  # 2^53 + 1
        _, string, integer, _, _ = self._run(f'{{"id": {snowflake}}}', "id")
        self.assertEqual(integer, snowflake)
        self.assertEqual(string, str(snowflake))

    def test_numeric_string_with_decimal(self):
        _, _, integer, number, _ = self._run('{"v": "1.5"}', "v")
        self.assertEqual(integer, 1)
        self.assertEqual(number, 1.5)

    def test_dict_string_output_is_valid_json(self):
        # 回归: dict/list 的 string 输出须为合法 JSON(双引号), 而非 Python repr
        _, string, _, _, _ = self._run('{"obj": {"a": "中文"}}', "obj")
        self.assertEqual(string, '{"a": "中文"}')

    def test_list_string_output_is_valid_json(self):
        _, string, _, _, _ = self._run('{"arr": [1, "x"]}', "arr")
        self.assertEqual(string, '[1, "x"]')

    def test_numeric_boolean_truthiness(self):
        # 数字按非零判定布尔
        self.assertTrue(self._run('{"v": 1}', "v")[4])
        self.assertFalse(self._run('{"v": 0}', "v")[4])
        self.assertTrue(self._run('{"v": 0.5}', "v")[4])

    def test_text_boolean_only_true_literal(self):
        self.assertTrue(self._run('{"v": "True"}', "v")[4])
        self.assertFalse(self._run('{"v": "yes"}', "v")[4])

    def test_bool_value_passthrough(self):
        _, string, integer, number, boolean = self._run('{"v": true}', "v")
        self.assertTrue(boolean)
        self.assertEqual(integer, 1)
        self.assertEqual(number, 1.0)
        self.assertEqual(string, "True")

    def test_unconvertible_falls_back_to_zero(self):
        _, _, integer, number, _ = self._run('{"v": "not a number"}', "v")
        self.assertEqual(integer, 0)
        self.assertEqual(number, 0.0)

    def test_infinity_int_falls_back_to_zero(self):
        # 回归: json.loads 接受 Infinity 字面量, int(inf) 抛 OverflowError
        # 须按 DESCRIPTION 契约回退 0 而非裸报错; float 输出保留 inf 原值
        any_val, _, integer, number, _ = self._run('{"v": Infinity}', "v")
        self.assertEqual(any_val, float("inf"))
        self.assertEqual(integer, 0)
        self.assertEqual(number, float("inf"))

    def test_missing_key_string_falls_back_to_empty(self):
        # 回归: key 未命中且未连 default 时 string 输出空串而非字面 "None"
        any_val, string, integer, number, boolean = self._run('{"a": 1}', "missing")
        self.assertIsNone(any_val)
        self.assertEqual(string, "")
        self.assertEqual(integer, 0)
        self.assertEqual(number, 0.0)
        self.assertFalse(boolean)


if __name__ == "__main__":
    unittest.main()
