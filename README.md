# 申诉主题分类的分案边界与选型

各表为本地合成业务记录，身份与机构代号不对应真实对象。

## 资料格式

资料为 UTF-8 编码、首行为列名的 CSV。空字段表示没有记录，不表示零。连续日号采用同一份内部日历。

| 文件 | 列 | 登记口径 |
| --- | --- | --- |
| `letters.csv` | `letter`、`arrived`、`channel`、`expression`、`urgent`、`text` | 本地合成文本；expression 由来件渠道登记是模板还是自由填写，urgent 由受理员标记。 |
| `forwarding.csv` | `copy`、`parent`、`known`、`state` | 随时间确认的转发来源，根来件不由文本是否完全相同决定。 |
| `labels.csv` | `letter`、`topic`、`policy`、`known`、`verdict` | 仅记人工逐项判定，空缺不是阴性；retracted 取消同项之前的结论。 |
| `topics.csv` | `policy`、`topic`、`parent`、`starts`、`desk` | v2 沿用上位类别，但该主题单独出现尚不能确定具体承办范围。 |
| `limits.csv` | `desk`、`known`、`from_day`、`through`、`quota`、`tolerance`、`min_support` | 按时段给名额、可接受误送率、独立验证至少支持的件数，人工补读使用 manual。 |
| `assignments.csv` | `letter`、`topic`、`desk`、`issued`、`ack`、`state` | 本地分送和承办桌签收回执；签收不等于人工标签。 |
| `windows.csv` | `name`、`train_before`、`select_from`、`select_to`、`audit_from`、`audit_to`、`fresh_from` | 半开训练截点，其他结束日含当天；window 仅用于约定哪批案件可参与选择和独立复核。 |

## 检查资料

使用 Python 3.11 和标准库。在项目根目录运行：

```sh
python3 -m unittest discover -s tests -v
```

这项检查核对资料格式与引用，不生成业务结论。
