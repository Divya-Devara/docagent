# Evaluation: results/eval_v2.md

Receipts: 50 (first 50 train IDs). Extraction failures: 0.

| Field | Scored | Exact | Fuzzy (>= 0.9) |
|---|---|---|---|
| company | 50 | 82% | 88% |
| date | 50 | 96% | 100% |
| address | 50 | 42% | 96% |
| total | 50 | 96% | 96% |

Latency per receipt: p50 5.4s, p95 6.6s

## Mistakes (fuzzy match failed)

| Receipt | Field | Predicted | Truth |
|---|---|---|---|
| X00016469620 | address | LOT 1851-A & 1851-B, JALAN KPB 6, KAWASAN PERINDUSTRIAN BALAKONG, 43300 SERI KEMBANGAN, SELANGOR | LOT 1851-A & 1851-B, JALAN KPB 6, KAWASAN PERINDUSTRIAN BALAKONG, 43300 SERI KEMBANGAN, SELANGOR (MR DIY TESCO TERBAU) |
| X00016469623 | company | TAN WOON YANN | MR D.I.Y. (M) SDN BHD |
| X51005361895 | total | 8.49 | 9.00 |
| X51005361950 | company | 99 SPEED MART S/B (519537-X) | 99 SPEED MART S/B |
| X51005433494 | company | AEON CO. (M) BHD (126926-H) | AEON CO. (M) BHD |
| X51005433522 | total | 8.2 |  |
| X51005441398 | company | BREWERY TAP DWNER BY CASTLE BLUE S/B | OWNER BY CASTLE BLUE S/B |
| X51005441401 | company | THREE STOOGES BISTRO & CAFE | THREE STOOGES |
| X51005441401 | address | 109, SS21/1A, DAMANSARA UTAMA GST ID : 001661886464 | 109, SS21/1A, DAMANSARA UTAMA |
| X51005441402 | company | THREE STOOGES BISTRO & CAFE | THREE STOOGES |
