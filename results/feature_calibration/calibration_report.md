# Presplit Feature-Parameter Calibration

This report contains feature-only calibration evidence. No classifier or experiment metric was used.

## Mutual Information

Selected k: `10`  
Knee method: `maximum_distance_to_endpoint_chord_normalized`  
Maximum chord distance: `0.253865910271`

Selected features: sttl, sbytes, dttl, sload, smeansz, ct_state_ttl, ct_dst_sport_ltm, ct_src_dport_ltm, dbytes, dmeansz

| Rank | Feature | MI score |
|---:|---|---:|
| 7 | ct_dst_sport_ltm | 0.330583112439 |
| 8 | ct_src_dport_ltm | 0.287608572886 |
| 9 | dbytes | 0.267645218633 |
| 10 | dmeansz | 0.245817149952 |
| 11 | dsport | 0.242587947808 |
| 12 | dpkts | 0.229850050933 |
| 13 | dur | 0.225949219697 |

## PCA

Selected components: `5`  
Cumulative variance at selection: `0.736035827547`  
Frozen representation recommendation: `component_count`.

| Component | Individual variance | Cumulative variance |
|---:|---:|---:|
| 2 | 0.119367547333 | 0.506537534297 |
| 3 | 0.105931892991 | 0.612469427288 |
| 4 | 0.070269100368 | 0.682738527656 |
| 5 | 0.0532972998917 | 0.736035827547 |
| 6 | 0.0472193881869 | 0.783255215734 |
| 7 | 0.0403767041862 | 0.82363191992 |
| 8 | 0.0319428741932 | 0.855574794114 |

## Spearman

Selected k: `4`  
Historical k: `4`  
Historical result reproduced: `True`

Selected features: ct_state_ttl, sttl, ct_dst_sport_ltm, state

| Rank | Feature | Absolute correlation |
|---:|---|---:|
| 1 | ct_state_ttl | 0.841011023034 |
| 2 | sttl | 0.682816449789 |
| 3 | ct_dst_sport_ltm | 0.475795749117 |
| 4 | state | 0.44678001834 |
| 5 | dload | 0.442892667624 |
| 6 | dttl | 0.407613617547 |
| 7 | ct_src_dport_ltm | 0.406147436554 |
