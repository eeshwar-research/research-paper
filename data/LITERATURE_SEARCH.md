# Literature search and citation verification

Documented reference set for the *Knowledge-Based Systems* short communication. This is **not** a Scopus/Web of Science systematic review; those proprietary databases were not available here. Records were retrieved from Crossref, DataCite, publisher landing pages (Elsevier ScienceDirect, IEEE Xplore), and RePEc, then checked against the DOI resolver on 23 October 2026.

## Protocol
- **Date:** 23 October 2026
- **Sources:** Crossref API, DataCite (Zenodo), IEEE Xplore, ScienceDirect/RePEc
- **Time window:** 2016–2026 for methods and TOD; knowledge-graph surveys from 2016–2022
- **Queries:** firm-specific technology opportunity discovery; patent knowledge graph; link prediction CPC/F-term; temporal / leakage-aware evaluation; knowledge-graph embedding surveys
- **Inclusion:** English peer-reviewed articles that (i) identify firm- or patent-level technology opportunities, (ii) use graphs or link prediction on patents, or (iii) survey knowledge-graph / graph-embedding methods used in the design rationale
- **Publisher screen:** ScienceDirect (Elsevier) and IEEE only, after an all-Q1 filter
- **Exclusion:** records whose DOI did not resolve to the cited title; conference papers; journals that are not Q1 in a field-relevant SCImago category

## Publisher and quartile screen
The manuscript cites **23 journal articles** plus **2 data/web sources**. Quartiles use SCImago best field-relevant quartile (2024). The GitHub archive is linked from the data-availability statement but is not a numbered reference, because it has no DOI.

| Publisher | Count | Journals (field-relevant best quartile) |
|---|---:|---|
| ScienceDirect / Elsevier | 18 | *Technol. Forecast. Soc. Change* (Q1); *Technovation* (Q1); *J. Informetr.* (Q1); *Adv. Eng. Inform.* (Q1); *Expert Syst. Appl.* (Q1); *Knowl.-Based Syst.* (Q1) |
| IEEE | 5 | *IEEE Trans. Eng. Manag.* (Q1); *IEEE Trans. Knowl. Data Eng.* (Q1); *IEEE Trans. Neural Netw. Learn. Syst.* (Q1) |
| Data / web | 2 | Zenodo PatentsView dump; USPTO PatentsView page |

**MDPI exclusion (not assumed away):** *Sustainability* is Q1 only in Geography, Planning and Development and Q2 in most computer-science and environmental categories. *Electronics* is Q2. *Applied Sciences* and *Big Data Cogn. Comput.* were inspected and not cited. No relevant MDPI journal cleared the all-Q1 bar for this topic, so MDPI titles were removed rather than forced into the list.

## Verified journal articles (Crossref metadata)

| # | Paper | DOI | Role |
|---|---|---|---|
| 1 | Lee, Ko, Yoon, Son (2021), *TFSC* 168, 120746 | 10.1016/j.techfore.2021.120746 | Firm-specific TOD via F-term link prediction |
| 2 | Park & Geum (2022), *TFSC* 183, 121934 | 10.1016/j.techfore.2022.121934 | GCN link prediction + firm-level validation |
| 3 | Choi, Lee, Yoon (2023), *TFSC* 186, 122161 | 10.1016/j.techfore.2022.122161 | Heterogeneous knowledge-graph TOD |
| 4 | Seol et al. (2023), *JOI* 17(4), 101464 | 10.1016/j.joi.2023.101464 | Firm-specific TOD from portfolio graphs / rules |
| 5 | Wu & Ji (2023), *JOI* 17(2), 101398 | 10.1016/j.joi.2023.101398 | Competitor-aware firm-specific TOD |
| 6 | Wu, Lin, Ji, Gu (2026), *Technovation* 151, 103405 | 10.1016/j.technovation.2025.103405 | Heterogeneous GNN firm-specific TOD |
| 7 | Zhao, Wei, Zhu (2025), *TFSC* 221, 124305 | 10.1016/j.techfore.2025.124305 | ML prediction of firm technology opportunities |
| 8 | Park & Yoon (2017), *TFSC* 118, 170–183 | 10.1016/j.techfore.2017.02.018 | Portfolio-based TOD + collaborative filtering |
| 9 | Liu, Feng, Uden (2023), *Technovation* 128, 102872 | 10.1016/j.technovation.2023.102872 | Hierarchical semantic networks + dual link prediction |
| 10 | Park & Yoon (2018), *JOI* 12(4), 1199–1222 | 10.1016/j.joi.2018.09.007 | Convergence TOD via knowledge-flow link prediction |
| 11 | Feng, Li, Wang (2026), *AEI* 76, 104850 | 10.1016/j.aei.2026.104850 | Firm-specific patent recommendation |
| 12 | Chang et al. (2025), *AEI* 66, 103498 | 10.1016/j.aei.2025.103498 | GAT link-prediction TOD |
| 13 | Wang et al. (2024), *AEI* 62, 102944 | 10.1016/j.aei.2024.102944 | From opportunities to solutions via ML link prediction |
| 14 | Geum & Kim (2020), *AEI* 46, 101155 | 10.1016/j.aei.2020.101155 | Keygraph CPC opportunity identification |
| 15 | Zhao, Li, Xi, Liu (2026), *TFSC* 229, 124701 | 10.1016/j.techfore.2026.124701 | Firm-level convergence on heterogeneous networks |
| 16 | Rodriguez et al. (2016), *IEEE TEM* 63(4), 426–437 | 10.1109/TEM.2016.2580619 | Citation-network TOD |
| 17 | Liu, Wang, Feng (2025), *IEEE TEM* 72, 2097–2115 | 10.1109/TEM.2025.3570659 | Firm-specific patent recommendation + topology link prediction |
| 18 | Sarica, Luo, Wood (2020), *ESWA* 142, 112995 | 10.1016/j.eswa.2019.112995 | TechNet patent semantic network |
| 19 | Wang, Mao, Wang, Guo (2017), *IEEE TKDE* 29(12), 2724–2743 | 10.1109/TKDE.2017.2754499 | Knowledge-graph embedding survey |
| 20 | Ji et al. (2022), *IEEE TNNLS* 33(2), 494–514 | 10.1109/TNNLS.2021.3070843 | Knowledge-graph survey |
| 21 | Goyal & Ferrara (2018), *KBS* 151, 78–94 | 10.1016/j.knosys.2018.03.022 | Graph-embedding survey (same journal) |
| 22 | Yang & Dong (2020), *KBS* 204, 106194 | 10.1016/j.knosys.2020.106194 | Explainable KG recommendation |
| 23 | Shimizu, Matsutani, Goto (2022), *KBS* 239, 107970 | 10.1016/j.knosys.2021.107970 | Explainable KG attention recommendation |
| 24 | USPTO PatentsView dump (2025) [dataset] | 10.5281/zenodo.15058362 | Public granted-patent metadata (DataCite) |
| 25 | USPTO PatentsView page (2026) | uspto.gov/.../patentsview | Current download location / coverage note |

## Corrections relative to earlier drafts
- Wu, Lin, Ji, Gu (2026) remains in **_Technovation_ 151, 103405**.
- Four *Sustainability* papers were removed because the journal is not Q1 in field-relevant categories.
- Ren & Zhao (2021) *Technovation* and Liu, Feng, Uden (2023) *TFSC* 192 were dropped in the short-communication cut; their role is covered by remaining TOD citations.
- Nickel et al. (2016) *Proc. IEEE* and Cai et al. (2018) *IEEE TKDE* were dropped to keep the short communication compact; Wang (2017) and Ji (2022) remain as the KG surveys.
- Park & Yoon (2017) *TFSC*, Feng et al. (2026) *AEI*, Chang et al. (2025) *AEI*, Rodriguez et al. (2016) *IEEE TEM*, and Sarica et al. (2020) *ESWA* were added as Q1 replacements.
- Every bibliography entry is cited in the text; numbering follows first appearance (Elsevier numbered style).
- The PatentsView dump is marked `[dataset]` as required by the KBS data-reference rule.
- Wang et al. (2024) *AEI* 62, 102944 includes Zhenghong Liu as the seventh author (Crossref, 23 October 2026).
- The GitHub archive is linked from Data availability; it is not a numbered software citation and does not replace a DOI deposit of the processed table.

## Gap after the expanded set
Prior firm-specific TOD studies typically validate opportunities for one or a few target firms, use random or unspecified splits, or evaluate GNN/link-prediction quality without a leakage-controlled, many-firm, out-of-time comparison of inspectable graph features against tabular baselines. That remains the scoped contribution of this short communication.
