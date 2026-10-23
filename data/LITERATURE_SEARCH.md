# Literature search and citation verification

Documented expansion of the manuscript reference set. This is **not** a Scopus/Web of Science systematic review; those proprietary databases were not available here. Records were retrieved from Crossref, DataCite, publisher landing pages (Elsevier ScienceDirect, IEEE Xplore, MDPI), and RePEc, then checked against the DOI resolver on 23 October 2026.

## Protocol
- **Date:** 23 October 2026
- **Sources:** Crossref API, DataCite (Zenodo), MDPI, IEEE Xplore, ScienceDirect/RePEc
- **Time window:** 2016–2026 for methods and TOD; knowledge-graph surveys from 2016–2022
- **Queries:** firm-specific technology opportunity discovery; patent knowledge graph; link prediction CPC/F-term; temporal / leakage-aware evaluation; knowledge-graph embedding surveys
- **Inclusion:** English peer-reviewed articles that (i) identify firm- or patent-level technology opportunities, (ii) use graphs or link prediction on patents, or (iii) survey knowledge-graph / graph-embedding methods used in the design rationale
- **Exclusion:** records whose DOI did not resolve to the cited title; conference papers unless an IEEE journal version was verified; non-patent market-forecasting papers

## Publisher and quartile screen
The manuscript cites **26 journal articles** plus **2 data/web sources**. Journal articles are restricted to **MDPI, IEEE, and ScienceDirect (Elsevier)**. Quartiles below use SCImago best quartile (2024/2025) unless noted.

| Publisher | Count | Journals (best quartile) |
|---|---:|---|
| ScienceDirect / Elsevier | 17 | *Technol. Forecast. Soc. Change* (Q1); *Technovation* (Q1); *J. Informetr.* (Q1); *Adv. Eng. Inform.* (Q1); *Knowl.-Based Syst.* (Q1) |
| IEEE | 5 | *IEEE Trans. Eng. Manag.* (Q1); *IEEE Trans. Knowl. Data Eng.* (Q1); *IEEE Trans. Neural Netw. Learn. Syst.* (Q1); *Proc. IEEE* (Q1) |
| MDPI | 4 | *Sustainability* (SJR Q1 in Geography, Planning and Development; Q2 in several other categories) |
| Data / web | 2 | Zenodo PatentsView dump; USPTO PatentsView page |

**Quartile caveat (not assumed away):** *Sustainability* is Q1 in one SCImago category and Q2 in others. It is retained because the four papers are directly on patent-based technology-opportunity discovery. *Applied Sciences* (CiteScore Q1 / JCR Q2) and *Big Data Cogn. Comput.* were inspected and **not** cited, to avoid weaker quartile claims.

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
| 8 | Liu, Feng, Uden (2023), *Technovation* 128, 102872 | 10.1016/j.technovation.2023.102872 | Hierarchical semantic networks + dual link prediction |
| 9 | Ren & Zhao (2021), *Technovation* 101, 102196 | 10.1016/j.technovation.2020.102196 | Domain knowledge-network TOD |
| 10 | Park & Yoon (2018), *JOI* 12(4), 1199–1222 | 10.1016/j.joi.2018.09.007 | Convergence TOD via knowledge-flow link prediction |
| 11 | Shi, Cai, Song (2019), *Sustainability* 11(22), 6381 | 10.3390/su11226381 | Firm-level patent-portfolio TOD |
| 12 | Wang, Chi, Hsin (2018), *Sustainability* 10(10), 3729 | 10.3390/su10103729 | Patent maps for technological opportunities |
| 13 | Feng et al. (2020), *Sustainability* 12(1), 136 | 10.3390/su12010136 | Keyword/morphology TOD |
| 14 | Yi & Song (2025), *Sustainability* 17(8), 3607 | 10.3390/su17083607 | CPC co-classification link prediction |
| 15 | Wang et al. (2024), *AEI* 62, 102944 | 10.1016/j.aei.2024.102944 | From opportunities to solutions via ML link prediction |
| 16 | Geum & Kim (2020), *AEI* 46, 101155 | 10.1016/j.aei.2020.101155 | Keygraph CPC opportunity identification |
| 17 | Liu, Feng, Uden (2023), *TFSC* 192, 122565 | 10.1016/j.techfore.2023.122565 | Cross-cutting patent analysis + link prediction |
| 18 | Zhao, Li, Xi, Liu (2026), *TFSC* 229, 124701 | 10.1016/j.techfore.2026.124701 | Firm-level convergence on heterogeneous networks |
| 19 | Liu, Wang, Feng (2025), *IEEE TEM* 72, 2097–2115 | 10.1109/TEM.2025.3570659 | Firm-specific patent recommendation + topology link prediction |
| 20 | Wang, Mao, Wang, Guo (2017), *IEEE TKDE* 29(12), 2724–2743 | 10.1109/TKDE.2017.2754499 | Knowledge-graph embedding survey |
| 21 | Ji et al. (2022), *IEEE TNNLS* 33(2), 494–514 | 10.1109/TNNLS.2021.3070843 | Knowledge-graph survey |
| 22 | Nickel, Murphy, Tresp, Gabrilovich (2016), *Proc. IEEE* 104(1), 11–33 | 10.1109/JPROC.2015.2483592 | Relational ML for knowledge graphs |
| 23 | Cai, Zheng, Chang (2018), *IEEE TKDE* 30(9), 1616–1637 | 10.1109/TKDE.2018.2807452 | Graph-embedding survey |
| 24 | Goyal & Ferrara (2018), *KBS* 151, 78–94 | 10.1016/j.knosys.2018.03.022 | Graph-embedding survey (same journal) |
| 25 | Yang & Dong (2020), *KBS* 204, 106194 | 10.1016/j.knosys.2020.106194 | Explainable KG recommendation |
| 26 | Shimizu, Matsutani, Goto (2022), *KBS* 239, 107970 | 10.1016/j.knosys.2021.107970 | Explainable KG attention recommendation |
| 27 | USPTO PatentsView dump (2025) [dataset] | 10.5281/zenodo.15058362 | Public granted-patent metadata (DataCite) |
| 28 | USPTO PatentsView page (2026) | uspto.gov/.../patentsview | Current download location / coverage note |

## Corrections relative to earlier drafts
- Wu, Lin, Ji, Gu (2026) is in **_Technovation_ 151, 103405** (DOI `10.1016/j.technovation.2025.103405`), not *Technological Forecasting and Social Change*.
- PLOS ONE, *Scientific Reports*, *Journal of Information Science*, *Technology Analysis & Strategic Management*, and the OECD REGPAT working paper were removed from the reference list because they are outside the requested MDPI / IEEE / ScienceDirect set.
- Every bibliography entry is cited in the text; numbering follows first appearance (Elsevier numbered style).

## Gap after the expanded set
Prior firm-specific TOD studies typically validate opportunities for one or a few target firms, use random or unspecified splits, or evaluate GNN/link-prediction quality without a leakage-controlled, many-firm, out-of-time comparison of inspectable graph features against tabular baselines. That remains the scoped contribution of this short communication.
