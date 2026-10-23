# Literature search log

This is a documented expansion of the manuscript reference set, not a Scopus/Web of Science systematic review. Those proprietary databases were not available in the execution environment. Records were retrieved from Crossref, publisher landing pages (Elsevier, Springer Nature, Taylor & Francis, USPTO), and RePEc, then checked against the DOI resolver.

## Protocol
- **Date:** 23 October 2026
- **Sources:** Crossref API, publisher pages, RePEc, USPTO PatentsView / Zenodo
- **Time window:** 2018–2026, plus foundational patent-data and explainability sources already in the manuscript
- **Queries used:**
  - `"technology opportunity discovery" AND (firm-specific OR firm-level) AND (patent OR CPC)`
  - `"technology opportunity" AND ("link prediction" OR "knowledge graph" OR GCN)`
  - `"patent recommendation" AND "knowledge graph"`
  - `PatentsView granted patent bulk download Zenodo`
- **Inclusion:** English peer-reviewed articles that (i) identify firm-specific technology opportunities from patents, or (ii) use graphs / link prediction / knowledge representations for patent recommendation or TOD, or (iii) document the PatentsView source.
- **Exclusion:** non-patent market-forecasting papers; unindexed blog posts; records whose DOI did not resolve to the cited title.

## Included after verification
| Paper | DOI | Role in manuscript |
|---|---|---|
| Lee, Ko, Yoon, Son (2021), *TFSC* | 10.1016/j.techfore.2021.120746 | Firm-specific TOD via F-term link prediction |
| Park & Geum (2022), *TFSC* | 10.1016/j.techfore.2022.121934 | GCN link prediction + firm-level validation |
| Choi, Lee, Yoon (2023), *TFSC* | 10.1016/j.techfore.2022.122161 | Heterogeneous knowledge-graph TOD |
| Seol, Lee, Kim, Yoon, Choi (2023), *JOI* | 10.1016/j.joi.2023.101464 | Firm-specific TOD from portfolio graphs / rules |
| Wu & Ji (2023), *JOI* | 10.1016/j.joi.2023.101398 | Competitor-aware firm-specific TOD |
| Chen & Deng (2023), *Scientific Reports* | 10.1038/s41598-023-28766-y | Interpretable patent recommendation with KGs |
| Liu, Wang, Feng (2025), *TASM* | 10.1080/09537325.2025.2492229 | Firm-centric patent recommendation / dual link prediction |
| Zhao, Wei, Zhu (2025), *TFSC* | 10.1016/j.techfore.2025.124305 | ML prediction of firm technology opportunities |
| Wu, Lin, Ji, Gu (2026), *Technovation* | 10.1016/j.technovation.2025.103405 | Heterogeneous GNN firm-specific TOD |
| He, Guan, Ran (2026), *Scientific Reports* | 10.1038/s41598-026-52263-7 | Personalized patent recommendation with knowledge maps |
| USPTO PatentsView Zenodo dump (2025) | 10.5281/zenodo.15058362 | Public granted-patent metadata used in this extract |

The Wu et al. article is in *Technovation* 151, not *Technological Forecasting and Social Change*. The DOI 10.1016/j.technovation.2025.103405 was confirmed via Crossref and RePEc.

## Gap after the expanded set
Prior firm-specific TOD studies typically validate opportunities for one or a few target firms, use random or unspecified splits, or evaluate GNN/link-prediction quality without a leakage-controlled, many-firm, out-of-time comparison of inspectable graph features against tabular baselines. That remains the scoped contribution of this short communication.
