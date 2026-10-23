# Temporal Knowledge-Graph Features for Firm-Specific Technology Opportunity Prediction

**Short Communication**

Eeshwar  
Independent Researcher  
Corresponding author: eeshwar.business@gmail.com

## Abstract
Firms pursuing digital innovation must decide which technology domains to explore beyond their patent portfolios. Graph-based opportunity discovery can look effective when future firm–technology relations leak into features, candidates, or random splits. This short communication tests whether temporally valid, inspectable knowledge-graph features improve out-of-time prediction of a firm's first entry into a previously unadopted Cooperative Patent Classification (CPC) subclass. A firm–technology graph is built from USPTO PatentsView records. Conventional activity features are compared with the same features plus common-neighbor, co-occurrence, and historical peer-adoption signals. Logistic regression and random forests are trained on identical candidates under chronological evaluation with an embargo year and a firm-clustered bootstrap. On 10,878 organization assignees (grant years 2010–2023), the graph-feature forest improved average precision versus the same-learner forest on 2021–2022 pairs (paired 95% interval 0.0155 to 0.0191). Absolute average precision remained low against a 1.11% positive rate, and logistic regression was a competitive ranking baseline. Relational features therefore add incremental signal, but they do not make technology-entry prediction decision-ready.

**Keywords:** knowledge graph; technology opportunity; patent analytics; temporal validation; decision support; explainable AI; link prediction

## 1. Introduction
A firm entering a new technology class cannot rely on its own patent history alone: classes co-occur across assignees, peers adopt neighboring technologies, and those relations change over time. A knowledge-based system can make the relations explicit only if every feature used at prediction time could have been observed then.

Firm-specific technology opportunity discovery (TOD) already uses patent graphs and link prediction [1–7]. Related work maps a firm's codes onto adjacent fields by collaborative filtering, semantic graphs, citation-flow prediction, or ranking of unconnected pairs [8–15]. Patent recommendation, citation-network outliers, and semantic networks such as TechNet treat inspectable relations as ranking evidence [16–18]. Knowledge-graph surveys separate representation from scoring [19–21], and explainable recommenders require the knowledge layer to be isolated before it is credited [22,23]. The failure mode here is temporal leakage: future classifications or future firm–class edges can make a retrospective model look stronger than a deployable one.

This paper asks whether compact relational features add out-of-time value over tabular baselines when candidates, feature timestamps, and cutoffs are controlled. The contribution is a many-firm, leakage-aware technology-entry benchmark on public PatentsView records [24,25], not a new graph neural network. Two hypotheses are pre-specified. **H1:** the graph-feature model achieves higher average precision than the strongest tabular baseline on the held-out period. **H2:** removing graph features reduces performance when the learner is held constant. H1 is read against the best tabular model on average precision. H2 is the same-learner contrast of the graph-feature forest versus the tabular forest.

## 2. Data and method
### 2.1 PatentsView extract
The source is the USPTO PatentsView granted-patent metadata dump through 31 December 2024, archived on Zenodo [24]. Later Open Data Portal files require authentication [25]. Files `g_patent.tsv.zip`, `g_assignee_disambiguated.tsv.zip`, and `g_cpc_current.tsv.zip` were downloaded on 23 October 2026 and joined on `patent_id`. One unique row is kept per firm–year–CPC subclass. The firm identifier is the disambiguated organization `assignee_id`. Year is the grant year. Withdrawn and non-utility patents are dropped; `Y10*` indexing codes are excluded. Joint assignees create multiple rows. Firms must have at least eight grant years, eight subclasses, twenty rows, and one G or H subclass in 2010–2023. The extract has 1,453,416 rows, 10,878 firms, and 668 subclasses. The outcome is first observed CPC-subclass entry in granted patents, not commercial innovation.

### 2.2 Knowledge graph and features
Let $F$ be firms and $C$ candidate CPC subclasses. For prediction year $t$,

\begin{equation}
G_t=(F,C,E_t),\quad E_t=\{(f,c): f\text{ has class }c\text{ in a grant year }\le t\}.
\end{equation}

Technology–technology relations are induced from co-occurrence in portfolios observed by $t$. The candidate set $C$ is the 30 most frequent subclasses in 2010–2012 and is then held fixed. A pair $(f,c,t)$ is eligible only if $c\notin P_{f,t}$, where $P_{f,t}$ is the firm's classes through year $t$. The target is $y_{f,c,t}=1$ if $c$ first appears in year $t+1$. Training uses prediction years 2013–2019, 2020 is embargoed, and 2021–2022 are the test window. Every feature for $(f,c,t)$ uses records dated $\le t$.

The conventional block is portfolio size $|P_{f,t}|$, historical firm–year–class activity, class prevalence $n_{c,t}/|F_t|$, and clipped recent growth comparing firm counts in $\{t-1,t\}$ with $\{t-3,t-2\}$. The graph block is inspectable. With $\omega_t(c,c')$ the number of historical portfolios containing both classes and $\mathrm{peers}_{f,t}$ other firms sharing at least one class with $f$,

\begin{equation}
x^{\mathrm{cn}}_{f,c,t}=\sum_{c'\in P_{f,t}}\omega_t(c,c'),\quad
x^{\mathrm{peer}}_{f,c,t}=\frac{|\{p\in\mathrm{peers}_{f,t}:c\in P_{p,t}\}|}{\max(1,|\mathrm{peers}_{f,t}|)},\quad
x^{\mathrm{deg}}_{c,t}=|\{c'\in C:\omega_t(c,c')>0\}|.
\end{equation}

$x^{\mathrm{cn}}$ is weighted overlap with the candidate, $x^{\mathrm{peer}}$ is peer adoption, and $x^{\mathrm{deg}}$ is co-occurrence degree. No graph neural network is trained.

### 2.3 Models and evaluation
Three models are compared on the same pairs (Table 1). **M1** is $\ell_2$-regularized logistic regression with balanced class weights and the conventional block. **M2** is a random forest (300 trees, $\sqrt{p}$ features, minimum leaf size 5, balanced subsample weights) with the same block. **M3** is the same forest with the graph block appended. The primary contrast is M3 versus M2. Hyperparameters were fixed before test scoring. Imputation and standardization are fitted on training rows only. The balanced-accuracy threshold is chosen on 2019, the last training year.

The primary metric is average precision, reported with positive prevalence. ROC-AUC is secondary. Balanced accuracy uses the training-year threshold. Brier score measures probability quality. For M3 versus M2, a paired bootstrap resamples held-out firms as clusters (1,000 replicates) and reports a 95% interval for the average-precision difference.

## 3. Results and discussion
Results come from `src/experiment.py` on the normalized PatentsView file. Demo-mode numbers are not reported. Training uses 1,540,698 pairs (2013–2019). The 2021–2022 test set has 455,458 pairs, 10,411 firms, 30 classes, and 5,057 first entries (prevalence 1.11%). Fig. 1 shows the pipeline; Fig. 2 shows the split.

**Table 1**  
Model comparison.

| Model | Feature block | Purpose |
|---|---|---|
| M1 logistic | Firm history and class activity | Transparent tabular baseline |
| M2 random forest | Same conventional features as M1 | Nonlinear tabular baseline |
| M3 graph-feature forest | M2 features plus Eq. (2) | Incremental value of graph features |

**Table 2**  
Held-out performance on 2021–2022. Prevalence $=0.0111$. The firm-cluster bootstrap for M3 minus M2 average precision is $\Delta\mathrm{AP}=0.0172$ (95% interval 0.0155 to 0.0191; 1,000 clusters). Higher is better except Brier score.

| Model | Average precision | ROC-AUC | Balanced accuracy | Brier score |
|---|---:|---:|---:|---:|
| M1 logistic | 0.0313 | 0.703 | 0.644 | 0.156 |
| M2 random forest | 0.0155 | 0.597 | 0.573 | 0.046 |
| M3 graph-feature forest | 0.0326 | 0.748 | 0.684 | 0.025 |

The M3–M2 interval excludes zero, so **H2 is supported**. M3 also has the best ROC-AUC, balanced accuracy, and Brier score. Absolute average precision remains low (0.033 versus a 1.11% base rate). **H1 is not supported:** the strongest tabular model on average precision is M1 (0.0313), not M2 (0.0155). M3's advantage over M1 is 0.0013 and has no pre-specified interval.

Graph features therefore add information that the forest does not recover from activity features alone [1–7,11,17], but they do not yield operational opportunity lists. M1 matches M3 on ranking and beats M2, yet its Brier score is much worse, so ranking and calibration must be reported together [22,23]. An average precision of 0.033 can prioritize review lists; it cannot authorize entry [8,15]. The knowledge layer is an auditable feature block, not an opaque embedding or a new GNN [2,6,9,19–21]. Assignee disambiguation, grant timing, the fixed 2010–2012 candidate universe, and the single embargoed cutoff remain limitations [24,25]. Scores should not be used for high-stakes decisions without independent validation.

**Fig. 1.** Feature pipeline from time-stamped PatentsView relations to the three-model comparison.

![Figure 1](figures/figure1_architecture.svg)

**Fig. 2.** Executed split: burn-in 2010–2012, training 2013–2019, embargo 2020, test 2021–2022.

![Figure 2](figures/figure2_evaluation.svg)

## 4. Conclusion
On PatentsView organization assignees from 2010–2023, temporal graph features improved the random forest on a 2021–2022 holdout, with a firm-clustered interval that excluded no gain. Logistic regression remained a strong ranking baseline, and absolute average precision stayed low. Relational features earn a place only as incremental, inspectable signals.

## CRediT authorship contribution statement
**Eeshwar:** Conceptualization, Methodology, Software, Formal analysis, Data curation, Writing – original draft, Writing – review and editing.

## Declaration of competing interest
The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process
During the preparation of this work the author used Cursor-hosted AI tools for literature metadata checking, language editing, and code documentation. The author reviewed the content and takes full responsibility for the published article.

## Data availability
PatentsView metadata: Zenodo DOI 10.5281/zenodo.15058362 [24]. Extract, code, and outputs: https://github.com/eeshwar-research/research-paper.

## Acknowledgements
Thanks to the USPTO PatentsView programme for disambiguated patent metadata.

## References
[1] J. Lee, N. Ko, J. Yoon, C. Son, An approach for discovering firm-specific technology opportunities: Application of link prediction to F-term networks, Technol. Forecast. Soc. Change 168 (2021) 120746. https://doi.org/10.1016/j.techfore.2021.120746.
[2] M. Park, Y. Geum, Two-stage technology opportunity discovery for firm-level decision making: GCN-based link-prediction approach, Technol. Forecast. Soc. Change 183 (2022) 121934. https://doi.org/10.1016/j.techfore.2022.121934.
[3] J. Choi, C. Lee, J. Yoon, Exploring a technology ecology for technology opportunity discovery: A link prediction approach using heterogeneous knowledge graphs, Technol. Forecast. Soc. Change 186 (2023) 122161. https://doi.org/10.1016/j.techfore.2022.122161.
[4] Y. Seol, S. Lee, C. Kim, J. Yoon, J. Choi, Towards firm-specific technology opportunities: A rule-based machine learning approach to technology portfolio analysis, J. Informetr. 17 (4) (2023) 101464. https://doi.org/10.1016/j.joi.2023.101464.
[5] Y. Wu, Y. Ji, Identifying firm-specific technology opportunities from the perspective of competitors by using association rule mining, J. Informetr. 17 (2) (2023) 101398. https://doi.org/10.1016/j.joi.2023.101398.
[6] Y. Wu, Z. Lin, Y. Ji, F. Gu, Identifying firm-specific technology opportunities: Heterogeneous graph neural network-based link prediction, Technovation 151 (2026) 103405. https://doi.org/10.1016/j.technovation.2025.103405.
[7] R. Zhao, W. Wei, Z. Zhu, Discovering firm technology opportunities by machine learning prediction method based on firm's technology portfolio, Technol. Forecast. Soc. Change 221 (2025) 124305. https://doi.org/10.1016/j.techfore.2025.124305.
[8] Y. Park, J. Yoon, Application technology opportunity discovery from technology portfolios: Use of patent classification and collaborative filtering, Technol. Forecast. Soc. Change 118 (2017) 170–183. https://doi.org/10.1016/j.techfore.2017.02.018.
[9] Z. Liu, J. Feng, L. Uden, Technology opportunity analysis using hierarchical semantic networks and dual link prediction, Technovation 128 (2023) 102872. https://doi.org/10.1016/j.technovation.2023.102872.
[10] I. Park, B. Yoon, Technological opportunity discovery for technological convergence based on the prediction of technology knowledge flow in a citation network, J. Informetr. 12 (4) (2018) 1199–1222. https://doi.org/10.1016/j.joi.2018.09.007.
[11] J. Feng, M. Li, X. Wang, Knowledge-driven patent recommendation toward firm-specific technology opportunities using multilayer cross-citation network and machine learning, Adv. Eng. Inform. 76 (2026) 104850. https://doi.org/10.1016/j.aei.2026.104850.
[12] Z.-X. Chang, W. Guo, H.-Y. Shao, L. Wang, Z.-L. Wang, Y.-R. Zhang, A framework for technology opportunity discovery using GAT-based link prediction and network analysis, Adv. Eng. Inform. 66 (2025) 103498. https://doi.org/10.1016/j.aei.2025.103498.
[13] Z. Wang, W. Guo, H. Shao, L. Wang, Z. Chang, Y. Zhang, From technology opportunities to solutions generation via patent analysis: Application of machine learning-based link prediction, Adv. Eng. Inform. 62 (2024) 102944. https://doi.org/10.1016/j.aei.2024.102944.
[14] Y. Geum, M. Kim, How to identify promising chances for technological innovation: Keygraph-based patent analysis, Adv. Eng. Inform. 46 (2020) 101155. https://doi.org/10.1016/j.aei.2020.101155.
[15] J. Zhao, J. Li, X. Xi, W. Liu, Embracing technological opportunities based on heterogeneous networks: An integrated link prediction approach for technology convergence of firms, Technol. Forecast. Soc. Change 229 (2026) 124701. https://doi.org/10.1016/j.techfore.2026.124701.
[16] A. Rodriguez, A. Tosyali, B. Kim, J. Choi, J.-M. Lee, B.-Y. Coh, M.K. Jeong, Patent clustering and outlier ranking methodologies for attributed patent citation networks for technology opportunity discovery, IEEE Trans. Eng. Manag. 63 (4) (2016) 426–437. https://doi.org/10.1109/TEM.2016.2580619.
[17] Z. Liu, X. Wang, J. Feng, Firm-specific patent recommendation using topology-based link prediction and SAO embedding method, IEEE Trans. Eng. Manag. 72 (2025) 2097–2115. https://doi.org/10.1109/TEM.2025.3570659.
[18] S. Sarica, J. Luo, K.L. Wood, TechNet: Technology semantic network based on patent data, Expert Syst. Appl. 142 (2020) 112995. https://doi.org/10.1016/j.eswa.2019.112995.
[19] Q. Wang, Z. Mao, B. Wang, L. Guo, Knowledge graph embedding: A survey of approaches and applications, IEEE Trans. Knowl. Data Eng. 29 (12) (2017) 2724–2743. https://doi.org/10.1109/TKDE.2017.2754499.
[20] S. Ji, S. Pan, E. Cambria, P. Marttinen, P.S. Yu, A survey on knowledge graphs: Representation, acquisition, and applications, IEEE Trans. Neural Netw. Learn. Syst. 33 (2) (2022) 494–514. https://doi.org/10.1109/TNNLS.2021.3070843.
[21] P. Goyal, E. Ferrara, Graph embedding techniques, applications, and performance: A survey, Knowl.-Based Syst. 151 (2018) 78–94. https://doi.org/10.1016/j.knosys.2018.03.022.
[22] Z. Yang, S. Dong, HAGERec: Hierarchical attention graph convolutional network incorporating knowledge graph for explainable recommendation, Knowl.-Based Syst. 204 (2020) 106194. https://doi.org/10.1016/j.knosys.2020.106194.
[23] R. Shimizu, M. Matsutani, M. Goto, An explainable recommendation framework based on an improved knowledge graph attention network with massive volumes of side information, Knowl.-Based Syst. 239 (2022) 107970. https://doi.org/10.1016/j.knosys.2021.107970.
[24] [dataset] United States Patent and Trademark Office, Final release of PatentsView metadata, pre-grant and granted (12/31/2024), Zenodo, 2025. https://doi.org/10.5281/zenodo.15058362.
[25] United States Patent and Trademark Office, PatentsView: Research-ready patent data and data downloads, 2026. https://www.uspto.gov/ip-policy/economic-research/patentsview (accessed 23 October 2026).
