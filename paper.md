# Temporal Knowledge-Graph Features for Firm-Specific Technology Opportunity Prediction

**Short Communication**

Eeshwar  
Independent Researcher  
Corresponding author: eeshwar.business@gmail.com

## Abstract
Firms that pursue digital innovation must decide which technology domains to explore beyond their current patent portfolios. Graph-based opportunity discovery can look effective when future firm–technology relations leak into features, candidate sets, or random train–test splits. This short communication tests whether temporally valid, inspectable knowledge-graph features improve out-of-time prediction of a firm's first entry into a previously unadopted Cooperative Patent Classification (CPC) subclass. A firm–technology graph is built from public USPTO PatentsView records. Conventional firm and class activity features are compared with the same features plus common-neighbor, co-occurrence, and historical peer-adoption signals. Logistic regression and random forests are trained on identical candidates under chronological evaluation with an embargo year, imbalance-aware metrics, and a firm-clustered bootstrap. On 10,878 organization assignees (grant years 2010–2023), the graph-feature forest improved average precision versus the same-learner forest on 2021–2022 pairs (paired 95% interval 0.0155 to 0.0191). Absolute average precision remained low against a 1.11% positive rate, and logistic regression was a competitive ranking baseline. Relational features therefore add incremental signal, but they do not make technology-entry prediction easy or automatically decision-ready.

**Keywords:** knowledge graph; technology opportunity; patent analytics; temporal validation; decision support; explainable AI; link prediction

## Highlights
- Temporally valid graph features are tested for firm technology-entry.
- A many-firm PatentsView holdout uses an embargoed chronological split.
- Graph features lift forest AP; logistic regression remains competitive.
- Absolute AP stays low against a 1.11% positive technology-entry rate.
- Relational scores are inspectable inputs, not mandates to enter a class.

## 1. Introduction
Digital innovation often requires a firm to enter technology classes that are absent from its current patent portfolio. The firm's own history is an incomplete map: classes co-occur across assignees, peers adopt neighboring technologies, and those relations change over time. A knowledge-based decision-support system can make those relations explicit, but only if the representation used at prediction time could actually have been observed then.

Firm-specific technology opportunity discovery (TOD) already uses patent graphs and link prediction. F-term co-occurrence networks, graph-convolutional convergence models, heterogeneous assignee–inventor–technology ecologies, competitor-aware association rules, and rule-based portfolio graphs have been used to propose new classes or patents for a target firm [1–7]. Closely related studies map a firm's existing codes onto adjacent fields through collaborative filtering, hierarchical semantic graphs, citation-flow prediction, or machine-learning ranking of previously unconnected classification pairs [8–15]. Engineering-management work on patent recommendation and citation-network outliers, and semantic patent networks such as TechNet, likewise treat inspectable relations as ranking evidence [16–18].

The existence of a graph is not, by itself, a novelty claim for *Knowledge-Based Systems*. Knowledge-graph and graph-embedding surveys distinguish representation, acquisition, relational scoring, and downstream use [19–21]. Explainable knowledge-graph recommenders show that inspectable relations can support ranking only when the knowledge layer is isolated and evaluated [22,23]. In innovation forecasting the corresponding failure mode is temporal leakage: future classifications, future firm–class edges, or labels that would not yet have been observable can make a retrospective model look stronger than a deployable one.

This paper therefore asks a narrower, falsifiable question. When candidate technologies, feature timestamps, and evaluation cutoffs are controlled, do compact relational features add out-of-time value over strong tabular baselines? The contribution is a many-firm, leakage-aware technology-entry benchmark on public PatentsView records [24,25], not a new graph neural network and not a claim that patenting equals innovation. Two hypotheses are pre-specified:

- **H1 (incremental value):** the graph-feature model achieves higher average precision than the strongest tabular baseline on the held-out future period.
- **H2 (relational contribution):** removing graph-derived features reduces performance relative to the full model when the learner is held constant.

H1 is interpreted against the best tabular model on average precision. H2 is the pre-specified same-learner contrast (graph-feature random forest versus tabular random forest). A second chronological cutoff would be required to test temporal robustness; only one embargoed split is executed here and is treated as a limitation rather than as a third hypothesis.

## 2. Data and method
### 2.1 PatentsView extract
The source is the USPTO PatentsView granted-patent metadata dump covering records through 31 December 2024, archived on Zenodo [24]. Later PatentsView files are hosted on the USPTO Open Data Portal [25] behind authenticated access; this study therefore uses the last fully public dump. Files `g_patent.tsv.zip`, `g_assignee_disambiguated.tsv.zip`, and `g_cpc_current.tsv.zip` were downloaded on 23 October 2026 and joined on `patent_id`. Provenance, hashes, and inclusion rules are stored with the normalized extract.

Normalization produces one unique row per firm–year–CPC subclass. The firm identifier is the PatentsView disambiguated `assignee_id`. Only assignees with a non-empty organization name are retained. Year is the grant year. Withdrawn patents and non-utility grants are excluded. CPC subclass codes are the first four alphanumeric characters of `cpc_subclass`; former-USPC indexing subclasses `Y10*` are dropped. Joint assignees create multiple firm–technology rows. Firms must have at least eight distinct grant years, eight distinct subclasses, twenty firm–year–CPC rows, and at least one G or H subclass in 2010–2023. After these filters the extract contains 1,453,416 rows, 10,878 firms, 14 years, and 668 subclasses.

Patent-based class entry is a proxy, not a measure of commercial success, non-patented innovation, or digital-transformation adoption. The target below is therefore first observed CPC-subclass entry in granted patents.

### 2.2 Knowledge graph and features
Let $F$ be the set of firms and $C$ the candidate CPC subclasses. For prediction year $t$, the time-indexed knowledge graph is

\begin{equation}
G_t = (F, C, E_t),\qquad E_t = \{(f,c)\in F\times C:\ \text{firm } f \text{ has class } c \text{ in a grant year }\le t\}.
\end{equation}

Technology–technology relations are induced from co-occurrence in portfolios observed by $t$. Future edges are not used to score a prediction made at $t$. The candidate universe $C$ is the 30 most frequent subclasses in the 2010–2012 burn-in window and is then held fixed. The design estimates entry into classes already visible in that window; it does not claim to predict entirely unseen subclasses.

The unit of analysis is a firm–candidate–year triple $(f,c,t)$ with $c\notin P_{f,t}$, where $P_{f,t}$ is the set of classes observed for $f$ by the end of year $t$. The binary target is first entry in the following year:

\begin{equation}
y_{f,c,t} = \mathbf{1}\bigl[c \notin P_{f,t}\ \text{and}\ c \in P_{f,t+1}\bigr].
\end{equation}

Eligible prediction years are 2013–2022. Training uses 2013–2019, 2020 is embargoed, and 2021–2022 form the test window. Every feature for $(f,c,t)$ is computed from records dated $\le t$.

Two feature blocks are used. The conventional block describes firm history and class activity: portfolio size $|P_{f,t}|$; the number of firm–year–class rows for $f$ with year $\le t$; class prevalence $n_{c,t}/|F_t|$; and clipped recent growth comparing firm counts in $\{t-1,t\}$ with those in $\{t-3,t-2\}$. The graph block is intentionally inspectable. Let $\omega_t(c,c')$ be the number of historical portfolios in $G_t$ that contain both $c$ and $c'$, and let $\mathrm{peers}_{f,t}$ be other firms that share at least one class with $f$ by year $t$. Then

\begin{align}
x^{\mathrm{cn}}_{f,c,t} &= \sum_{c'\in P_{f,t}}\omega_t(c,c'),\\
x^{\mathrm{peer}}_{f,c,t} &= \bigl|\{p\in\mathrm{peers}_{f,t}: c\in P_{p,t}\}\bigr|/\max\bigl(1,|\mathrm{peers}_{f,t}|\bigr),\\
x^{\mathrm{deg}}_{c,t} &= \bigl|\{c'\in C: \omega_t(c,c')>0\}\bigr|.
\end{align}

Common-neighbor strength $x^{\mathrm{cn}}$ is a weighted overlap between the firm's portfolio and the candidate. Peer prevalence $x^{\mathrm{peer}}$ is the historical adoption rate among technological neighbors. Degree $x^{\mathrm{deg}}$ is the candidate's co-occurrence connectivity. No graph neural network is trained.

**Algorithm 1.** Leakage-aware feature construction at prediction year $t$  
**Input:** firm–year–CPC records with grant year $\le t$; fixed candidate set $C$  
**Output:** feature vectors for eligible pairs $(f,c)$  
1: Build portfolios $P_{f,t}$ and edge set $E_t$ from records dated $\le t$.  
2: Count co-occurrence weights $\omega_t(c,c')$ inside those portfolios.  
3: Compute class prevalence and clipped recent growth from years $\le t$.  
4: For each firm $f$ with $P_{f,t}\neq\emptyset$ and each $c\in C\setminus P_{f,t}$:  
5:     compute the conventional block and the graph block in Eqs. (3)–(5);  
6:     set $y_{f,c,t}$ from the firm's classes in year $t+1$ (never used as a feature).  
7: Return the pair panel for year $t$.

### 2.3 Models and evaluation
Three models are compared on the same candidate pairs (Table 1):

1. **M1**, $\ell_2$-regularized logistic regression with balanced class weights, using only the conventional block.
2. **M2**, a random forest (300 trees, $\sqrt{p}$ features, minimum leaf size 5, balanced subsample weights), using the same conventional block.
3. **M3**, the same forest as M2 with the graph block appended.

The primary contrast is M3 versus M2 because it isolates the relational features while holding the learner fixed. M1 is a transparent linear reference. Hyperparameters were fixed before the test window was scored. Numeric preprocessing (median imputation and standardization) is fitted on training rows only. The decision threshold for balanced accuracy is chosen on prediction year 2019, the last training year, by maximizing balanced accuracy on that year; test labels are not used to choose a threshold.

Technology-entry is rare. The primary metric is average precision, always reported with the test positive prevalence. ROC-AUC is secondary because it can look favorable under severe imbalance. Balanced accuracy is reported at the training-year threshold. Brier score is reported for probability quality. For M3 versus M2, a paired bootstrap resamples held-out firms as clusters, retaining all of a firm's test pairs together, and forms a 95% interval for the average-precision difference from 1,000 replicates. H1 is supported only if the graph model beats the strongest tabular baseline on average precision with uncertainty. H2 is supported if the M3–M2 interval excludes no improvement.

## 3. Results and discussion
Results are taken from the versioned outputs of `src/experiment.py` on the normalized PatentsView file. Synthetic `--demo` numbers are not reported.

The candidate universe is the 30 most common CPC subclasses in 2010–2012. Training uses prediction years 2013–2019 (1,540,698 pairs). Year 2020 is embargoed. The test window is 2021–2022 (455,458 pairs; 10,411 firms; 30 classes). Test-set positive prevalence is 1.11% (5,057 first entries). Thresholds were chosen on 2019. Fig. 1 records the pipeline used to obtain Table 2; Fig. 2 records the executed split.

**Table 1**  
Model comparison and the question each contrast addresses.

| Model | Feature block | Purpose |
|---|---|---|
| M1 logistic regression | Firm history and class activity | Transparent tabular baseline |
| M2 random forest | Same conventional features as M1 | Nonlinear tabular baseline; same learner as M3 |
| M3 graph-feature forest | M2 features plus Eqs. (3)–(5) | Incremental value of inspectable graph features |

**Table 2**  
Held-out performance on prediction years 2021–2022. Positive prevalence $=0.0111$. Random ranking would yield average precision near that base rate. The paired firm-cluster bootstrap for M3 minus M2 average precision is $\Delta\mathrm{AP}=0.0172$ (95% interval 0.0155 to 0.0191; 1,000 firm clusters).

| Model | Average precision | ROC-AUC | Balanced accuracy | Brier score |
|---|---:|---:|---:|---:|
| M1 logistic | 0.0313 | 0.703 | 0.644 | 0.156 |
| M2 random forest | 0.0155 | 0.597 | 0.573 | 0.046 |
| M3 graph-feature forest | 0.0326 | 0.748 | 0.684 | 0.025 |

Note: Higher is better for average precision, ROC-AUC, and balanced accuracy; lower is better for Brier score.

The interval for M3 minus M2 excludes zero, so the pre-specified same-learner comparison supports **H2**: removing the graph block reduces average precision. M3 also has the best ROC-AUC, balanced accuracy, and Brier score. Absolute average precision remains low for all models (0.033 against a 1.11% base rate).

**H1 is not supported on the pre-specified reading.** The strongest tabular model on average precision is M1 (0.0313), not M2 (0.0155). M3's advantage over M1 is 0.0013 and has no pre-specified cluster interval. Logistic regression is therefore a competitive ranking baseline; the forest without graph features is not.

On this extract, inspectable graph features added average-precision value when the learner was held constant. Co-occurrence, common-neighbor, and peer-adoption signals therefore contain information that the forest does not recover from portfolio size, activity, class prevalence, and recent growth alone. That finding is consistent with TOD studies that treat relational structure as informative [1–7,11,17], but it does not reproduce their typical claim of operational opportunity lists for one focal firm. The many-firm, embargoed protocol is stricter, and the absolute scores are modest.

The result also warns against treating a nonlinear model as a substitute for a linear baseline. M1 matched M3 closely on average precision and beat M2 by a wide margin. Class-weighted logistic regression, however, had a much worse Brier score than either forest, so its scores should not be read as frequencies. For decision support, ranking and calibration must be reported together [22,23]. Average precision of 0.033 against a 1.11% base rate is better than random ranking, but most highly scored firm–class pairs will still not enter next year. The system can prioritize review lists; it cannot authorize entry. Opportunity ranking remains an input to strategy, cost, regulation, capability, and demand, none of which are in the target [8,15]. The knowledge layer is a small, auditable feature block rather than an opaque embedding [19–21]. The contribution is therefore not another GNN case study [2,6,9], but a compact test of whether a relational block earns its place next to standard activity features when leakage is controlled.

**Fig. 1.** Knowledge-graph feature pipeline. Time-stamped PatentsView relations become the graph $G_t$, inspectable features, and a three-model comparison under one chronological protocol.

![Figure 1. Knowledge-graph feature pipeline](figures/figure1_architecture.svg)

**Fig. 2.** Executed leakage-aware split. Burn-in years 2010–2012 fix the candidate universe; training uses 2013–2019; 2020 is embargoed; 2021–2022 are scored but never used for fitting. This is a protocol diagram, not a performance chart.

![Figure 2. Temporal evaluation protocol](figures/figure2_evaluation.svg)

## 4. Limitations
Assignee disambiguation and grant timing introduce measurement error [24,25]. The fixed candidate universe excludes classes absent from 2010–2012. Graph features may favor patent-intensive firms, and holdout performance can deteriorate under drift. Only one embargoed cutoff was run, so temporal robustness across windows remains untested. Model scores should not be used to deny funding or to make high-stakes decisions without independent validation and human review.

## 5. Conclusion
This short communication tests whether temporal knowledge-graph features improve firm-specific technology-entry prediction over conventional models. On USPTO PatentsView organization assignees from 2010–2023, graph features improved the random forest on a 2021–2022 holdout, with a firm-clustered interval that excluded no gain. Logistic regression remained a strong ranking baseline, and absolute average precision stayed low. Relational features therefore earn a place in this decision-support stack only as incremental, inspectable signals—not as a claim of easy or causal opportunity discovery.

## CRediT authorship contribution statement
**Eeshwar:** Conceptualization, Methodology, Software, Formal analysis, Data curation, Writing – original draft, Writing – review and editing.

## Declaration of competing interest
The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process
During the preparation of this work the author used Cursor-hosted AI tools to support literature metadata checking, manuscript language editing, and code documentation. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article. The tools were not used to generate primary experimental images or to invent sources, data, or results.

## Data availability
The PatentsView granted-patent metadata used in this study are available from Zenodo, DOI [10.5281/zenodo.15058362](https://doi.org/10.5281/zenodo.15058362) [24]. The normalized `firm_id,year,cpc_code` extract, provenance file, preparation script, experiment code, and measured outputs are in the accompanying repository: https://github.com/eeshwar-research/research-paper. Raw TSV dumps are reproduced with `src/prepare_patentsview.py` and are not stored in git.

## Acknowledgements
The author thanks the USPTO PatentsView programme for publishing disambiguated patent metadata.

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
