# Temporal knowledge-graph features for firm-specific technology-opportunity prediction

**Article type:** Short Communication  
**Target journal:** *Knowledge-Based Systems*  
**Evidence status:** The experimental protocol is implemented and audited against the repository code. Real PatentsView results are not yet available. No performance values are fabricated.

Eeshwar P.<sup>a,*</sup>  
<sup>a</sup>Independent Researcher  

<sup>*</sup>Corresponding author.  
E-mail address: eeshwar.business@gmail.com

## Abstract
Firms that pursue digital innovation must decide which technology domains to explore beyond their current capabilities. Patent portfolios provide observable traces of technological activity, and firm–technology relations provide a structured basis for knowledge-aware decision support. Graph-based opportunity discovery is nevertheless vulnerable to a practical evaluation failure: a model can appear effective when future technology relations leak into features, candidate sets, or random train–test splits. This Short Communication specifies a temporally constrained benchmark for predicting whether a firm will enter a previously unadopted Cooperative Patent Classification (CPC) subclass in the following year. A time-indexed firm–technology knowledge graph is constructed from public patent records. The proposed model augments conventional firm and class activity features with inspectable graph signals: portfolio–class co-occurrence strength, historical peer adoption, and candidate-class graph degree. Logistic regression and random-forest baselines are trained on identical firm–technology candidates under chronological evaluation, an embargo between training labels and the test period, paired firm-cluster bootstrap intervals, and imbalance-aware metrics. The contribution is deliberately bounded. The study tests whether transparent relational features add out-of-time predictive value, rather than assuming that graph complexity is beneficial. Because no real-data run has been completed, this manuscript reports no empirical superiority claim. Measured findings must be inserted only after the documented experiment is executed and audited.

**Keywords:** knowledge graph; technology opportunity; patent analytics; temporal validation; explainable AI; decision support

## Highlights
Highlights are submitted as a separate file (`highlights.txt`), as required by the journal.

## 1. Introduction
Digital innovation often requires a firm to move beyond the technology domains in which it already operates. A firm may possess relevant capabilities for an adjacent domain, yet its internal patent history alone does not reveal all useful connections. Technology classes co-occur in portfolios, firms share technological neighbours, and adoption patterns evolve over time. A knowledge representation can make those relations explicit and support a prediction system whose recommendations are grounded in identifiable evidence.

Technology-opportunity discovery (TOD) has a substantial patent-analytic tradition. Early work used keyword maps and function-based representations to surface candidate opportunities from existing technologies and products [1,2]. Patent citation networks were later used to rank outliers as opportunity signals [3]. More recent studies move from descriptive mapping to firm-specific link prediction, including F-term networks [4], graph-convolutional prediction of technology convergence with firm-level validation [5], deep-learning text mining combined with knowledge graphs [6], heterogeneous knowledge-graph ecology models [7], multilayer supply-chain networks [8], machine-learning prediction of technology-convergence solutions [9], and heterogeneous graph neural networks for firm-specific opportunities [10]. Collectively, these studies establish that graph-based TOD is no longer novel by itself.

The relevant question for a knowledge-based decision-support system is narrower. Knowledge graphs can represent entities, relations, and temporal dynamics [11–15], and compact graph features can be used for link prediction without embedding-based opacity [16–19]. The existence of a graph does not, however, make a model interpretable or reliable. Explainable AI research distinguishes feature extraction, relation extraction, graph construction, and reasoning, and treats explanations as an explicit design requirement rather than a side effect of using a graph [20,21]. Temporal leakage is especially consequential in innovation forecasting: future patent classifications, future firm–technology links, or labels that would not yet have been observable can make a retrospective model look stronger than a deployable one.

This paper therefore proposes a leakage-aware, firm-specific technology-entry benchmark. It does not claim that patenting captures all innovation, that prediction is causal, or that the approach is the first graph-based TOD method. Its contribution is a reproducible experimental design that tests the incremental value of auditable relational features under a common temporal protocol.

The research question is:

**RQ.** Do temporally available knowledge-graph features improve out-of-time prediction of a firm's entry into a previously unadopted technology class, compared with conventional firm- and technology-level predictors?

The pre-specified hypotheses are:

- **H1 (incremental value).** The graph-feature model achieves higher average precision than the strongest tabular baseline on the held-out future period.
- **H2 (relational contribution).** Removing graph-derived features reduces performance relative to the full model.
- **H3 (temporal robustness).** Any observed improvement is directionally stable across additional chronological cutoffs, rather than being confined to one test window.

These are hypotheses to test, not findings. H1 or H2 may be rejected. A null result is informative if data quality, sample size, and evaluation power are adequately described.

## 2. Methodology
### 2.1 Data source and operational definitions
The planned source is the United States Patent and Trademark Office (USPTO) PatentsView research dataset, distributed through the USPTO Open Data Portal [24]. PatentsView provides structured patent records and disambiguated organization and technology information. The exact downloaded release, file names, download date, and licence terms must be recorded at execution time.

The analysis requires a normalized firm–year–technology table with one row per unique firm–year–CPC subclass relation and fields `firm_id`, `year`, and `cpc_code`. This is an input contract, not a claim that a processed dataset has already been downloaded. The preparation log must record the PatentsView tables and join keys, the treatment of joint assignees, the handling of missing or ambiguous firm identifiers, and the harmonization of CPC codes across years. A patent assigned to multiple firms may create multiple firm–technology relations; that choice must be disclosed and checked in sensitivity analysis. CPC subclass codes are normalized to the first four alphanumeric characters (for example, A01B).

Patent activity is an imperfect proxy for innovation. It undercounts non-patented, service, organizational, and process innovations; differs by sector and firm size; and reflects patenting strategy as well as inventive output [1,2]. The outcome is therefore interpreted as **technology-class entry observed in patent data**, not as a comprehensive measure of entrepreneurship or innovation.

The unit of analysis is a firm–candidate-technology–prediction-year triple. For firm $f$, candidate technology $c$, and prediction year $t$, the binary target is

$$
y_{f,c,t}=
\begin{cases}
1, & \text{if } c\notin P_{f,\le t}\ \text{and}\ c\in P_{f,t+1},\\
0, & \text{otherwise among eligible candidates,}
\end{cases}
\tag{1}
$$

where $P_{f,\le t}$ is the set of CPC subclasses observed for firm $f$ in years up to and including $t$. Firms must have patent history before the prediction year. The candidate-technology universe $C$ is fixed from a pre-specified burn-in window and retained throughout evaluation; it is not selected using the future test period. The design therefore estimates entry into classes already visible in the initial data window and does not claim to predict entirely unseen classes.

### 2.2 Temporal knowledge graph and inspectable features
Let

$$
G_t=(F,C,E_t)
\tag{2}
$$

denote the time-indexed knowledge graph, where $F$ is the set of firms, $C$ is the fixed candidate CPC universe, and $E_t$ contains firm–technology edges observed no later than year $t$. Technology–technology relations are induced from co-occurrence in firm portfolios observed by $t$. Future edges are not used to calculate a prediction made at $t$. This construction follows the temporal-knowledge-graph principle that each relation must carry an availability time [11,14,15].

Graph features are intentionally simple and inspectable. For a firm with historical portfolio $P_{f,\le t}$ and a candidate $c\notin P_{f,\le t}$, the implemented relational features are:

$$
\mathrm{CN}(f,c,t)=\sum_{k\in P_{f,\le t}} w_t(c,k),
\tag{3}
$$

$$
\mathrm{Peer}(f,c,t)=\frac{1}{|N_f(t)|}\sum_{g\in N_f(t)}\mathbf{1}[c\in P_{g,\le t}],
\tag{4}
$$

$$
\mathrm{Deg}(c,t)=\bigl|\{k\in C:w_t(c,k)>0\}\bigr|,
\tag{5}
$$

where $w_t(c,k)$ is the number of historical firm portfolios, dated no later than $t$, in which classes $c$ and $k$ co-occur, and $N_f(t)$ is the set of peer firms that share at least one technology with $f$ by time $t$. Eq. (3) is a weighted co-occurrence, or common-neighbour, strength rather than an opaque embedding. Eq. (4) is a historical peer-adoption rate. Eq. (5) is the candidate-class degree in the induced technology graph.

Conventional features describe firm portfolio size $|P_{f,\le t}|$, historical firm–technology activity volume, candidate-class prevalence in $G_t$, and recent class growth calculated from the two years ending at $t$ versus the two preceding years. Growth is clipped to $[-5,5]$ to limit extreme ratios in sparse classes. The full model adds the relational features to the same conventional predictors used by the tabular baseline. No graph neural network is claimed or implemented in this compact benchmark. The design therefore occupies a different position from GCN- or heterogeneous-GNN-based TOD models [5,7,10]: those studies increase representational capacity, whereas this study isolates whether a small set of time-valid relational features is incrementally useful.

### 2.3 Models and controlled comparison
Three models are compared on the same candidate pairs (Table 1):

1. **M1 — Logistic regression:** a transparent linear baseline.
2. **M2 — Random forest:** a nonlinear tabular baseline using only conventional features.
3. **M3 — Graph-feature random forest:** the same random-forest learner augmented with the pre-specified graph-derived features.

The primary comparison is M3 against M2 because it isolates the incremental value of the relational feature block while holding the learning algorithm constant. M1 provides a simpler reference point. Hyperparameters, class weighting, candidate universe, and feature availability are fixed before the final test is evaluated. If tuning is performed, it must occur only within the training period.

### 2.4 Temporal split, leakage controls, and metrics
Rows are ordered by prediction year. After an initial three-year burn-in that defines $C$, eligible prediction years exclude the final calendar year because the target requires year $t+1$. The last two eligible prediction years form the test window. One intervening prediction year is embargoed. Training examples have outcome years that end before the embargo. Every feature for a firm–candidate pair is calculated from records available by that pair's prediction cutoff. Preprocessing is fitted only on training rows. Test labels are used only for final scoring and uncertainty estimation. Thresholds are selected on the latest training prediction year, never by random row splitting and never on the test period. Fig. 2 summarizes the protocol.

Technology-entry prediction is imbalanced: most firm–candidate pairs are expected not to become positive in any one year [22,23]. The primary metric is average precision, reported with the test positive prevalence. Receiver-operating-characteristic area under the curve (ROC-AUC) is secondary because it can appear favourable under severe imbalance [22]. Balanced accuracy is reported at a threshold selected only on training data. Brier score is reported for probability quality. The study reports the number of firms, candidate classes, firm–candidate–year rows, positives, and prevalence for each split.

For the primary M3–M2 comparison, paired bootstrap resampling of held-out firms is used, preserving all of a sampled firm's held-out pairs, and a 95% interval is reported for the average-precision difference. H1 is supported only if the pre-specified primary comparison is positive and uncertainty is reported; a point estimate alone is insufficient. Multiple time cutoffs and feature ablations are robustness analyses, not opportunities to select the most favourable result.

## 3. Experimental protocol and evidence requirements
No empirical results are reported in this manuscript version because the real PatentsView extract has not yet been processed and the experiment has not been run on it. Table 1 defines the controlled comparison. Table 2 specifies the metrics and inference plan. The repository can generate a model-comparison chart after execution; that chart must display observed performance and must not be populated with synthetic or illustrative values presented as real evidence.

**Table 1.** Model comparison and the inferential question each comparison addresses.

| Model | Input feature blocks | Comparison purpose |
|---|---|---|
| M1: Logistic regression | Firm history and candidate-class activity | Establishes a transparent linear baseline |
| M2: Random forest | Same conventional features as M1 | Tests nonlinear tabular prediction |
| M3: Graph-feature random forest | M2 features plus temporal relational features | Primary test of incremental graph-feature value |

**Table 2.** Pre-specified evaluation and robustness protocol.

| Dimension | Required report | Interpretation safeguard |
|---|---|---|
| Primary discrimination | Average precision and positive prevalence | Prevalence must accompany average precision |
| Secondary ranking | ROC-AUC | Do not use alone for imbalanced data |
| Thresholded prediction | Balanced accuracy; threshold chosen on training data | No test-set threshold tuning |
| Probability quality | Brier score and calibration assessment | Ranking is not calibration |
| Incremental value | Paired, firm-clustered bootstrap interval for M3 minus M2 | Same held-out observations; uncertainty shown |
| Temporal robustness | At least two cutoffs where feasible; report every cutoff | Avoid selecting only the best period |
| Component contribution | Ablation removing graph features | Distinguishes relational value from model complexity |
| Data validity | Firm count, class count, rows, positives, prevalence, exclusions | Makes selection and imbalance visible |

**Fig. 1.** Knowledge-graph feature pipeline and decision-support architecture. Time-stamped firm–year–CPC records are converted into a cutoff-dated graph $G_t$. Conventional activity features and inspectable relational features are then compared under one shared evaluation protocol.

![Figure 1. Knowledge-graph feature pipeline](figures/figure1_architecture.svg)

**Fig. 2.** Leakage-aware temporal evaluation protocol. The figure distinguishes the historical burn-in used to freeze the candidate universe, training labels, the embargo year, and later test outcomes. It is a protocol diagram, not a measured performance plot.

![Figure 2. Temporal evaluation protocol](figures/figure2_evaluation.svg)

After execution, this section must be completed with the actual output from the versioned dataset: (a) descriptive sample statistics; (b) all three models' test metrics; (c) the M3–M2 confidence interval; (d) the results of each pre-specified temporal cutoff and ablation; and (e) calibration and subgroup findings. Do not infer or write that H1–H3 are supported before these outputs exist. If M3 does not improve on M2, report that result and investigate sparsity, data coverage, and drift rather than changing the test protocol after seeing the outcome.

The implemented pipeline (`src/experiment.py`) matches this protocol. Candidate classes are the most frequent CPC subclasses in the first three years, subject to a minimum-firm filter. Features at prediction year $t$ use only records dated $\le t$. The target is first entry in $t+1$. The last two eligible prediction years are held out, with one embargo year. Thresholds are selected on a later training-period validation year. The paired bootstrap resamples firms. The repository `--demo` mode is a software smoke test only; synthetic results must never be reported as empirical findings.

## 4. Discussion
The intended contribution is a test of whether explicit relational knowledge adds useful predictive information for firm-specific technology-entry decisions. If the graph-feature model improves average precision consistently across temporal cutoffs and the firm-clustered interval excludes no improvement, the evidence would support the claim that the chosen relations contain incremental predictive signal for the observed patent-based task. Such a result would not establish that a firm should enter a technology, that graph features cause innovation, or that the model generalizes to all sectors or countries. Opportunity ranking is a decision-support input. Strategic fit, cost, regulation, capabilities, and market demand remain outside this prediction target unless they are measured and included.

If performance does not improve, that result is still informative. It would suggest that simple firm and technology histories capture most of the signal available in this dataset, or that the graph is too sparse or noisy to help. A transparent negative result is more useful than a complexity claim based on one favourable split. The primary model comparison and ablation are therefore central to the contribution.

This positioning is consistent with the TOD literature and with knowledge-based systems research. Prior firm-specific graph models have shown that link prediction can generate candidate opportunities [4,5,7,8,10]. Those studies typically optimize discovery quality for a target firm or industry case. The present benchmark asks a different, more falsifiable question: after candidate construction, timestamps, and learner class are controlled, do inspectable graph features add out-of-time value? That question is closer to the journal's requirement that a knowledge-based system make its representation explicit, evaluate the contribution of that representation, and disclose when the knowledge layer does not improve decisions [20,21].

The novelty claim is therefore scoped. The paper does not claim the first use of patents, graphs, or temporal holdouts in TOD. It claims a compact, auditable protocol in which candidate construction, feature timestamps, incremental model value, and uncertainty are reported together. A systematic review of a larger corpus would be required to establish an exhaustive novelty claim. Before a final submission that reports empirical results, the literature search should be extended in Scopus or Web of Science and the gap statement revised against the retrieved corpus.

### 4.1 Limitations and responsible use
First, patent-based class entry is not equivalent to entrepreneurial success, commercial innovation, or digital-transformation adoption. Second, assignee disambiguation and patent timing introduce measurement error [24]. Third, the fixed candidate universe excludes technologies absent from the burn-in window; results do not cover open-ended discovery of entirely new classes. Fourth, graph features may favour firms in patent-intensive sectors. Fifth, temporal holdout performance can deteriorate under technology and market drift. Sixth, patent data are not a substitute for confidential strategy, and model scores should not be used to deny funding or make high-stakes decisions without independent validation and human review.

### 4.2 Reproducibility
Archive the PatentsView release date, download links, raw-to-normalized data-preparation script or exact documented steps, dataset hashes where permitted, inclusion criteria, random seed, package versions, commit hash, temporal cutoffs, and all model outputs. Before submission of an empirical version, execute the pipeline on real data, run unit checks on temporal boundaries, and verify that no outcome-year or future graph edge enters features.

## 5. Conclusion
This Short Communication defines a reproducible test of whether temporal knowledge-graph features improve firm-specific technology-entry prediction over conventional models. Its strongest claim is conditional and testable: relational features deserve inclusion in innovation decision support only if they provide stable out-of-time value under a transparent protocol. The manuscript is not yet an empirical paper ready for formal submission because the real-data extraction, execution, and measured results remain outstanding. Completing those steps—not adding more terminology or visual decoration—is necessary to establish a credible contribution.

## Data availability
The planned source is the public USPTO PatentsView / Open Data Portal [24]. The exact release and normalized data file have not yet been downloaded or archived for this study. The analysis code is available at https://github.com/eeshwar-research/research-paper. Real-data empirical reproduction is pending.

## CRediT authorship contribution statement
**Eeshwar P.:** Conceptualization, Methodology, Software, Validation, Writing – original draft, Writing – review and editing.

## Declaration of competing interest
The author has nothing to declare. This statement must be confirmed in Elsevier's declarations tool at submission.

## Funding
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. Confirm or replace this sentence if funding exists.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process
During the preparation of this work the author used Cursor-hosted generative AI tools to support language editing, reference formatting, and code review. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article. Confirm the exact tool names before submission.

## Acknowledgements
None.

## References
[1] S. Lee, B. Yoon, Y. Park, An approach to discovering new technology opportunities: Keyword-based patent map approach, Technovation 29 (6–7) (2009) 481–497. https://doi.org/10.1016/j.technovation.2008.10.006.

[2] J. Yoon, H. Park, W. Seo, J.-M. Lee, B.-Y. Coh, J. Kim, Technology opportunity discovery (TOD) from existing technologies and products: A function-based TOD framework, Technol. Forecast. Soc. Change 100 (2015) 153–167. https://doi.org/10.1016/j.techfore.2015.04.012.

[3] A. Rodriguez, A. Tosyali, B. Kim, J. Choi, J.-M. Lee, B.-Y. Coh, M.K. Jeong, Patent clustering and outlier ranking methodologies for attributed patent citation networks for technology opportunity discovery, IEEE Trans. Eng. Manag. 63 (4) (2016) 426–437. https://doi.org/10.1109/TEM.2016.2580619.

[4] J. Lee, N. Ko, J. Yoon, C. Son, An approach for discovering firm-specific technology opportunities: Application of link prediction to F-term networks, Technol. Forecast. Soc. Change 168 (2021) 120746. https://doi.org/10.1016/j.techfore.2021.120746.

[5] M. Park, Y. Geum, Two-stage technology opportunity discovery for firm-level decision making: GCN-based link-prediction approach, Technol. Forecast. Soc. Change 183 (2022) 121934. https://doi.org/10.1016/j.techfore.2022.121934.

[6] M. Lee, S. Kim, H. Kim, J. Lee, Technology opportunity discovery using deep learning-based text mining and a knowledge graph, Technol. Forecast. Soc. Change 180 (2022) 121718. https://doi.org/10.1016/j.techfore.2022.121718.

[7] J. Choi, C. Lee, J. Yoon, Exploring a technology ecology for technology opportunity discovery: A link prediction approach using heterogeneous knowledge graphs, Technol. Forecast. Soc. Change 186 (2023) 122161. https://doi.org/10.1016/j.techfore.2022.122161.

[8] Y. Wu, Y. Ji, F. Gu, Identifying firm-specific technology opportunities in a supply chain: Link prediction analysis in multilayer networks, Expert Syst. Appl. 213 (2023) 119053. https://doi.org/10.1016/j.eswa.2022.119053.

[9] Z. Wang, W. Guo, H. Shao, L. Wang, Z. Chang, Y. Zhang, Z. Liu, From technology opportunities to solutions generation via patent analysis: Application of machine learning-based link prediction, Adv. Eng. Inform. 62 (2024) 102944. https://doi.org/10.1016/j.aei.2024.102944.

[10] Y. Wu, Z. Lin, Y. Ji, F. Gu, Identifying firm-specific technology opportunities: Heterogeneous graph neural network-based link prediction, Technovation 151 (2026) 103405. https://doi.org/10.1016/j.technovation.2025.103405.

[11] S. Ji, S. Pan, E. Cambria, P. Marttinen, P.S. Yu, A survey on knowledge graphs: Representation, acquisition, and applications, IEEE Trans. Neural Netw. Learn. Syst. 33 (2) (2022) 494–514. https://doi.org/10.1109/TNNLS.2021.3070843.

[12] Q. Wang, Z. Mao, B. Wang, L. Guo, Knowledge graph embedding: A survey of approaches and applications, IEEE Trans. Knowl. Data Eng. 29 (12) (2017) 2724–2743. https://doi.org/10.1109/TKDE.2017.2754499.

[13] M. Nickel, K. Murphy, V. Tresp, E. Gabrilovich, A review of relational machine learning for knowledge graphs, Proc. IEEE 104 (1) (2016) 11–33. https://doi.org/10.1109/JPROC.2015.2483592.

[14] K. Liang, L. Meng, M. Liu, Y. Liu, W. Tu, S. Wang, S. Zhou, X. Liu, F. Sun, K. He, A survey of knowledge graph reasoning on graph types: Static, dynamic, and multi-modal, IEEE Trans. Pattern Anal. Mach. Intell. 46 (12) (2024) 9456–9478. https://doi.org/10.1109/TPAMI.2024.3417451.

[15] H. Huang, L. Xie, M. Liu, J. Lin, H. Shen, An embedding model for temporal knowledge graphs with long and irregular intervals, Knowl.-Based Syst. 296 (2024) 111893. https://doi.org/10.1016/j.knosys.2024.111893.

[16] X. Zhao, J. Miao, F. Yang, S. Pang, Geometry interaction embeddings for interpolation temporal knowledge graph completion, Mathematics 12 (13) (2024) 2022. https://doi.org/10.3390/math12132022.

[17] C. Zhang, K. Li, S. Wang, B. Zhou, L. Wang, F. Sun, Learning heterogeneous graph embedding with metapath-based aggregation for link prediction, Mathematics 11 (3) (2023) 578. https://doi.org/10.3390/math11030578.

[18] L. Shi, W. Liu, Y. Wu, C. Dai, Z. Ji, I. Ganchev, Knowledge graph embedding using a multi-channel interactive convolutional neural network with triple attention, Mathematics 12 (18) (2024) 2821. https://doi.org/10.3390/math12182821.

[19] X. Yuan, W. Wang, B. Gao, L. Zhao, R. Ma, F. Ding, Explicit and implicit feature contrastive learning model for knowledge graph link prediction, Sensors 24 (22) (2024) 7353. https://doi.org/10.3390/s24227353.

[20] A. Barredo Arrieta, N. Díaz-Rodríguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. Garcia, S. Gil-Lopez, D. Molina, R. Benjamins, R. Chatila, F. Herrera, Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI, Inf. Fusion 58 (2020) 82–115. https://doi.org/10.1016/j.inffus.2019.12.012.

[21] I. Tiddi, S. Schlobach, Knowledge graphs as tools for explainable machine learning: A survey, Artif. Intell. 302 (2022) 103627. https://doi.org/10.1016/j.artint.2021.103627.

[22] H. He, E.A. Garcia, Learning from imbalanced data, IEEE Trans. Knowl. Data Eng. 21 (9) (2009) 1263–1284. https://doi.org/10.1109/TKDE.2008.239.

[23] S. Gan, M. Alshahrani, S. Liu, Positive-unlabeled learning for network link prediction, Mathematics 10 (18) (2022) 3345. https://doi.org/10.3390/math10183345.

[24] United States Patent and Trademark Office (USPTO), PatentsView: Research-ready patent data and data downloads. https://www.uspto.gov/ip-policy/economic-research/patentsview, 2026 (accessed 23 October 2026).
