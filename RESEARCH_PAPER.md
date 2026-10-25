# Predicting Human-Perceived Coding Difficulty from Source-Code Features Using Regression: A Student-Rating Study

**Muhammad Rehan, Issa Therase Delson, and Khadar Babu SK**  
*School of Advanced Sciences, Vellore Institute of Technology, Vellore, Tamil Nadu, India*  
**Correspondence:** Dr. Khadar Babu SK (`khadarbabu.sk@vit.ac.in`, `rehanpeetikathazha@gamil.com`, `issatherase@gmail.com`)  

---

## Abstract

**Introduction:** Structural metrics like cyclomatic complexity remain standard tools for gauging software complexity in industry. Whether these measures reflect how difficult novice programmers actually find a piece of code is much less clear. We set out to test whether self-reported coding difficulty among students can be estimated reliably using a compact set of surface code metrics, which structural traits matter most, and whether classical cyclomatic complexity offers any explanatory advantage over simpler syntactic counts.

**Methods:** We curated 100 self-contained Python snippets spanning elementary routines to nested algorithmic problems, extracting seven structural properties: lines of code (LOC), nesting depth, loop count, condition count, function definitions, unique variable count, and cyclomatic complexity. Thirty-two computing students rated snippet difficulty (1–10) and their confidence (1–5) across four balanced forms. Each student reviewed 25 distinct snippets and one repeat verification item, yielding 832 total ratings (8 evaluations per snippet). After data quality verification and reliability checks (test-retest consistency and one-way intraclass correlation), we trained OLS linear regression, ridge, lasso, elastic net, and random forest models. All models underwent repeated 5-fold cross-validation (5 repeats, 25 folds) benchmarked against mean-only, LOC-only, and cyclomatic-only baselines. Predictor influence was assessed using bootstrap standardised coefficients and permutation importance.

**Results:** Every submitted rating passed our quality filters. Students showed strong self-consistency, averaging an absolute test-retest difference of 0.94 points, with no student exceeding the 3-point threshold. Inter-rater reliability reached $\text{ICC}(1) = 0.67$ (95% CI [0.60, 0.74]) for single ratings and $\text{ICC}(1,k) = 0.94$ (95% CI [0.92, 0.96]) for the snippet-aggregated means. Linear and regularised models delivered the best predictive accuracy (Linear: $\text{RMSE} = 0.685 \pm 0.129$, $R^2 = 0.877 \pm 0.061$; Elastic Net: $\text{RMSE} = 0.689 \pm 0.120$, $R^2 = 0.877 \pm 0.057$; Ridge: $\text{RMSE} = 0.690 \pm 0.123$, $R^2 = 0.875 \pm 0.061$; Lasso: $\text{RMSE} = 0.690 \pm 0.122$, $R^2 = 0.876 \pm 0.057$). These approaches cut intercept-only baseline error by 67% and clearly outperformed random forest ($\text{RMSE} = 0.837 \pm 0.103$, $R^2 = 0.822 \pm 0.060$). Nesting depth emerged as the primary driver (ridge $\beta = 0.89$ [95% CI: 0.69, 1.07]), followed by lines of code ($0.47$ [0.23, 0.69]), loop count ($0.42$ [0.23, 0.62]), and variable count ($0.38$ [0.15, 0.61]). Cyclomatic complexity provided zero additional predictive value over raw loop and condition counts because of exact mathematical collinearity ($\text{VIF} > 10^{14}$).

**Discussion:** How deep control flow nests and how long the code runs dominate how difficult students perceive a program to be. McCabe's cyclomatic complexity offers no predictive benefit beyond basic counts of loops and conditional branches. Because regularised linear models achieve strong accuracy and remain easy to interpret, educational platforms can use simple structural metrics to order exercises by difficulty without relying on composite software metrics.

**Keywords:** code difficulty; perceived complexity; cyclomatic complexity; regression analysis; regularised regression; programming education; novice programmers.

---

## 1. Introduction

Knowing how challenging a snippet of code feels to a reader serves immediate practical needs. Instructors need to order assignments logically, curriculum tools need to spot confusing examples early, and maintenance teams need to allocate review time where code is hardest to trace. For decades, software engineering has tried to quantify this through automated source-code metrics. McCabe proposed cyclomatic complexity to count independent execution paths [1], Halstead focused on operand and operator volumes [2], and Campbell later introduced Cognitive Complexity to penalise nested breaks in linear reading flow [3].

Another line of inquiry asks developers directly. Buse and Weimer pioneered empirical readability modelling from human reviews of short code fragments [4], a technique later refined using smaller metric sets [5] and lexical features [6]. Yet when researchers evaluated standard structural metrics against objective developer comprehension, the correlation was frequently weak or inconsistent [7]. Compounding the problem, most structural indicators scale directly with overall code length, making it hard to tell whether a metric measures cognitive friction or merely line count [8].

Three main issues motivate our work. First, most earlier datasets capture professional developers or general readability, leaving novice learners understudied despite being the group most sensitive to structural hurdles. Second, many studies collapse ratings into binary buckets ("readable" vs "unreadable"), discarding granular variance that continuous regression models can exploit. Third, because basic structural features overlap heavily, fitting standard least-squares regression invites severe multicollinearity; whether composite metrics like cyclomatic complexity contribute any unique signal beyond raw lines, loops, and branches is rarely tested directly under regularised conditions.

This paper makes an empirical contribution rather than proposing a new metric. We apply standard regression pipelines under a transparent protocol: automated data cleaning, test-retest and inter-rater reliability checks, multicollinearity diagnostics, repeated cross-validation against simple baselines, and bootstrap feature importance. Given our 100-snippet sample, we prioritize parsimony, validation stability, and honest uncertainty intervals. Three questions guide this evaluation:

* **RQ1:** Which source-code features carry the greatest weight in student assessments of coding difficulty?
* **RQ2:** Does cyclomatic complexity improve difficulty predictions when basic structural counts are already present?
* **RQ3:** Which regression family—linear, penalised, or tree-based—yields the most accurate cross-validated predictions?

---

## 2. Methodology

![Figure 1: Study Pipeline](file:///d:/personal/code/python/ds/jupyter/fig1_study_pipeline.png)  
*Figure 1. Overview of the study pipeline, from snippet collection and feature extraction through rating collection, cleaning, modelling and interpretation.*

### 2.1 Snippet Corpus
We assembled 100 short Python programs (S001 to S100) drawn from introductory CS coursework, algorithmic repositories, and textbook problem sets. The corpus covers a wide structural span, ranging from basic sequential math routines and string scripts to multi-way branching, recursive calls, nested loops, and dynamic programming tables. To ensure participants judged program logic rather than visual presentation, all snippets were stripped of comments and docstrings, standard formatted to PEP 8, and kept short enough to read within one to two minutes.

### 2.2 Feature Extraction
We parsed each snippet with Python's built-in `ast` module and calculated complexity metrics via `radon` (Table 1). In line with McCabe's standard definition, cyclomatic complexity was computed as one plus the number of decision points (including `if`, `elif`, `for`, `while`, and short-circuit boolean operations). We verified the script by hand-counting metrics across 10 randomly drawn snippets; the manual and automated counts matched perfectly across all dimensions.

### 2.3 Collection of Difficulty Ratings
Ratings were gathered via four Google Forms surveys (Forms A through D), each containing 25 distinct snippets stratified by cyclomatic complexity to balance difficulty across forms. Every form included 26 rating tasks: the 25 target snippets plus a duplicate of the first snippet placed at the very end to check intra-rater consistency. 

Thirty-two computing students (undergraduate and postgraduate) participated, yielding 8 independent ratings per form and exactly 8 ratings per snippet across 832 total submissions. For each snippet, participants rated difficulty on a 1-to-10 scale ($1 = \text{very easy}$, $10 = \text{very hard}$) and indicated confidence on a 1-to-5 scale ($1 = \text{very unconfident}$, $5 = \text{highly confident}$). Students were instructed to inspect snippets visually without executing code or consulting outside resources. Participation was voluntary, uncompensated, and approved under institutional ethics clearance using randomized participant IDs.

### 2.4 Data Cleaning
Submissions were converted into a long-format table (one row per student-snippet pair). We set pre-registered exclusion criteria: absence of consent, out-of-range difficulty or confidence entries, malformed participant identifiers, unrecognized snippet keys, or duplicate attempts outside the designated verification check (where only the initial entry was kept). Table 2 logs the cleaning results.

### 2.5 Reliability Assessment
We evaluated data reliability on two levels. For individual consistency, we computed the absolute score shift between the first and repeat presentations of the check snippet; students deviating by more than 3 points were flagged for sensitivity checks. For group-level consistency across raters, we calculated one-way random-effects intraclass correlation coefficients: $\text{ICC}(1)$ for individual submissions and $\text{ICC}(1,k)$ for the mean across $k=8$ raters [9, 10].

### 2.6 Construction of the Modelling Table
The regression target was each snippet's mean difficulty rating, derived exclusively from the non-repeat entries. The final modelling table contained 100 rows, with seven structural predictors and one continuous outcome. Identifiers, form labels, and rating counts were omitted from the predictor matrix to prevent data leakage. Averaging across raters smoothed out idiosyncratic scoring habits, yielding a stable snippet-level benchmark.

### 2.7 Exploratory Analysis and Outlier Screening
We inspected target distributions via histograms and boxplots, generated scatter plots for each predictor against mean difficulty, and evaluated both Pearson and Spearman correlation matrices. We screened for potential data anomalies using the $1.5 \times \text{IQR}$ rule. Any flagged observations were reviewed manually; genuine code snippets were preserved regardless of their influence on overall fit. To detect undue leverage during linear fitting, we calculated Cook's distance [11].

### 2.8 Multicollinearity Diagnostics
Because cyclomatic complexity is built from loop and conditional statements, strong collinearity was inevitable. We computed Variance Inflation Factors (VIF) for each predictor, treating values above 5 to 10 as clear evidence of severe collinearity [12].

### 2.9 Regression Models
We benchmarked five regression architectures: Ordinary Least Squares (OLS), Ridge ($L_2$ regularisation), Lasso ($L_1$ shrinkage and selection), Elastic Net (blended $L_1/L_2$), and Random Forest (300 trees, `min_samples_leaf` = 2). Hyperparameters ($\alpha$ and $l_1$ ratio) were tuned strictly through internal cross-validation nested within each training fold.

### 2.10 Baselines and Validation
To contextualize performance, we tracked four reference models: a dummy intercept-only baseline predicting the training mean, a single-variable LOC model, a single-variable cyclomatic complexity model, and a linear model containing all features except cyclomatic complexity (targeting RQ2). With 100 snippets, simple train-test splits risk severe partition variance. We therefore ran repeated 5-fold cross-validation (5 repeats, 25 distinct validation folds). Standardisation was fitted only on the training slice of each fold to prevent information leakage [17].

### 2.11 Performance Metrics
Models were judged on root mean squared error (RMSE), mean absolute error (MAE), coefficient of determination ($R^2$), and Spearman rank correlation between actual and out-of-fold predicted scores. Results are reported as mean $\pm$ standard deviation across folds. Given fold dependencies, model differences were evaluated descriptively; models within one standard deviation were considered statistically comparable.

### 2.12 Feature Importance
To answer RQ1, we inspected standardised coefficients from Ridge and Lasso models fitted across the complete corpus, calculating 95% confidence intervals from 1,000 bootstrap resamples. For non-linear reference, we computed permutation feature importance from the Random Forest on out-of-fold data. RQ2 was addressed by contrasting the full feature model against the model omitting cyclomatic complexity and the cyclomatic-only baseline.

### 2.13 Sensitivity Analyses
We verified stability via four checks: re-running fits after removing raters with the largest test-retest gaps, swapping snippet means for snippet medians as the target, fitting models without high Cook's distance observations, and running a linear mixed-effects model on the raw 800 individual ratings with random intercepts per student to control for rater clustering.

### 2.14 Software Environment
Pipelines were built in Python 3.14.6 using pandas 3.0.3 and NumPy 2.5.1 for data handling, `ast` and `radon` for metric extraction, scikit-learn 1.9.0 [19] for cross-validation and models, statsmodels 0.15.0 [20] for diagnostics, and Matplotlib 3.11.1 for plots.

---

## 3. Results

### 3.1 Dataset Verification and Cleaning
All 32 enrolled students completed their assigned questionnaires, producing 832 total ratings (800 unique and 32 repeat checks). The data was exceptionally clean: no records were missing, no ratings fell outside the 1–10 or 1–5 bounds, all participant codes were valid, and no accidental duplicates occurred (Table 2). The final dataset retained all 100 snippets, each supported by exactly 8 independent first-pass evaluations ($\text{mean} = 8.0, \text{range} = 8\text{ to }8$).

### 3.2 Reliability of Ratings
Intra-rater consistency proved strong. Across all 32 students, the mean absolute gap between initial and repeat snippet ratings was 0.94 points ($\text{SD} = 0.91$). Thirteen students scored the repeat snippet identically, 11 differed by 1 point, 5 differed by 2 points, and 3 differed by 3 points. Not a single participant exceeded the 3-point exclusion limit. 

Inter-rater agreement supported this trend. The single-rater intraclass correlation was $\text{ICC}(1) = 0.67$ (95% CI [0.60, 0.74]), indicating reasonable consistency for individual students. When averaged across 8 raters, reliability rose to $\text{ICC}(1,k) = 0.94$ (95% CI [0.92, 0.96]), providing a solid, low-noise dependent variable for regression modelling.

### 3.3 Descriptive Statistics and Correlations
Table 3 outlines descriptive statistics for all variables. Snippet mean difficulty spanned almost the entire scale, from 1.500 to 9.375 ($\text{mean} = 5.258, \text{SD} = 2.068, \text{median} = 5.125, \text{IQR} = 3.125$). Snippet lengths ranged from 3 to 39 lines ($\text{mean} = 20.02, \text{SD} = 9.82$), nesting reached up to 4 levels ($\text{mean} = 1.97, \text{SD} = 1.23$), and cyclomatic complexity ranged between 1 and 9 ($\text{mean} = 4.65, \text{SD} = 2.15$).

![Figure 2: Distribution of Mean Difficulty](file:///d:/personal/code/python/ds/jupyter/fig2_difficulty_distribution.png)  
*Figure 2. Distribution of mean difficulty ratings across the snippets (N = 100).*

Every structural metric correlated positively and significantly ($p < 0.001$) with perceived difficulty. The strongest associations came from lines of code (Pearson $r = 0.887$; Spearman $\rho = 0.890$) and nesting depth ($r = 0.886$; $\rho = 0.887$), followed by cyclomatic complexity ($r = 0.844$), variable count ($r = 0.826$), loop count ($r = 0.758$), condition count ($r = 0.690$), and function count ($r = 0.614$). 

![Figure 3: Scatter Plots](file:///d:/personal/code/python/ds/jupyter/fig3_scatter_plots.png)  
*Figure 3. Scatter plots of each source-code feature against the mean difficulty rating.*

![Figure 4: Correlation Matrix](file:///d:/personal/code/python/ds/jupyter/fig4_correlation_matrix.png)  
*Figure 4. Correlation matrix of the seven features and the mean difficulty rating.*

The $1.5 \times \text{IQR}$ screen detected zero distributional outliers across any feature or the target. In the OLS fit, six snippets exceeded the $4/N = 0.04$ threshold for Cook's distance (S015, S022, S032, S042, S056, and S092), led by S056 at $D = 0.419$. Manual inspection showed these were legitimate code samples with genuine ratings, and none approached the cutoff of $D \ge 1.0$. We retained all of them.

### 3.4 Multicollinearity
Table 4 displays the Variance Inflation Factors. Fitting all seven predictors at once produced infinite VIFs ($> 10^{14}$) for loops, conditions, and cyclomatic complexity. This reflects exact structural rank deficiency: across our snippets, cyclomatic complexity is literally $\text{loops} + \text{conditions} + 1$, leaving the design matrix at rank 7 of 8. 

Dropping cyclomatic complexity resolved the issue completely. VIFs for the remaining six predictors fell well below conventional risk thresholds: LOC was 5.92, nesting depth was 3.45, variables was 3.96, conditions was 2.59, loops was 2.13, and functions was 1.92.

### 3.5 Model Comparison
Cross-validation benchmarks across the 25 evaluation folds appear in Table 5. Standard linear regression and elastic net delivered the lowest errors and highest explained variance (Linear: $\text{RMSE} = 0.685 \pm 0.129, \text{MAE} = 0.533 \pm 0.082, R^2 = 0.877 \pm 0.061$; Elastic Net: $\text{RMSE} = 0.689 \pm 0.120, R^2 = 0.877 \pm 0.057$). Ridge and lasso yielded near-identical figures (both $\text{RMSE} = 0.690, R^2 \approx 0.876$). Compared to the intercept-only baseline ($\text{RMSE} = 2.061 \pm 0.186$), the linear family slashed prediction error by 67%. 

Performance differences across all four linear variants were tiny (within 0.005 on RMSE and $R^2$), sitting comfortably inside one standard deviation. By contrast, Random Forest lagged noticeably behind ($\text{RMSE} = 0.837 \pm 0.103, \text{MAE} = 0.690 \pm 0.081, R^2 = 0.822 \pm 0.060$). On a dataset of this size and dimensionality, regularised linear models generalise better than decision trees.

![Figure 5: Model Comparison](file:///d:/personal/code/python/ds/jupyter/fig5_model_comparison.png)  
*Figure 5. Cross-validated RMSE and R-squared of each model and baseline, with error bars showing the standard deviation across folds.*

![Figure 7: Actual vs Predicted](file:///d:/personal/code/python/ds/jupyter/fig7_actual_vs_predicted_best.png)  
*Figure 7. Observed versus out-of-fold predicted mean difficulty for the best-performing model.*

### 3.6 Role of Cyclomatic Complexity
Addressing RQ2, removing cyclomatic complexity from the full linear model caused no loss in predictive power. Both specifications yielded identical cross-validated scores ($\text{RMSE} = 0.685 \pm 0.129, \text{MAE} = 0.533 \pm 0.082, R^2 = 0.877 \pm 0.061$). 

Looking at single-feature baselines confirms this redundancy. Lines of code alone ($\text{RMSE} = 0.957 \pm 0.105, R^2 = 0.769 \pm 0.066$) substantially outperformed cyclomatic complexity alone ($\text{RMSE} = 1.121 \pm 0.124, R^2 = 0.674 \pm 0.139$). Cyclomatic complexity adds nothing when constituent counts are present, and acts as a weaker difficulty proxy than basic code length when evaluated in isolation.

### 3.7 Feature Influence
Table 6 reports standardised coefficients (with 1,000 bootstrap resamples) alongside permutation importances. Nesting depth had by far the largest impact on student difficulty ratings (Ridge: $0.89$, 95% CI [0.69, 1.07]; Lasso: $0.90$, 95% CI [0.67, 1.12]). A one standard deviation increase in nesting depth drove perceived difficulty up by roughly 0.9 points. Lines of code (Ridge: $0.47$ [0.23, 0.69]), loops ($0.42$ [0.23, 0.62]), and variable counts ($0.38$ [0.15, 0.61]) also maintained positive, non-zero weights across 100% of bootstrap runs.

In contrast, isolated conditions and function definitions had negligible influence. Ridge coefficients hovered near zero ($0.06$ and $-0.06$), while Lasso retained conditions in only 44% of resamples and functions in 24%. Random Forest permutation rankings agreed with these trends, placing lines of code first (mean RMSE increase $= 0.777$), followed by student background ($0.239$) and nesting depth ($0.210$).

![Figure 6: Feature Influence](file:///d:/personal/code/python/ds/jupyter/fig6_feature_influence.png)  
*Figure 6. Feature importance from standardised coefficients (with 95% bootstrap CIs) and random forest permutation importance.*

### 3.8 Sensitivity Analyses
All four sensitivity checks confirmed the stability of these findings:
- Excluding the three students with the maximum test-retest gap of 3 points ($N=29$ raters) produced almost identical cross-validation scores ($\text{RMSE} = 0.682 \pm 0.127, R^2 = 0.878 \pm 0.058$).
- Switching the prediction target from the mean to the median snippet difficulty yielded congruent results ($\text{RMSE} = 0.826, R^2 = 0.840$, Pearson $r$ between mean and median $= 0.989$) with unchanged feature order: nesting depth led ($0.924$), followed by variables ($0.512$), LOC ($0.467$), and loops ($0.451$).
- Removing influential snippet S056 ($D = 0.419$) lowered error to $\text{RMSE} = 0.627 \pm 0.086$ ($R^2 = 0.893$) without shifting feature weights. Removing all six snippets with $D > 0.04$ further improved fit ($\text{RMSE} = 0.565, R^2 = 0.912$) while maintaining the same relative hierarchy.
- A linear mixed-effects model fitted on all 800 raw ratings corroborated significant fixed effects for nesting depth ($B = 0.783, p < 0.001$), LOC ($B = 0.044, p < 0.001$), loops ($B = 0.399, p < 0.001$), variables ($B = 0.161, p < 0.001$), and student experience level ($B = -0.605, p < 0.001$). Conditions and functions remained non-significant, showing that our conclusions hold whether looking at aggregated snippet means or hierarchical student-level ratings.

---

## 4. Discussion

Our results provide direct answers to the three research questions. For RQ1, nesting depth and code length dominate student-perceived difficulty, with loop and variable counts playing secondary roles; standalone conditionals and function counts had little effect. For RQ2, classical cyclomatic complexity contributed no unique predictive value over elementary structural counts, suffering from exact linear dependency on loops and branches. For RQ3, simple regularised linear models (OLS, Ridge, Lasso, Elastic Net) predicted perceived difficulty with strong accuracy ($R^2 = 0.877, \text{RMSE} = 0.685$), handily outperforming Random Forest.

These outcomes align closely with cognitive models of programming. Following deeply nested code forces learners to juggle multiple execution contexts in working memory, creating immediate cognitive friction. Simple line count matters for a similar reason: longer programs demand more sustained attention and state-tracking. 

The complete redundancy of McCabe's metric in this setting has direct educational relevance. Cyclomatic complexity has been taught for years as a general benchmark for how hard code is to understand. Yet in our data, it offers no information beyond separate counts of loops and branch statements, and it performs noticeably worse than plain line count as a standalone indicator. This mirrors Gil and Lalouche's broader observation that many popular complexity metrics primarily mirror program size rather than intrinsic structural challenge [8]. 

Our findings also demonstrate that complex non-linear models are not always necessary for educational metrics. Random forest overfitted the smaller training subsets and generalised less effectively than regularised linear models. Linear and elastic net models not only produced smaller prediction errors, but their coefficients translate directly into actionable guidelines for curriculum designers.

**Limitations.** Our findings should be interpreted with several caveats in mind. First, the corpus was restricted to 100 short Python scripts. While representative of common introductory programming coursework, these short fragments do not mirror the modular architectures, class hierarchies, or third-party dependencies of large software systems. Second, our participant pool drew from computing students at a single university. While suitable for studying novice perception, these patterns may not reflect how seasoned software engineers read code, as senior developers rely on richer mental templates and idiom recognition. Third, we measured subjective perceived difficulty rather than objective comprehension performance (e.g., bug-fix speed, tracing error rates, or gaze fixations). Even though aggregated ratings proved highly reliable ($\text{ICC}(1,k) = 0.94$), perceived effort does not always correlate with actual task failure. Fourth, we restricted our feature set strictly to syntax, omitting semantic factors like variable name quality, algorithm familiarity, and visual formatting. Finally, each snippet received eight independent evaluations, leaving some minor residual noise in individual snippet means.

**Future Work.** A natural next step is replicating this protocol across other languages (such as Java, C++, or functional languages) to test whether the dominant role of nesting depth holds regardless of syntax style. Combining subjective difficulty scores with objective metrics—like code-tracing response times or comprehension error rates—would clarify the relationship between felt difficulty and actual task performance. Finally, pairing structural metrics with semantic indicators, such as identifier clarity metrics or large language model prompt representations, could capture the interplay between syntax and conceptual complexity.

---

## 5. Conclusion

This study tested whether student-perceived coding difficulty can be predicted from automatically extractable source-code metrics across 100 Python snippets evaluated by 32 computing students. We found that perceived difficulty is highly predictable using simple linear and regularised models ($R^2 = 0.877, \text{RMSE} = 0.685$). Control-flow nesting depth and lines of code served as the primary drivers of student difficulty ratings. Cyclomatic complexity provided no predictive benefit over simple structural counts and proved inferior to raw code length as an isolated metric. These results suggest that educators, curriculum planners, and automated assessment platforms can evaluate and sequence coding exercises using transparent, elementary structural features without relying on composite software metrics.

---

## Declarations & Statements

**Data Availability Statement:**  
The full snippet corpus, extracted feature tables, anonymised student ratings, analysis scripts, and trained model objects are available at https://github.com/empirical-cs-ed/code-difficulty-prediction.

**Ethics Statement:**  
This research was approved by the Institutional Review Board (ref: CS-EDU-2026-0814). All participants gave written informed consent before completing the questionnaires. Participation was voluntary, uncompensated, and fully anonymised.

**Author Contributions:**  
* **Muhammad Rehan:** Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualization, Writing – original draft.
* **Issa Therase Delson:** Investigation, Data curation, Software, Methodology, Writing – review & editing.
* **Khadar Babu SK:** Conceptualization, Project administration, Supervision, Validation, Writing – review & editing.

**Funding:**  
The authors received no external funding for the research, authorship, or publication of this article.

**Acknowledgements:**  
We thank the 32 computer science students who participated in this study, along with the faculty department for facilitating data collection.

**Conflict of Interest Statement:**  
The authors declare no competing financial interests or personal relationships that could have influenced this work.

---

## References

[1] McCabe TJ. A complexity measure. *IEEE Transactions on Software Engineering*. 1976;SE-2(4):308-320.  
[2] Halstead MH. *Elements of Software Science*. New York: Elsevier; 1977.  
[3] Campbell GA. Cognitive Complexity: an overview and evaluation. In: *Proceedings of the International Conference on Technical Debt*. 2018.  
[4] Buse RPL, Weimer WR. Learning a metric for code readability. *IEEE Transactions on Software Engineering*. 2010;36(4):546-558.  
[5] Posnett D, Hindle A, Devanbu P. A simpler model of software readability. In: *Proceedings of the 8th Working Conference on Mining Software Repositories*. 2011.  
[6] Scalabrino S, Linares-Vasquez M, Poshyvanyk D, Oliveto R. Improving code readability models with textual features. In: *Proceedings of the 24th IEEE International Conference on Program Comprehension*. 2016.  
[7] Scalabrino S, Bavota G, Vendome C, Linares-Vasquez M, Poshyvanyk D, Oliveto R. Automatically assessing code understandability. *IEEE Transactions on Software Engineering*. 2021;47(3):595-613.  
[8] Gil JY, Lalouche G. On the correlation between size and metric validity. *Empirical Software Engineering*. 2017;22:2585-2611.  
[9] Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. *Psychological Bulletin*. 1979;86(2):420-428.  
[10] Koo TK, Li MY. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*. 2016;15(2):155-163.  
[11] Cook RD. Detection of influential observation in linear regression. *Technometrics*. 1977;19(1):15-18.  
[12] O'Brien RM. A caution regarding rules of thumb for variance inflation factors. *Quality and Quantity*. 2007;41:673-690.  
[13] Hoerl AE, Kennard RW. Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*. 1970;12(1):55-67.  
[14] Tibshirani R. Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*. 1996;58(1):267-288.  
[15] Zou H, Hastie T. Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*. 2005;67(2):301-320.  
[16] Breiman L. Random forests. *Machine Learning*. 2001;45:5-32.  
[17] Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91.  
[18] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*. 2017.  
[19] Pedregosa F, et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*. 2011;12:2825-2830.  
[20] Seabold S, Perktold J. Statsmodels: econometric and statistical modeling with Python. In: *Proceedings of the 9th Python in Science Conference*. 2010.

---

## Tables

### Table 1: Source-code features used as independent variables
| Feature | Definition | Tool |
| :--- | :--- | :--- |
| Lines of code (LOC) | Number of source lines excluding blank lines and comments | `radon` |
| Nesting depth | Maximum depth of nested loops and conditional blocks | `ast` |
| Loops | Number of `for` and `while` statements | `ast` |
| Conditions | Number of `if`, `elif` and conditional expressions | `ast` |
| Functions | Number of function definitions | `ast` |
| Variables | Number of distinct assigned identifiers | `ast` |
| Cyclomatic complexity | Number of decision points plus one | `radon` |

### Table 2: Data cleaning log (Rows removed under each rule)
| Rule | Rows removed |
| :--- | :--- |
| Consent not given | 0 |
| Missing difficulty rating | 0 |
| Difficulty outside 1 to 10 | 0 |
| Confidence outside 1 to 5 | 0 |
| Invalid participant code | 0 |
| Unknown snippet identifier | 0 |
| Duplicate submission | 0 |
| **Total removed / retained** | **0 / 832 (800 non-repeat ratings retained)** |

### Table 3: Descriptive statistics of features and mean difficulty rating ($N = 100$ snippets)
| Variable | Mean | SD | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| Lines of Code (LOC) | 20.02 | 9.82 | 3.00 | 39.00 |
| Nesting depth | 1.97 | 1.23 | 0.00 | 4.00 |
| Loops | 1.65 | 1.09 | 0.00 | 4.00 |
| Conditions | 2.00 | 1.44 | 0.00 | 5.00 |
| Functions | 1.94 | 0.83 | 1.00 | 4.00 |
| Variables | 5.75 | 2.61 | 1.00 | 11.00 |
| Cyclomatic complexity | 4.65 | 2.15 | 1.00 | 9.00 |
| **Mean difficulty rating** | **5.26** | **2.07** | **1.50** | **9.38** |

### Table 4: Variance inflation factors (VIF) of the seven features
| Feature | VIF (All 7 Features) | VIF (Excluding Cyclomatic) |
| :--- | :--- | :--- |
| LOC | 5.92 | 5.92 |
| Nesting depth | 3.45 | 3.45 |
| Loops | $> 10^{14}$* | 2.13 |
| Conditions | $> 10^{14}$* | 2.59 |
| Functions | 1.92 | 1.92 |
| Variables | 3.96 | 3.96 |
| Cyclomatic complexity | $> 10^{14}$* | — *(Omitted: Collinear)* |

*\*Note: Exact mathematical collinearity ($\text{cyclomatic\_complexity} = \text{loops} + \text{conditions} + 1$). Rank is 7 of 8 with intercept.*

### Table 5: Cross-validated performance of regression models and baselines (Mean ± SD across 25 folds)
| Model | RMSE | MAE | $R^2$ | Spearman $\rho$ |
| :--- | :--- | :--- | :--- | :--- |
| Intercept only | 2.061 (0.186) | 1.742 (0.172) | -0.034 (0.040) | 0.000 (0.000) |
| LOC only | 0.957 (0.105) | 0.789 (0.093) | 0.769 (0.066) | 0.866 (0.044) |
| Cyclomatic only | 1.121 (0.124) | 0.914 (0.127) | 0.674 (0.139) | 0.825 (0.074) |
| **All features except cyclomatic** | **0.685 (0.129)** | **0.533 (0.082)** | **0.877 (0.061)** | **0.928 (0.031)** |
| **Linear regression (Full)** | **0.685 (0.129)** | **0.533 (0.082)** | **0.877 (0.061)** | **0.928 (0.031)** |
| Ridge | 0.690 (0.123) | 0.540 (0.083) | 0.875 (0.061) | 0.926 (0.033) |
| Lasso | 0.690 (0.122) | 0.542 (0.077) | 0.876 (0.057) | 0.928 (0.033) |
| Elastic net | 0.689 (0.120) | 0.541 (0.077) | 0.877 (0.057) | 0.927 (0.033) |
| Random forest | 0.837 (0.103) | 0.690 (0.081) | 0.822 (0.060) | 0.891 (0.042) |

### Table 6: Feature influence from standardised coefficients and permutation importance
| Feature | Ridge coefficient (95% CI) | Lasso coefficient (95% CI) | Lasso selected (% resamples) | RF Permutation Importance (RMSE inc, Mean (SD)) |
| :--- | :--- | :--- | :--- | :--- |
| **LOC** | 0.47 [0.23, 0.69] | 0.46 [0.12, 0.69] | 100% | 0.777 (0.190) |
| **Nesting depth** | 0.89 [0.69, 1.07] | 0.90 [0.67, 1.12] | 100% | 0.210 (0.117) |
| **Loops** | 0.42 [0.23, 0.62] | 0.39 [0.19, 0.61] | 100% | 0.047 (0.066) |
| **Variables** | 0.38 [0.15, 0.61] | 0.36 [0.11, 0.59] | 100% | 0.020 (0.033) |
| **Conditions** | 0.06 [-0.14, 0.27] | 0.00 [0.00, 0.24] | 44% | -0.011 (0.021) |
| **Functions** | -0.06 [-0.24, 0.14] | -0.00 [-0.08, 0.08] | 24% | -0.002 (0.018) |
| **Cyclomatic complexity** | — *(Omitted: Collinear)* | — *(Omitted: Collinear)* | — | 0.032 (0.015) |
| *Student Experience Level* | -0.54 [-0.63, -0.44] | -0.53 [-0.61, -0.43] | 100% | 0.239 (0.070) |
| *Python Comfort (1-5)* | -0.07 [-0.17, 0.01] | -0.02 [-0.12, 0.00] | 64% | 0.039 (0.020) |
