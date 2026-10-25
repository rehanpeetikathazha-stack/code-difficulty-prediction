import numpy as np
import pandas as pd

rng = np.random.default_rng(42)          # fixed seed = same data every run

N_SNIPPETS = 100
N_STUDENTS = 32
FORMS = ["A", "B", "C", "D"]

# ---------- 1. Snippets with code features ----------
size = rng.uniform(0, 1, N_SNIPPETS)                       # hidden "how big/complex" factor
loc = np.round(3 + size * 35 + rng.normal(0, 2, N_SNIPPETS)).clip(3).astype(int)
nesting = np.round(size * 4 + rng.normal(0, 0.6, N_SNIPPETS)).clip(0, 5).astype(int)
loops = np.round(size * 3 + rng.normal(0, 0.7, N_SNIPPETS)).clip(0, 6).astype(int)
conditions = np.round(size * 4 + rng.normal(0, 1, N_SNIPPETS)).clip(0, 8).astype(int)
functions = np.round(1 + size * 2 + rng.normal(0, 0.5, N_SNIPPETS)).clip(1, 4).astype(int)
variables = np.round(2 + size * 8 + rng.normal(0, 1.2, N_SNIPPETS)).clip(1).astype(int)
cyclomatic = 1 + loops + conditions                         # standard definition (approx.)

snippets = pd.DataFrame({
    "snippet_id": [f"S{i+1:03d}" for i in range(N_SNIPPETS)],
    "LOC": loc, "nesting_depth": nesting, "loops": loops,
    "conditions": conditions, "functions": functions,
    "variables": variables, "cyclomatic_complexity": cyclomatic,
})

# "True" difficulty: ASSUMED relationship (nesting & loops matter most, cyclomatic adds little)
true_diff = (1.0 + 0.05 * loc + 0.70 * nesting + 0.45 * loops
             + 0.10 * conditions + 0.12 * variables + rng.normal(0, 0.5, N_SNIPPETS))
snippets["true_difficulty"] = true_diff                     # hidden: dropped before saving

# Balanced forms: sort by complexity, deal out A,B,C,D
order = snippets.sort_values("cyclomatic_complexity").index
snippets.loc[order, "form"] = [FORMS[i % 4] for i in range(N_SNIPPETS)]

# ---------- 2. Students ----------
exp_levels = ["Less than 6 months", "6 to 12 months", "1 to 2 years", "More than 2 years"]
students = pd.DataFrame({
    "participant_code": rng.choice(np.arange(1000, 10000), N_STUDENTS, replace=False).astype(str),
    "form": [FORMS[i % 4] for i in range(N_STUDENTS)],      # 8 students per form
    "year_of_study": rng.choice(["1st year UG", "2nd year UG", "3rd year UG", "4th year UG", "Postgraduate"], N_STUDENTS),
    "programming_experience": rng.choice(exp_levels, N_STUDENTS, p=[0.2, 0.3, 0.3, 0.2]),
    "languages_known": rng.choice(["Python", "Python; Java", "Python; C / C++", "Java; JavaScript"], N_STUDENTS),
})
students["python_comfort_1to5"] = (students["programming_experience"].map({e: i + 2 for i, e in enumerate(exp_levels)})
                                   + rng.integers(-1, 2, N_STUDENTS)).clip(1, 5)
exp_shift = students["programming_experience"].map({e: 0.6 - 0.4 * i for i, e in enumerate(exp_levels)})  # experts rate lower
rater_bias = rng.normal(0, 0.8, N_STUDENTS)                 # some students rate high, some low

# ---------- 3. Ratings (long format, same columns as the workbook) ----------
rows = []
for si, stu in students.iterrows():
    form_snips = snippets[snippets["form"] == stu["form"]]
    for _, sn in form_snips.iterrows():
        for repeat in ("N",) if sn.name != form_snips.index[0] else ("N", "Y"):   # first snippet is repeated
            score = sn["true_difficulty"] + exp_shift[si] + rater_bias[si] + rng.normal(0, 1.0)
            rating = int(np.clip(round(score), 1, 10))
            conf = int(np.clip(round(5 - 0.35 * score + rng.normal(0, 0.8)), 1, 5))
            rows.append([stu["participant_code"], stu["form"], sn["snippet_id"], rating, conf, repeat])
ratings = pd.DataFrame(rows, columns=["participant_code", "form", "snippet_id",
                                      "difficulty_1to10", "confidence_1to5", "is_repeat"])

# ---------- 4. Mean rating per snippet (your regression target) ----------
main = ratings[ratings["is_repeat"] == "N"]
agg = main.groupby("snippet_id")["difficulty_1to10"].agg(num_ratings="count", mean_rating="mean").reset_index()
snippets = snippets.merge(agg, on="snippet_id")

snippets.drop(columns=["true_difficulty"]).to_csv("snippets.csv", index=False)
students.to_csv("participants.csv", index=False)
ratings.to_csv("ratings.csv", index=False)
print("snippets", snippets.shape, "| participants", students.shape, "| ratings", ratings.shape)