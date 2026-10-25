import os
import ast
import importlib
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

try:
    joblib = importlib.import_module("joblib")
except ModuleNotFoundError:
    joblib = None


def load_model(path):
    if joblib is not None:
        return joblib.load(path)
    with open(path, "rb") as model_file:
        return pickle.load(model_file)

# ==============================================================================
# 1. LOAD BEST TRAINED MODELS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LINEAR_MODEL_PATH = os.path.join(BASE_DIR, "linear_model.pkl")
LINEAR_REG_USER_PATH = os.path.join(BASE_DIR, "linear_regression_model.pkl")
RANDOM_FOREST_PATH = os.path.join(BASE_DIR, "random_forest_difficulty.pkl")

# Best model: Linear Regression (R² = 0.877, RMSE = 0.685 on snippet test-folds)
linear_model = None
user_adjusted_model = None
rf_model = None

try:
    if os.path.exists(LINEAR_MODEL_PATH):
        linear_model = load_model(LINEAR_MODEL_PATH)
        print("[OK] Loaded best linear_model.pkl (7 code metrics)")
    if os.path.exists(LINEAR_REG_USER_PATH):
        user_adjusted_model = load_model(LINEAR_REG_USER_PATH)
        print("[OK] Loaded user_adjusted linear_regression_model.pkl (9 metrics)")
    if os.path.exists(RANDOM_FOREST_PATH):
        rf_model = load_model(RANDOM_FOREST_PATH)
        print("[OK] Loaded random_forest_difficulty.pkl")
except Exception as e:
    print(f"Warning loading models: {e}")

# ==============================================================================
# 2. AST CODE METRICS ANALYZER
# ==============================================================================
class CodeMetricsAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.loops = 0
        self.conditions = 0
        self.functions = 0
        self.variables = set()
        self.cyclomatic_complexity = 1  # Base complexity of module

    def visit_If(self, node):
        self.conditions += 1
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.loops += 1
        self.cyclomatic_complexity += 1
        if isinstance(node.target, ast.Name):
            self.variables.add(node.target.id)
        elif isinstance(node.target, (ast.Tuple, ast.List)):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.variables.add(elt.id)
        self.generic_visit(node)

    def visit_While(self, node):
        self.loops += 1
        self.cyclomatic_complexity += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.functions += 1
        self.cyclomatic_complexity += 1
        for arg in node.args.args:
            self.variables.add(arg.arg)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.functions += 1
        self.cyclomatic_complexity += 1
        for arg in node.args.args:
            self.variables.add(arg.arg)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.cyclomatic_complexity += 1
        if node.name:
            self.variables.add(node.name)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # 'and' / 'or' increase decision points
        self.cyclomatic_complexity += len(node.values) - 1
        for val in node.values:
            self.visit(val)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.variables.add(elt.id)
        self.generic_visit(node)


def calculate_max_nesting(node, current_depth=0):
    """Calculates the maximum nesting depth of control flow blocks."""
    is_block = isinstance(
        node,
        (
            ast.If,
            ast.For,
            ast.While,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.With,
            ast.Try,
        ),
    )
    next_depth = current_depth + 1 if is_block else current_depth
    max_d = next_depth

    for child in ast.iter_child_nodes(node):
        max_d = max(max_d, calculate_max_nesting(child, next_depth))
    return max_d


def extract_code_features(source_code: str):
    """Parses Python source code and extracts 7 structural complexity metrics."""
    if source_code is None:
        source_code = ""

    lines = source_code.splitlines()
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

    if not source_code.strip():
        return {
            "lines_of_code": 0,
            "code_lines": 0,
            "nesting_depth": 0,
            "loops": 0,
            "conditions": 0,
            "functions": 0,
            "variables_count": 0,
            "variables_list": [],
            "cyclomatic_complexity": 1,
        }

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return {"error": f"Syntax Error on line {e.lineno}: {e.msg}"}
    except Exception as e:
        return {"error": f"Invalid Python syntax: {e}"}

    analyzer = CodeMetricsAnalyzer()
    analyzer.visit(tree)
    nesting_depth = max(0, calculate_max_nesting(tree) - 1)

    return {
        "lines_of_code": total_lines,
        "code_lines": code_lines,
        "nesting_depth": nesting_depth,
        "loops": analyzer.loops,
        "conditions": analyzer.conditions,
        "functions": analyzer.functions,
        "variables_count": len(analyzer.variables),
        "variables_list": sorted(list(analyzer.variables)),
        "cyclomatic_complexity": analyzer.cyclomatic_complexity,
    }


# ==============================================================================
# 3. PREDICTION & DIAGNOSTIC REASONING ENGINE
# ==============================================================================
EXPERIENCE_MAPPING = {
    "Beginner": 0,      # Less than 6 months
    "Intermediate": 1,  # 6 to 12 months / 1-2 years
    "Advanced": 2,      # 1 to 2 years
    "Professional": 3   # More than 2 years
}

def predict_difficulty_profile(features, experience="Intermediate", confidence=5):
    """
    Applies the best regression model (Linear Regression, R² = 0.877)
    to predict perceived coding difficulty and quality score.
    """
    if "error" in features or features.get("code_lines", 0) == 0:
        return None

    # Code features DataFrame matching training schema
    X_code = pd.DataFrame([{
        "LOC": features["code_lines"],
        "nesting_depth": features["nesting_depth"],
        "loops": features["loops"],
        "conditions": features["conditions"],
        "functions": features["functions"],
        "variables": features["variables_count"],
        "cyclomatic_complexity": features["cyclomatic_complexity"]
    }])

    # 1. Normative Difficulty Prediction (1 to 10 scale)
    if linear_model is not None:
        raw_pred = float(linear_model.predict(X_code)[0])
    else:
        # Fallback to empirical formula derived from the best linear model fit
        # intercept: 1.216, LOC: 0.0487, nesting: 0.754, loops: 0.255, conds: -0.120, funcs: -0.074, vars: 0.165, cc: 0.134
        raw_pred = (
            1.2157
            + 0.0487 * features["code_lines"]
            + 0.7543 * features["nesting_depth"]
            + 0.2547 * features["loops"]
            - 0.1203 * features["conditions"]
            - 0.0737 * features["functions"]
            + 0.1655 * features["variables_count"]
            + 0.1344 * features["cyclomatic_complexity"]
        )

    normative_score = float(np.clip(raw_pred, 1.0, 10.0))

    # 2. Personalized Difficulty Prediction (adjusting for experience & confidence)
    personalized_score = None
    exp_code = EXPERIENCE_MAPPING.get(experience, 1)
    comfort_1to5 = int(np.clip(round(confidence / 2.0), 1, 5))

    if user_adjusted_model is not None:
        X_user = X_code.copy()
        X_user["experience_level"] = exp_code
        X_user["python_comfort_1to5"] = comfort_1to5
        user_pred = float(user_adjusted_model.predict(X_user)[0])
        personalized_score = round(float(np.clip(user_pred, 1.0, 10.0)), 2)

    # 3. Categorization & Quality Score
    quality_score = int(np.clip(round(100.0 - (normative_score - 1.0) * 10.0), 10, 98))

    if normative_score < 3.5:
        category = "Introductory (Novice-Friendly)"
        tag = "Optimal Code Score"
        badge_class = "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
        text_color = "text-emerald-400"
        quality_label = "High Readability"
    elif normative_score < 5.5:
        category = "Moderate Complexity"
        tag = "Balanced Structure"
        badge_class = "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
        text_color = "text-cyan-400"
        quality_label = "Good Maintainability"
    elif normative_score < 7.5:
        category = "Elevated Cognitive Load"
        tag = "High Working Memory Load"
        badge_class = "bg-amber-500/20 text-amber-400 border border-amber-500/30"
        text_color = "text-amber-400"
        quality_label = "Refactoring Recommended"
    else:
        category = "High Structural Complexity"
        tag = "Critical Cognitive Burden"
        badge_class = "bg-rose-500/20 text-rose-400 border border-rose-500/30"
        text_color = "text-rose-400"
        quality_label = "Deep Nesting Alert"

    # Descriptive diagnostics based on primary drivers (Nesting Depth, LOC, Loops)
    reasons = []
    if features["nesting_depth"] >= 3:
        reasons.append(f"deep block nesting (level {features['nesting_depth']})")
    elif features["nesting_depth"] == 2:
        reasons.append("moderate block nesting (level 2)")

    if features["loops"] >= 2:
        reasons.append(f"{features['loops']} loop structures")
    elif features["loops"] == 1:
        reasons.append("a single loop")

    if features["code_lines"] > 25:
        reasons.append(f"lengthy implementation ({features['code_lines']} LOC)")

    if features["variables_count"] >= 6:
        reasons.append(f"tracking {features['variables_count']} active variables")

    driver_text = " and ".join(reasons) if reasons else "straightforward sequential control flow"
    description = (
        f"Difficulty prediction is driven by {driver_text}. "
        f"In our empirical study, nesting depth (β = 0.89) and lines of code (β = 0.47) "
        f"were identified as the strongest determinants of student-perceived difficulty."
    )

    return {
        "score": round(normative_score, 2),
        "personalized_score": personalized_score,
        "quality_score": quality_score,
        "category": category,
        "tag": tag,
        "badge_class": badge_class,
        "text_color": text_color,
        "quality_label": quality_label,
        "description": description
    }


# ==============================================================================
# 4. FLASK APPLICATION ROUTE
# ==============================================================================
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    features = None
    prediction = None
    error = None
    sample_code = ""
    experience = "Intermediate"
    confidence = 5

    if request.method == 'POST':
        sample_code = request.form.get("code-editor", "")
        experience = request.form.get("experience", "Intermediate")
        try:
            confidence = int(request.form.get("confidence", 5))
        except (ValueError, TypeError):
            confidence = 5

        if sample_code.strip():
            features = extract_code_features(sample_code)
            if "error" in features:
                error = features["error"]
                features = None
            else:
                prediction = predict_difficulty_profile(features, experience, confidence)

    return render_template(
        "index.html",
        features=features,
        prediction=prediction,
        error=error,
        sample_code=sample_code,
        experience=experience,
        confidence=confidence
    )


if __name__ == '__main__':
    app.run(debug=True)