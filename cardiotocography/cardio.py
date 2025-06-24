import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


LABEL_COL = 'CLASS'
FEATURE_COLS = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']


def load_data(filepath):
    df = pd.read_excel(filepath, sheet_name='Raw Data')
    df = df[FEATURE_COLS + [LABEL_COL]]
    df = df.dropna(subset=[LABEL_COL])
    return df


def explore_data(df):
    print("Rozmiar danych:", df.shape)
    print("\nRozkład klas:")
    print(df[LABEL_COL].value_counts().sort_index())
    print("\nStatystyki opisowe:")
    print(df.describe(include='all'))
    print("\nBraki danych:")
    print(df.isnull().sum())


def prepare_variants(df):
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    variants = {'no_processing': (X_train, X_test)}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    variants['standardized'] = (X_train_scaled, X_test_scaled)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    variants['pca'] = (X_train_pca, X_test_pca)

    return variants, y_train, y_test


def evaluate_model(y_true, y_pred, model_name, variant_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"{model_name} ({variant_name}): Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

    return acc, prec, rec, f1


def plot_confusion_matrix(y_true, y_pred, model_name, variant_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Macierz pomyłek - {model_name} ({variant_name})")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"confusion_{model_name}_{variant_name}.png")
    plt.close()


def run_experiments(df):
    variants, y_train, y_test = prepare_variants(df)
    results = []

    for variant_name, (X_tr, X_te) in variants.items():
        for smooth in [1e-9, 1e-3, 1.0]:
            model = GaussianNB(var_smoothing=smooth)
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            acc, prec, rec, f1 = evaluate_model(y_test, y_pred, "NaiveBayes", variant_name)
            plot_confusion_matrix(y_test, y_pred, "NaiveBayes", variant_name)
            results.append(["NaiveBayes", variant_name, smooth, "-", acc, prec, rec, f1])

        for depth, crit in [(3, 'gini'), (5, 'entropy'), (None, 'gini')]:
            model = DecisionTreeClassifier(max_depth=depth, criterion=crit, random_state=42)
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            acc, prec, rec, f1 = evaluate_model(y_test, y_pred, "DecisionTree", variant_name)
            plot_confusion_matrix(y_test, y_pred, "DecisionTree", variant_name)
            results.append(["DecisionTree", variant_name, depth, crit, acc, prec, rec, f1])

    df_results = pd.DataFrame(results, columns=["Model", "Variant", "Param1",
                                                "Param2", "Accuracy", "Precision",
                                                "Recall", "F1"])
    df_results.to_csv("classification_results.csv", index=False)
    print("\nWyniki zapisane do classification_results.csv")


if __name__ == "__main__":
    df = load_data("CTG.xls")
    print("Eksploracja danych")
    explore_data(df)
    print("\nUruchamianie eksperymentów")
    run_experiments(df)