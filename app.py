
import streamlit as st
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="Forest Fire Prediction", page_icon="🌲", layout="wide")

st.title("🌲 Algerian Forest Fire Prediction")
st.write("Compare SVM, KNN and Naive Bayes using weather conditions.")

FEATURES = [
    "Temperature", "RH", "Ws", "Rain", "FFMC",
    "DMC", "DC", "ISI", "BUI", "FWI", "Region"
]
NUMERIC = FEATURES[:-1]

@st.cache_resource
def train_models():
    df = pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv", header=1)
    df.columns = [
        "day", "month", "year", "Temperature", "RH", "Ws",
        "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "Classes"
    ]
    df["Region"] = (df.index > 122).astype(int)
    df = df.dropna()
    df = df[df["day"] != "day"].copy()

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Classes"] = df["Classes"].str.strip()
    df = df.dropna(subset=NUMERIC + ["Classes"])
    df["Fire"] = (df["Classes"].str.lower() == "fire").astype(int)

    X = df[FEATURES]
    y = df["Fire"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grids = {
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf"))
            ]),
            {
                "classifier__C": [0.1, 1, 10, 100],
                "classifier__gamma": ["scale", 0.001, 0.01, 0.1]
            }
        ),
        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier())
            ]),
            {
                "classifier__n_neighbors": list(range(3, 31, 2)),
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"]
            }
        ),
        "Naive Bayes": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB())
            ]),
            {
                "classifier__var_smoothing": [
                    1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
                    1e-4, 1e-3, 1e-2, 1e-1
                ]
            }
        )
    }

    fitted = {}
    for name, (pipeline, params) in grids.items():
        search = GridSearchCV(
            pipeline, params, cv=cv, scoring="f1", n_jobs=-1
        )
        search.fit(X_train, y_train)
        fitted[name] = search

    return fitted, len(df)

try:
    with st.spinner("Preparing and tuning the three models..."):
        models, rows = train_models()
except Exception as e:
    st.error("Could not load or train the models. Check the CSV filename and data format.")
    st.exception(e)
    st.stop()

st.success(f"Models ready. Cleaned dataset contains {rows} observations.")

st.subheader("Enter weather conditions")

c1, c2, c3 = st.columns(3)

with c1:
    temperature = st.number_input("Temperature (°C)", 0.0, 55.0, 35.0)
    humidity = st.number_input("Relative Humidity (%)", 0.0, 100.0, 30.0)
    wind = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 15.0)
    rain = st.number_input("Rain (mm)", 0.0, 100.0, 0.0)

with c2:
    ffmc = st.number_input("FFMC", 0.0, 101.0, 90.0)
    dmc = st.number_input("DMC", 0.0, 300.0, 40.0)
    dc = st.number_input("DC", 0.0, 1000.0, 150.0)
    isi = st.number_input("ISI", 0.0, 100.0, 8.0)

with c3:
    bui = st.number_input("BUI", 0.0, 300.0, 50.0)
    fwi = st.number_input("FWI", 0.0, 100.0, 15.0)
    region = st.selectbox(
        "Region",
        [0, 1],
        format_func=lambda x: "Bejaia" if x == 0 else "Sidi-Bel Abbes"
    )

if st.button("Predict with all models", type="primary"):
    sample = pd.DataFrame([{
        "Temperature": temperature, "RH": humidity, "Ws": wind,
        "Rain": rain, "FFMC": ffmc, "DMC": dmc, "DC": dc,
        "ISI": isi, "BUI": bui, "FWI": fwi, "Region": region
    }], columns=FEATURES)

    st.subheader("Prediction results")
    results = []

    for name, search in models.items():
        prediction = int(search.best_estimator_.predict(sample)[0])
        label = "🔥 Fire" if prediction == 1 else "🌿 No Fire"
        results.append({
            "Algorithm": name,
            "Prediction": label,
            "CV F1-score": round(search.best_score_, 4)
        })

    st.dataframe(pd.DataFrame(results), use_container_width=True)
    st.caption(
        "Educational demonstration only. Predictions are not a real-time "
        "fire warning or a validated operational forecast."
    )

st.divider()
st.subheader("Best tuned parameters")
for name, search in models.items():
    with st.expander(name):
        st.write("Cross-validation F1-score:", round(search.best_score_, 4))
        st.json(search.best_params_)
