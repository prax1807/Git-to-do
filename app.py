import pandas as pd
import streamlit as st

from ml_demo import DATASETS, MODELS, train_and_evaluate

st.set_page_config(page_title="ML Lab", layout="wide")
st.title("ML Lab")
st.caption("Explore classic classification models on familiar datasets.")

with st.sidebar:
    st.header("Experiment")
    dataset_name = st.selectbox("Dataset", list(DATASETS))
    model_name = st.selectbox("Model", MODELS)
    test_size = st.slider("Test set", min_value=0.15, max_value=0.4, value=0.25, step=0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    neighbors = 5
    if model_name == "K-nearest neighbors":
        neighbors = st.slider("Neighbors", min_value=1, max_value=15, value=5)

@st.cache_resource(show_spinner="Training model...")
def get_experiment(dataset, model, test_fraction, seed, neighbor_count):
    return train_and_evaluate(dataset, model, test_fraction, seed, neighbor_count)

result = get_experiment(dataset_name, model_name, test_size, int(random_state), neighbors)

metric_columns = st.columns(4)
for column, label, key in zip(
    metric_columns,
    ("Accuracy", "Precision", "Recall", "F1 score"),
    ("accuracy", "precision", "recall", "f1"),
):
    column.metric(label, f"{result[key]:.1%}")

st.subheader("Test set")
left, right = st.columns([1, 1])
with left:
    st.markdown("**Confusion matrix**")
    class_names = [str(name) for name in result["target_names"]]
    confusion = pd.DataFrame(
        result["confusion_matrix"],
        index=class_names,
        columns=class_names,
    )
    st.dataframe(confusion, use_container_width=True)
with right:
    st.markdown("**Per-class report**")
    report = pd.DataFrame(result["classification_report"]).T
    st.dataframe(report, use_container_width=True)

st.subheader("Inspect a prediction")
sample_index = st.slider(
    "Test example",
    min_value=0,
    max_value=len(result["actual"]) - 1,
    value=0,
)
actual_name = class_names[int(result["actual"][sample_index])]
predicted_name = class_names[int(result["predictions"][sample_index])]
status = "Correct" if actual_name == predicted_name else "Incorrect"
st.write(f"**{status}** · Actual: {actual_name} · Predicted: {predicted_name}")
