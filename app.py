import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import sys
from io import BytesIO
from agonyStreamlitDAG import get_results

running_process = None
if "results_content" not in st.session_state:
    st.session_state["results_content"] = None
if "numpy_data" not in st.session_state:
    st.session_state["numpy_data"] = None

# CSV validation
def validate_csv(file):
    if file is not None:
        if file.name.endswith('.csv'):
            return True
        else:
            st.sidebar.error("Invalid file type. Please upload a CSV file.")
            return False
    return True
    
# Terminate the running subprocess if any
def terminate_process():
    global running_process
    if running_process and running_process.poll() is None: 
        running_process.terminate()
        running_process = None

# Page configuration
st.set_page_config(page_title="AniDomNet", layout="wide")
st.title("AniDomNet")
st.write("Adjust the parameters on the left and press 'Run' to execute the program.")

# Sidebar for parameters
st.sidebar.title("Input Parameters")

uploaded_file = st.sidebar.file_uploader("Dataset file", type="csv", help="Upload a CSV file with the appropriate format.")
if uploaded_file is not None and validate_csv(uploaded_file):
    original_name = os.path.splitext(uploaded_file.name)[0] 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
else:
    st.sidebar.warning("Using default file: demo.csv")
    original_name = "demo"
    file_path = os.path.abspath("demo.csv")

anomaly_detection = st.sidebar.checkbox("Anomaly Detection", help="Check to run Anomaly Detection instead of base AniDomNet.")
n_initializations = st.sidebar.number_input("Number of Random Initializations (-n)", value=10, min_value=1, help="Set the number of random initializations (integer > 0).")
n_cores = st.sidebar.number_input("Number of Cores (-ncores)", value=1, min_value=1, max_value=2, help="Set the number of cores to use (3 > integer > 0).")

small_col, left_col, right_col = st.columns([1, 2, 3])

# Display uploaded dataset
with small_col:
    if uploaded_file is not None or os.path.exists(file_path):
        csv_data = pd.read_csv(file_path)
        indices = sorted(set(list(csv_data.iloc[:, 0]) + list(csv_data.iloc[:, 1])))
        pivot = pd.crosstab(index=csv_data.iloc[:, 0],
                            columns=csv_data.iloc[:, 1],
                            dropna=True).reindex(indices,
                                                 fill_value=0,
                                                 axis=0).reindex(indices,
                                                                 fill_value=0,
                                                                 axis=1)
        st.subheader("Interactions")
        st.dataframe(csv_data, height=530)

with left_col:
    if uploaded_file is not None or os.path.exists(file_path):
        st.subheader("Dominance Matrix")
        st.dataframe(pivot, height=530)

# Results pane
with right_col:
    st.subheader("Results")
    if st.sidebar.button("Run"):
        executable_path = os.path.abspath("AniDomNetAnomaly.py" if anomaly_detection else "AniDomNet.py")
        python_executable = sys.executable
        if not os.path.isfile(file_path):
            st.error("File not found! Please upload a valid CSV file.")
        else:
            cmd = [python_executable, executable_path,
                   "-file", file_path,
                   "-outfolder", "./",
                   "-n", str(n_initializations),
                   "-ncores", str(n_cores)]

            with st.spinner("Processing..."):
                try:
                    running_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = running_process.communicate()
                    #subprocess.run(cmd, check=True)

                    # Read and display results from the text file
                    name = os.path.basename(file_path).rsplit('.', 1)[0]
                    results_file_path = f"./{name}.txt"
                    
                    if os.path.exists(results_file_path):
                        with open(results_file_path, "r") as f:
                            st.session_state["results_content"] = f.read()

                    # Check for generated numpy file
                    if not anomaly_detection:
                        numpy_file_path = f"./{name}_scores.npy"
                        if os.path.exists(numpy_file_path):
                            with open(numpy_file_path, "rb") as f:
                                st.session_state["numpy_data"] = f.read()
                        else:
                            st.session_state["numpy_data"] = None
                            st.error("NumPy file not found.")

                except subprocess.CalledProcessError as e:
                    st.error(f"An error occurred during execution: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    if st.session_state.get("results_content"):
        st.text_area(
            label="Results",
            value=st.session_state["results_content"],
            height=530,
            key="results_text_area"  
        )

    if not anomaly_detection and st.session_state.get("numpy_data"):
        st.download_button(
            label="Download Scores (NumPy)",
            data=st.session_state["numpy_data"],
            file_name=f"{original_name}_scores.npy",
            mime="application/octet-stream",
            key="download_numpy_button"
        )
    
    if st.sidebar.button("Reset"):
        terminate_process()
        st.session_state.clear()
        uploaded_file = None
        st.rerun()

# Define 3 columns for the bottom section
bottom_col1, bottom_col2, bottom_col3 = st.columns([2, 2, 2])

# Ensure NumPy data exists in session state before processing
if "numpy_data" in st.session_state and st.session_state["numpy_data"] is not None:
    with st.spinner("Processing DAG matrices..."):
        try:
            if not anomaly_detection:
                numpy_file_path = f"./{name}_scores.npy"
                if os.path.exists(numpy_file_path):
                    scoreMatrix = np.load(numpy_file_path)

            dagc_matrix, dagnc_matrix, fig = get_results(scoreMatrix)
            st.session_state["dagc_matrix"] = dagc_matrix
            st.session_state["dagnc_matrix"] = dagnc_matrix
            st.session_state["dag_figure"] = fig
            
        except Exception as e:
            st.error(f"Error processing DAG matrices: {e}")


# Display results in the 3 bottom columns
with bottom_col1:
    st.subheader("Dominance Matrix: Cycles")
    if "dagc_matrix" in st.session_state:
        st.write(st.session_state["dagc_matrix"])
    else:
        st.write("Waiting for computation...")

with bottom_col2:
    st.subheader("Dominance Matrix: DAG")
    if "dagnc_matrix" in st.session_state:
        st.write(st.session_state["dagnc_matrix"])
    else:
        st.write("Waiting for computation...")

with bottom_col3:
    st.subheader("Hierarchical Structure")
    if "dag_figure" in st.session_state:
        st.pyplot(st.session_state["dag_figure"])
    else:
        st.write("Waiting for computation...")