import streamlit as st
import requests

# Replace with your actual deployed FastAPI backend URL
BACKEND_URL = "https://shl-assessment-recommendation-s2wx.onrender.com"

st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")
st.title("🔍 SHL Assessment Recommender")
st.markdown("Enter a job description to get the most relevant SHL assessments.")

query = st.text_area("Job Description / Query", height=200)

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a job description or query.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/recommend",
                    json={"query": query}
                )
                if response.status_code == 200:
                    results = response.json().get("recommended_assessments", [])
                    if results:
                        for i, rec in enumerate(results, 1):
                            st.markdown(f"### {i}. [{rec['url']}]({rec['url']})")
                            st.markdown(f"**Description**: {rec['description']}")
                            st.markdown(f"**Duration**: {rec['duration']} minutes")
                            st.markdown(f"**Remote Support**: {rec['remote_support']}")
                            st.markdown(f"**Adaptive Support**: {rec['adaptive_support']}")
                            st.markdown(f"**Test Types**: {', '.join(rec['test_type'])}")
                            st.markdown("---")
                    else:
                        st.info("No recommendations found.")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
