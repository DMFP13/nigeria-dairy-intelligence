import streamlit as st


st.set_page_config(page_title="Nigeria Dairy Intelligence", layout="wide")

st.title("Nigeria Dairy Intelligence")
st.caption("A lightweight decision-support shell for dairy market, farm, herd, and scenario insights.")

sections = {
    "Market Signals": "Placeholder for milk price trends, feed-cost movement, weather risk, and demand-supply indicators.",
    "Farm Economics": "Placeholder for farm-level unit economics, margin tracking, and cost-to-production analysis.",
    "Herd Intelligence": "Placeholder for herd performance monitoring, health signals, and operational anomaly surfacing.",
    "Scenario Lab": "Placeholder for what-if planning across prices, productivity, feed strategy, and investment choices.",
}

st.sidebar.header("Modules")
selected_section = st.sidebar.radio("Go to", list(sections.keys()))

for name, description in sections.items():
    if name == selected_section:
        st.subheader(name)
        st.write(description)
    else:
        st.markdown(f"### {name}")
        st.caption(description)
