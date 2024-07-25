import streamlit as st
import pandas as pd
import plotly_express as px
import time


def load_data(data):
    return pd.read_csv(data)


def stream_text(text: str, delay: float = 0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)


def stream_df(df, delay: float = 0.02):
    for i in range(len(df)):
        yield df.iloc[[i]]
        time.sleep(delay)


df = load_data("data/llm_clean_list.csv")
df_leaderboard = load_data("data/llm_clean_leaderboard.csv")


def about_page():
    st.title("About")


def ask_llm_page():
    st.title("Ask LLM")

    search_model = st.chat_input("Search for LLM Model")
    if search_model:
        st.write(f"Here is some info about: {search_model}")

        result_df = df[df["Model Name"].str.contains(search_model.title())]
        result_as_dict = result_df.to_dict(orient="records")

        for i in result_as_dict:
            model_name = i.get("Model Name")
            maintainer = i.get("Maintainer")
            size = i.get("Size")
            score = i.get("Score")
            context_length = i.get("Context Length")
            summary_df = pd.DataFrame([i])
            template_text = f"""
            {model_name} is the latest version of {maintainer}'s large language model (LLM). 
            It is designed to handle a more extensive array of tasks, including text, image, and video processing. 
            {model_name} has a context length of {context_length}, allowing for more nuanced understanding and generation of content.
            It has {size} parameters.
            Below is a summary table of the model
            """
            with st.chat_message("user"):
                result = template_text.format(
                    model_name=model_name, maintainer=maintainer
                )
                st.write_stream(stream_text(result))
                st.write_stream(stream_df(summary_df))


def llm_list_page():
    st.title("LLM List")
    quick_filters = st.multiselect("Filters", df.columns.tolist())
    st.dataframe(df[quick_filters])


def llm_stats_page():
    st.title("LLM stats")
    metric = st.selectbox("Metric", ["Downloads", "Likes", "Context Length"])

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("LLM Stats"):
            show_llm_stats_as_table()

    with col2:
        if st.button("LLM List"):
            show_llm_list_as_table()

    with col3:
        if st.button("LLM Leaderboard"):
            show_llm_stats_as_table()

    # Select the top 10 models by metric
    top_models = df.nlargest(columns=metric, n=10)[["Model Name", "Maintainer", metric]]

    # Bar Chart
    bar_fig = px.bar(top_models, x="Model Name", y=metric, color="Maintainer")
    # Create a Streamlit app
    st.title(f"Top 10 Models by {metric} - Bar Chart")
    st.plotly_chart(bar_fig)

    # Pie Chart
    pie_fig = px.pie(top_models, names="Model Name", values=metric, color="Maintainer")
    # Create a Streamlit app
    st.title(f"Top 10 Models by {metric} - Pie Chart")
    st.plotly_chart(pie_fig)

    # Scatter Chart
    scatter_fig = px.scatter(top_models, x="Model Name", y=metric, color="Maintainer")
    # Create a Streamlit app
    st.title(f"Top 10 Models by {metric} - Scatter Plot")
    st.plotly_chart(scatter_fig)


def llm_leaderboard_page():
    st.title("LLM Leaderboard")

    quick_filters = st.multiselect(
        "Filters ",
        [
            "Model Name",
            "Maintainer",
            "License",
            "Context Length",
            "Mt Bench",
            "Humaneval",
            "Input Priceusd/1M Tokens",
        ],
        default=["Model Name", "Maintainer", "License", "Context Length", "Humaneval"],
    )

    st.dataframe(df_leaderboard[quick_filters])

    metric = st.selectbox("Metric", ["Context Length", "Humaneval", "MT Bench"])

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("LLM Leaderboard Stats "):
            show_stats_as_table_for_leaderboard()

    # Top N Models

    # Select the top 10 models by metric
    top_models = df_leaderboard.nlargest(columns=metric, n=10)[
        ["Model Name", "Maintainer", metric]
    ]

    # Bar Chart
    bar_fig = px.bar(top_models, x="Model Name", y=metric, color="Maintainer")
    # Create a Streamlit app
    st.title(f"Top 10 Models by {metric} - Bar Chart")
    st.plotly_chart(bar_fig)

    # Pie Chart
    pie_fig = px.pie(top_models, names="Model Name", values=metric, color="Maintainer")
    # Create a Streamlit app
    st.title(f"Top 10 Models by {metric} - Pie Chart")
    st.plotly_chart(pie_fig)

    # Scatter Chart
    scatter_fig = px.scatter(top_models, x="Model Name", y=metric, color="Maintainer")
    # Create a Streamlit app
    st.title(f"Top 10 Models by {metric} - Scatter Plot")
    st.plotly_chart(scatter_fig)


@st.experimental_dialog("LLM List - Stats")
def show_llm_stats_as_table():
    metric = st.selectbox("Metric ", ["Downloads", "Likes", "Context Length"])
    # Select the top 10 models by metric
    top_models = df.nlargest(columns=metric, n=10)[["Model Name", "Maintainer", metric]]

    st.dataframe(top_models)


@st.experimental_dialog("LLM List ")
def show_llm_list_as_table():
    st.dataframe(df)


@st.experimental_dialog("LLM Leaderboard - Stats")
def show_stats_as_table_for_leaderboard():
    metric = st.selectbox("Metric  ", ["Context Length", "Humaneval", "MT Bench"])
    # Select the top 10 models by metric
    top_models = df_leaderboard.nlargest(columns=metric, n=10)[
        ["Model Name", "Maintainer", metric]
    ]

    st.dataframe(top_models)


about = st.Page(about_page, title="About", icon=":material/info:")
ask_llm = st.Page(ask_llm_page, title="Ask LLM", icon=":material/chat:")
llm_stats = st.Page(llm_stats_page, title="LLM Stats", icon=":material/list:")
llm_list = st.Page(llm_list_page, title="LLM List", icon=":material/analytics:")
llm_leaderboard = st.Page(
    llm_leaderboard_page, title="Leaderboard", icon=":material/favorite:"
)


# Navigations
pg = st.navigation(
    {"Home": [llm_list, ask_llm, llm_stats, llm_leaderboard], "About": [about]}
)


pg.run()
