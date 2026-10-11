# app.py
import pickle
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


@st.cache_resource
def load_artifacts():
    with open("movielens_implicit.pkl", "rb") as f:
        return pickle.load(f)


art = load_artifacts()
user_factors = np.asarray(art["user_factors"])
item_factors = np.asarray(art["item_factors"])
item_content_norm = art["item_content_norm"]
train = art["train"].tocsr()
best_alpha = art["best_alpha"]
user_idx, item_idx, idx_item = art["user_idx"], art["item_idx"], art["idx_item"]
titles, seen = art["titles"], art["seen"]
idx_user = {i: u for u, i in user_idx.items()}


def hybrid_scores(u, alpha=best_alpha):
    """Same logic as hybrid_scorer in the notebook."""
    cf = item_factors @ user_factors[u]
    seen_idx = train[u].indices
    if len(seen_idx) == 0:
        return cf
    profile = item_content_norm[seen_idx].mean(axis=0)
    profile /= np.linalg.norm(profile) + 1e-9
    content = item_content_norm @ profile
    cf_z = (cf - cf.mean()) / (cf.std() + 1e-9)
    ct_z = (content - content.mean()) / (content.std() + 1e-9)
    return (1 - alpha) * cf_z + alpha * ct_z


def recommend(user_id, n):
    u = user_idx[user_id]
    scores = hybrid_scores(u)
    for iid in seen.get(user_id, []):
        scores[item_idx[iid]] = -np.inf
    top = np.argsort(-scores)[:n]
    return [(titles[idx_item[i]], idx_item[i]) for i in top]


def similar_movies(item_id, n=5):
    idx = item_idx[item_id]
    sims = cosine_similarity(item_factors[idx:idx + 1], item_factors)[0]
    sims[idx] = -1
    top = np.argsort(-sims)[:n]
    return [(titles[idx_item[i]], float(sims[i])) for i in top]


st.title("🎬 Hybrid Movie Recommender")
st.caption(
    "Implicit BPR collaborative filtering on MovieLens 100K, blended with "
    "genre and decade content similarity."
)

with st.sidebar:
    st.header("Settings")
    user_id = st.selectbox("Choose a user", sorted(user_idx.keys()))
    n = st.slider("Number of recommendations", 5, 20, 10)
    st.markdown(f"Content weight (alpha): **{best_alpha}**")

tab1, tab2 = st.tabs(["Recommendations", "Similar movies"])

with tab1:
    rated = seen.get(user_id, set())
    st.write(f"User {user_id} has rated **{len(rated)}** movies in the dataset.")
    with st.expander("Some movies this user rated highly"):
        st.write(", ".join(titles[i] for i in list(rated)[:10]) or "None")

    st.subheader("Recommended for you")
    for rank, (title, _) in enumerate(recommend(user_id, n), 1):
        st.write(f"**{rank}. {title}**")

with tab2:
    query_titles = {t: i for i, t in titles.items()}
    choice = st.selectbox("Pick a movie", sorted(query_titles.keys()))
    for title, score in similar_movies(query_titles[choice]):
        st.write(f"{title}  \n*similarity: {score:.3f}*")