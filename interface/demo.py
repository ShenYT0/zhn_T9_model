import streamlit as st
import asyncio
# from dummy import input_predict
from predict import input_predict

if "text" not in st.session_state:
    st.session_state.text = ""
if "cursor" not in st.session_state:
    st.session_state.cursor = len(st.session_state.text)
if "code_input" not in st.session_state:
    st.session_state.code_input = ""
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "page" not in st.session_state:
    st.session_state.page = 0

async def update_candidates():
    query = st.session_state.text[:st.session_state.cursor] + st.session_state.code_input
    st.session_state.candidates = await input_predict(query)
    st.session_state.page = 0
    st.rerun()

def insert_text_at_cursor(new_text):
    before = st.session_state.text[:st.session_state.cursor]
    after = st.session_state.text[st.session_state.cursor:]
    st.session_state.text = before + new_text + after
    st.session_state.cursor += len(new_text)

def handle_delete():
    if st.session_state.code_input:
        st.session_state.code_input = st.session_state.code_input[:-1]
    elif st.session_state.cursor > 0:
        st.session_state.text = (
            st.session_state.text[:st.session_state.cursor - 1] +
            st.session_state.text[st.session_state.cursor:]
        )
        st.session_state.cursor -= 1

text_val = st.text_area(
    "",
    value=st.session_state.text,
    height=150,
)

if text_val != st.session_state.text:
    st.session_state.text = text_val
    st.session_state.cursor = len(text_val)

left, cand_cols, right = st.columns([1, 8, 1])
with left:
    if st.button("⬅️") and st.session_state.page > 0:
        st.session_state.page -= 1
with right:
    if st.button("➡️") and (st.session_state.page + 1) * 5 < len(st.session_state.candidates):
        st.session_state.page += 1

with cand_cols:
    cand_row = st.columns(5)
    start = st.session_state.page * 5
    for i, candidate in enumerate(st.session_state.candidates[start:start + 5]):
        if cand_row[i].button(candidate, key=f"cand_{i}"):
            insert_text_at_cursor(candidate)
            st.session_state.code_input = ""
            asyncio.run(update_candidates())

st.text(f"当前输入: {st.session_state.code_input}")

keypad = [
    ["1\n，。", "2\nABC", "3\nDEF"],
    ["4\nGHI", "5\nJKL", "6\nMNO"],
    ["7\nPQRS", "8\nTUV", "9\nWXYZ"],
    ["", "0\n␣", "⌫"]
]
for row in keypad:
    cols = st.columns(3)
    for i, label in enumerate(row):
        if label == "":
            continue
        key_label = label.split("\n")[0]
        if cols[i].button(label):
            if key_label == "1":
                pass
            elif key_label == "⌫":
                handle_delete()
                asyncio.run(update_candidates())
            elif key_label == "0":
                insert_text_at_cursor(" ")
            else:
                st.session_state.code_input += key_label
                asyncio.run(update_candidates())