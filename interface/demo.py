import streamlit as st
import asyncio
from dummy import code_to_pinyin, code_to_hanzi, pinyin_to_hanzi, predict_next_hanzi

# Initialize session state variables
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'code' not in st.session_state:
    st.session_state.code = ""
if 'selected_pinyin' not in st.session_state:
    st.session_state.selected_pinyin = ""
if 'hanzi_page' not in st.session_state:
    st.session_state.hanzi_page = 0
if 'pinyin_page' not in st.session_state:
    st.session_state.pinyin_page = 0
if 'show_punct' not in st.session_state:
    st.session_state.show_punct = False

st.session_state.text = st.text_area("", st.session_state.text, height=150)

async def get_predictions():
    hanzi_candidates = []
    pinyin_candidates = []

    if st.session_state.show_punct:
        hanzi_candidates = ["，", "。"]
    elif st.session_state.code:
        if st.session_state.selected_pinyin:
            hanzi_candidates = await pinyin_to_hanzi(st.session_state.selected_pinyin)
        else:
            hanzi_candidates = await code_to_hanzi(st.session_state.code)
            pinyin_candidates = await code_to_pinyin(st.session_state.code)
    elif st.session_state.text:
        hanzi_candidates = await predict_next_hanzi(st.session_state.text)

    return hanzi_candidates, pinyin_candidates

hanzi_candidates, pinyin_candidates = asyncio.run(get_predictions())

cols = st.columns(7)
start = st.session_state.hanzi_page * 5
for i, hanzi in enumerate(hanzi_candidates[start:start+5]):
    if cols[i+1].button(hanzi):
        st.session_state.text += hanzi
        st.session_state.code = ""
        st.session_state.selected_pinyin = ""
        st.session_state.hanzi_page = 0
        st.session_state.pinyin_page = 0
        st.session_state.show_punct = False
        st.rerun()

with cols[0]:
    if st.button("⬅️") and st.session_state.hanzi_page > 0:
        st.session_state.hanzi_page -= 1
        st.rerun()
with cols[-1]:
    if st.button("➡️") and (start + 5) < len(hanzi_candidates):
        st.session_state.hanzi_page += 1
        st.rerun()

# show predict pinyin candidates
if st.session_state.code and not st.session_state.selected_pinyin:
    cols = st.columns(7)
    start = st.session_state.pinyin_page * 5
    for i, py in enumerate(pinyin_candidates[start:start+5]):
        if cols[i+1].button(py):
            st.session_state.selected_pinyin = py
            st.session_state.hanzi_page = 0
            st.rerun()
    with cols[0]:
        if st.button("⬅️ ", key="pinyin_prev") and st.session_state.pinyin_page > 0:
            st.session_state.pinyin_page -= 1
            st.rerun()
    with cols[-1]:
        if st.button("➡️ ", key="pinyin_next") and (start + 5) < len(pinyin_candidates):
            st.session_state.pinyin_page += 1
            st.rerun()

# T9 Keypad
def handle_keypress(key):
    if key == "⌫":
        if st.session_state.code:
            st.session_state.code = ""
            st.session_state.selected_pinyin = ""
        elif st.session_state.text:
            st.session_state.text = st.session_state.text[:-1]
        st.session_state.show_punct = False
        st.rerun()
    elif key == "space":
        if st.session_state.code:
            st.session_state.text += hanzi_candidates[0] if hanzi_candidates else ""
            st.session_state.code = ""
            st.session_state.selected_pinyin = ""
        st.session_state.text += " "
        st.session_state.show_punct = False
        st.rerun()
    elif key == "punct":
        if st.session_state.code:
            st.session_state.text += hanzi_candidates[0] if hanzi_candidates else ""
            st.session_state.code = ""
            st.session_state.selected_pinyin = ""
        st.session_state.show_punct = True
        st.rerun()
    elif key in [str(i) for i in range(10)]:
        if key == "0":
            handle_keypress("space")
        elif key == "1":
            handle_keypress("punct")
        else:
            st.session_state.code += key
            st.session_state.selected_pinyin = ""
            st.session_state.show_punct = False
            st.rerun()

keys = [
    ["1", "2", "3"],
    ["4", "5", "6"],
    ["7", "8", "9"],
    ["", "0", "⌫"]
]
labels = {
    "1": "，。", "2": "ABC", "3": "DEF",
    "4": "GHI", "5": "JKL", "6": "MNO",
    "7": "PQRS", "8": "TUV", "9": "WXYZ",
    "0": "␣", "⌫": ""
}

for row in keys:
    cols = st.columns(3)
    for i, key in enumerate(row):
        if key and cols[i].button(f"{key}\n\n{labels[key]}"):
            handle_keypress(key)
