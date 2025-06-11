# Streamlit T9 Demo

## How to start

```
streamlit run demo.py
```

## Functions need to replace in dummy

- pinyin_to_hanzi
    - input: pinyin string (exp: "di")
    - output: hanzi predict list (exp: "["第", "地", "底", "低", ...]")
- code_to_pinyin
    - input: T9 code (exp: "3426")
    - output: hanzi predict list 
- code_to_hanzi
    - input: T9 code (exp: "3426")
    - output: pinyin predict list (exp: "["dian", "dibo", ...]")
- predict_next_hanzi
    - input : context string
    - output : a hanzi