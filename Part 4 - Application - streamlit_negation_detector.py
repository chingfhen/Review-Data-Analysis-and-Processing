

import streamlit as st
import NegationDetector 


s = st.text_input("Input Sentence", value = "Alice had hardly done anything.")
if s:
    st.text(f"Negation: {NegationDetector.predict_negation_span(s)}")








