#Algonauts devkit https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link#scrollTo=I8aRowcQxM8E
import streamlit as st
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import time
import numpy as np
import sys
from .. import utils

data_dir='/home/maria/Algonauts2023'
parent_submission_dir='/home/maria/Algonauts2023_submission'
subj=1
args = utils.argObj(data_dir, parent_submission_dir, subj)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(50, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")