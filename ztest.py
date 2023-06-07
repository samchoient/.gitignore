import streamlit as st
import pandas as pd
from statistics import NormalDist
from statsmodels.stats.weightstats import ztest
from scipy.stats import shapiro

st.title('ZTEST')

with st.expander('view table'):
    df = pd.read_csv(
        'https://raw.githubusercontent.com/ethanweed/pythonbook/main/Data/zeppo.csv')
    st.dataframe(df.transpose())

with st.expander('View Statistic'):
    st.dataframe(df.describe().transpose())

st.write('## Constructing Hypothesis')
st.latex('H_{0}: \mu = \mu_{0}')
st.latex('H_{1}: \mu \neq \mu_{0}')  # neq tidak sama dengan

alpha = st.number_input('masukkan nilai alpha',
                        step=0.000001, min_value=0., max_value=1.)
null_mean = st.number_input('masukkan nilai mu_0')
clicked = st.button('do the Z test')

if clicked:
    alpha_z = NormalDist().inv_cdf(p=1-alpha/2)
    z_score, p_value = ztest(
        df['grades'], value=null_mean, alternative='two-sided')
    if abs(z_score) > alpha_z:
        st.latex('REJECT H_{O}')
    else:
        st.latex('CAN NOT REJECT H_{0}')
        st.latex(f'{z_score} dan {p_value}')
        st.write(
            f'titik kritis = {alpha_z}, hitung z = {z_score}, p_value = {p_value}')
st.write('## Check Normality')
clicked_2 = st.button('Do The Shapiro Test')

if clicked_2:
    result = shapiro(df['grades'])
    st.write(result)
    st.bar_chart(df['grades'])
