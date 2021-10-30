import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

data = [766,28781,7157,87050] # şunlar hesaplanıp eklenecek
ingredients = ["Twitter","Şikayet Var","Ekşi Sözlük","Yemek Sepeti",]

def func(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n({:d} yorum)".format(pct, absolute)

wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
textprops=dict(color="w"))

ax.legend(wedges, ingredients,
title="Kanallar",
loc="center left",
bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")
fig.tight_layout()

st.pyplot(fig)