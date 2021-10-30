import matplotlib.pyplot as plt
import re
from string import digits
s = "string. With. 1Punctuation 132 ?"
remove_digits = str.maketrans('', '', digits)
s = s.translate(remove_digits)
print(s)