import pdoc
import os
import nbformat as nbf

import pynlcontrol

with open('../README.md', 'r') as fw:
    readme_content = fw.read()

with open('pyNLControl_Manual.md', 'w') as fw:
    fw.write(readme_content + "\n\n")
    fw.write(pdoc.text("pynlcontrol.BasicUtils") + "\n\n")
    fw.write(pdoc.text("pynlcontrol.Estimator") + "\n\n")
    fw.write(pdoc.text("pynlcontrol.Controller") + "\n\n")
    fw.write(pdoc.text("pynlcontrol.QPInterface") + "\n\n")

with open('pyNLControl_Manual.md', 'r') as fw:
    z = fw.read()

nb = nbf.v4.new_notebook()

nb['cells'] = [nbf.v4.new_markdown_cell(z)]

fname = 'UserGuide.ipynb'

with open(fname, 'w') as f:
    nbf.write(nb, f)

os.system("jupyter nbconvert --execute --to html UserGuide.ipynb")
os.system("jupyter nbconvert --to webpdf --allow-chromium-download UserGuide.ipynb")