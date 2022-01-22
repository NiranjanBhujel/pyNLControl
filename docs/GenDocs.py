import pdoc
import os

import pyNLControl

with open('../README.md', 'r') as fw:
    readme_content = fw.read()

with open('pyNLControl_Manual.md', 'w') as fw:
    fw.write(readme_content + "\n\n")
    fw.write(pdoc.text("pyNLControl.BasicUtils") + "\n\n")
    fw.write(pdoc.text("pyNLControl.Estimation"))


# input_filename = 'pyNLControl_Manual.md'
# output_filename = 'pyNLControl_Manual.docx'

# os.system(f'pandoc -V geometry:margin=1in -f markdown -t docx {input_filename} -o {output_filename}')
