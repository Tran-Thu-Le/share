# Install and run Python package online with Colab

Demo package: `deep_inv_opt`, [source link][1].
-->Vietnamese user

In this document, we are going to see how to install the Python package `deep_inv_opt` and run their jupyter notebook [Example 1][2] using Google Colab.

**Step 1.**
Create a Drive folder that contains your code, say "python"

**Step 1.**
- Go to [Example 1][2]
- In the url, change "github" to "githubtocolab" and hit enter. A new window will pop up and show the codes in "Example 1" notebook.
- Press `Ctrl+S` and choose "Luu ban sao trong Drive"
- In tab, "Tep", choose  "Di chuyen". Browse to your "python" folder, hit "Chon". Your "Example 1" file is now in your "python" folder
- Create a code cell: In tab "Chen", choose "O chua ma"
- Copy and past the following code to that "code cell" and hit `Ctrl+Enter` to execute it
```python
!git clone https://github.com/yingcongtan/deep_inv_opt.git # download project
import os 
os.chdir("deep_inv_opt") # enter the folder deep_inv_opt
!pip install -e . # install deep_inv_opt
```
- Highlight the above code and *comment out* it by `Ctrl+?` (this is important)

Now the notebook is ready to run. For example,
- To execute one by one, enter that code cell that you want to run and hit `Ctrl+Enter`
- To execute the whole doc, in tab "Thoi gian chay", choose "Chay tat ca"



[1]: https://github.com/yingcongtan/deep_inv_opt?fbclid=IwAR3_wGflQZjKXTUQHxZ6wkFqsofUOOhw1IfrbQJKRdmbubA6ixDCohahN40
[2]: https://github.com/yingcongtan/deep_inv_opt/blob/master/examples/Example%201%20-%20Using%20deep_inv_opt%20with%20a%20single%20target%20point.ipynb

 
