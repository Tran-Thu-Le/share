## How to run a python project on your computer?

Demo project:  [`deep_inv_opt`][1]

This documentation is for Linux user
(Remark: may use `ctrl+L` to clear the terminal if needed)


**Part 1. download project**
 * Create and enter the folder that we want to save the code 
 * Open terminal by right-click and choose "open terminal"
 *  Check whether python3, virtual environment, git, pip were installed in your computer or not
	 * `python3 --version`
	 * `virtualenv --version`
	 * `git --version`
	 * `pip --version`
	 * If everything goes well, move on! If not, try installing them manually!
 - Copy their project: `git clone https://github.com/yingcongtan/deep_inv_opt.git` 

**Part 2. setup virtual environment**
 - Enter the project: `cd deep_inv_opt`
 - Create virtual environment: `virtualenv venv -p python3` 
 - Open virtual environment: `source venv/bin/activate`. Now you will see that a new word "(venv)" before the link of the current directory

**Part 3. install python packages**
* Install some python packages required by deep_inv_opt project
	* numpy (matrix vector calculation): `pip install numpy`
	* matplotlib (plots) `pip install matplotlib`
	* Pytorch (machine learning tools) `pip install torch` 
	* ipython (to run jupyter notebook) `pip install ipython` 
	* ipykernal (to run virtual jupyter notebook) `pip install ipykernel`
	* install deep_inv_opt: `pip install -e .`
- (Optional) check the packages by `pip list`
- Create virtualenv kernel for jupyter notebook `
ipython kernel install --user --name=venv`
- Open jupyter notebook: `jupyter notebook`
- A window will pop up, browse to "examples" folder and then double click on the "Example 1 - Example 1 - Using deep_inv_opt with a single target point.ipynb"
- On tab kernel choose "change Kernel", then "venv"

**Part 4. rune code**
- Open tab "Cell" choose "Run all" and enjoy the result (I hope)






[1]: https://github.com/yingcongtan/deep_inv_opt?fbclid=IwAR3_wGflQZjKXTUQHxZ6wkFqsofUOOhw1IfrbQJKRdmbubA6ixDCohahN40
