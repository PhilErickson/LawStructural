### Program Files for "Information Asymmetry in Profit-generating Graduate Education Markets: A Structural Approach to Law Schools" by Philip J. Erickson

This module contains data and code for all the structural estimation discussed in this paper. An active license for the KNITRO package from Ziena Optimization, LLC is required for the optimization routines.

To run the program, run the script lawdriver.py from a folder with edit permission. Eg.

./$python lawdriver.py

The program has options with the following keys:

Option | Key | Description
------ | --- | -----------
data | 'd' | recompile the datasets
tables | 't' | generate result tables
react | 'r' | include 25th and 75th percentiles of undergraduate profiles as choice variables for schools (deprecated)
entrance | 'e' | include potential entrants
verbose | 'v' | display results during runtime
nose | 'n' | run unit tests
pylint | 'p' | run pylint on code to check formatting
speed | 's' | profile the base program
latex | 'l' | typeset the paper
beamer | 'b' | typeset slides
generate | 'g' | re-generate simulation-based estimates
multiprocessing | 'm' | parallelize where possible
clean | 'c' | clean out all previous results before run

If, for example, you would like to recompile the school-level data and display output, include keys 'r' and 'v' as follows

./$python lawdriver.py rv

If option 't' is specified, the program will create a new 'Results' folder in the directory containing lawdriver.py with results. Each table will be stored in the "Tables" subdirectory with one latex file per table as well as a file "lawtextables.tex" containing all the tables combined. Each figure will be stored in the "Figures" subdirectory.

#### Notes on Program Modification

I have attempted to make this program fairly abstracted. That being said, there were some hard-coding shortcuts made given the specific nature of the problem being estimated. These should be taken into account if trying to modify the program to analyze other possible specifications. Here are a few of the major ones with notes concerning the way to make modifications.

1. First stage estimates

To modify first-stage estimates, they must be changed in the 'firststage' module for the initial estimation and the 'secondstage' module in the 'Market' class to specify market evolution, ' _ _ init _ _ ' method, and after it evolves. If more variables are being used that weren't previously (for example, more interaction terms), the 'format' module should also be modified to generate the new variables. In order to preserve consistency as results are passed through to different classes, no variables used for first-stage estimation are dynamically generated during analysis.

2. Second stage estimates

If the functional form for the net-donation function needs to be changed, most of the modification should happen in the 'secondstage' module. Places are in the 'SecondStage' class in the following methods: 'gen _ basis' for generation, 'objective _ q' for parameter break-down, 'min _ q' for initial guess. Also, the 'Market' class under the 'demand' method.