#!/usr/bin/python

""" Driver module """

from __future__ import print_function
print("* ---------------------------------------------------- *")
print("*    IMPORTING R LIBRARIES AND COMPILING EXTENSIONS    *")
print("* ---------------------------------------------------- *")
import pandas as pd
import numpy as np
import sys
import getopt
import cProfile
import os
import re
import shutil
from subprocess import call
from os.path import join, dirname, realpath
import lawstructural.lawstructural.format as lf
import lawstructural.lawstructural.firststage as fs
import lawstructural.scripts.firststageappadmit as fst
from lawstructural.lawstructural.bbl import BBL


def driver(opts):
    """
    Replicate paper results

    Args:
        opts: Dict with members:
            data: 1 to format data
            tables: 1 to recreate latex tables and figures
            react: 'simple' to use median quality reaction variables
                   'full' to use 25, 50, and 75 quantiles
            entrance: 1 to estimate with entrance/exit decision
            verbose: 1 to print estimation and simulation output to screen
                     during runtime
            nose: 1 to run tests before program
    """
    path = dirname(realpath(__file__))

    if opts['pylint']:
        run_pylint()
        return 0

    if opts['nose']:
        run_nosetests()
        return 0

    if opts['clean']:
        #clear_results()
        opts['generate'] = 1  # SS results need to be regenerated now
        reset_results = ResetResults(path)
        reset_results.reset()

    if opts['tables']:
        initial_tables(path)
    
    if opts['data']:
        data_format()

    my_bbl = BBL(opts)
    if not my_bbl.data_check():
        my_bbl.estimate()
        my_bbl.simulate()

    if opts['tables']:
        post_tables(path)

    if opts['latex']:
        typeset_latex(path)

    if opts['beamer']:
        typeset_beamer(path)

    print("* ------------------ *")
    print("*    RUN COMPLETE    *")
    print("* ------------------ *")


def run_pylint():
    """ Run Pylint on module """
    print("* -------------------- *")
    print("*    RUNNING PYLINT    *")
    print("* -------------------- *")
    call(["pylint", "--disable=anomalous-backslash-in-string",
          "--disable=no-member",
          "lawstructural"])
    print("* --------------------- *")
    print("*    PYLINT FINISHED    *")
    print("* --------------------- *")


def run_nosetests():
    """ Run unit tests on module """
    print("* ----------------------- *")
    print("*    RUNNING NOSETESTS    *")
    print("* ----------------------- *")
    call(['nosetests',
          'lawstructural.tests.lawstructural_secondstage_tests'])
    #call(['nosetests'])
    print("* -------------------- *")
    print("*    TESTS COMPLETE    *")
    print("* -------------------- *")


class ResetResults(object):
    """ Clears out previous results if applicable and recreates
    directory tree
    """
    def __init__(self, path):
        self.path = path

    def clear_results(self):
        """ Clear results for new run """
        path_results = join(self.path, 'Results')
        if os.path.exists(path_results):
            shutil.rmtree(path_results)

    @staticmethod
    def check_make(path):
        """ Check for path, generate if non-existent.

        Returns
        -------
        indicator: 1 if path generated, 0 if not
        """
        if not os.path.exists(path):
            os.makedirs(path)
            return 1
        return 0

    def dir_gen(self):
        """ Generate directories for results """
        action = 0
        path_results = join(self.path, 'Results')
        folders = ('Initial', 'FirstStage', 'SecondStage')
        subfolders = ('Tables', 'Figures')
        os.makedirs(path_results)
        for folder in folders:
            os.makedirs(join(path_results, folder))
            for subfolder in subfolders:
                os.makedirs(join(path_results, folder, subfolder))

    def reset(self):
        """ Driver to run other two methods """
        self.clear_results()
        self.dir_gen()


def initial_tables(path):
    """ Generate charts and tables before the first stage """
    print("* -------------------------------------- *")
    print("*    GENERATING PRE-ESTIMATION TABLES    *")
    print("* -------------------------------------- *")
    path_script = join(path, 'lawstructural', 'scripts')
    os.chdir(path_script)
    call(['Rscript', 'initialresultsUSN.R'])
    call(['Rscript', 'initialresultsLSN.R'])
    call(['Rscript', 'firststageLSN.R'])
    if os.path.isfile('Rplots.pdf'):
        # Take care of R leftover
        call(['unlink', 'Rplots.pdf'])
    os.chdir(path)


def post_tables(path):
    """ Generate charts and tables for the first stage """
    print("* --------------------------------------- *")
    print("*    GENERATING POST-ESTIMATION TABLES    *")
    print("* --------------------------------------- *")
    fst.main()


def data_format():
    """ Run data-formatting routines """
    print("FORMATTING DATA")
    f = lf.Format()
    f.format()
    f.entrance_format()
    f.data_out()
    f.lsn_format()
    print("DATA FORMATTED")


def table_compile():
    """ Compile all tables into one Latex file """
    fpath = path_grab('Tables', 'lawtextables.tex')
    try:
        os.remove(fpath)  # Make sure it isn't picked up in list of files
    except OSError:
        pass
    path = dirname(dirname(dirname(__file__)))
    path = join(path, 'Results', 'Tables')
    filenames = [f for f in os.listdir(path) if isfile(join(path, f))]
    filenames = [path_grab('Tables', f) for f in filenames]
    with open(fpath, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


def typeset_latex(path):
    """ Typeset paper """
    print("* ----------------------- *")
    print("*    TYPESETTING PAPER    *")
    print("* ----------------------- *")
    path_paper = join(path, 'Paper')
    os.chdir(path_paper)
    call(['pdflatex', '-interaction=batchmode', 'LawStructuralPaper.tex'])
    call(['bibtex', 'LawStructuralPaper.aux'])
    call(['pdflatex', '-interaction=batchmode', 'LawStructuralPaper.tex'])
    call(['pdflatex', '-interaction=batchmode', 'LawStructuralPaper.tex'])
    os.chdir(path)


def typeset_beamer(path):
    """ Typeset slides """
    print("* ------------------------ *")
    print("*    TYPESETTING SLIDES    *")
    print("* ------------------------ *")
    for fpath in ['Slides30', 'Slides90']:
        path_paper = join(path, 'Slides', fpath)
        os.chdir(path_paper)
        call(['pdflatex', '-interaction=batchmode', 'LawStructuralSlides.tex'])
        call(['pdflatex', '-interaction=batchmode', 'LawStructuralSlides.tex'])
        os.chdir(path)


def main(args):
    """ Main function for running the program. Uses first letters of
    opts keys as possible argument values in args
    """
    opts = {'data': 0, 'tables': 0, 'reaction': 0, 'entrance': 0, 'verbose': 0,
            'nose': 0, 'pylint': 0, 'speed': 0, 'latex': 0, 'beamer': 0,
            'generate': 0, 'multiprocessing': 0, 'clean': 0}
    if args:
        args = args[0]
        for key in opts:
            if key[0] in args:
                opts[key] = 1
    if opts['reaction']:
        opts['reaction'] = 'full'
    else:
        opts['reaction'] = 'simple'
    if opts['speed']:
        cprofile_arg = ', '.join("'{}': {}".format(key, val)
                                 for key, val in opts.items())
        cprofile_arg = '{' + cprofile_arg + '}'
        cprofile_arg = re.sub('simple', "'simple'", cprofile_arg)
        print(cprofile_arg)
        cProfile.run("driver(" + cprofile_arg + ")")
    else:
        driver(opts)


if __name__ == '__main__':
    main(sys.argv[1:])