#pylint: skip-file
from __future__ import division, print_function
from nose.tools import *
import numpy as np
import pandas as pd
import os
import lawstructural.lawstructural.format as lf

class TestFormat(object):
    """ Test the data formatting routines (make sure values are correct) """
    def __init__(self):
        #self.initialize_data()  # Comment if data is already compiled
        self.dir_name = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(self.dir_name, 'data', 'lawData.csv')
        self.data = pd.read_csv(data_path)
        self.incumb = self.data.loc[
            (self.data.school == 'Albany Law School') &
            (self.data.year == 2006),
        ].reset_index(drop=True)
        self.entrant = self.data.loc[
            (self.data.school == 'Arizona Summit Law School') &
            (self.data.year == 2005),
        ].reset_index(drop=True)

    @staticmethod
    def initialize_data():
        f = lf.Format()
        f.format()
        f.entrance_format()
        f.data_out()

    def test_cpi_format(self):
        assert self.incumb['cpi'][0] == 201.6

    def test_elec_format_state(self):
        assert self.incumb['p_elec_state_nominal'][0] == 13.46

    def test_elec_format_us(self):
        assert self.incumb['p_elec_us_nominal'][0] == 8.37

    def test_fill_ranks_ranked_true(self):
        assert self.incumb['Ranked'][0] == 1

    def test_fill_ranks_ranked_false(self):
        assert self.entrant['Ranked'][0] == 0

    def test_fill_ranks_inside_ranked_true(self):
        assert self.incumb['InsideRanked'][0] == 1

    def test_fill_ranks_inside_ranked_false(self):
        assert self.entrant['InsideRanked'][0] == 0

    def test_fill_ranks_squared(self):
        assert self.incumb['OverallRankSquared'][0] == 121**2

    def test_combine_tuition_tuitionandfees(self):
        assert np.allclose(self.incumb['Tuition_nominal'][0], 35.079)

    def test_lags_rank(self):
        assert self.incumb['OverallRankL'][0] == 115

    def test_lags_medianlsat(self):
        assert self.incumb['MedianLSATL'][0] == 155

    def test_add_entrants_entrant(self):
        assert self.entrant['entry'][0] == 1

    def test_add_entrants_incumbent(self):
        assert self.incumb['entry'][0] == 0

    def test_combine_dsets_state(self):
        assert self.incumb['state'][0] == 'NY'

    def test_combine_dsets_id(self):
        assert self.incumb['id'][0] == 2946

    def test_price_adjust_real_tuition(self):
        """ Test price adjusting Tuition for real dollars """
        expected = 35.079 * (172.2 / 201.6)
        assert np.allclose(self.incumb['Tuition'][0], expected)

    def test_price_adjust_real_p_elec_state(self):
        """ Test price adjusting state elec prices for real dollars """
        expected = 13.46 * (172.2 / 201.6)
        assert np.allclose(self.incumb['p_elec_state'][0], expected)

    def test_competition(self):
        """ Still need to work out what this should be """
        pass

    def test_dummies_true(self):
        assert self.incumb['NY'][0] == 1

    def test_dummies_false(self):
        assert self.incumb['UT'][0] == 0

    def test_ratio_11(self):
        """ For 2011 .csv file, for year 2009 in aggregate dataset """
        case = self.data.loc[
            (self.data.school == 'Albany Law School') &
            (self.data.year == 2010),
        ].reset_index(drop=True)
        weights = np.array([73, 73, 73, 86, 91, 110, 130, 130, 0]) / 130
        expected = np.array([4, 54, 18, 11, 0, 5, 8, 6, 0])
        expected = np.dot(weights, expected) / sum(expected)
        assert np.allclose(case['ratio'][0], expected)

    def test_ratio_13(self):
        """ For 2013 .csv file, for year 2011 in aggregate dataset """
        case = self.data.loc[
            (self.data.school == 'Brigham Young University (Clark)') &
            (self.data.year == 2012),
        ].reset_index(drop=True)
        weights = np.array([73, 73, 73, 86, 91, 110, 130, 130, 0]) / 130
        expected = np.array([3, 24, 6, 6, 4, 5, 1, 5, 0])
        expected = np.dot(weights, expected) / sum(expected)
        print("Ratio")
        print(case['ratio'][0])
        print("Expected")
        print(expected)
        assert np.allclose(case['ratio'][0], expected)


class TestEntranceFormat(object):
    """ Test the entrance_format method of Format """
    def __init__(self):
        self.dir_name = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(self.dir_name, 'data', 'entData.csv')
        self.data = pd.read_csv(data_path)

    def test_entrants(self):
        self.data.loc[self.data['year'] == 2000, 'entry_s'] == 3


class TestLawSchoolNumbersFormat(object):
    """ Test LawSchoolNumbers class """
    def __init__(self):
        self.subjects = [
            {'year': 2003, 'user': 'BYU17', 'state': 'Texas', 'LSAT': 170,
             'LSDAS_GPA': 3.8, 'Degree_GPA': 3.83, 'schools':
             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
             },
            {'year': 2004, 'user': 'yuplawyup', 'state': 'Alaska', 'LSAT': 157,
             'LSDAS_GPA': 3.4, 'Degree_GPA': 3.3, 'schools':
             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
             },
            {'year': 2011, 'user': 'happierfeet', 'state': None, 'LSAT': 180,
             'LSDAS_GPA': 3.85, 'Degree_GPA': 3.85, 'schools':
             np.array([5, 4, 4, 4, 4, 4, 0, 0, 4, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
             },
            {'year': 2014, 'user': 'sal2484', 'state': 'Texas', 'LSAT': 156,
             'LSDAS_GPA': 3.52, 'Degree_GPA': 3.52, 'schools':
             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
             }
        ]
        self.schoolnames = [
            "Yale University", "Harvard University", "Stanford University", "Columbia University", "University of Chicago", "New York University", "University of Pennsylvania", "University of Virginia", "University of California--Berkeley", "Duke University", "University of Michigan--Ann Arbor", "Northwestern University", "Cornell University", "Georgetown University", "University of Texas--Austin", "University of California--Los Angeles", "Vanderbilt University", "Washington University in St. Louis", "Emory University", "George Washington University", "University of Minnesota--Twin Cities", "University of Southern California (Gould)", "University of Alabama", "College of William and Mary (Marshall-Wythe)", "University of Washington", "University of Notre Dame", "Boston University", "University of Iowa", "Indiana University--Bloomington (Maurer)", "University of Georgia", "Arizona State University (O'Connor)", "Ohio State University (Moritz)", "University of North Carolina--Chapel Hill", "University of Wisconsin--Madison", "Wake Forest University", "Boston College", "Brigham Young University (Clark)", "Fordham University", "University of California--Davis", "University of Arizona (Rogers)", "University of Illinois--Urbana-Champaign", "Southern Methodist University (Dedman)", "University of Colorado--Boulder", "Washington and Lee University", "Florida State University", "George Mason University", "Tulane University", "University of Maryland (Carey)", "University of Florida (Levin)", "University of Utah (Quinney)", "Baylor University", "Pennsylvania State University (Dickinson)", "University of Richmond (Williams)", "Pepperdine University", "University of California (Hastings)", "University of Connecticut", "University of Nebraska--Lincoln", "University of Houston", "University of Kentucky", "University of Oklahoma", "Temple University (Beasley)", "University of Arkansas--Fayetteville", "University of Miami", "Case Western Reserve University", "Georgia State University", "University of Missouri", "Yeshiva University (Cardozo)", "Loyola University Chicago", "Seton Hall University", "University of Denver (Sturm)", "University of Kansas", "American University (Washington)", "Illinois Institute of Technology (Chicago-Kent)", "Lewis & Clark College (Northwestern)", "Louisiana State University--Baton Rouge (Hebert)", "University of New Mexico", "University of Tennessee--Knoxville", "University of Tulsa", "University of Cincinnati", "University of San Diego", "Rutgers, the State University of New Jersey--Camden", "University of Pittsburgh", "Brooklyn Law School", "Rutgers, the State University of New Jersey--Newark", "University of Nevada--Las Vegas", "West Virginia University", "Indiana University--Indianapolis (McKinney)", "Loyola Marymount University", "Michigan State University", "Seattle University", "University of Louisville (Brandeis)", "Wayne State University", "Marquette University", "Northeastern University", "Stetson University", "St. Louis University", "University of New Hampshire School of Law", "University of South Carolina", "Villanova University", "Florida International University", "SUNY Buffalo Law School", "University of Hawaii--Manoa (Richardson)", "University of Oregon", "Mercer University (George)", "University of Mississippi", "University of Missouri--Kansas City", "The Catholic University of America", "Gonzaga University", "Santa Clara University", "St. John's University", "Syracuse University", "Texas Tech University", "CUNY", "Drake University", "Cleveland State University (Cleveland-Marshall)", "Creighton University", "Washburn University", "Albany Law School", "Quinnipiac University", "University of Idaho", "Campbell University", "DePaul University", "Duquesne University", "Hamline University", "University of Akron", "University of Arkansas--Little Rock (Bowen)", "University of Montana", "Willamette University (Collins)", "Drexel University", "University of Maine", "University of North Dakota", "University of St. Thomas", "University of Wyoming", "Vermont Law School", "Hofstra University (Deane)", "Howard University", "Samford University (Cumberland)", "University of Baltimore", "William Mitchell College of Law", "Chapman University (Fowler)", "New York Law School", "Pace University", "University of Memphis (Humphreys)", "University of Toledo", "University of South Dakota", "South Texas College of Law", "University of the Pacific (McGeorge)", "Appalachian School of Law", "Arizona Summit Law School", "Atlanta's John Marshall Law School", "Ave Maria School of Law", "Barry University", "California Western School of Law", "Capital University", "Charleston School of Law", "Charlotte School of Law", "Elon University", "Faulkner University (Jones)", "Florida A&M University", "Florida Coastal School of Law", "Golden Gate University", "The John Marshall Law School", "Liberty University", "Loyola University New Orleans", "Mississippi College", "New England Law Boston", "North Carolina Central University", "Northern Illinois University", "Northern Kentucky University (Chase)", "Nova Southeastern University (Broad)", "Ohio Northern University (Pettit)", "Oklahoma City University", "Regent University", "Roger Williams University", "Southern Illinois University--Carbondale", "Southern University Law Center", "Southwestern Law School", "St. Mary's University", "St. Thomas University", "Suffolk University", "Texas A&M University", "Texas Southern University (Marshall)", "Thomas Jefferson School of Law", "Thomas M. Cooley Law School", "Touro College (Fuchsberg)", "University of Dayton", "University of Detroit Mercy", "University of San Francisco", "University of the District of Columbia (Clarke)", "Valparaiso University", "Western New England University", "Western State College of Law at Argosy University", "Whittier College", "Widener University", "Catholic University", "Inter-American University", "University of California--Irvine", "University of La Verne", "University of Massachusetts--Dartmouth", "University of Puerto Rico"
        ]
        self.gen_dict = {
            '_worst': np.nanmax,
            '_best': np.nanmin,
            '_mean': np.nanmean,
            '_std': np.nanstd
        }
        warning_list = [
            "Mean of empty slice", "Degrees of freedom <= 0 for slice.",
            "All-NaN axis encountered", "All-NaN slice encountered"
        ]
        for warn in warning_list:
            warnings.filterwarnings('ignore', warn)
    
    def test():
        assert True
