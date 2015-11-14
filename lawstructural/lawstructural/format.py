""" Format data """

from __future__ import division, print_function
import pandas as pd
import numpy as np
import re
from os.path import dirname, join
from copy import deepcopy
import lawstructural.lawstructural.constants as lc
import lawstructural.lawstructural.utils as lu

#TODO: Take out entrant stuff from lawData
class Format(object):
    """ Basic class for formatting dataset """
    def __init__(self):
        self.dpath = join(dirname(dirname(__file__)), 'data')
        self.data_sets = self.data_imports()
        self.data = self.data_sets['usn']
        self.ent_data = pd.DataFrame([])

    @staticmethod
    def _col_fix(col):
        """ Fix column strings to be R-readable as well and to be consistent
        with older datasets. Think about changing name through rest of program
        instead.
        """
        col = re.sub('%', 'Percent', col)
        col = re.sub('[() -/]', '', col)
        if col[0].isdigit():
            col = re.sub('thpercentile', '', col)
            col = 'p' + col
        if col == 'Name':
            col = 'school'
        if col == 'Issueyear':
            col = 'year'
        if col == 'Overallrank':
            col = 'OverallRank'
        return col

    @staticmethod
    def _fix_bad_values(data):
        """ Fix known USN data typos """
        data.loc[(data['school'] == 'University of Miami') &
                 (data['year'] == 2000), 'Tuitionandfeesfulltime'] = 21000
        data.loc[(data['school'] == 'Emory University') &
                 (data['year'] == 2006), 'Employmentrateatgraduation'] = 72.4
        data.loc[(data['school'] == 'Northwestern University') &
                 (data['year'] == 2006),
                 'EmploymentRate9MonthsafterGraduation'] = 99.5
        data.loc[(data['school'] == 'Michigan State University') &
                 (data['year'] == 2001), 'BarpassageRateinJurisdiction'] = 75
        data.loc[(data['school'] == 'Mississippi College') &
                 (data['year'] == 2001), 'BarpassageRateinJurisdiction'] = 80
        return data

    def usn_format(self):
        """ Basic USN import and format """
        #data = pd.read_csv(join(self.dpath, 'usn2015.csv'))
        data = pd.read_csv(join(self.dpath, 'Law1988-2015.csv'))
        data = data[['Name', 'Value', 'Metric description', 'Issue year']]
        data = pd.pivot_table(data, values='Value',
                              index=['Name', 'Issue year'],
                              columns='Metric description')
        data = data.reset_index()
        names = data.columns.tolist()
        data.columns = [self._col_fix(el) for el in names]
        data = self._fix_bad_values(data)
        data = data.sort(['school', 'year'])
        data['year'] = data['year'].astype(int)
        return data

    def cpi_format(self):
        """ Basic CPI import and format """
        data = pd.read_csv(join(self.dpath, 'lawCPI.csv'))
        # Make up for reporting vs data year in USNews and BLS
        data['year'] = data['year'] + 2
        data = data[data['year'] <= 2015]
        data = data.reset_index(drop=True)
        return data

    @staticmethod
    def _id_name_fix(col):
        """ Fix outdated names of schools from id dataset """
        #FIXME: Find out why this doesn't work for Drexel, Cath U
        old = ['Phoenix School of Law',
               'Chapman University',
               'Drexel University (Mack)',
               'Indiana University--Indianapolis',
               'Texas Wesleyan University',
               'Catholic University of America (Columbus)',
               'John Marshall Law School']
        new = ['Arizona Summit Law School',
               'Chapman University (Fowler)',
               'Drexel University',
               'Indiana University--Indianapolis (McKinney)',
               'Texas A&M University',
               'The Catholic University of America',
               'The John Marshall Law School']
        for i in xrange(len(old)):
            col = re.sub(old[i], new[i], col)
        return col

    def id_format(self):
        """ Import LSAC id's. Note that Irvine doesn't have an id. """
        data = pd.read_csv(join(self.dpath, 'USNewsNameStateID.csv'))
        data['name'] = [self._id_name_fix(col) for col in data['name']]
        return data

    def elec_format(self):
        """ Import yearly electricity prices """
        data = pd.read_csv(join(self.dpath, 'lawElectricity.csv'))
        states = pd.read_csv(join(self.dpath, 'lawStateAbbr.csv'))
        # Change state names to abbreviations
        data = pd.merge(data, states)
        data = data.drop('state', 1)
        columns = data.columns.tolist()
        index = columns.index('abbr')
        columns[index] = 'state'
        data.columns = columns
        data['year'] = data['year'] + 2
        return data

    def data_imports(self):
        """ Import dictionary of initially formatted datasets

        Datasets are as follows with corresponding sources/scrapers

        usn
        ---
            - Data: US News and World Report
            - Source: ai.usnews.com

        cpi
        ---
            - Data: CPI data from BLS
            - Source: http://data.bls.gov/cgi-bin/dsrv?cu
                      Series Id:	CUUR0000SA0,CUUS0000SA0
                      Not Seasonally Adjusted
                      Area:	U.S. city average
                      Item:	All items
                      Base Period:	1982-84=100
                      Years:	1986 to 2015
            - Used to be data.bls.gov/timeseries/LNS14000000

        wage
        ----
            - Data: Market wages for lawyers from BLS
            - Source: bls.gov/oes

        states
        ------
            - Data: US News name/state combinations
            - Source: US News Top Law Schools
            - Scraper: StateScraper.py

        id
        --
            - Data: US News names and LSAC ID combinations
            - Source: http://www.lsac.org/lsacresources/publications/
                      official-guide-archives
            - Scraper: NameScraperLSAC.py

        entrants
        --------
            - Data: School entrants, with id's and dates
            - Source: http://www.americanbar.org/groups/legal_education/
              resources/aba_approved_law_schools/in_alphabetical_order.html
              via
              http://en.wikipedia.org/
              wiki/List_of_law_schools_in_the_United_States
            - Scraper: entryscraper.py

        electricity
        -----------
            - Data: State/Country level electricity prices
            - Source: eia.gov/electricity/monthly/backissues.html
            - Scraper: ElecScraper.py


        Returns
        -------
        data_sets: dict; data sets from specified sources
        """
        data_sets = {
            'usn': self.usn_format(),
            'cpi': self.cpi_format(),
            'states': pd.read_csv(join(self.dpath, 'lawNameState.csv')),
            'id': self.id_format(),
            'entrants': pd.read_csv(join(self.dpath, 'lawEntrants.csv')),
            'electricity': self.elec_format(),
            'stateregions': pd.read_csv(join(self.dpath, 'StateRegions.csv')),
            'aaup_comp_region': pd.read_csv(join(self.dpath,
                                                 'aaup_comp_region.csv')),
            'aaup_comp': pd.read_csv(join(self.dpath, 'aaup_comp.csv')),
            'aaup_salary_region': pd.read_csv(join(self.dpath,
                                                   'aaup_salary_region.csv')),
            'aaup_salary': pd.read_csv(join(self.dpath, 'aaup_salary.csv'))
        }

        return data_sets

    def fill_ranks(self):
        """ Generate top/bottom/inside/squared rank variables,
        fill in unranked schools
        """
        # Indicate top/bottom ranked schools
        self.data['TopRanked'] = 1 * (self.data['OverallRank'] == 1)
        self.data['BottomRanked'] = 1 * (self.data['OverallRank'] ==
                                         np.nanmax(self.data['OverallRank']))
        self.data['InsideRanked'] = 1 * ((self.data['OverallRank'] > 1) &
                                         (self.data['OverallRank'] <
                                          np.nanmax(self.data['OverallRank'])))

        # Squared rank
        self.data['OverallRankSquared'] = self.data['OverallRank']**2

        # Fill in un-ranked schools as max(rank) + 1 or lc.UNRANKED
        mask = pd.isnull(self.data['OverallRank'])
        #unranked = np.nanmax(self.data['OverallRank']) + 1
        unranked = lc.UNRANKED
        self.data['OverallRank'][mask] = unranked
        self.data['Ranked'] = 1 * (self.data['OverallRank'] != unranked)

    def combine_tuition(self):
        """ Combine Full-time and Out-of-State Tuitions """
        self.data['Tuition'] = self.data['Tuitionandfeesfulltime']
        self.data['Tuition'] = self.data['Tuition'].fillna(
            value=self.data['Outofstatetuitionandfeesfulltime']
        )

    def lags(self):
        """ Generate various lags (including tuition alignment) """
        lag_vars = ['OverallRank', 'Ranked']
        lag_vars = [lag_vars, lu.reaction_spec('full')[0]]
        lag_vars.append(lag_vars[1])
        lag_vars[2] = [el + '_comp' for el in lag_vars[2]]
        lag_vars = [el for sublist in lag_vars for el in sublist]
        for lvar in lag_vars:
            self.data[lvar + 'L'] = self.data.groupby('school').apply(
                pd.DataFrame.shift)[lvar]

    def add_entrants(self):
        """ Add indicators for when schools entered """
        self.data_sets['entrants']['Founded'] = \
            self.data_sets['entrants']['Founded'] + 2
        self.data['entry'] = 0
        zipped = zip(self.data_sets['entrants']['Founded'],
                     self.data_sets['entrants']['SchoolUS'])
        for enter in zipped:
            self.data.loc[(self.data['school'] == enter[1]) &
                          (self.data['year'] == enter[0]), 'entry'] = 1

    def combine_dsets(self):
        """ Add in other members of self.data_sets to self.data """
        # Location and id
        self.data['id'] = 0
        for name in self.data['school'].unique():
            self.data.ix[self.data['school'] == name, 'state'] = \
                self.data_sets['states']['place'].where(
                    self.data_sets['states']['name'] == name).max()
            self.data.ix[self.data['school'] == name, 'id'] = \
                self.data_sets['id']['id'].where(
                    self.data_sets['id']['name'] == name).max()
        # Electricity
        self.data = pd.merge(self.data, self.data_sets['electricity'],
                             how='outer', sort=True)
        # CPI
        self.data = pd.merge(self.data, self.data_sets['cpi'])

        # Regions
        self.data = pd.merge(self.data, self.data_sets['stateregions'])

        # AAUP data sets
        aaup_dsets = ['aaup_salary_region',
                      'aaup_salary']
        for dset in aaup_dsets:
            self.data = pd.merge(self.data, self.data_sets[dset], how='outer')

    def price_adjust(self):
        """ Convert nominal to real prices (base year is 2000 - remember,
        using USNews dating convention at this point, will be adjusted later)
        and change some to thousands
        """
        dollar_vars = [
            'Tuition', 'p25privatesectorstartingsalary',
            'p75privatesectorstartingsalary',
            'Averageindebtednessofgraduateswhoincurredlawschooldebt',
            'Medianpublicservicestartingsalary',
            'Roomboardbooksmiscellaneousexpenses'
        ]
        for dvar in dollar_vars:
            self.data[dvar] = self.data[dvar] / 1000.0
        dollar_vars.append('p_elec_state')
        dollar_vars.append('p_elec_us')
        base = self.data_sets['cpi'].loc[
            self.data_sets['cpi']['year'] == 2002, 'cpi']
        base = base.reset_index(drop=True)[0]
        for dvar in dollar_vars:
            self.data[dvar + '_nominal'] = self.data[dvar]
            self.data[dvar] = self.data[dvar] * (base / self.data['cpi'])


    def competition(self):
        """ Generate averages in competition sets """
        # Generate competition variables
        reac_vars = lu.reaction_spec('full')[0]
        comp_vars = ['OverallRank']
        comp_vars = [reac_vars, comp_vars]
        comp_vars = [el for sublist in comp_vars for el in sublist]
        comp_vars_comp = [el + '_comp' for el in comp_vars]
        comp_add = pd.DataFrame(
            np.zeros((self.data.shape[0], len(comp_vars_comp))),
            columns=comp_vars_comp)
        self.data = self.data.join(comp_add)

        for year in self.data['year'].unique():
            for cvar in comp_vars:
                mask = (1 - np.isnan(self.data[cvar])).astype(bool)
                mdata = deepcopy(self.data[mask])
                comp_mat = lu.comp_mat_gen(mdata.loc[
                    mdata['year'] == year, 'OverallRank'])
                mdata.loc[mdata['year'] == year, cvar + '_comp'] = \
                    np.dot(comp_mat, mdata.loc[mdata['year'] == year, cvar])
                self.data[mask] = mdata

    def treatment(self):
        """ Generate treatment variables """
        self.data['treat'] = 1 * (self.data['year'] >= 2012)
        self.data['RankTreat'] = self.data['OverallRank'] * self.data['treat']
        self.data['RankTreat_comp'] = (self.data['OverallRank_comp'] *
                                       self.data['treat'])
        self.data['GPATreat'] = (self.data['UndergraduatemedianGPA'] *
                                 self.data['treat'])
        self.data['LSATTreat'] = (self.data['MedianLSAT'] *
                                  self.data['treat'])
        self.data['TuitionTreat'] = (self.data['Tuition'] *
                                     self.data['treat'])


    def ratios(self):
        """ Generate transparency ratios """
        emp = EmploymentData(self.data, self.dpath)
        emp.id_extend()
        emp.ratio_gen()
        self.data = emp.data_merge()
        # Currently next line drops Drexel, Catholic U of Amer., and UC Irvine
        self.data = self.data[pd.notnull(self.data['id'])]
        self.data = self.data[pd.notnull(self.data['school'])]
        for name in np.unique(self.data['school']):
            # Find earliest year with calculated ratio, and reverse impute
            selection = self.data.loc[
                self.data['school'] == name,
                ['year', 'ratio']
            ].reset_index(drop=True)
            year = np.min(
                selection['year'][np.where(selection['ratio'].notnull())[0]]
            )
            self.data.loc[
                (self.data['school'] == name) &
                (self.data['year'] < year),
                'ratio'
            ] = self.data.loc[
                (self.data['school'] == name) &
                (self.data['year'] == year),
                'ratio'
            ].reset_index(drop=True)[0]

    def dummies(self):
        """ Generate state dummies """
        dummies = pd.get_dummies(self.data['state'])
        self.data = self.data.join(dummies)
        self.data = self.data.reset_index(drop=True)

    def var_name_change(self):
        """ Change names of variables to match previous data dump """
        previous = [
            'p25LSATfulltime',
            'MedianLSATfulltime', 'p75LSATfulltime',
            'p25undergraduateGPAfulltime',
            'MedianundergraduateGPAfulltime',
            'p75undergraduateGPAfulltime', 'Enrollmentfulltime'
        ]

        new = [
            'p25LSATScore',
            'MedianLSAT', 'p75LSATScore',
            'p25UndergraduateGPA',
            'UndergraduatemedianGPA',
            'p75UndergraduateGPA', 'FreshmanEnrolled'
        ]
        columns = self.data.columns.tolist()
        for i in xrange(len(previous)):
            columns = [re.sub(previous[i], new[i], el) for el in columns]
        self.data.columns = columns

    def format(self):
        """ Driver function """
        print("    * Formatting primary dataset.")
        self.var_name_change()
        self.fill_ranks()
        self.combine_dsets()
        self.combine_tuition()
        self.price_adjust()
        self.competition()
        self.lags()
        self.treatment()
        #FIXME: Find out why ratios not keeping Atlanta's John Marshall
        self.ratios()
        self.dummies()
        self.data['const'] = 1
        self.data['year'] = self.data['year'] - 2
        self.data = self.data.sort(['school', 'year']).reset_index(drop=True)
        self.data = self.data.drop_duplicates(subset=['school', 'year'])
        n_applicants_format(self.dpath)

    def entrance_format(self):
        """ Generate dataset for entrance estimates """
        print("    * Formatting entrance dataset.")
        data = self.data[[
            'year', 'Numberofmatriculantsfulltime', 'Tuition'
        ]]
        data['Revenue'] = (data['Numberofmatriculantsfulltime'] *
                           data['Tuition'])
        grouped = data.groupby('year')
        self.ent_data = pd.DataFrame(grouped.mean())
        data = pd.read_csv(join(self.dpath, 'lawEntrants.csv'))
        #pylint: disable=maybe-no-member
        data = data.loc[data['Founded'] >= min(self.data['year']), 'Founded']
        summed = pd.DataFrame(data.value_counts(), columns=['entry'])
        summed = summed.sort().reset_index()
        summed.columns = ['year', 'entry']
        for year in xrange(int(min(self.data['year'])),
                           int(max(self.data['year']))+1):
            if year not in summed.year.values:
                summed = summed.append({'year': year, 'entry': 0},
                                       ignore_index=True)
        summed = summed.sort('year').reset_index(drop=True)
        self.ent_data = pd.merge(self.ent_data.reset_index(),
                                 summed.reset_index())
        diffed = self.ent_data.set_index('year').diff()
        diffed.columns = [el + '_d' for el in diffed.columns]
        self.ent_data['treat'] = 1 * (self.ent_data['year'] >= 2010)
        self.ent_data = pd.merge(self.ent_data, diffed.reset_index())

    def data_out(self):
        """ Save dataset as .csv file """
        self.data.to_csv(join(self.dpath, 'lawData.csv'), encoding='utf-8',
                         index=False)
        self.ent_data.to_csv(join(self.dpath, 'entData.csv'), index=False)

    def lsn_format(self):
        """ Format Law School Numbers dataset. Must be run AFTER data_out """
        print("    * Formatting Law School Numbers dataset.")
        data = pd.read_csv(join(self.dpath, 'lawschoolnumbersSCHOOLS.csv'))
        ranks = pd.read_csv(join(self.dpath, 'lawData.csv'))
        lsn = LawSchoolNumbers(self.dpath, data, ranks)
        lsn.gen()
        lsn.save()


def n_applicants_format(dpath):
    """ Format end-of-year ABA Volume summary for yearly number of
    applicants

    Source: http://www.lsac.org/lsacresources/data/lsac-volume-summary
    """
    print("    * Formatting ABA year-end summary")
    data = pd.read_excel(join(dpath, 'eoy-spreadsheet.xlsx'), index_col=None)
    data = data.ix[1, :].reset_index()
    data.columns = ['year', 'n_apps']
    data['year'] = data['year'].str.replace(r'\D+', '').astype('int')
    def sub_format(arg):
        """ Sub-function to clean out unicode """
        try:
            return int(re.sub(',', '', re.sub(u'\xa0', '', arg)))
        except TypeError:
            return arg
    data['n_apps'] = data['n_apps'].map(sub_format)
    data.to_csv(join(dpath, 'abaeoy.csv'))




class EmploymentData(object):
    """ Construct and merge transparency ratios. Note that each year in raw
    data corresponds to the previous year's graduating class. Example: 2011
    data was reported in 2011 about the 2010 graduating class associated with
    the 2009 entering class. This is adjusted by one year with respect to the
    "master data" set to correspond to students using last year's placement to
    make decisions for the current year. This is equivalent to Ranking being
    based on previous year's school attributes.
    """
    def __init__(self, master_data, path):
        self.max_year = 2014
        self.master_data = master_data
        self.data = self.data_import(path)
        self.weights = self.weight_gen()
        self.ratios = None

    @staticmethod
    def col_filter(col):
        """ Filter column names from 2011 data """
        col = re.sub('university', 'SchoolName', col)
        col = re.sub('Emp Solo', 'Solo', col)
        col = re.sub('Emp 501', '501-PLUS', col)
        col = re.sub('Emp Type Unk', 'Unknown', col)
        col = re.sub('Emp ', '', col)
        col = re.sub('\+', '', col)
        col = col.strip()
        return col

    def data_import(self, path):
        """ Import employment datasets, format 2011 columns to math other
        years, and export as dictionary

        Parameters
        ----------
        path: string, directory with data files
        """
        data = {}
        for year in xrange(2011, (self.max_year + 1)):
            dpath = join(path, 'EmploymentSummary-' + str(year) + '.csv')
            data[str(year)] = pd.read_csv(dpath)
        data['2011'].columns = [
            self.col_filter(col) for col in data['2011'].columns
        ]
        return data

    @staticmethod
    def weight_gen():
        """ Generate weights for different employment types. Weights taken
        from expected salaries for new law graduates by firm size reported at
        http://www.nalp.org/new_associate_sal_oct2011

        Returns
        -------
        dictionary: types of employment with corresponding weights
        """
        types = ['Solo', '2-10', '11-25', '26-50', '51-100', '101-250',
                 '251-500', '501-PLUS', 'Unknown']
        salaries = np.array([73, 73, 73, 86, 91, 110, 130, 130, 0])
        salaries = salaries / salaries[-2]
        return dict(zip(types, salaries))

    def id_extend(self):
        """ Extend LSAC ids to all schools to facilitate merger with US News
        dataset. Previously only associated with 2011-2012.
        """
        idvars = ['SchoolName', 'id']
        nameid = pd.concat([self.data['2011'][idvars],
                            self.data['2012'][idvars]])
        nameid = nameid.drop_duplicates()
        for name in nameid['SchoolName']:
            for year in xrange(2013, (self.max_year + 1)):
                self.data[str(year)].loc[
                    self.data[str(year)]['SchoolName'] == name, 'id'
                ] = nameid.loc[nameid['SchoolName'] == name, 'id'].tolist()[0]

    def ratio_gen(self):
        """ Generate transparency ratios """
        ratios = []
        for year in xrange(2011, (self.max_year + 1)):
            data_weight = deepcopy(self.data[str(year)][self.weights.keys()])
            denominator = data_weight.sum(1)
            for column in data_weight.columns:
                data_weight[column] = (data_weight[column] *
                                       self.weights[column])
            numerator = data_weight.sum(1)
            ratio = numerator / denominator
            ratio = pd.DataFrame({
                'id': self.data[str(year)]['id'],
                'year': year,
                'ratio': ratio
            })
            ratios.append(ratio)
        self.ratios = pd.concat(ratios, ignore_index=True)

    def data_merge(self):
        """ Merge ratios with master dataset and return """
        self.ratios['year'] = self.ratios['year'] + 1
        data_out = pd.merge(self.master_data, self.ratios, on=['id', 'year'],
                            how='outer')
        return data_out


class LawSchoolNumbers(object):
    """ Primary class. Keeps updated dataset as attribute """
    def __init__(self, dpath, data, ranks):
        self.dpath = dpath
        self.data = data
        self.ranks = ranks[['school', 'year', 'OverallRank', 'Tuition']]
        initial_vars = ['year', 'user', 'state', 'LSAT', 'LSDAS_GPA',
                        'Degree_GPA']
        self.school_names = np.array(self.data.columns)
        self.school_names = self.get_school_names(initial_vars)
        self.data = self.data.loc[
            ~(data[self.school_names] == 0).all(axis=1)
        ].reset_index(drop=True)
        self.new_data = deepcopy(self.data[initial_vars])

    def get_school_names(self, initial_vars):
        """ Isolate school names from column names in data """
        self.school_names = np.delete(self.school_names, 0)
        original_indices = self.school_names.argsort()
        remove = original_indices[
            np.searchsorted(self.school_names[original_indices],
                            initial_vars)
        ]
        return np.delete(self.school_names, remove)

    def get_not_applied(self, year):
        """ For a given year, return boolean mask of schools not applied to.
        Returned mask is a pandas dataframe object.
        """
        not_applied = (
            self.data.loc[self.data.year == year, self.school_names] == 0
        ).reset_index(drop=True)
        return not_applied

    def get_not_admitted(self, year):
        """ For a given year, return boolean mask of schools not admitting
        applicant. Returned mask is a pandas dataframe object.
        """
        not_admitted = (
            self.data.loc[self.data.year == year, self.school_names] < 4
        ).reset_index(drop=True)
        return not_admitted

    def get_not_matric(self, year):
        """ For a given year, return boolean mask of schools not matriculating
        applicant. Returned mask is a pandas dataframe object.
        """
        not_matric = (
            self.data.loc[self.data.year == year, self.school_names] != 5
        ).reset_index(drop=True)
        return not_matric

    def max_applied(self, rank, data_schools, year):
        """ Select maximum (worst) rank of applied-to schools """
        data_applied = deepcopy(data_schools)
        not_applied = self.get_not_applied(year)
        data_applied[not_applied] = 0
        worst_schools = np.array(data_applied.idxmax(1))
        self.new_data.loc[self.new_data['year'] == year, 'app_worst'] = \
            np.array(
                rank.loc[worst_schools, 'OverallRank']
            )

    def min_applied(self, rank, data_schools, year):
        """  Select minimum (best) rank of applied-to schools"""
        data_applied = deepcopy(data_schools)
        not_applied = self.get_not_applied(year)
        data_applied[not_applied] = 500
        best_schools = np.array(data_applied.idxmin(1))
        self.new_data.loc[self.new_data['year'] == year, 'app_best'] = \
            np.array(
                rank.loc[best_schools, 'OverallRank']
            )

    def max_admitted(self, rank, data_schools, year):
        """ Select maximum (worst) rank of admitted-to schools """
        data_admitted = deepcopy(data_schools)
        not_admitted = self.get_not_admitted(year)
        data_admitted[not_admitted] = np.nan
        worst_schools = np.array(data_admitted.idxmax(1))
        was_admitted = True - pd.isnull(worst_schools)
        this_year = self.new_data['year'] == year
        self.new_data.loc[this_year, 'admit_worst'] = np.nan
        selection = self.new_data.loc[this_year, 'admit_worst']
        selection.loc[was_admitted] = \
            np.array(
                rank.loc[worst_schools[was_admitted], 'OverallRank']
            )
        self.new_data.loc[this_year, 'admit_worst'] = selection

    def min_admitted(self, rank, data_schools, year):
        """  Select minimum (best) rank of admitted-to schools"""
        data_admitted = deepcopy(data_schools)
        not_admitted = self.get_not_admitted(year)
        data_admitted[not_admitted] = np.nan
        best_schools = np.array(data_admitted.idxmin(1))
        was_admitted = True - pd.isnull(best_schools)
        this_year = self.new_data['year'] == year
        self.new_data.loc[this_year, 'admit_best'] = np.nan
        selection = self.new_data.loc[this_year, 'admit_best']
        selection.loc[was_admitted] = \
            np.array(
                rank.loc[best_schools[was_admitted], 'OverallRank']
            )
        self.new_data.loc[this_year, 'admit_best'] = selection

    def dist_applied(self, data_dict, year):
        """ Get distribution parameters for applied-to schools:
        (rank mean, rank std, tuition mean, n)
        """
        for data_var in ('rank', 'tuition'):
            data_applied = deepcopy(data_dict[data_var])
            not_applied = self.get_not_applied(year)
            data_applied[not_applied] = np.nan
            if data_var == 'rank':
                self.new_data.loc[self.new_data['year'] == year, 'app_n'] = \
                    np.array(data_applied.count(1))
            self.new_data.loc[
                self.new_data['year'] == year, 'app_mean_' + data_var
            ] = np.array(data_applied.mean(axis=1))
            self.new_data.loc[
                self.new_data['year'] == year, 'app_std_' + data_var
            ] = np.array(data_applied.std(axis=1))

    def dist_admitted(self, data_schools, year):
        """ Get distribution parameters for admitted-to schools (mean, std, n)
        """
        data_admitted = deepcopy(data_schools)
        not_admitted = self.get_not_admitted(year)
        data_admitted[not_admitted] = np.nan
        self.new_data.loc[self.new_data['year'] == year, 'admit_n'] = \
            np.array(data_admitted.count(1))
        self.new_data.loc[self.new_data['year'] == year, 'admit_mean'] = \
            np.array(data_admitted.mean(axis=1))
        self.new_data.loc[self.new_data['year'] == year, 'admit_std'] = \
            np.array(data_admitted.std(axis=1))

    def get_matric(self, data_schools, year):
        """ Get Rank of matriculated school """
        data_matric = deepcopy(data_schools)
        not_matric = self.get_not_matric(year)
        data_matric[not_matric] = 0
        data_matric = data_matric.sum(1)
        data_matric = np.array(data_matric.replace(0, np.nan))
        self.new_data.loc[self.new_data['year'] == year, 'matric'] = \
            data_matric

    def gen_interactions(self):
        """ Generate interaction terms """
        self.new_data['app_mean_tuition_year'] = (
            self.new_data['app_mean_tuition'] *
            self.new_data['year']
        )
        self.new_data['LSAT_treat'] = (
            self.new_data['LSAT'] * self.new_data['treat']
        )
        self.new_data['LSDAS_GPA_treat'] = (
            self.new_data['LSDAS_GPA'] * self.new_data['treat']
        )
        self.new_data['LSAT_LSDAS_GPA_treat'] = (
            self.new_data['LSAT'] * self.new_data['LSDAS_GPA'] *
            self.new_data['treat']
        )


    def gen(self):
        """ Driver method """
        for year in np.unique(self.data['year']):
            data = self.data.loc[self.data['year'] == year, :]
            rank = self.ranks.loc[self.ranks['year'] == year, :]
            if not rank.index.tolist():
                # self.data has more years than rank
                break
            rank = rank.set_index('school')
            data_schools_rank = pd.DataFrame(
                np.tile(np.array(rank.loc[self.school_names, 'OverallRank']),
                        (data.shape[0], 1)),
                columns=self.school_names
            )
            data_schools_tuition = pd.DataFrame(
                np.tile(np.array(rank.loc[self.school_names, 'Tuition']),
                        (data.shape[0], 1)),
                columns=self.school_names
            )
            data_dict = {'rank': data_schools_rank,
                         'tuition': data_schools_tuition}
            self.max_applied(rank, data_schools_rank, year)
            self.min_applied(rank, data_schools_rank, year)
            self.dist_applied(data_dict, year)
            self.max_admitted(rank, data_schools_rank, year)
            self.min_admitted(rank, data_schools_rank, year)
            self.dist_admitted(data_schools_rank, year)
            self.get_matric(data_schools_rank, year)
        self.new_data['admit_binary'] = 1 - np.isnan(self.new_data.admit_mean)
        self.new_data['matric_binary'] = 1 - np.isnan(self.new_data.matric)
        nan_subset = ['app_std_rank', 'app_std_tuition', 'admit_worst',
                      'admit_best', 'admit_mean', 'admit_std', 'matric']
        self.new_data[nan_subset] = self.new_data[nan_subset].fillna(0)
        # Only use observations that had at least one school report tuition
        self.new_data = self.new_data.dropna(
            subset=['LSAT', 'LSDAS_GPA', 'app_mean_tuition']
        )
        self.new_data['treat'] = 1 * (self.new_data.year >= 2010)
        self.gen_interactions()
        self.new_data['const'] = 1
        self.gen_long_dset()

    def gen_long_dset(self):
        """ Generate data from lawschoolnumbersSCHOOLS with one unit of
        observation being an application decision
        """
        print("    * Formatting LSN Application dataset.")
        ids = np.array(self.data.columns)
        ids = ids[np.in1d(ids, self.school_names, invert=True)]
        new_data = pd.melt(self.data, id_vars=ids.tolist(),
                           value_vars=self.school_names.tolist(),
                           var_name='school', value_name='event')
        new_data = pd.merge(new_data, self.ranks, how='outer')
        new_data['app'] = 1 * (new_data['event'] > 0)
        new_data['admit'] = 1 * (new_data['event'] >= 4)
        new_data['matric'] = 1 * (new_data['event'] == 5)
        new_data['treat'] = 1 * (new_data['year'] >= 2010)
        new_data.to_csv(join(self.dpath, 'lsnLong.csv'), index=False)

    def save(self):
        """ Save dataset as .csv file """
        self.new_data.to_csv(join(self.dpath, 'lawschoolnumbers.csv'))
