"""Functions for analysis of US census Age data"""

import os
import datetime

import numpy as np
import pandas as pd


MAX_AGE = 110
RESAMPLE_AGES = list(range(0, MAX_AGE + 1))
TARGET_AGES = list(range(0, 101))


# Tell pylint that I'm going to have some bad variable names...
#pylint: disable=C0103

#
# These are all the population files downloaded from the US Census
#

DATA_DIR = '~/GitHub/sundries/age_distro/data_raw'

filelist_historical = [
    'pe-11-1900s.xls',
    'pe-11-1910s.xls',
    'pe-11-1920s.xls',
    'pe-11-1930s.xls',
    'pe-11-1940s.xls',
    'pe-11-1950s.xls',
    'pe-11-1960s.xls',
    'pe-11-1970s.xls',
]

filelist_1980s = [
    'E8081PQI.TXT',
    'E8182PQI.TXT',
    'E8283PQI.TXT',
    'E8384PQI.TXT',
    'E8485PQI.TXT',
    'E8586PQI.TXT',
    'E8687PQI.TXT',
    'E8788PQI.TXT',
    'E8889PQI.TXT',
    'E8990PQI.TXT',
]

filelist_1990s = 'us-est90int-08.csv'
file_2000s = 'us-est00int-01.xls'

file_projections = 'US Population Projections by Age and Sex.xlsx'



def strage_to_intage(str_age):
    """Turn a string-age into an integer age"""

    if str_age[-1] == '+':
        return int(str_age[:-1])
    return int(str_age)


def upsample_population(df_in):
    """Given the distribution of ages for a given year, interpolate the
        values to match `target_values`
    """

    series_out = (
        df_in.set_index('Age')['Population'].
        dropna().
        iloc[:-1].
        append(pd.Series(0, index=[MAX_AGE])).
        apply(lambda x: np.log(x + 1)).
        reindex(pd.Index(RESAMPLE_AGES, name='Age')).
        interpolate(
            method='piecewise_polynomial',
            limit=None,
        ).
        loc[TARGET_AGES].
        apply(lambda x: np.exp(x) - 1)    
    )
    return series_out


def get_future_pop(filename, first_projection_year=2010):
    """Import data from the projections file into a DataFrame"""

    df = pd.read_excel(
        os.path.join(DATA_DIR, filename),
        skiprows=1,
        skipfooter=50,
    )

    df = df.drop(0, 0)
    df['Age'] = [(i - 1) * 5 for i in df.index.tolist()]
    df = df.drop('Sex and age', 1)
    df = df.set_index('Age')
    df = pd.melt(df.reset_index(), id_vars=['Age'], var_name='Year', value_name='Population')

    df_list = []
    for year, _df in df.groupby('Year'):
        if year >= first_projection_year:
            _df['Year'] = year
            df_list.append(_df)

    df_full = pd.concat(df_list, axis=0)
    # Buckets span 5 years, so lets downwieght
    df_full['Population'] *= (1000 / 5.0)
    return df_full


def get_2000s_pop(filename):
    """Import data from the 2000s into a DataFrame"""

    df = pd.read_excel(
        os.path.join(DATA_DIR, filename),
        skiprows=3,
        skipfooter=103,
    )

    df = df.drop(0, 0)
    df = df.drop([c for c in df.columns if isinstance(c, str) and c.startswith('Unnamed: ')], 1)
    df['Age'] = [(i - 1) * 5 for i in df.index.tolist()]
    df = pd.melt(df, id_vars=['Age'], var_name='Year', value_name='Population')

    df_list = []
    for year, _df in df.groupby('Year'):
        if year > 2000:
            _df.assign(Year=int(year))
            df_list.append(_df)

    df_full = pd.concat(df_list, axis=0)

    # Buckets span 5 years, so lets downwieght
    df_full['Population'] /= 5.0
    return df_full


def get_1990s_pop(filename):
    """Import data from the 1990s into a DataFrame"""

    df = pd.read_csv(
        os.path.join(DATA_DIR, filename),
        skiprows=3,
        header=None,
        names=['Date', 'Age', 'Population', 'pop_female', 'pop_male'],
        usecols=['Date', 'Age', 'Population']
    )

    df = df[~df.Date.str.startswith('Inter')]
    df = df[~df.Age.str.startswith('All ')]

    df.Age = df.Age.apply(strage_to_intage)
    df.Date = pd.to_datetime(df.Date)
    df['Year'] = df.Date.apply(lambda x: x.year)

    df = (
        df.groupby(['Year', 'Age'])[['Population']].
        mean().sort_index().reset_index()
    )

    df_list = []
    for year, _df in df.groupby('Year'):
        _df.assign(Year=int(year))
        df_list.append(_df)

    df_full = pd.concat(df_list, axis=0)
    return df_full


def get_1980s_pop(filename_list):
    """Import data from the 1980s into a DataFrame"""

    col_widths = [(0, 2), (2, 6), (6, 9), (9, 21), (21, 221)]

    df = pd.concat([
        pd.read_fwf(
            os.path.join(DATA_DIR, fname),
            colspecs=col_widths,
            delim_whitespace=True,
            names=['reportcode', 'Datecode', 'Age', 'Population', 'tmp'],
            dtype='str'
        )
        for fname in filename_list
    ], axis=0)
    df = df[~df.Age.isin(['999', 'nan'])]
    df.Age = df.Age.astype(int)
    df = df.drop(['tmp', 'reportcode'], 1)

    df.Population = df.Population.astype(float)

    df['Date'] = (
        df.Datecode.astype(str).str.strip().
        apply(lambda x: datetime.datetime(
            int('19' + x[-2:]),
            int(x[:-2]),
            1
        ))
    )

    df['Year'] = df.Date.apply(lambda x: x.year)
    df = (
        df.groupby(['Year', 'Age'])[['Population']].
        mean().sort_index().reset_index()
    )

    df_list = []
    for year, _df in df.groupby('Year'):
        if year < 1990:
            df_list.append(_df)

    df_full = pd.concat(df_list, axis=0)        
    return df_full


def get_pre80s_pop(filename_list):
    """Import data from the pre-1980s into a DataFrame"""

    df_list = []
    for fname in filename_list:
        dict_df = pd.read_excel(
            os.path.join(DATA_DIR, fname),
            skiprows=8,
            skipfooter=15,
            parse_cols=[0, 1],
            names=['Age', 'Population'],
            sheetname=None
        )

        for year in dict_df:
            df = dict_df[year]
            df.Age = df.Age.apply(strage_to_intage)
            df['Year'] = int(year)
            df_list.append(df)

    df_full = pd.concat(df_list, axis=0)        
    return df_full


def import_all_population_data():
    """Import all datafiles into a single DataFrame and return
    two dataframe -- one with the raw data, and one with the age interped
    to a common age range
    """

    df_list = [
        get_future_pop(file_projections),
        get_2000s_pop(file_2000s),
        get_1990s_pop(filelist_1990s),
        get_1980s_pop(filelist_1980s),
        get_pre80s_pop(filelist_historical),    
    ]

    df_full = pd.concat(df_list, axis=0).reset_index()
    df_full['Age'] = df_full.Age.astype(int)

    # run interpolation for each year of data
    df_list_interp = []
    for year, df in df_full.groupby('Year'):    
        _df = pd.DataFrame()
        _df['Population'] = upsample_population(df)
        _df = _df.reset_index()
        _df['Year'] = year
        df_list_interp.append(_df)
    df_full_interp = pd.concat(df_list_interp, axis=0).reset_index()
    df_full_interp['Age'] = df_full_interp.Age.astype(int)

    return df_full, df_full_interp
