import pandas as pd


def setup(df):
    df.rename(columns={df.columns[0]: 'hole'}, inplace=True)
    df.rename(columns={df.columns[1]: 'substrate'}, inplace=True)
    return df.set_index('hole')


def changed_order(df):
    df_first_trial = df.filter(regex='01$|02$|03$|04$', axis=0)
    df_second_trial = df.filter(regex='05$|06$|07$|08$', axis=0)
    df_third_trial = df.filter(regex='09$|10$|11$|12$', axis=0)

    final = pd.concat([df_first_trial, df_second_trial, df_third_trial])
    return final.set_index(['substrate'], append=True)


def subtract_control(df):
    mean_water = df.query("substrate == 'Water'").mean()
    return (df - mean_water).query("substrate != 'Water'")


def subtract_start(df):
    return df.sub(df['0'], axis=0)


def truncate_under_threshold(df, threshold=0.1):
    return df.applymap(lambda x: 0 if x < threshold else x)


def add_trial_info(df):
    trial = []
    for x in [1, 2, 3]:
        trial = trial + [x] * 31
    df['trial'] = trial
    return df.set_index('trial', append=True)


def mean_each_trial(df):
    return df.groupby('trial').mean()
