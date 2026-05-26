import pandas as pd
import matplotlib.pyplot as plt

def energy_over_time():
     # used this to make graphs of energy / year  ( energy / March of each year )
    df = pd.read_csv('wind-2018-uur-data-daily.csv')
    df['date'] = pd.to_datetime(df['date'])

    df_spring = df[df['date'].dt.month.isin([3])]

    plt.figure(figsize=(14, 5))

    # plt.bar(df_spring['date'], df_spring['capacity (kW)'], color='green', width=1)
    plt.bar(df['date'], df['capacity (kW)'], color='green', width=1)


    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Date')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    plt.ylabel('Capacity (MW)')
    plt.title('Daily Wind Energy 2018')
    # plt.title('Daily Wind Energy in March 2018')
    plt.tight_layout()
    plt.show()
# energy_over_time()

def line_chart_per_5years():
    files = [
        'wind-2017-uur-data-daily.csv',
        'wind-2018-uur-data-daily.csv',
        'wind-2019-uur-data-daily.csv',
        'wind-2020-uur-data-daily.csv',
        'wind-2021-uur-data-daily.csv',
    ]

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    plt.figure(figsize=(16, 5))
    plt.plot(df['date'], df['capacity (kW)'], color='purple', linewidth=0.8)

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Date')
    plt.ylabel('Capacity (kW)')
    plt.title('Daily Wind Energy 2017 - 2021')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 1e6:.0f}M'))
    plt.tight_layout()
    plt.show()

line_chart_per_5years()


