import re
import pandas as pd
def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:am|pm)\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    dates = [d.replace("\u202f", " ") for d in dates]
    df = pd.DataFrame({'user_message': messages, 'date': dates})
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([^:]+):\s', message, maxsplit=1)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notofication')
            messages.append(entry[0])
    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    df['clean_date'] = df['date'].str.extract(r'(\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2} [ap]m)')
    df['datetime'] = pd.to_datetime(df['clean_date'], format="%d/%m/%Y, %I:%M %p")
    df['year'] = df['datetime'].dt.year
    df['month_num'] = df['datetime'].dt.month
    df['month'] = df['datetime'].dt.month_name()
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['only_date'] = df['datetime'].dt.date
    df['day_name'] = df['datetime'].dt.day_name()
    df = df.drop(columns=['date'])
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('clean_date')))
    df = df[cols]
    df = df.rename(columns={'clean_date': 'date'})
    return df
