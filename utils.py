import pandas as pd
import numpy as np
import requests
import json
import csv
import time
import tqdm
from datetime import timedelta, date,datetime

pd.options.display.max_columns = None

def update_ratings():
    def daterange(start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)
    ratings = pd.read_csv('D:/dota_bet/ratings.csv', sep='\t')
    start_date = (datetime.fromtimestamp(ratings['Date'][-1:]))    #2021, 1, 1
    end_date = (datetime.fromtimestamp(time.time()+90000))
    for single_date in daterange(start_date, end_date):
        time.sleep(1)
        dic = requests.get('https://www.datdota.com/api/ratings?date='+single_date.strftime("%d-%m-%Y")).json()
        l=[]
        tn=[]
        date =[]
        for d in dic['data']:
            date.append(single_date.timestamp())
            l.append(d['glicko2'])
            tn.append(d['teamName'])
        tdf = pd.DataFrame(l)
        tdf.drop(columns=['sigma'],inplace=True)
        tdf['teamName'] = tn
        tdf['Date'] = date
        ratings=pd.concat([ratings,tdf],ignore_index=True)
        ratings.applymap(str)
        ratings.drop_duplicates(inplace=True)
        ratings.to_csv('ratings.csv', index=False, sep="\t")
    print('ratings updated')

def get_heroes_by_id(id):
    heroes = []
    response = requests.get('https://api.opendota.com/api/matches/' + str(id))
    print(response.status_code)
    if response.status_code == 502:
        time.sleep(5)
        response = requests.get('https://api.opendota.com/api/matches/' + str(id))
    if response.status_code != 200:
        with open('D:\dota_bet\heroes.json', 'r', encoding='utf-8') as outfile:
            data = json.load(outfile)
        data = list(map(lambda d: d['id'], data))
        one_hoted_list_radiant_heroes = np.zeros((1, len(data)))
        one_hoted_list_dire_heroes = np.zeros((1, len(data)))
        one_hoted = np.concatenate((one_hoted_list_radiant_heroes, one_hoted_list_dire_heroes), axis=1)
        fuckmylife = pd.DataFrame(one_hoted)
        fuckmylife.to_csv('D:\dota_bet\FUCK.csv', sep='\t', index=False)
        df = pd.read_csv('D:\dota_bet\FUCK.csv', sep='\t')
        return df
    else:
        dic = response.json( )
        heroes.append(dic['players'][0]['hero_id'])
        heroes.append(dic['players'][1]['hero_id'])
        heroes.append(dic['players'][2]['hero_id'])
        heroes.append(dic['players'][3]['hero_id'])
        heroes.append(dic['players'][4]['hero_id'])
        heroes.append(dic['players'][5]['hero_id'])
        heroes.append(dic['players'][6]['hero_id'])
        heroes.append(dic['players'][7]['hero_id'])
        heroes.append(dic['players'][8]['hero_id'])
        heroes.append(dic['players'][9]['hero_id'])
        with open('D:\dota_bet\heroes.json', 'r', encoding='utf-8') as outfile:
            data = json.load(outfile)
        data = list(map(lambda d: d['id'], data))
        one_hoted_list_radiant_heroes = np.zeros((1, len(data)))
        one_hoted_list_dire_heroes = np.zeros((1, len(data)))
        for j, item in enumerate(heroes):
            if j <= 4:
                one_hoted_list_radiant_heroes[0][data.index(int(item))] = 1
            else:
                one_hoted_list_dire_heroes[0][data.index(int(item))] = 1
        one_hoted = np.concatenate((one_hoted_list_radiant_heroes, one_hoted_list_dire_heroes),axis=1)
        fuckmylife = pd.DataFrame(one_hoted)
        fuckmylife.to_csv('D:\dota_bet\FUCK.csv',sep='\t',index=False)
        df = pd.read_csv('D:\dota_bet\FUCK.csv',sep='\t')
        return df
'''
for i, matchid in tqdm.tqdm(enumerate(df["match_id"])):
    if i < c * 100:
        time.sleep(1.1)
        response = requests.get('https://api.opendota.com/api/matches/' + str(matchid))
        dic = response.json()

        rh1 = (dic['players'][0]['hero_id'])
        rh2 = (dic['players'][1]['hero_id'])
        rh3 = (dic['players'][2]['hero_id'])
        rh4 = (dic['players'][3]['hero_id'])
        rh5 = (dic['players'][4]['hero_id'])
        dh1 = (dic['players'][5]['hero_id'])
        dh2 = (dic['players'][6]['hero_id'])
        dh3 = (dic['players'][7]['hero_id'])
        dh4 = (dic['players'][8]['hero_id'])
        dh5 = (dic['players'][9]['hero_id'])
        rp1 = (dic['players'][0]['name'])
        rp2 = (dic['players'][1]['name'])
        rp3 = (dic['players'][2]['name'])
        rp4 = (dic['players'][3]['name'])
        rp5 = (dic['players'][4]['name'])
        dp1 = (dic['players'][5]['name'])
        dp2 = (dic['players'][6]['name'])
        dp3 = (dic['players'][7]['name'])
        dp4 = (dic['players'][8]['name'])
        dp5 = (dic['players'][9]['name'])
        patch = (dic['patch'])
        radiant_win = (dic['radiant_win'])
        data = [[rh1,
                    rh2,
                    rh3,
                    rh4,
                    rh5,
                    dh1,
                    dh2,
                    dh3,
                    dh4,
                    dh5,
                    rp1,
                    rp2,
                    rp3,
                    rp4,
                    rp5,
                    dp1,
                    dp2,
                    dp3,
                    dp4,
                    dp5,
                    patch,
                    radiant_win]]

        small_df = pd.DataFrame(data,columns = ['radiant_hero1','radiant_hero2','radiant_hero3','radiant_hero4','radiant_hero5',
                                                    'dire_hero1','dire_hero2','dire_hero3','dire_hero4','dire_hero5',
                                                    'radiant_player1', 'radiant_player2', 'radiant_player3', 'radiant_player4','radiant_player5',
                                                    'dire_player1', 'dire_player2', 'dire_player3', 'dire_player4', 'dire_player5',
                                                    'patch','radiant_win'])
        big_df = pd.read_csv('promatches_parsed.csv',sep='\t')
        big_df = pd.concat([small_df, big_df],ignore_index=True)
        big_df.drop_duplicates( )
        big_df.to_csv('promatches_parsed.csv', index=False, sep="\t")
'''

def hero_one_hot(df,ddf):
    if input('First pick ') =='r':
        rh1 = int(input('Radiant hero 1 '))
        dh1 = int(input('Dire hero 1 ')   )
        dh2 = int(input('Dire hero 2 ')   )
        rh2 = int(input('Radiant hero 2 '))
        dh3 = int(input('Dire hero 3 ')   )
        rh3 = int(input('Radiant hero 3 '))
        rh4 = int(input('Radiant hero 4 '))
        dh4 = int(input('Dire hero 4 ')   )
        rh5 = int(input('Radiant hero 5 '))
        dh5 = int(input('Dire hero 5 '))
    else:
        dh1 = int(input('Dire hero 1 ')   )
        rh1 = int(input('Radiant hero 1 '))
        rh2 = int(input('Radiant hero 2 '))
        dh2 = int(input('Dire hero 2 '))
        rh3 = int(input('Radiant hero 3 '))
        dh3 = int(input('Dire hero 3 '))
        dh4 = int(input('Dire hero 4 ')   )
        rh4 = int(input('Radiant hero 4 '))
        dh5 = int(input('Dire hero 5 '))
        rh5 = int(input('Radiant hero 5 '))

    df['Radiant_hero1'] = [rh1]
    df['Radiant_hero2'] = [rh2]
    df['Radiant_hero3'] = [rh3]
    df['Radiant_hero4'] = [rh4]
    df['Radiant_hero5'] = [rh5]
    df['Dire_hero1'] = [dh1]
    df['Dire_hero2'] = [dh2]
    df['Dire_hero3'] = [dh3]
    df['Dire_hero4'] = [dh4]
    df['Dire_hero5'] = [dh5]

    ddf['Radiant_hero1'] = [dh1]
    ddf['Radiant_hero2'] = [dh2]
    ddf['Radiant_hero3'] = [dh3]
    ddf['Radiant_hero4'] = [dh4]
    ddf['Radiant_hero5'] = [dh5]
    ddf['Dire_hero1'] = [rh1]
    ddf['Dire_hero2'] = [rh2]
    ddf['Dire_hero3'] = [rh3]
    ddf['Dire_hero4'] = [rh4]
    ddf['Dire_hero5'] = [rh5]


    return [np.asarray(df),np.asarray(ddf)]

def date_encode(df):
    df['year'] = list(map(int,pd.to_datetime(df['startDate'], unit='s').dt.year))
    df['year']-=2000
    df['day_of_year'] = pd.to_datetime(df['startDate'], unit='s').dt.day_of_year
    one_hoted_year = np.zeros((len(df), 30))
    one_hoted_day = np.zeros((len(df), 366))
    for j, item in enumerate(df['year']):
        one_hoted_year[j][item] = 1
    for j, item in enumerate(df['day_of_year']):
        one_hoted_day[j][item] = 1
    q = np.concatenate((one_hoted_year, one_hoted_day), axis=1)
    new_df = pd.DataFrame(q)
    df.drop(columns=['year', 'day_of_year','startDate'], inplace=True)
    new_df = pd.concat([new_df, df], ignore_index=True, axis=1)
    return new_df

def feature_encode_no_heroes(df):
    teams = list(pd.unique(pd.read_csv('D:/dota_bet/ratings.csv', sep='\t')['teamName']))
    rteams = []
    dteams = []
    for item in df['Radiant_Team']:
        rteams.append(teams.index(item))
    for item in df['Dire_Team']:
        dteams.append(teams.index(item))
    df.drop(columns=['Radiant_Team', 'Dire_Team'], inplace=True)
    df['Radiant_Team']=rteams
    df['Dire_Team']=dteams
    return df

def feature_encode(df):
    teams = list(pd.unique(pd.read_csv('D:/dota_bet/ratings.csv', sep='\t')['teamName']))
    rteams = []
    dteams = []
    rhero1 = []
    dhero1 = []
    rhero2 = []
    dhero2 = []
    rhero3 = []
    dhero3 = []
    rhero4 = []
    dhero4 = []
    rhero5 = []
    dhero5 = []
    for item in df['Radiant_Team']:
        rteams.append(teams.index(item))
    for item in df['Dire_Team']:
        dteams.append(teams.index(item))
    df.drop(columns=['Radiant_Team', 'Dire_Team'], inplace=True)
    df['Radiant_Team']=rteams
    df['Dire_Team']=dteams
    radteamheroes = df.iloc[:,:123]
    radteamheroes=radteamheroes.to_numpy()
    radteamheroes=np.nonzero(radteamheroes)[1]
    direteamheroes = df.iloc[:, 123:246]
    direteamheroes = direteamheroes.to_numpy( )
    direteamheroes = np.nonzero(direteamheroes)[1]

    for i in range(0,len(direteamheroes),5):
        rhero1.append(radteamheroes[i])
        rhero2.append(radteamheroes[i+1])
        rhero3.append(radteamheroes[i+2])
        rhero4.append(radteamheroes[i+3])
        rhero5.append(radteamheroes[i+4])
        dhero1.append(direteamheroes[i])
        dhero2.append(direteamheroes[i + 1])
        dhero3.append(direteamheroes[i + 2])
        dhero4.append(direteamheroes[i + 3])
        dhero5.append(direteamheroes[i + 4])

    df[f'Radiant_hero1'] = rhero1
    df[f'Radiant_hero2'] = rhero2
    df[f'Radiant_hero3'] = rhero3
    df[f'Radiant_hero4'] = rhero4
    df[f'Radiant_hero5'] = rhero5
    df[f'Dire_hero1'] = dhero1
    df[f'Dire_hero2'] = dhero2
    df[f'Dire_hero3'] = dhero3
    df[f'Dire_hero4'] = dhero4
    df[f'Dire_hero5'] = dhero5
    df = df.iloc[:,246:]
    return df

def new_onehot(df):
    teams = list(pd.unique(pd.read_csv('D:/dota_bet/ratings.csv',sep='\t')['teamName']))
    one_hoted_list_radiant_team=np.zeros((len(df),len(teams)))
    one_hoted_list_dire_team = np.zeros((len(df), len(teams)))
    for j, item in enumerate(df['Radiant_Team']):
        one_hoted_list_radiant_team[j][teams.index(item)] = 1
    for j, item in enumerate(df['Dire_Team']):
        one_hoted_list_dire_team[j][teams.index(item)] = 1
    q = np.concatenate((one_hoted_list_radiant_team, one_hoted_list_dire_team),axis=1)
    new_df = pd.DataFrame(q)
    df.drop(columns=['Radiant_Team','Dire_Team'],inplace = True)
    new_df = pd.concat([new_df,df],ignore_index=True,axis=1)
    return new_df

def onehot(df, ishero):
    epochlist = np.zeros((len(df), 250))
    for j, epoch in enumerate(df["epoch"]):
        epochlist[j][epoch] = 1
    patchlist = np.zeros((len(df),1))
    for j, patch in enumerate(df["patch"]):
        if int(patch)>1:
            patchlist[j][0] = 1
    with open('D:\dota_bet\proplayers.json', 'r', encoding='utf-8') as outfile:
        data = json.load(outfile)
    data = list(map(lambda d: d['name'],data))
    one_hoted_list_radiant_players = np.zeros((len(df),len(data)))
    one_hoted_list_dire_players = np.zeros((len(df),len(data)))
    for i in range(1, 6):
        for j,item in enumerate(df[f"radiant_player{i}"]):
            one_hoted_list_radiant_players[j][data.index(item)] = 1
        for j,item in enumerate(df[f"dire_player{i}"]):
            one_hoted_list_dire_players[j][data.index(item)] = 1
    if ishero:
        with open('D:\dota_bet\heroes.json', 'r', encoding='utf-8') as outfile:
            data = json.load(outfile)
        data = list(map(lambda d: d['id'], data))
        one_hoted_list_radiant_heroes = np.zeros((len(df), len(data)))
        one_hoted_list_dire_heroes = np.zeros((len(df), len(data)))
        for i in range(1, 6):
            for j, item in enumerate(df[f"radiant_hero{i}"]):
                one_hoted_list_radiant_heroes[j][data.index(int(item))] = 1
            for j, item in enumerate(df[f"dire_hero{i}"]):
                one_hoted_list_dire_heroes[j][data.index(int(item))] = 1
        one_hoted = np.concatenate((one_hoted_list_radiant_players, one_hoted_list_dire_players,one_hoted_list_radiant_heroes,one_hoted_list_dire_heroes,patchlist,epochlist),axis=1)
    else:
        one_hoted = np.concatenate((one_hoted_list_radiant_players, one_hoted_list_dire_players,patchlist,epochlist),axis=1)
    return one_hoted

def update_hero_data():
    ratings = pd.read_csv('D:/dota_bet/ratings.csv', sep='\t')
    microl = ['premium','professional','semipro']
    for tier in microl:
        df = pd.DataFrame( )
        rmu = []
        rphi = []
        rratingSevenDaysAgo=[]
        dmu = []
        dphi = []
        dratingSevenDaysAgo = []
        Match_id = []
        Date = []
        Radiant_Team = []
        Dire_Team = []
        Radiant_win = []
        Radiant_Team_Rating = []
        Dire_Team_Rating = []
        dic = requests.get(f'https://www.datdota.com/api/matches?tier={tier}').json( )
        for d in dic['data']:
            cur_date = float(d['startDate'] / 1000)
            m_id = d['matchId']
            time.sleep(1)
            df = pd.concat([get_heroes_by_id(m_id),df],ignore_index=True)
            Match_id.append(m_id)
            Date.append(str(cur_date))
            rad_team = d['radiant']['name']
            Radiant_Team.append(rad_team)
            dire_team = d['dire']['name']
            Dire_Team.append(dire_team)
            tempdf = ratings[(ratings['Date'] >= cur_date - 86400) &
                             (ratings['Date'] < (cur_date))]
            if not list(tempdf[tempdf['teamName'] == dire_team]['rating']):
                e = None
                emu = None
                ephi = None
                eratingSevenDaysAgo = None
            else:
                e = list(tempdf[tempdf['teamName'] == dire_team]['rating'])[0]
                emu = list(tempdf[tempdf['teamName'] == dire_team]['mu'])[0]
                ephi = list(tempdf[tempdf['teamName'] == dire_team]['phi'])[0]
                eratingSevenDaysAgo = list(tempdf[tempdf['teamName'] == dire_team]['ratingSevenDaysAgo'])[0]
            if not list(tempdf[tempdf['teamName'] == rad_team]['rating']):
                w = None
                wmu = None
                wphi = None
                wratingSevenDaysAgo = None
            else:
                w = list(tempdf[tempdf['teamName'] == rad_team]['rating'])[0]
                wmu = list(tempdf[tempdf['teamName'] == rad_team]['mu'])[0]
                wphi = list(tempdf[tempdf['teamName'] == rad_team]['phi'])[0]
                wratingSevenDaysAgo = list(tempdf[tempdf['teamName'] == rad_team]['ratingSevenDaysAgo'])[0]
            rmu.append(wmu)
            rphi.append(wphi)
            rratingSevenDaysAgo.append(wratingSevenDaysAgo)
            Radiant_Team_Rating.append(w)
            dmu.append(emu)
            dphi.append(ephi)
            dratingSevenDaysAgo.append(eratingSevenDaysAgo)
            Dire_Team_Rating.append(e)
            Radiant_win.append(d['radiantVictory'])
        df['startDate'] = Date
        df['Radiant_Team'] = Radiant_Team
        df['Radiant_Team_Rating'] = Radiant_Team_Rating
        df['Radiant_Team_mu'] = rmu
        df['Radiant_Team_phi'] = rphi
        df['Radiant_Team_ratingSevenDaysAgo'] = rratingSevenDaysAgo
        df['Dire_Team'] = Dire_Team
        df['Dire_Team_Rating'] = Dire_Team_Rating
        df['Dire_Team_mu'] = dmu
        df['Dire_Team_phi'] = dphi
        df['Dire_Team_ratingSevenDaysAgo'] = dratingSevenDaysAgo
        df['Radiant_win'] = Radiant_win
        print(df)
        df.dropna(inplace=True)
        print(df)
        df.to_csv(f'D:\dota_bet\Data_{tier}.csv', index=False, sep='\t')

def update_data():
    ratings = pd.read_csv('D:/dota_bet/ratings.csv', sep='\t')
    microl = ['premium','professional','semipro']
    for tier in microl:
        df = pd.DataFrame( )
        rmu = []
        rphi = []
        rratingSevenDaysAgo=[]
        dmu = []
        dphi = []
        dratingSevenDaysAgo = []
        Match_id = []
        Date = []
        Radiant_Team = []
        Dire_Team = []
        Radiant_win = []
        Radiant_Team_Rating = []
        Dire_Team_Rating = []
        dic = requests.get(f'https://www.datdota.com/api/matches?tier={tier}').json( )
        for d in dic['data']:
            cur_date = float(d['startDate'] / 1000)
            m_id = d['matchId']
            #time.sleep(1)
            #df = pd.concat([get_heroes_by_id(m_id),df],ignore_index=True)
            Match_id.append(m_id)
            Date.append(str(cur_date))
            rad_team = d['radiant']['name']
            Radiant_Team.append(rad_team)
            dire_team = d['dire']['name']
            Dire_Team.append(dire_team)
            tempdf = ratings[(ratings['Date'] <= cur_date)]
            Last_date = tempdf['Date'][-1:].values[0]
            tempdf = tempdf[(tempdf['Date'] == Last_date)]
            if not list(tempdf[tempdf['teamName'] == dire_team]['rating']):
                e = None
                emu = None
                ephi = None
                eratingSevenDaysAgo = None
            else:
                e = list(tempdf[tempdf['teamName'] == dire_team]['rating'])[0]
                emu = list(tempdf[tempdf['teamName'] == dire_team]['mu'])[0]
                ephi = list(tempdf[tempdf['teamName'] == dire_team]['phi'])[0]
                eratingSevenDaysAgo = list(tempdf[tempdf['teamName'] == dire_team]['ratingSevenDaysAgo'])[0]
            if not list(tempdf[tempdf['teamName'] == rad_team]['rating']):
                w = None
                wmu = None
                wphi = None
                wratingSevenDaysAgo = None
            else:
                w = list(tempdf[tempdf['teamName'] == rad_team]['rating'])[0]
                wmu = list(tempdf[tempdf['teamName'] == rad_team]['mu'])[0]
                wphi = list(tempdf[tempdf['teamName'] == rad_team]['phi'])[0]
                wratingSevenDaysAgo = list(tempdf[tempdf['teamName'] == rad_team]['ratingSevenDaysAgo'])[0]
            rmu.append(wmu)
            rphi.append(wphi)
            rratingSevenDaysAgo.append(wratingSevenDaysAgo)
            Radiant_Team_Rating.append(w)
            dmu.append(emu)
            dphi.append(ephi)
            dratingSevenDaysAgo.append(eratingSevenDaysAgo)
            Dire_Team_Rating.append(e)
            Radiant_win.append(d['radiantVictory'])
        df['startDate'] = Date
        df['Radiant_Team'] = Radiant_Team
        df['Radiant_Team_Rating'] = Radiant_Team_Rating
        df['Radiant_Team_mu'] = rmu
        df['Radiant_Team_phi'] = rphi
        df['Radiant_Team_ratingSevenDaysAgo'] = rratingSevenDaysAgo
        df['Dire_Team'] = Dire_Team
        df['Dire_Team_Rating'] = Dire_Team_Rating
        df['Dire_Team_mu'] = dmu
        df['Dire_Team_phi'] = dphi
        df['Dire_Team_ratingSevenDaysAgo'] = dratingSevenDaysAgo
        df['Radiant_win'] = Radiant_win
        old_df = pd.read_csv(f'D:\dota_bet\Data_{tier}_no_hero.csv', sep='\t')
        df = pd.concat([old_df,df],ignore_index=True)
        df.dropna(inplace=True)
        df = df.applymap(str)
        df.drop_duplicates(inplace=True)
        df.to_csv(f'D:\dota_bet\Data_{tier}_no_hero.csv', index=False, sep='\t')

def compose_data():
    pro = pd.read_csv('D:\dota_bet\Data_professional.csv',sep='\t')
    prem = pd.read_csv('D:\dota_bet\Data_premium.csv', sep='\t')
    q = pd.concat([prem,pro],ignore_index=True)
    q.to_csv('D:\dota_bet\heroData.csv',sep='\t',index=False)
    q = q.iloc[:,246:]
    q.to_csv('D:\dota_bet\DATA.csv',sep='\t',index=False)

def compose_no_hero_data():
    pro = pd.read_csv('D:\dota_bet\Data_professional_no_hero.csv', sep='\t')
    prem = pd.read_csv('D:\dota_bet\Data_premium_no_hero.csv', sep='\t')
    #semi = pd.read_csv('D:\dota_bet\Data_semipro_no_hero.csv', sep='\t')
    q = pd.concat([prem, pro], ignore_index=True)
    #q = pd.concat([q,semi],ignore_index=True)
    q["startDate"] = q["startDate"].values.astype('float')
    q.sort_values(by=['startDate'],ignore_index=True,inplace=True)
    q.to_csv('No_hero_data_all.csv',sep='\t',index=False)

def update_all():
    update_ratings()
    update_data()
    #compose_data()
    compose_no_hero_data()

def get_player_stat():
    dic = requests.get('https://www.datdota.com/api/players/performances?after=01%2F01%2F2011&before=26%2F08%2F2022&duration=0%3B200&duration-value-from=0&duration-value-to=200&threshold=201').json()
    l=[]
    for d in dic['data']:
        l.append(d)
    tdf = pd.DataFrame(l)
    print(tdf)
    tdf.dropna(inplace=True)
    tdf.to_csv('player_full_stat.csv', index=False, sep="\t")
