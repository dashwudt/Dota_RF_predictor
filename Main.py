from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import joblib
from sklearn.metrics import log_loss
import time
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt


utils.get_tsm_data()
time.sleep(1123)

utils.get_player_stat()

q = input('qweq')
#utils.update_all()
pd.options.display.max_columns = None
#pd.options.display.max_rows = None


#Data = pd.read_csv('D:\dota_bet\heroData.csv',sep='\t')

Data = pd.read_csv('No_hero_data_all.csv',sep='\t')
ratings = pd.read_csv('D:/dota_bet/ratings.csv',sep='\t')

def kelly(bank, classifier, with_heroes,bo):
    d = {}
    d_kostil = {}
    Radiant_Team = input('Radiant_Team ')
    #d['Radiant_Team'] = Radiant_Team
    d['Radiant_Team_Rating'] = ratings[ratings['teamName'] == f'{Radiant_Team}']['rating'][-1:].values[0]
    d['Radiant_Team_mu'] = ratings[ratings['teamName'] == f'{Radiant_Team}']['mu'][-1:].values[0]
    d['Radiant_Team_phi'] = ratings[ratings['teamName'] == f'{Radiant_Team}']['phi'][-1:].values[0]
    d['Radiant_Team_ratingSevenDaysAgo'] = ratings[ratings['teamName'] == f'{Radiant_Team}']['ratingSevenDaysAgo'][-1:].values[0]
    Dire_Team = input('Dire_Team ')
    #d['Dire_Team'] = Dire_Team
    d['Dire_Team_Rating'] = ratings[ratings['teamName'] == f'{Dire_Team}']['rating'][-1:].values[0]
    d['Dire_Team_mu'] = ratings[ratings['teamName'] == f'{Dire_Team}']['mu'][-1:].values[0]
    d['Dire_Team_phi'] = ratings[ratings['teamName'] == f'{Dire_Team}']['phi'][-1:].values[0]
    d['Dire_Team_ratingSevenDaysAgo'] = ratings[ratings['teamName'] == f'{Dire_Team}']['ratingSevenDaysAgo'][-1:].values[0]

    #d_kostil['Radiant_Team'] = Dire_Team
    d_kostil['Radiant_Team_Rating'] = ratings[ratings['teamName'] == f'{Dire_Team}']['rating'][-1:].values[0]
    d_kostil['Radiant_Team_mu'] = ratings[ratings['teamName'] == f'{Dire_Team}']['mu'][-1:].values[0]
    d_kostil['Radiant_Team_phi'] = ratings[ratings['teamName'] == f'{Dire_Team}']['phi'][-1:].values[0]
    d_kostil['Radiant_Team_ratingSevenDaysAgo'] = ratings[ratings['teamName'] == f'{Dire_Team}']['ratingSevenDaysAgo'][-1:].values[0]
    #d_kostil['Dire_Team'] = Radiant_Team
    d_kostil['Dire_Team_Rating'] = ratings[ratings['teamName'] == f'{Radiant_Team}']['rating'][-1:].values[0]
    d_kostil['Dire_Team_mu'] = ratings[ratings['teamName'] == f'{Radiant_Team}']['mu'][-1:].values[0]
    d_kostil['Dire_Team_phi'] = ratings[ratings['teamName'] == f'{Radiant_Team}']['phi'][-1:].values[0]
    d_kostil['Dire_Team_ratingSevenDaysAgo'] = ratings[ratings['teamName'] == f'{Radiant_Team}']['ratingSevenDaysAgo'][-1:].values[0]


    ddf = pd.DataFrame(data=[d_kostil])
    df = pd.DataFrame(data=[d])
    #ddata = utils.feature_encode_no_heroes(ddf)
    #data = utils.feature_encode_no_heroes(df)
    ddata = np.asarray(ddf)
    data = np.asarray(df)
    if with_heroes:
        dfs = utils.hero_one_hot(data,ddata)
        normal_df = dfs[0]
        reverse_df = dfs[1]
        hero_pred = classifier.predict_proba(normal_df)
        reverse_hero_pred = classifier.predict_proba(reverse_df)
        print('Side diff Radiant', abs(hero_pred[0][1]-reverse_hero_pred[0][0]),' Dire ', abs(hero_pred[0][0]-reverse_hero_pred[0][1]))
        print(f'Reverse {Radiant_Team}', reverse_hero_pred[0][0],f' {Dire_Team} ', reverse_hero_pred[0][1])
        Radiant_prob = (hero_pred[0][1] + reverse_hero_pred[0][0])/2
        Dire_prob = (hero_pred[0][0] + reverse_hero_pred[0][1]) / 2
        print(f'{Dire_Team}:', Dire_prob, f'{Radiant_Team}:', Radiant_prob)
    else:
        prob = classifier.predict_proba(np.asarray(data))
        prob1 = classifier.predict_proba(np.asarray(ddata))
        Radiant_prob = (prob[0][1]+prob1[0][0])/2
        Dire_prob = (prob[0][0]+prob1[0][1])/2


    if bo ==2:
        draw = Dire_prob*Radiant_prob*2
        rw = Radiant_prob**2
        dw = Dire_prob**2
        print(f'{Dire_Team}: {int(dw*100)}% {Radiant_Team}: {int(rw*100)}% Draw: {int(draw*100)}%')
        if (draw-dw)>.05 and (draw-rw)>.05:
            coefficient_draw = float(input(f'Coefficient draw '))
            if draw <= 1/coefficient_draw:
                print(f"Bet if кэф>{1/draw}:)")
                return
            else:
                print(f'Bet on Draw: ', int(bank * (coefficient_draw * draw - 1)/(coefficient_draw-1)))
                return
        if rw>max(dw,.36):
            coefficient_radiant = float(input(f'Coefficient {Radiant_Team} '))
            if rw <= 1/coefficient_radiant:
                print(f"Bet if кэф>{1/rw}:)")
                return
            else:
                print(f'Bet on {Radiant_Team}:', int(bank * (coefficient_radiant * rw - 1)/(coefficient_radiant-1)))
                return
        elif dw > .36:
            coefficient_dire = float(input(f'Coefficient {Dire_Team} '))
            if dw <= 1 / coefficient_dire:
                print(f"Bet if кэф>{1 / dw}:)")
                return
            else:
                print(f'Bet on {Dire_Team}', int(bank * (coefficient_dire * dw - 1) / (coefficient_dire - 1)))
                return
        print('Not worth')
        return

    if bo == 3:
        rw = Radiant_prob**2 + 2 * (Radiant_prob**2) * Dire_prob
        dw = Dire_prob**2 + 2 * (Dire_prob**2) * Radiant_prob
        print(f'{Dire_Team}: {int(dw*100)}% {Radiant_Team}: {int(rw*100)}% 2:0 {Radiant_Team} {Radiant_prob**2} 2:0 {Dire_Team} {Dire_prob**2}')
        return
    print(f'{Dire_Team}:', Dire_prob, f'{Radiant_Team}:', Radiant_prob)
    if Radiant_prob > max(Dire_prob,.52):
        coefficient_radiant = float(input(f'Coefficient {Radiant_Team} '))
        fora = float(input('Фора? '))
        if fora:
            foracaef = float(input('Кэф форы '))
            if Radiant_prob**2 <= 1/foracaef:
                print(f"Bet if кэф>{1 / (Radiant_prob**2)}:)")
                return
            else:
                print(f'Bet on {Radiant_Team}:', int(bank * (foracaef * Radiant_prob**2 - 1)/(foracaef-1)))
                return
        else:
            if Radiant_prob <= 1/coefficient_radiant:
                print(f"Bet if кэф>{1/Radiant_prob}:)")
                return
            else:
                print(f'Bet on {Radiant_Team}:', int(bank * (coefficient_radiant * Radiant_prob - 1)/(coefficient_radiant-1)))
                return
    elif Dire_prob>.52:
        coefficient_dire = float(input(f'Coefficient {Dire_Team} '))
        fora = float(input('Фора? '))
        if fora:
            foracaef = float(input('Кэф форы '))
            if Dire_prob**2 <= 1 / foracaef:
                print(f"Bet if кэф>{1/(Dire_prob**2)}:)")
                return
            else:
                print(f'Bet on {Dire_Team}', int(bank * (foracaef * Dire_prob**2 - 1) / (foracaef - 1)))
                return
        else:
            if Dire_prob <= 1 / coefficient_dire:
                print(f"Bet if кэф>{1/Dire_prob}:)")
                return
            else:
                print(f'Bet on {Dire_Team}', int(bank * (coefficient_dire * Dire_prob - 1) / (coefficient_dire - 1)))
                return
    print('Not worth')
Data = Data[(Data['Radiant_Team_Rating']>1700) | (Data['Dire_Team_Rating']>1700 )]
target = list(Data['Radiant_win'])
Data.drop(columns=['Radiant_win','Radiant_Team','Dire_Team','startDate'],inplace=True)

#Data = utils.feature_encode(Data)
#Data.drop(columns=[],inplace=True)

# should be recursive
def balance_for_hero_position(df,target):
    X=pd.DataFrame()
    new_df = df.copy()
    X_test = pd.DataFrame( )
    full=pd.DataFrame()
    heropos1 = [1,2,3,4,5]
    for i in heropos1:
        heropos2 = heropos1.copy()
        heropos2.remove(i)
        new_df['Radiant_hero1'] = df[f'Radiant_hero{i}']
        new_df['Dire_hero1'] = df[f'Dire_hero{i}']
        for j in heropos2:
            new_df['Radiant_hero2'] = df[f'Radiant_hero{j}']
            new_df['Dire_hero2'] = df[f'Dire_hero{j}']
            heropos3 = heropos2.copy()
            heropos3.remove(j)
            for w in heropos3:
                new_df['Radiant_hero3'] = df[f'Radiant_hero{w}']
                new_df['Dire_hero3'] = df[f'Dire_hero{w}']
                heropos4 = heropos3.copy( )
                heropos4.remove(w)
                for e in heropos4:
                    new_df['Radiant_hero4'] = df[f'Radiant_hero{e}']
                    new_df['Dire_hero4'] = df[f'Dire_hero{e}']
                    heropos5 = heropos4.copy( )
                    heropos5.remove(e)
                    for ss in heropos5:
                        new_df['Radiant_hero5'] = df[f'Radiant_hero{ss}']
                        new_df['Dire_hero5'] = df[f'Dire_hero{ss}']
                        new_df['startDate'] = df['startDate']
                        new_df['target'] = target
                        X_test = new_df[int(len(new_df)*.75):]
                        qq = new_df[:int(len(new_df)*.75)]
                        full = pd.concat([full, new_df], ignore_index=True)
                        X=pd.concat([X,qq],ignore_index=True)
    Y_test = X_test['target']

    Y=X['target']
    X_test.drop(columns=['target'],inplace=True)
    full.drop(columns=['target'], inplace=True)
    X.drop(columns=['target'],inplace=True)

    return X, Y, X_test, Y_test, full
'''
X_hero, Y_hero, X_test_hero, Y_test_hero, full = balance_for_hero_position(q,target)

full=np.asarray(full)
X_test_hero =  np.asarray(X_test_hero)
Y_test_hero =  np.asarray(Y_test_hero)
X_hero      =  np.asarray(X_hero)
Y_hero      =  np.asarray(Y_hero)



lgloss=100
small_bar = tqdm(range(2*4*5*3),desc='Total',position=0,leave=False)
for criterion in ['gini','entropy']:
    for n_estimators in [20,40,60,80,100]:
        for samples in [.66,.77,.88,1]:
            for features in ['sqrt','log2',None]:
                small_bar.update( )
                hero_model = RandomForestClassifier(n_estimators=n_estimators,random_state=42,criterion=criterion,max_features=features,max_samples=samples,n_jobs=-1)
                hero_model.fit(X_hero,Y_hero)
                if log_loss(Y_test_hero,hero_model.predict_proba(X_test_hero))<lgloss and balanced_accuracy_score(Y_test_hero,hero_model.predict(X_test_hero))>.5 and hero_model.feature_importances_[-1]<.5:
                    lgloss=log_loss(Y_test_hero,hero_model.predict_proba(X_test_hero))
                    crit=criterion
                    trees=n_estimators
                    sampl=samples
                    fet=features
                    print('Hero model score: ', balanced_accuracy_score(Y_test_hero, hero_model.predict(X_test_hero)))
                    print(log_loss(Y_test_hero, hero_model.predict_proba(X_test_hero)))
                    print(hero_model.feature_importances_)
                    #print('Hero model score: ', balanced_accuracy_score(Y_test_hero,hero_model.predict(X_test_hero)))
                    #print(log_loss(Y_test_hero, hero_model.predict_proba(X_test_hero)))
                    #print(hero_model.feature_importances_)

hero_model = RandomForestClassifier(n_estimators=trees,random_state=42,criterion=crit,max_features=fet,max_samples=sampl)
hero_model.fit(X_hero,Y_hero)

print('Hero model score: ', balanced_accuracy_score(Y_test_hero,hero_model.predict(X_test_hero)))
print(log_loss(Y_test_hero,hero_model.predict_proba(X_test_hero)))
print(hero_model.feature_importances_)

filename = 'hero_model.sav'
joblib.dump(hero_model, filename)
'''


#razbiv = float(input('РазбивОчка '))
razbiv = .92
X_test_team =  np.asarray(Data[int(len(Data)*razbiv+.01) :])
Y_test_team =  np.asarray(target[int(len(Data)*razbiv+.01) :])
X_team      =  np.asarray(Data[:int(len(Data)*razbiv) ])
Y_team      =  np.asarray(target[:int(len(Data)*razbiv) ])

print(len(Y_test_team))
#filename = 'hero_model.sav'
#hero_model = joblib.load(filename)

lgloss=100
small_bar = tqdm(range(2*30*2*3),desc='Total',position=0,leave=False)
for criterion in ['gini','entropy']:
    for n_estimators in range(21,3021,100):
        for samples in [.66, 1]:
            for features in ['sqrt','log2',None]:
                small_bar.update( )
                team_model = RandomForestClassifier(n_estimators = n_estimators,random_state=42,criterion=criterion,max_features=features,max_samples=samples,n_jobs=-1)
                team_model.fit(X_team,Y_team)
                if log_loss(Y_test_team,team_model.predict_proba(X_test_team))<lgloss and team_model.score(X_test_team, Y_test_team)>.5 and team_model.feature_importances_[0]>0:
                    lgloss=log_loss(Y_test_team,team_model.predict_proba(X_test_team))
                    crit=criterion
                    trees=n_estimators
                    sampl=samples
                    fet=features
                    print('team_model score: ', team_model.score(X_test_team, Y_test_team))
                    print(log_loss(Y_test_team, team_model.predict_proba(X_test_team)))
                    print(team_model.feature_importances_)

team_model = RandomForestClassifier(n_estimators =trees,random_state=42,criterion=crit,max_features=fet,max_samples=sampl,n_jobs=-1)
team_model.fit(X_team,Y_team)

print(crit,sampl,fet,trees)
print('team_model score: ', team_model.score(X_test_team, Y_test_team))
print(log_loss(Y_test_team, team_model.predict_proba(X_test_team)))
print(team_model.feature_importances_)
time.sleep(1121)
filename = 'team_model.sav'
joblib.dump(team_model, filename)

team_model = RandomForestClassifier(n_estimators =200, random_state=42, n_jobs=-1)
team_model.fit(X_team,Y_team)
#filename = 'team_model.sav'
#joblib.dump(team_model, filename)


#team_model = joblib.load(filename)
print('team_model score: ', team_model.score(X_test_team, Y_test_team))
print(log_loss(Y_test_team, team_model.predict_proba(X_test_team)))
print(team_model.feature_importances_)

#
#lr=[]
#ld=[]
#l=[]
#predictions = team_model.predict_proba(X_test_team)
#print(len(predictions))
#for i, pred in enumerate(predictions):
#    if Y_test_team[i] and pred[1]<.5:
#        lr.append(pred[1])
#        l.append(pred[1])
#        #print('Radiant ',1-pred[1])
#    if not Y_test_team[i] and pred[0]<.5:
#        l.append(pred[0])
#        ld.append(pred[0])
#        #print('Dire ',1 - pred[0])
#
#
#
#
#
#
#print('Radiant mean',np.average(np.asarray(lr)),len(lr))
#print('Dire mean',np.average(np.asarray(ld)),len(ld))
#plt.figure(figsize=(15,8))
#plt.hist(np.asarray(l),density=True,stacked=True)
#plt.show()




#filename = 'Final_Model.sav'
#filename = 'Final_Model_heroes.sav'

#joblib.dump(classifier, filename)

#loaded_model = joblib.load(filename)

filename = 'team_model.sav'
team_model = joblib.load(filename)

kelly(15,team_model,0,bo = 3)
