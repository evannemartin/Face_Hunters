import pandas as pds
import numpy as np


db = pds.read_csv('../database/list_attr_celeba.csv', sep=",") # load a pandas dataframe from csv in current directory
print(db.head())
print(db.shape)

#sorting male/female

# print(len(db['Male']))
# print(db['Male'].value_counts())
#
# lmale=db['Male'].value_counts()[1]
# lfemale=db['Male'].value_counts()[-1]
# print(lmale)
# print(lfemale)
# male=np.chararray(shape=lmale,itemsize=10)
# female=np.chararray(shape=lfemale,itemsize=10)
#
# j=0
# k=0
#
# #for i in range (len(db['Male'])) :
#     #if (db['Male'][i]==1) :
#         #male[j]=db['image_id'][i]
#         #j+=1
#     #else :
#         #female[k]=db['image_id'][i]
#         #k+=1

db_male=db.loc[db['Male']==1]
db_female=db.loc[db['Male']==-1]


#sorting young/old

db_male_young=db_male.loc[db['Young']==1]
db_male_old=db_male.loc[db['Young']==-1]
print("Liste des hommes jeunes :",db_male_young)

db_female_young=db_female.loc[db['Young']==1]
db_female_old=db_female.loc[db['Young']==-1]
print("Liste des femmes jeunes :", db_female_young)

#sorting beard/no beard

db_male_young_nobeard=db_male_young.loc[db['No_Beard']==1]
# Enregister notre structure de données dans un fichier.csv
db_male_young_nobeard.to_csv('male_young_nobeard.csv',mode='w+')
db_male_young_beard=db_male_young.loc[db['No_Beard']==-1]
db_male_young_beard.to_csv('male_young_beard.csv',mode='w+')

db_male_old_nobeard=db_male_old.loc[db['No_Beard']==1]
db_male_old_nobeard.to_csv('male_old_nobeard.csv',mode='w+')
db_male_old_beard=db_male_old.loc[db['No_Beard']==-1]
db_male_old_beard.to_csv('male_old_beard.csv',mode='w+')

print("Liste des hommes jeunes avec barbe :", db_male_young_beard)
print("Liste des hommes vieux sans barbe :", db_male_old_nobeard)

#sorting wavy/straight hair

db_female_young_wavy=db_female_young.loc[db['Straight_Hair']==-1]
db_female_young_wavy.to_csv('female_young_wavy.csv',mode='w+')
db_female_young_straight=db_female_young.loc[db['Straight_Hair']==1]
db_female_young_straight.to_csv('female_young_straight.csv',mode='w+')

db_female_old_wavy=db_female_old.loc[db['Straight_Hair']==-1]
db_female_old_wavy.to_csv('female_old_wavy.csv',mode='w+')
db_female_old_straight=db_female_old.loc[db['Straight_Hair']==1]
db_female_old_straight.to_csv('female_old_straight.csv',mode='w+')

print("Liste des femmes jeunes avec straight :", db_female_young_straight)
print("Liste des femmes jeunes sans straight :", db_female_young_wavy)
print("Liste des femmes vieilles sans straight :", db_female_old_wavy)
