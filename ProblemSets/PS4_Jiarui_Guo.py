
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import scipy.optimize as sc
from scipy.optimize import differential_evolution
import geopy
from geopy.distance import vincenty


# In[2]:

data=pd.read_excel('radio_merger_data.xlsx')


# In[3]:

data=data.drop_duplicates(['buyer_id','year','buyer_lat','buyer_long'])
list(data.columns)


# By observation, for each case of I[.], I have factual matach buyer_id= factual match target_id, and counter-factual match buyer_id =counter-factual target_id, and factual buyer_id and target_id stay the same until I cover all cases for counter-factual matches. Therefore, I start with creating id's for a table contains all factual and counter-factual matches. 

# In[48]:

Counter_factual_table=pd.DataFrame(data={'year':[],'buyer_id':[],'real_target_id':[],                                   'CF_target_id':[]})
Counter_factual_table


# In[49]:

a=[len(data[data.year==2007]),len(data[data.year==2008])]
b=data.year.unique()


# In[50]:

CFT_given_year=pd.DataFrame(data={'year':[],'buyer_id':[],'real_target_id':[],\ # real_target_id is the target_id in a factual match
                                   'CF_target_id':[]})
for j in a:
    for i in list(reversed(range(1,j))):
       
        CFT_given_year=CFT_given_year.append(pd.DataFrame(data={'buyer_id':np.array((j-i)*np.ones(i)),                                'real_target_id':np.array((j-i)*np.ones(i)),                    'CF_target_id':list(range(j+1-i,j+1))}))#buyer_id and real_target_id stays the same in each loop while counter
                                                            #factual id's are growing by one as I go down each row in the table.
    CFT_given_year['year']=b[a.index(j)]*np.ones(len(CFT_given_year))
Counter_factual_table=Counter_factual_table.append(CFT_given_year)
#I realized that I don't have to create real_target_id, rather, I could set it euqual to buyer_id after I create this table.


# In[8]:

Counter_factual_table['CF_buyer_id']=Counter_factual_table.CF_target_id


# Then, I create four tables, two of them contain buyer information and the other contain target information. Although, the two buyer tables contain the same information, but one is for factual matches the other is for counter-factual matches. The same for target information tables. This makes it easier to merge on different id's later.

# In[9]:

buyer_characteristic_table=pd.DataFrame([data.year,data.buyer_id,data.buyer_lat,data.buyer_long,                                                      data.num_stations_buyer,data.corp_owner_buyer]).transpose()


# In[10]:

CF_buyer_characteristic_table=pd.DataFrame([data.year,data.buyer_id,data.buyer_lat,data.buyer_long,                                                      data.num_stations_buyer,data.corp_owner_buyer]).transpose()
CF_buyer_characteristic_table.columns=['year','CF_buyer_id','CF_buyer_lat','CF_buyer_long','CF_num_stations_buyer',                                                                     'CF_corp_owner_buyer']


# In[11]:

CF_target_characteristic_table=pd.DataFrame([data.year,data.target_id,data.target_lat,data.target_long,                                                      data.hhi_target,data.population_target]).transpose()
CF_target_characteristic_table.columns=['year','CF_target_id','CF_target_lat','CF_target_long','CF_hhi_target',                                                                     'CF_population_target']


# In[12]:

real_target_characteristic_table=pd.DataFrame([data.year,data.target_id,data.target_lat,data.target_long,                                                      data.hhi_target,data.population_target]).transpose()
real_target_characteristic_table.columns=['year','real_target_id','real_target__lat','real_target__long','real_hhi_target',                                                                     'real_population_target']


# In[13]:

###WIP:change names of variables for cf and real targets

Counter_factual_table1=Counter_factual_table.merge(buyer_characteristic_table,on=['year','buyer_id'],how='right')
Counter_factual_table1=Counter_factual_table1.merge(CF_target_characteristic_table,on=['year','CF_target_id'],how='right')
Counter_factual_table1=Counter_factual_table1.merge(real_target_characteristic_table,on=['year','real_target_id'],how='right')
Counter_factual_table1=Counter_factual_table1.merge(CF_buyer_characteristic_table,on=['year','CF_buyer_id'],how='right')


# I re-organized the table below to make it look better and easier to check for issues  

# In[14]:

Counter_factual_table1.columns


# In[15]:

Counter_factual_table1=Counter_factual_table1[['year', 'buyer_id','CF_buyer_id','real_target_id','CF_target_id', 
       'buyer_lat', 'buyer_long', 'num_stations_buyer', 'corp_owner_buyer',
       'CF_target_lat', 'CF_target_long', 'CF_hhi_target', 'CF_population_target',
        'real_target__lat', 'real_target__long', 'real_hhi_target', 'real_population_target',
       'CF_buyer_lat','CF_buyer_long', 'CF_num_stations_buyer', 'CF_corp_owner_buyer']]


# In[16]:

Counter_factual_table1=Counter_factual_table1.sort_values(['year','buyer_id'],ascending=[True,True])


# In[17]:

Counter_factual_table1=Counter_factual_table1[pd.notnull(Counter_factual_table1['buyer_id'])]
#drop "nan" to be on the safe-side. But I would like to argue that "nan" does not influence my results becuase they would not
#produce false scores.


# In[20]:

Counter_factual_table1['distance_bt']=Counter_factual_table1.apply(lambda x:                         vincenty((x['buyer_lat'],x['buyer_long']),(x['real_target__lat'],x['real_target__long'])).miles,axis=1)
Counter_factual_table1['distance_cfbcft']=Counter_factual_table1.apply(lambda x:                        vincenty((x['CF_buyer_lat'],x['CF_buyer_long']),(x['CF_target_lat'],x['CF_target_long'])).miles,axis=1)
Counter_factual_table1['distance_bcft']=Counter_factual_table1.apply(lambda x:                         vincenty((x['buyer_lat'],x['buyer_long']),(x['CF_target_lat'],x['CF_target_long'])).miles,axis=1)
Counter_factual_table1['distance_cfbt']=Counter_factual_table1.apply(lambda x:                         vincenty((x['CF_buyer_lat'],x['CF_buyer_long']),(x['real_target__lat'],x['real_target__long'])).miles,axis=1)


# In[32]:

def i_function(coefficients):
    '''
    f's are components of indication function.
    Indication function: I[f(b,t)+f(b',t')>f(b,t')+f(b't)], where I[.]=1 if the inequality is true and I[.]=0 if the inequality
    is false.
    f_bt=f(b,t)
    f_cfbcft=f(b',t')
    f_bcft=f(b,t')
    f_cfbt=f(b't)
    Usually I would put "data" as another input of this function, but it would cause error in differential_evolution function.
    '''
    
    f_bt=data['num_stations_buyer']*data['real_population_target']+    coefficients[0]*data['corp_owner_buyer']*data['real_population_target']+    coefficients[1]*data['distance_bt']
    
    f_cfbcft=data['CF_num_stations_buyer']*data['CF_population_target']+    coefficients[0]*data['CF_corp_owner_buyer']*data['CF_population_target']+    coefficients[1]*data['distance_cfbcft']
    
    f_bcft=data['num_stations_buyer']*data['CF_population_target']+    coefficients[0]*data['corp_owner_buyer']*data['CF_population_target']+    coefficients[1]*data['distance_bcft']
    
    f_cfbt=data['CF_num_stations_buyer']*data['real_population_target']+    coefficients[0]*data['CF_corp_owner_buyer']*data['real_population_target']+    coefficients[1]*data['distance_cfbt']
    
    a=f_bt+f_cfbcft
    b=f_bcft+f_cfbt
    
    i=(a>b)
    to_min_score=-i.sum()
    return  to_min_score


# In[42]:

bounds=[(-1,1),(-1,1)]
for i in [2007,2008]:
    data=Counter_factual_table1[Counter_factual_table1.year==i]
    result=differential_evolution(i_function,bounds)
    print(result.x)


# In[39]:

def i_function1(coefficients):
    '''
    f's are components of indication function.
    Indication function: I[f(b,t)+f(b',t')>f(b,t')+f(b't)], where I[.]=1 if the inequality is true and I[.]=0 if the inequality
    is false.
    f_bt=f(b,t)
    f_cfbcft=f(b',t')
    f_bcft=f(b,t')
    f_cfbt=f(b't)
    Usually I would put "data" as another input of this function, but it would cause error in differential_evolution function.
    '''
    
    f_bt=coefficients[0]*data['num_stations_buyer']*data['real_population_target']+    coefficients[1]*data['corp_owner_buyer']*data['real_population_target']+coefficients[2]*data['real_hhi_target']+    coefficients[3]*data['distance_bt']
    
    f_cfbcft=coefficients[0]*data['CF_num_stations_buyer']*data['CF_population_target']+coefficients[2]*data['CF_hhi_target']+    coefficients[1]*data['CF_corp_owner_buyer']*data['CF_population_target']+    coefficients[3]*data['distance_cfbcft']
    
    f_bcft=coefficients[0]*data['num_stations_buyer']*data['CF_population_target']+coefficients[2]*data['CF_hhi_target']+    coefficients[1]*data['corp_owner_buyer']*data['CF_population_target']+    coefficients[3]*data['distance_bcft']
    
    f_cfbt=coefficients[0]*data['CF_num_stations_buyer']*data['real_population_target']+    coefficients[1]*data['CF_corp_owner_buyer']*data['real_population_target']+coefficients[2]*data['real_hhi_target']+    coefficients[3]*data['distance_cfbt']
    
    a=f_bt+f_cfbcft
    b=f_bcft+f_cfbt
    
    i=(a>b)
    to_min_score=-i.sum()
    return  to_min_score


# In[43]:

bounds1=[(-1,1),(-1,1),(-1,1),(-1,1)]
for i in [2007,2008]:
    data=Counter_factual_table1[Counter_factual_table1.year==i]
    result=differential_evolution(i_function1,bounds1)
    print(result.x)


# In[ ]:




# In[ ]:



