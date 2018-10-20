import pandas as pd
import numpy as np

"""playground"""
a = pd.DataFrame(columns =["test1","test2"])
b = pd.DataFrame([[5,3]],columns = ["test1","test2"])
print(a)
print(b)
c = a.append(b,ignore_index=True,sort=False)
c = c.append(b,ignore_index=True,sort=False)
print(c)
print(c.columns.values)
print(c.columns.get_loc("test1"))


"""
Some useful dataframe commands:

df.columns.values # returns a list of all column names in df
df.columns.get_loc("colname")) # returns the column index of a col in df

df.drop(["Gender","Group"],axis=1) # NOT inplace!
df.drop([1,2],axis=0) # drops rows with idx 1 and 2. geht nur bei axis =0

df.loc[:,df.notnull().all()] # selects only cols without NaNs

#reshaping
df_reshaped = df.pivot(index="Students",columns="Gender",values="Age") # change index to students, create 2 gender cols
df.stack() # creates a multiindex with the colnames. NOT inplace. converts df to Series
df.unstack() # unstack level from index onto col axis
pd.melt(df,id_vars = ["Students"],value_vars=["Gender","Group"],value_name="NewCol")# gather columns into rows --> wide to long


# sorting
df.sort_index(axis=1) # alphabetic column sorting
df.sort_values(by = "Group").reset_index() # not inplace

# functions on dfs
f = lambda x: x*2
df.apply(f) # repeats value in every cell (doubles for numeric). NOT inplace
df.applymap(f) # same

# Boolean Indexing
df[df["Age"] >20] # everyone aged > 20
df[~(df["Age"]>20)] # everyone except those aged >20
df[~(df["Gender"]=="M")] # only Females
df[(df["Age"]>20) & (df["Age"]<34)] 

"""