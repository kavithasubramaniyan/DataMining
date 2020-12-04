import math

import pandas as pd
import os

from ipywidgets import VBox, HBox, Label, interactive
from matplotlib import pyplot as plt
#Which youtube channel has the most trending videos
os.chdir('C:/Users/kavis/OneDrive/Desktop/Data Mining/Project')
df=pd.read_csv('CAvideos.csv')
#group by channel_title and video_idgm fuy54ab
y = df.groupby(['channel_title','video_id']).size().reset_index(name='counts')
#Sort the values based on count
y.sort_values(by=['counts'], inplace=True, ascending=False)
X=y['channel_title'].head(10)
Y=y['counts'].head(10)
#Plotting the graph
plt.bar(X, Y, color ='green', width = 0.4)
plt.xlabel("channel_title")
plt.ylabel("No.of videos")
plt.title('Channel and count of trending videos',color='b')
plt.xticks(rotation=90)
plt.savefig("Top 10 Channel-Trending videos")
plt.show()

#Identify at which day of the week more videos are published
import calendar
date=pd.DataFrame()
df['date']=df.publish_time.str.slice(0, 10)
#Extracting day information
date['day']=pd.to_datetime(df['date']).dt.day
#Extracting day name information
df['day_name']=pd.to_datetime(df['date']).dt.day_name()
import matplotlib.pyplot as plt
#Taking the day_name and grouping it by and taking counts
y = df.groupby(['day_name']).size().reset_index(name='counts')
slices = y['counts']
activities = y['day_name']
colors  = ("red", "green", "orange", "cyan", "brown","grey","blue")
#Plotting pie chart
plt.pie(slices,labels=activities,colors=colors,startangle=90,shadow= True,autopct='%1.1f%%')
plt.title('Days and percentage of videos published ')
plt.show()


#Identify the most watched videos category wise
from matplotlib import pyplot as plt
from matplotlib import style
file=pd.read_excel("Trending_CA_videos_List.xlsx")
y=file.sort_values('views',ascending = True).groupby('category_name')['views','channel_title','category_name'].tail(1)
style.use('ggplot')
#Defining X and Y variable
X = y['category_name']
Y = y['views']
low = min(Y)#Minimum value of views
high = max(Y)#Maximum value of views
plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+0.5*(high-low))])
plt.bar(X, Y, color ='red', width = 0.4)
plt.xlabel("category")
plt.ylabel("No.of views")
plt.title('Most viewed -category',color='b')
plt.xticks(rotation=90)
plt.savefig("Category-most liked")
plt.show()

#Is there a relationship between views,likes,dislikes
import seaborn as sns
plt.figure(figsize=(8, 12))
sns.heatmap(df[['views','likes','dislikes']].corr(), annot = True, fmt='.2g',cmap= 'coolwarm')
plt.show()

#Check if the video had error removed issues
import matplotlib.pyplot as plt
#Finding the counts
kk = df['video_error_or_removed'].value_counts()
slices = kk
activities = ['Yes','No']
colors  = ("grey","blue")
#Plotting the graph
plt.pie(slices,labels=activities,colors=colors,startangle=90,shadow= True,autopct='%1.1f%%')
plt.title('Do the video has issues(error or removed)')
plt.show()

#Wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
cloud = WordCloud().generate(' '.join(file['description']))
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()