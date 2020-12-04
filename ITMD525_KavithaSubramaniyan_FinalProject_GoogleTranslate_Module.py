from json import JSONDecodeError

import pandas as pd
import numpy as np
import os
import json
import openpyxl
import temp as temp
from googletrans import Translator
#from googletrans-temp import Translator
from translator import translator

s=[]
os.chdir('C:/Users/kavis/OneDrive/Desktop/Data Mining/Project')
#Reading the file
df=pd.read_csv('CAvideos.csv')
df.loc[df['tags']=='[none]','tags']=' '
#Removing duplicates
df=df.drop_duplicates('video_id',keep='first')
#Combining title,tags,description
df['description']=df['title'].astype(str)+' '+df['tags'].astype(str)+' '+df['description'].astype(str)

array=[]
array1=[]
#Taking the category description by doing inner join with json file
with open('CA_category_id.json','r') as myfile:
    data=myfile.read()
    #parse file
    obj=json.loads(data)
    items_info=obj.get("items")
    for i in range(len(items_info)):
        id=items_info[i]['id']
        array.append(id)
        a=items_info[i]['snippet']
        b=a.get("title")
        array1.append(b)
myfile.close()
df2=pd.DataFrame()
df2["category_id"]=array
df2["category_name"]=array1
df2['category_id']=pd.to_numeric(df2['category_id'])

#Some columns will be removed which does not have category in json file
merged_df=pd.merge(df,df2,on='category_id',how='inner')

merged_df.to_excel("Trending_CA_videos_List.xlsx",index=False,sheet_name='CA')

#Translating the description
# Importa pandas library for inmporting CSV
import pandas as pd
from google.api_core.protobuf_helpers import get_messages
# Imports the Google Cloud client library
from google.cloud import translate
from google.cloud import translate_v2
from google.protobuf import timestamp_pb2
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/kavis/Downloads/My First Project-f6a05cec63b4.json"
# Instantiates a client
translate_client = translate_v2.Client()
# Translating the text to specified target language
def translate(word):
    # Target language
    target_language = 'en'  # Add here the target language that you want to translate to
    # Translates some text into Russian
    translation = translate_client.translate(word,
                                             target_language=target_language)
    # print([translation['translatedText'],translation['detectedSourceLanguage'])
    return translation['translatedText'], translation['detectedSourceLanguage']
# Import data from CSV
def importCSV():
    # data = pd.read_csv('PATH/TO/THE/CSV/FILE/FILE_NAME.csv')
    file = pd.read_excel('Trending_CA_videos_List.xlsx')
    data = file['description']
    data = pd.DataFrame(data)
    countRows = (len(data))
    # Create a dictionary with translated words
    translatedCSV = {"description": []}
    # Translated word one by one from the CSV file and save them to the dictionary
    t_text_lang_list = []
    t_text_list = []
    t_lang_list = []
    for index, row in data.iterrows():
        t_text = ''
        t_lang = ''
        print(row)
        t_text, t_lang = translate(row["description"])
        t_text_list.append(t_text)
        t_lang_list.append(t_lang)
        t_text_lang_list.append([t_text, t_lang])
    # Create a Dataframe from Dictionary, # Save the DataFrame to a CSV file,# df = pd.DataFrame(data=translatedCSV)
    file['Translated_Description'] = t_text_list
    file['source_language'] = t_lang_list
    file.to_excel("Trending_Videos_CA_Final_Translated_List.xlsx",index=False)
    print("Successfully translated all the records in description column")
# Call the function
importCSV()

