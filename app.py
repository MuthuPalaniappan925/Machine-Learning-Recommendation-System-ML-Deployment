import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import streamlit.components.v1 as stc
import difflib
import matplotlib.pyplot as plt



def load_data(data):
    data = pd.read_csv(data)
    return data

def data_preprocessing(data):
    #data.drop(columns='\nCo-ordinating Institute',inplace=True)
    data = CS_data = data[(data['Discipline'] == 'Computer Science and Engineering') | (data['Discipline'] == 'Computer Science & Engineering') |(data['Discipline'] == 'Electrical and Electronics Engineering,\n Computer Science and Engineering,\n VLSI Specialization')]
    data['credits'] = data['credits'].astype('str')
    data['Applicable NPTEL Domain'].fillna(' ',inplace=True)
    feature1=data['Course Name']
    li = [i for i in range(len(data))]
    data['index'] = li
    return data,feature1

def text_to_cosine_mat(data,feature1):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(feature1)
    cosine_sim_mat1= cosine_similarity(cv_mat)
    course_indices = pd.Series(data.index,index=data['Course Name']).drop_duplicates()
    return cosine_sim_mat1,course_indices

def get_input_manipulation():
    cname = input("Enter Course name ")
    inst = input("Institue Name ")
    dur = input("Enter Course Duartion")
    up = input("UG/PG")
    ce = input("Core/Elective")
    dom = input("Applicable NPTEL Domain")
    cr = input("Enter Credits")
    dfl = input("Enter Difficulty level")

    title = inst+" "+dur+" "+up+" "+ce+" "+dom+" "+cr+" "+dfl
    return title,cname,inst,dur,up,ce,dom,cr,dfl

def countplot(data):
    fig = plt.figure(figsize=(10,4))
    sns.countplot(x=data['Duration'],data=data)
    st.pyplot(fig)
    
def pieplot(data):
    fig = plt.figure(figsize=(10,4))
    #plt.pie(x=data['Difficulty Level'].value_counts())
    data['Difficulty Level'].value_counts().plot(kind='pie')
    st.pyplot(fig)
    
def histplot(data):
    fig = plt.figure(figsize=(10,4))
    sns.histplot(x=data['credits'],hue=data['Difficulty Level'],data=data)
    st.pyplot(fig)

#a-cname
#b - feature1
def get_course_recommendation(a,b,cosine_sim_mat1,course_indices,number,data):
    close_match = difflib.get_close_matches(a,b)[0]
    j=0
    for i in b:
        if i==close_match:
            break
        else:
            j+=1
    similarity_score = list(enumerate(cosine_sim_mat1[j]))
    sorted_sim_Score = sorted(similarity_score,key=lambda x : x[1],reverse=True)
    selected_course_indices = [i[0] for i in sorted_sim_Score[0:number]]
    result_data = data.iloc[selected_course_indices]
    return result_data

@st.cache
def search_term_if_not_found(term,df):
	result_df = df[df['Course Name'].str.contains(term)]
	return result_df


def main():
    
    st.title("NPTEL - CS - Course Recommendation System ML")
    menu = ["Home","Description of the Dataset","Exploratory Data Analysis","Recommendation-System","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    data = load_data("nptel_electives.csv")
    
    if choice == "Home":
        st.subheader("Home")
        feat = st.sidebar.number_input("Number of records",4,30,7)
        st.text("Data Frame")
        st.dataframe(data[(data['Discipline'] == 'Computer Science and Engineering') | (data['Discipline'] == 'Computer Science & Engineering') |(data['Discipline'] == 'Electrical and Electronics Engineering,\n Computer Science and Engineering,\n VLSI Specialization')].head(feat))
        
    elif choice == "Description of the Dataset":
        st.subheader("Description of the Dataset")
        st.text("Course Id - Unique ID of the particular Course")
        st.text("Discipline - Discipline of that particular Course")
        st.text("Course Name - Name of the Course")
        st.text("SME Name - Name of the Faculty offering Course")
        st.text("Institute - Name of the institue offering Course")
        st.text("Duration - Time period to complete particular Course")
        st.text("Credits - Number of credits allocated to Course")
        st.text("Difficulty Level - Level of difficulty")
        
    elif choice == "Exploratory Data Analysis":
        st.subheader("Countplot of Duration")
        new_df = data[(data['Discipline'] == 'Computer Science and Engineering') | (data['Discipline'] == 'Computer Science & Engineering') |(data['Discipline'] == 'Electrical and Electronics Engineering,\n Computer Science and Engineering,\n VLSI Specialization')]
        countplot(new_df)
        st.subheader("Pie Plot of Difficulty level")
        pieplot(new_df)
        st.subheader("Hist Plot")
        histplot(new_df)
        #fig = plt.figure(figsize=(10,4))
        #sns.countplot(x=new_df['credits'],data=new_df)
        #st.pyplot(fig)
        
    elif choice == "Recommendation-System":
        st.subheader("Course Recommendation for you")
        number_of_rec = st.sidebar.number_input("Number of recommended courses",4,15,5)
        cname = st.text_input("Course")
        inst = st.text_input("Institute")
        cr = st.radio('Credits',['1','2','3'])
        
        if st.button("Show Recommendations"):
            try:
                data,feature1 = data_preprocessing(data)
                cosine_sim_mat1,course_indices = text_to_cosine_mat(data,feature1)
                result_df = get_course_recommendation(cname, feature1, cosine_sim_mat1, course_indices,number_of_rec,data)
                st.dataframe(result_df)
                try:
                    
                    result_df_2 = result_df[result_df['credits']==cr]
                    st.text("Recommendations based on number of credits offered")
                    st.dataframe(result_df_2)
                except:
                    st.warning("No Course related to opted credits")
                    
                try:
                    result_df_2 = result_df[result_df['Institute']==inst]
                    st.text("Recommendations based on Institution offered")
                    st.dataframe(result_df_2)
                except:
                    st.warning("No Course offered by opted institute")
                
            except:
                st.warning("Course")
                result_df = search_term_if_not_found(cname,data)
                st.dataframe(result_df.head(number_of_rec))
            
                
        else:    
            print("")
            
    elif choice == "About":
        st.text("")
    
if __name__ == '__main__':
	main()