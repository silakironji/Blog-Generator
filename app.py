import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


#function to get response from LLAma 2 model

def getLLAmaresponse(input_text,no_words,blog_style):
    #LLAma2 model
    llm=CTransformers(model='G:\LLM\BlogGeneration\model\llama-2-7b-chat.ggmlv3.q8_0.bin' ,
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})

    #Prompt Template

    template=""""
        Write a blog for {blog_style} job profile fo a topic {input_text}
        with {no_words} words
            """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text", 'no_words'],
                          template=template)

## Generate the rensponse from the llma 2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text=st.text_input("Enter the Blog Topic")

#creating two more columns for 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for' ,
                            ('Researchers', 'Data scientist', 'Common people'), index=0)
        
submit=st.button("Generate")

# Final Response
if submit:
    st.write(getLLAmaresponse(input_text,no_words,blog_style))