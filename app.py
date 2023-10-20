import streamlit as st
from pathlib import Path
import os
from doc_comparer import doc_compare
import time
# from dotenv import load_dotenv

# load environment variables
# load_dotenv()
# title of the streamlit app
st.title(f""":rainbow[Long Document Summarization with Amazon Bedrock]""")

# default container that houses the document upload field
with st.container():
    # header that is shown on the web UI
    st.header('Single File Upload')
    # the file upload field, the specific ui element that allows you to upload the file
    File1 = st.file_uploader('Upload File 1', type=["pdf"], key="doc_1")
    File2 = st.file_uploader('Upload File 2', type=["pdf"], key="doc_2")
    # when a file is uploaded it saves the file to the directory, creates a path, and invokes the
    # Chunk_and_Summarize Function
    if File1 and File2 is not None:
        # determine the path to temporarily save the PDF file that was uploaded
        save_folder = "/Users/rdoty/PycharmProjects/Amazon-Bedrock-Document-Comparison-POC"
        # create a posix path of save_folder and the file name
        save_path_1 = Path(save_folder, File1.name)
        save_path_2 = Path(save_folder, File2.name)
        # write the uploaded PDF to the save_folder you specified
        with open(save_path_1, mode='wb') as w:
            w.write(File1.getvalue())
        with open(save_path_2, mode='wb') as w:
            w.write(File2.getvalue())
        # once the save path exists...
        if save_path_1.exists() and save_path_2.exists():
            # write a success message saying the file has been successfully saved
            st.success(f'File {File1.name} is successfully saved!')
            st.success(f'File {File2.name} is successfully saved!')
            # creates a timer to time the length of the summarization task and starts the timer
            start = time.time()
            # running the summarization task, and outputting the results to the front end
            st.write(doc_compare(save_path_1, save_path_2))
            # st.write("success")
            # ending the timer
            end = time.time()
            # using the timer, we calculate the minutes and seconds it took to perform the summarization task
            seconds = int(((end - start) % 60))
            minutes = int((end - start) // 60)
            # string to highlight the amount of time taken to complete the summarization task
            total_time = f"""Time taken to generate a summary:
            Minutes: {minutes} Seconds: {round(seconds, 2)}"""
            # sidebar is created to display the total time taken to complete the summarization task
            with st.sidebar:
                st.header(total_time)
            # removing the PDF that was temporarily saved to perform the summarization task
            os.remove(save_path_1)
            os.remove(save_path_2)