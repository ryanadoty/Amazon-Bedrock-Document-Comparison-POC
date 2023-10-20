import boto3
import json
import botocore.config
from pypdf import PdfReader
import yaml
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.example_selector.semantic_similarity import (
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma

# loading environment variables
# load_dotenv()
# configure Bedrock client
boto3.setup_default_session(profile_name="bedrock")
config = botocore.config.Config(connect_timeout=120, read_timeout=120)
bedrock = boto3.client('bedrock-runtime', 'us-east-1', endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                       config=config)


def llm_compare(prompt_data) -> str:
    """
    This function creates the summary of each individual chunk as well as the final summary.
    :param prompt_data: This is the prompt along with the respective chunk of text, at the end it contains all summary chunks combined.
    :return: A summary of the respective chunk of data passed in or the final summary that is a summary of all summary chunks.
    """
    # setting the key parameters to invoke Amazon Bedrock
    body = json.dumps({"prompt": prompt_data,
                       "max_tokens_to_sample": 8191,
                       "temperature": 0,
                       "top_k": 250,
                       "top_p": 0.5,
                       "stop_sequences": []
                       })
    # the specific Amazon Bedrock model you are using
    modelId = 'anthropic.claude-v2'
    # type of data that should be expected upon invocation
    accept = 'application/json'
    contentType = 'application/json'
    # the invocation of bedrock, with all of the parameters you have configured
    response = bedrock.invoke_model(body=body,
                                    modelId=modelId,
                                    accept=accept,
                                    contentType=contentType)
    # gathering the response from bedrock, and parsing to get specifically the answer
    response_body = json.loads(response.get('body').read())
    answer = response_body.get('completion')
    # returning the final summary for that chunk of text
    return answer


def load_samples():
    """
    Load the generic examples for few-shot prompting.
    :return: The generic samples from the generic_samples.yaml file
    """
    # initializing the generic_samples variable, where we will store our samples once they are read in
    generic_samples = None
    # opening and reading the sample prompts file
    with open("sample_prompts/sample_prompt_data.yaml", "r") as stream:
        # storing the sample files in the generic samples variable we initialized
        generic_samples = yaml.safe_load(stream)
    # returning the string containing all the sample prompts
    return generic_samples


def prompt_finder(question):
    """
    This function performs a semantic search based on the users question against all the sample prompts stored in the
    sample_prompts/generic_samples.yaml file. It finds the three most relevant prompts and formats them into a single prompt
    along with the users question.
    :param question: This is the question that is passed in through the streamlit frontend from the user.
    :return: This function returns a final prompt that contains three semantically similar prompts, the chat history if
    there is any and the users question all formatted in a single prompt ready to be passed into Amazon Bedrock.
    """
    # loading the sample prompts from sample_prompts/generic_samples.yaml
    examples = load_samples()
    # instantiating the hugging face embeddings model to be used to produce embeddings of user queries and prompts
    local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # The example selector loads the examples, creates the embeddings, stores them in Chroma (vector store) and a
    # semantic search is performed to see the similarity between the question and prompts, it returns the 3 most similar
    # prompts as defined by k
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # This is the list of examples available to select from.
        examples,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        local_embeddings,
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        Chroma,
        # This is the number of examples to produce.
        # TODO: Can change this number to determine how many prompts you want to retrieve
        k=3
    )
    # This is formatting the prompts that are retrieved from the sample_prompts/generic_samples.yaml file
    example_prompt = PromptTemplate(input_variables=["prompt", "assistant"],
                                    template="\n\nHuman: {prompt} \n\nAssistant: "
                                             "{assistant}")
    # This is orchestrating the example selector (finding similar prompts to the question), example_prompt (formatting
    # the retrieved prompts, and formatting the chat history and the user input
    prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        suffix="{input}",
        input_variables=["input"]
    )
    # This is calling the prompt method and passing in the users question to create the final multi-shot prompt,
    # with the semantically similar prompts, and chat history
    question_with_prompt = prompt.format(input=question)
    print(question_with_prompt)
    # we return the finalized prompt, ready to be passed into Amazon Bedrock to generate a response
    return llm_compare(question_with_prompt)


def doc_compare(uploaded_file_1, uploaded_file_2):
    # using PyPDF PdfReader to read in the PDF file as text
    document_1 = PdfReader(uploaded_file_1)
    document_2 = PdfReader(uploaded_file_2)
    # creating an empty string for us to append all the text extracted from the PDF
    doc_1_text = ""
    doc_2_text = ""
    # a simple for loop to iterate through all pages of the PDF we uploaded
    for (page_1, page_2) in zip(document_1.pages, document_2.pages):
        # as we loop through each page, we extract the text from the page and append it to the "text" string
        doc_1_text += page_1.extract_text() + "\n"
        doc_2_text += page_2.extract_text() + "\n"

    prompt = f"""\n\nHuman: Please compare Document A and Document B sentence by sentence, with the granularity of a single word change. 
        Listing every change that took place. Double check your work.
        
        Document A: {doc_1_text}
        
        Document B: {doc_2_text}
        
        \n\nAssistant:"""

    changes = prompt_finder(prompt)
    print("-------------------------------------------------------------------------------------------------------")

    return changes
