import gradio as gr
from transformers import pipeline

# using sentiment analysis first since there is a lot of documentation on it
task = "sentiment-analysis"

# this is the default model for sentiment-analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# initializing our nlp model
classifier = pipeline(task, model=model_name)


# creating this helper function, which we'll use below with gradio
def predict_sentiment(input):
    """
    A function that takes a user input in the form of text and provides
    a sentiment prediction using the nlp model above.

    Returns
    -------
    List of tuples where the first element is the user input
    and the second element is the prediction.
    """

    # generating the response from the model, will return a 1 element list of a dictionary
    response = classifier(input)

    # NOTE: the next few lines are based off the returned results from this model
    # if the model changes, the outputs will likely change, and therefore this code will break.
    label = response[0]["label"]

    score = response[0]["score"]
    score = round(score, 4)

    modified_response = f"The predicted label for the text is {label} with a confidence score of {score}"

    # now that we have input and reponse, lets apppend them as a tuple to our message history
    # if we dont do this, then we wont see message history in the UI
    global message_history
    message_history.append((input, modified_response))

    return message_history


message_history = []

# this is where the gradio code starts
with gr.Blocks() as demo:
    title = """ # This chatbot is designed to provide sentiment classifications to text you provide.
    To get started, type your text below and hit 'enter' to submit your response.
    """
    gr.Markdown(title)

    chatbot = gr.Chatbot()

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Type your message here").style(
            container=False
        )
        txt.submit(predict_sentiment, txt, chatbot)
        txt.submit(
            lambda: "", None, txt
        )  # this is needed so the textbox is cleared after user submits it

demo.launch()
