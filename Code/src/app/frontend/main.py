import requests
import streamlit as st
import pandas as pd


# TODO: zamijeni s localhost!!!
ENDPOINT_URL = "http://172.24.16.20:8080/api/predict"

# defines an h1 header
st.title("Audio classification app")

class_mappings = {"cel": 0, "cla": 1, "flu": 2, "gac": 3, "gel": 4, "org": 5, "pia": 6, "sax": 7, "tru": 8, "vio": 9,
                  "voi": 10}

class_names_mappings = {"cel": "Cello", "cla": "Clarinet", "flu": "Flute", "gac": "Guitar - acoustic", "gel": "Guitar - electric",
                        "org": "Organ", "pia": "Piano", "sax": "Saxophone", "tru": "Trumpet", "vio": "Violin", "voi": "Voice"}

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            table { border-collapse:collapse }
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

def send_file_to_endpoint(audio_file):
    """Sends the audio file to the endpoint and returns the response."""
    files = {"audio_file": audio_file.getvalue()}
    response = requests.post(ENDPOINT_URL, files=files)
    return response.json()
    # with open(audio_file, "rb") as f:
    #     files = {"audio_file": f}
    #     response = requests.post(ENDPOINT_URL, files=files)
    #     return response.json()


# Add a file upload form
audio_file = st.file_uploader("Upload an audio file")

if (audio_file is not None):
    audio_bytes = audio_file.getvalue()
    st.audio(audio_bytes, format='audio/*')

# Add a submit button
if st.button("Submit"):
    # Make sure the user has uploaded a file
    if audio_file is not None:
        # Send the file to the endpoint


        response = send_file_to_endpoint(audio_file)
        # Convert the response to a dictionary
        response_dict = response
        # Create a list of the labels and values
        labels = list(response_dict.keys())
        values = list(response_dict.values())
        # # Create a bar chart
        # fig, ax = plt.subplots()
        # ax.bar(labels, values)
        # st.pyplot(fig)
        table_data = []
        for key, value in response_dict.items():
            if value == 1:
                table_data.append({"Instrument": class_names_mappings[key], "Prediction": "✅"})
            else:
                table_data.append({"Instrument": class_names_mappings[key], "Prediction": "❌"})

        df = pd.DataFrame(table_data)
        st.table(df)
    else:
        st.write("Please upload an audio file")
