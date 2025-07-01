# %%
import pandas as pd
pd.options.mode.chained_assignment = None
df = pd.read_json('./data/image_indexing_modified.json')
filtro = (df['files'].str[-3:] =='jpg') | (df['files'].str[-3:] =='JPG')
df = df[filtro]
columns_to_keep = ['files'] + [col for col in df.columns if col[-4:] == 'term' and col!='thema_term']
df = df[columns_to_keep]
df = df.dropna()


# %%
import json
with open('./data/translated_terms.json') as f:
  dictionary = json.load(f)

# %%
import unicodedata

def normalize_text(text):
    # Normalizza unicode (NFKC), rimuove escape, unifica apici
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\\', '')
    text = text.replace('“', "'").replace('”', "'").replace('"', "'")
    text = text.replace("’", "'").replace("`", "'")
    return text.strip()

def process_dataframe(df, dictionary):
    # Normalizza le chiavi del dizionario
    normalized_dict = {normalize_text(k): v for k, v in dictionary.items()}

    def map_and_concat(row):
        col_to_term = {
            'nature_place_term': 'Natural elements',
            'object_architecture_term': 'Architecture and objects elements:',
            'character_term': 'Character elements:',
            'subject_term': 'Subject elements:'
        }
        result = {}
        for col in ['nature_place_term', 'object_architecture_term', 'character_term', 'subject_term']:
            value = row[col]
            if isinstance(value, list):
                mapped = [normalized_dict[normalize_text(item)] for item in value]
                result[col_to_term[col]] = mapped

        return json.dumps(result, ensure_ascii=False)

    df['processed_terms'] = df.apply(map_and_concat, axis=1)
    return df

# %%
df = df.replace('Joab tuant Absalom', 'Joab tué Absalom')
df = df.replace('gahom/Pierpont m805.105r.jpg', 'gahom/Pierpont_m805.105r.jpg')
df = df.replace('gahom/Pierpont m805.109r.jpg', 'gahom/Pierpont_m805.109r.jpg')
df = process_dataframe(df, dictionary)

# %%
df.iloc[1,-1]

# %%
df.iloc[1]['files']

# %%
from PIL import Image

# %%
from openai import OpenAI
client = OpenAI(api_key='YOUR_API_KEY')

def get_response(input_text):  
  response = client.responses.create(
    prompt={
      "id": "pmpt_685821e905dc8197964d665e339611b702097471bbc3d5ef",
      "version": "1"
    },
    input=[
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": input_text,
          }
        ]
      }
    ],
    text={
      "format": {
        "type": "json_schema",
        "name": "description",
        "strict": True,
        "schema": {
          "type": "object",
          "properties": {
            "caption": {
              "type": "string",
              "description": "A descriptive caption providing details about the scene depicted."
            }
          },
          "required": [
            "caption"
          ],
          "additionalProperties": False
        }
      }
    },
    reasoning={},
    max_output_tokens=2048,
    store=True
  )

  return response

# %%
from tqdm import tqdm
import traceback

def apply_openai_caption(df):
    try:
        # Inizializza la colonna 'openai_caption' con valori None
        df['openai_caption'] = None

        # Itera su ogni riga del dataframe
        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                #response = get_response(row['processed_terms'])
                #caption = response.output_text  # Estrai la caption dalla risposta
                caption = "This is a placeholder caption for the image."  # Placeholder per il test
                df.at[index, 'openai_caption_raw'] = caption
            except Exception as e:
                # Se c'è un errore per una riga specifica, stampa un messaggio
                print(f"Errore nella riga {index}: {e}")

        # Salva il dataframe modificato in un file JSON
        df.to_json('../data/image_indexing_modified_openai.json', orient='records', force_ascii=False)
        print("Fatto!")
        return df

    except Exception as e:
        # Salva il dataframe modificato fino a quel momento in caso di errore
        print(f"Errore durante l'elaborazione: {e}")
        print(traceback.format_exc())
        df.to_json('../data/image_indexing_modified_openai.json', orient='records', force_ascii=False)
        raise e  # Rilancia l'errore per ulteriori debug

# Applicazione della funzione al dataframe

#df = apply_openai_caption(df)

# %%
_ = apply_openai_caption(df)


