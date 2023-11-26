""" chromadb """
import os
import sys
import time
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from dotenv import load_dotenv
import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
import tiktoken


MODEL = 'gpt-4-1106-preview'

# maksymalna wielkość odpowiedzi
OUTPUT_TOKENS = 1000

# ceny gpt-4-1106-preview w dolarach
INPUT_PRICE_GPT = 0.01
OUTPUT_PRICE_GPT = 0.03


@retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(6))
def get_answer_with_backoff(**kwargs):
    """ add exponential backoff to requests using the tenacity library """
    client = OpenAI()
    response = client.chat.completions.create(**kwargs)
    return response


def get_answer(prompt:str='', text:str='', model:str=MODEL) -> str:
    """ funkcja konstruuje prompt do modelu GPT dostępnego przez API i zwraca wynik """
    result = ''
    prompt_tokens = completion_tokens = 0

    try:
        completion = get_answer_with_backoff(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant, a specialist in history, genealogy, biographies of famous figures."},
                        {"role": "user", "content": f"{prompt}"}
                    ],
                    temperature=0.0,
                    top_p = 1.0,
                    seed=2,
                    max_tokens=OUTPUT_TOKENS)

        result = completion.choices[0].message.content
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens

    except Exception as request_error:
        print(request_error)
        sys.exit(1)

    return result, prompt_tokens, completion_tokens


def count_tokens(text:str, model:str = "gpt-4") -> int:
    """ funkcja zlicza tokeny """
    num_of_tokens = 0
    enc = tiktoken.encoding_for_model(model)
    num_of_tokens = len(enc.encode(text))

    return num_of_tokens


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    # Przykładowe pytania:
    # Czy kaprowie byli właścicielami okrętów?
    # Czy okręty kaprów były dobrze uzbrojone w broń palną, posiadały dużo dział?
    # Czy kaprowie napadali na statki inne niż szwedzkie czy duńskie?
    user_query = input('Pytanie: ')

    print('Przygotowanie danych...')

    # pomiar czasu wykonania
    start_time = time.time()

    # api key
    env_path = Path(".") / ".env"
    load_dotenv(dotenv_path=env_path)

    OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    client = chromadb.PersistentClient(path="../emb/")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=OPENAI_API_KEY,
                    model_name="text-embedding-ada-002"
                )

    collection = client.get_collection(name="bodniak_v3", embedding_function=openai_ef)

    result = collection.query(
        query_texts=[user_query],
        n_results=10,
        #where={"metadata_field": {"$eq": "search_string"}},
        #where_document={"$contains":"search_string"}
    )

    # dokumenty w odpowiedzi sortowane są wg odległości od pytania
    # do GPT przekazywane są 3 najbliższe

    documents = result['documents'][0][:3]
    metadatas = result['metadatas'][0][:3]

    documents = [x.strip() for x in documents]
    text = '\n'.join(documents)

    prompt = f"""Na podstawie podanego przed pytaniem kontekstu odpowiedz proszę na pytanie użytkownika.
    Jeżeli w podanym tekście nie ma informacji pozwalających na odpowiedź na pytanie, napisz:
    'Niestety nie posiadam informacji na ten temat'.
    Użyj języka formalnego, encyklopedycznego. W odpowiedzi nie powołuj się na tekst, nie używaj sformułowań typu:
    w tekście napisano itp. po prostu odpowiedz, nie krócej niż w 2-3 zdaniach.
    ###
    Tekst: {text}
    ###
    Pytanie {user_query}
"""
    print('Zapytanie do GPT...')
    llm_result, llm_prompt_tokens, llm_compl_tokens = get_answer(prompt, model=MODEL)

    # obliczenie kosztów
    price_gpt4 = (((llm_prompt_tokens/1000) * INPUT_PRICE_GPT) +
                    ((llm_compl_tokens/1000) * OUTPUT_PRICE_GPT))
    # wynik
    print(f'Odpowiedź:\n{llm_result}\n')

    print('Źródło:\n')

    for i in range(0,3):
        if len(documents[i]) > 150:
            max = 150
        else:
            max = len(documents[i])
        print(f'{i+1}. "{documents[i][:max]}..."')
        print(f"[{metadatas[i]['source']}, rozdział: {metadatas[i]['chapter']}, str. {metadatas[i]['page']}]")
        print()

    print(f'Liczba tokenów: ({llm_prompt_tokens}), koszt: {price_gpt4:.2f}\n')

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
