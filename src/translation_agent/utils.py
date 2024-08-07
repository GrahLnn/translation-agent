import json
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import List

import requests
import tiktoken
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


load_dotenv()  # read local .env file
MODEL = os.getenv("MODEL_NAME")
VERTEX_URL = os.getenv("VERTEX_API_URL")
IAM_FILE = os.getenv("IAM_PATH")
MAX_TOKENS_PER_CHUNK = (
    1000  # if text is more than this many tokens, we'll break it up into
)
OPENAI_URL = os.getenv("OPENAI_API_URL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
USE_VERTEX = os.getenv("USE_VERTEX") == "True"
GEMINI_API = os.getenv("GEMINI_API")
API_LIMIT = int(os.getenv("API_DALAY_LIMIT"))
GEMINI_KEYS = list(map(str.strip, os.getenv("GEMINI_API_KEY").split(",")))
API_INTERVEL_TIME = int(os.getenv("API_INTERVEL_TIME"))
SAVE_CACHE = os.getenv("SAVE_CACHE") == "True"
filename = ""
textname = ""


def load_credentials(filename):
    with open(filename) as file:
        credentials_info = json.load(file)
    return credentials_info


def store_token(auth_token: str, auth_time: datetime):
    with open("asset/google_auth.json", "w") as file:
        data = {
            "auth_token": auth_token,
            "auth_time": auth_time.astimezone(timezone.utc).isoformat(),
        }
        json.dump(data, file)


def read_stored_token():
    try:
        with open("asset/google_auth.json") as file:
            data: dict = json.load(file)
            auth_token = data.get("auth_token")
            auth_time = data.get("auth_time")
            if auth_token and auth_time:
                auth_time = datetime.fromisoformat(auth_time).astimezone(
                    timezone.utc
                )
                return auth_token, auth_time
    except FileNotFoundError:
        pass
    return None, None


def update_limit(key: str, limit: int):
    cur_day_utc = datetime.now(timezone.utc).date()
    data: dict = {}

    if os.path.exists("asset/limit.json"):
        with open("asset/limit.json") as file:
            data = json.load(file)
            day_record = data.get("cur_day")
    else:
        day_record = None

    is_same_day = day_record == cur_day_utc.isoformat()

    if not is_same_day:
        for k in data.get("limits", {}):
            data["limits"][k] = API_LIMIT
        data["cur_day"] = cur_day_utc.isoformat()

    if "limits" not in data:
        data["limits"] = {}
    data["limits"][key] = limit

    with open("asset/limit.json", "w") as file:
        json.dump(data, file, indent=4)


def choose_key():
    if os.path.exists("asset/limit.json"):
        with open("asset/limit.json") as file:
            data = json.load(file)
            limits: dict = data["limits"]
    else:
        limits = {}

    random.shuffle(GEMINI_KEYS)

    for key in GEMINI_KEYS:
        if limits.get(key, None) is None:
            update_limit(key, API_LIMIT)
            limit = API_LIMIT
            return key, limit
        elif limits[key] > 0:
            limit = limits[key]
            return key, limit

    sleep_to_next_refresh()
    return choose_key()


def sleep_to_next_refresh():
    now = datetime.now(timezone.utc)
    tomorrow = now + timedelta(days=1)
    tomorrow = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    sleep_time = (tomorrow - now).total_seconds()
    for i in reversed(range(int(sleep_time))):
        hours, remainder = divmod(i, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        print(
            f"Sleep to {tomorrow} refresh at {time_str}.", flush=True, end="\r"
        )
        time.sleep(1)
    print("Wake up! Keep working!                    ", flush=True)


def save_cache(dir, text, name):
    os.makedirs(dir, exist_ok=True)
    output = dir + f"{name}.md"

    with open(output, "w", encoding="utf-8") as output_file:
        output_file.write(text)


def get_access_token(credentials_info):
    stored_token, stored_time = read_stored_token()
    if (
        stored_token
        and stored_time
        and (datetime.now(timezone.utc) < stored_time + timedelta(minutes=10))
    ):
        return stored_token

    credentials = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    request = Request()
    credentials.refresh(request)
    new_token = credentials.token
    new_time = datetime.now(timezone.utc)
    store_token(new_token, new_time)
    for _ in range(240):
        print(f"Token refreshed at {new_time}.", flush=True, end="\r")
        time.sleep(1)
    return new_token


def call_api(url, access_token, data):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    max_retries = 10
    wait_time = 60

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

            try:
                res = response.json()
                return res
            except Exception as e:
                print(f"Error decoding JSON response: {e}\n{data}\n{res}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        if attempt < max_retries - 1:
            for remaining in range(wait_time, 0, -1):
                print(
                    f"Retrying in {remaining} seconds...", flush=True, end="\r"
                )
                time.sleep(1)
            print("Retrying...                   ", flush=True)
        else:
            print("Exceeded maximum retries.")
            sys.exit(1)


def call_api_without_authhead(url, data):
    headers = {
        "Content-Type": "application/json",
    }
    max_retries = 10
    wait_time = 60

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

            try:
                res = response.json()
                return res
            except Exception as e:
                print(f"Error decoding JSON response: {e}\n{data}\n{res}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        if attempt < max_retries - 1:
            for remaining in range(wait_time, 0, -1):
                print(
                    f"Retrying in {remaining} seconds...", flush=True, end="\r"
                )
                time.sleep(1)
            print("Retrying...                   ", flush=True)
        else:
            print("Exceeded maximum retries.")
            sys.exit(1)


def gemini_completion(prompt, system_message, temperature, model, key, limit):
    os.makedirs("asset", exist_ok=True)

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": system_message}],
        },
        "generationConfig": {"temperature": temperature},
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ],
    }
    if USE_VERTEX:
        api_url = VERTEX_URL + f"{model}:generateContent"
        credentials_info = load_credentials(IAM_FILE)
        access_token = get_access_token(credentials_info)
        res = call_api(api_url, access_token, payload)
    else:
        api_url = GEMINI_API + f"{model}:generateContent" + "?key=" + key
        res = call_api_without_authhead(api_url, payload)
        update_limit(key, limit - 1)
    try:
        answer = res["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        print(
            "unexpect output, most are judged to be in violation, you can remove related text and retry, related prompt bellow: \n------prompt------\n",
            prompt,
            "\n------result------\n",
            res,
        )
        sys.exit(1)
    return answer


def openai_completion(prompt, system_message, temperature, model):
    data = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    res = call_api(OPENAI_URL, OPENAI_KEY, data)
    answer = res["choices"][0]["message"]["content"]

    return answer


def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = MODEL,
    temperature: float = 0.3,
) -> str:
    answer = ""
    if "gemini" in model:
        key, limit = choose_key()
        answer = gemini_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            model=model,
            key=key,
            limit=limit,
        )
    else:
        answer = openai_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            model=model,
        )
    if API_INTERVEL_TIME > 0:
        time.sleep(API_INTERVEL_TIME)
    return answer


def one_chunk_initial_translation(
    source_lang: str, target_lang: str, source_text: str
) -> str:
    """
    Translate the entire text as one chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.

    Returns:
        str: The translated text.
    """

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""

    prompt = translation_prompt.format(source_text=source_text)

    translation = get_completion(prompt, system_message=system_message)

    return translation


def one_chunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    country: str = "",
) -> str:
    """
    Use an LLM to reflect on the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        country (str): Country specified for target language.

    Returns:
        str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.
    """

    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text, and the content needs to be consistent.),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text, and the content needs to be consistent.),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    prompt = reflection_prompt.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation_1=translation_1,
    )
    reflection = get_completion(prompt, system_message=system_message)
    return reflection


def one_chunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    reflection: str,
) -> str:
    """
    Use the reflection to improve the translation, treating the entire text as one chunk.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The original text in the source language.
        translation_1 (str): The initial translation of the source text.
        reflection (str): Expert suggestions and constructive criticism for improving the translation.

    Returns:
        str: The improved translation based on the expert suggestions.
    """

    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

    prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text, and the content needs to be consistent.),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""

    translation_2 = get_completion(prompt, system_message)

    return translation_2


def one_chunk_translate_text(
    source_lang: str, target_lang: str, source_text: str, country: str = ""
) -> str:
    """
    Translate a single chunk of text from the source language to the target language.

    This function performs a two-step translation process:
    1. Get an initial translation of the source text.
    2. Reflect on the initial translation and generate an improved translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for the translation.
        source_text (str): The text to be translated.
        country (str): Country specified for target language.
    Returns:
        str: The improved translation of the source text.
    """
    translation_1 = one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )

    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )
    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection
    )

    return translation_2


def num_tokens_in_string(
    input_str: str, encoding_name: str = "cl100k_base"
) -> int:
    """
    Calculate the number of tokens in a given string using a specified encoding.

    Args:
        str (str): The input string to be tokenized.
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base",
            which is the most commonly used encoder (used by GPT-4).

    Returns:
        int: The number of tokens in the input string.

    Example:
        >>> text = "Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


def multichunk_initial_translation(
    source_lang: str, target_lang: str, source_text_chunks: List[str]
) -> List[str]:
    """
    Translate a text in multiple chunks from the source language to the target language.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): A list of text chunks to be translated.

    Returns:
        List[str]: A list of translated text chunks.
    """

    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    translation_prompt = """Your task is to provide a professional translation from {source_lang} to {target_lang} of PART of a text.

To reiterate, you should translate only this part and ALL from this of the text, shown here between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

Guidelines for translate:
1. Translate ALL content between <TRANSLATE_THIS> and </TRANSLATE_THIS> part.
2. Maintain paragraph structure and line breaks.
3. Preserve all markdown, image links, LaTeX code, and titles.
4. Do not remove any single line from the <TRANSLATE_THIS> and </TRANSLATE_THIS> part.

Output only the translation of the portion you are asked to translate, and nothing else.
"""
    done_idx = -1
    translation_chunks = []

    cache_file = "cache/init_translation.json"
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            cache_data: dict = json.load(f)
            done_idx = cache_data.get("done_idx", 0)
            translation_chunks = cache_data.get("translation_chunks", [])

    if done_idx == len(source_text_chunks) - 1:
        return translation_chunks

    for i in tqdm(
        range(done_idx + 1, len(source_text_chunks)), desc="1:init translating"
    ):
        tagged_text = (
            ("".join(source_text_chunks[max(i - 2, 0) : i]) if i > 0 else "")
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + (
                "".join(
                    source_text_chunks[
                        i + 1 : min(i + 2, len(source_text_chunks))
                    ]
                )
                if i < len(source_text_chunks) - 1
                else ""
            )
        )
        prompt = translation_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            chunk_to_translate=source_text_chunks[i],
        )

        translation = get_completion(prompt, system_message=system_message)
        translation = (
            translation.replace("<TRANSLATION>", "")
            .replace("</TRANSLATION>", "")
            .replace("</TRANSLATE_THIS>", "")
            .replace("<TRANSLATE_THIS>", "")
            .strip()
        )
        print(
            "\n",
            len(source_text_chunks[i].split("\n\n"))
            - len(translation.split("\n\n"))
        )
        translation_chunks.append(translation)

        cache_data = {
            "done_idx": i,
            "translation_chunks": translation_chunks,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)

        if SAVE_CACHE:
            save_cache(
                f"saved_cache/{textname}/{MODEL}/chunk_{i}/",
                source_text_chunks[i],
                "source_t",
            )
            save_cache(
                f"saved_cache/{textname}/{MODEL}/chunk_{i}/",
                translation,
                f"init_t_{i}",
            )

    if SAVE_CACHE:
        save_cache(
            f"saved_cache/{textname}/{MODEL}/",
            "".join(translation_chunks),
            "init_t",
        )
    return translation_chunks


def multichunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    country: str = "",
) -> List[str]:
    """
    Provides constructive criticism and suggestions for improving a partial translation.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language of the translation.
        source_text_chunks (List[str]): The source text divided into chunks.
        translation_1_chunks (List[str]): The translated chunks corresponding to the source text chunks.
        country (str): Country specified for target language.

    Returns:
        List[str]: A list of reflections containing suggestions for improving each translated chunk.
    """

    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text as context for critiquing the translated part. Retain all markdown image links, Latex code and multi-level title in their positions and relationships within the text.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text, and the content needs to be consistent.),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text as context for critiquing the translated part. Retain all markdown image links, Latex code and multi-level title in their positions and relationships within the text. Retain all markdown image links, Latex code and multi-level title in their positions and relationships within the text.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text, and the content needs to be consistent.),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    done_idx = -1
    reflection_chunks = []

    cache_file = "cache/reflection_chunks.json"
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            cache_data: dict = json.load(f)
            done_idx = cache_data.get("done_idx", 0)
            reflection_chunks = cache_data.get("reflection_chunks", [])

    if done_idx == len(source_text_chunks) - 1:
        return reflection_chunks

    for i in tqdm(
        range(done_idx + 1, len(source_text_chunks)),
        desc="2:reflect translating",
    ):
        # Will translate chunk i
        tagged_text = (
            ("".join(source_text_chunks[max(i - 2, 0) : i]) if i > 0 else "")
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + (
                "".join(
                    source_text_chunks[
                        i + 1 : min(i + 2, len(source_text_chunks))
                    ]
                )
                if i < len(source_text_chunks) - 1
                else ""
            )
        )
        if country != "":
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
                country=country,
            )
        else:
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
            )

        reflection = get_completion(prompt, system_message=system_message)
        reflection_chunks.append(reflection)

        cache_data = {
            "done_idx": i,
            "reflection_chunks": reflection_chunks,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)
        if SAVE_CACHE:
            save_cache(
                f"saved_cache/{textname}/{MODEL}/chunk_{i}/",
                reflection,
                f"reflection_t_{i}",
            )

    if SAVE_CACHE:
        save_cache(
            f"saved_cache/{textname}/{MODEL}/",
            "".join(reflection_chunks),
            "reflection_t",
        )

    return reflection_chunks


def multichunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    reflection_chunks: List[str],
) -> List[str]:
    """
    Improves the translation of a text from source language to target language by considering expert suggestions.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The source text divided into chunks.
        translation_1_chunks (List[str]): The initial translation of each chunk.
        reflection_chunks (List[str]): Expert suggestions for improving each translated chunk.

    Returns:
        List[str]: The improved translation of each chunk.
    """

    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

    improvement_prompt = """Improve the translation from {source_lang} to {target_lang} based on expert suggestions. Use the provided source text and initial translation as reference.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

Translate only the part within <TRANSLATE_THIS> tags:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

Initial translation:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

Expert suggestions:
<EXPERT_SUGGESTIONS>
{reflection_chunk}
- Even if it is a single title or a title containing incomplete paragraphs, it still needs to be translated.
</EXPERT_SUGGESTIONS>

Guidelines for improvement:
1. Cover ALL paragraph and infomation from the part within <TRANSLATE_THIS> tags.
2. Maintain ALL paragraph structure and line breaks, and titles.
3. Preserve all markdown, image links, LaTeX code, and titles.
4. Prioritize completeness over conflicting expert suggestions.
5. Do not remove any single line from the part within <TRANSLATE_THIS> tags.
6. All initial translation content should be included in the final translation!

Output only the new translation of the indicated part and nothing else.
"""

    done_idx = -1
    translation_2_chunks = []

    cache_file = "cache/imporove_chunks.json"
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            cache_data: dict = json.load(f)
            done_idx = cache_data.get("done_idx", 0)
            translation_2_chunks = cache_data.get("translation_2_chunks", [])

    if done_idx == len(source_text_chunks) - 1:
        return translation_2_chunks

    for i in tqdm(
        range(done_idx + 1, len(source_text_chunks)),
        desc="3:improve translating",
    ):
        # Will translate chunk i
        tagged_text = (
            ("".join(source_text_chunks[max(i - 2, 0) : i]) if i > 0 else "")
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + (
                "".join(
                    source_text_chunks[
                        i + 1 : min(i + 2, len(source_text_chunks))
                    ]
                )
                if i < len(source_text_chunks) - 1
                else ""
            )
        )

        prompt = improvement_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            translation_1_chunk=translation_1_chunks[i],
            reflection_chunk=reflection_chunks[i],
        )

        translation_2 = get_completion(prompt, system_message=system_message)
        translation_2_chunks.append(translation_2)

        cache_data = {
            "done_idx": i,
            "translation_2_chunks": translation_2_chunks,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)

        if SAVE_CACHE:
            save_cache(
                f"saved_cache/{textname}/{MODEL}/chunk_{i}/",
                translation_2,
                f"improvition_t_{i}",
            )
        print("\n", len(translation_2) - len(translation_1_chunks[i]))
        print(
            len(translation_2.split("\n\n"))
            - len(translation_1_chunks[i].split("\n\n"))
        )

    if SAVE_CACHE:
        save_cache(
            f"saved_cache/{textname}/{MODEL}/",
            "".join(translation_2_chunks),
            "improvition_t",
        )

    return translation_2_chunks


def multichunk_translation(
    source_lang, target_lang, source_text_chunks, country: str = ""
):
    """
    Improves the translation of multiple text chunks based on the initial translation and reflection.

    Args:
        source_lang (str): The source language of the text chunks.
        target_lang (str): The target language for translation.
        source_text_chunks (List[str]): The list of source text chunks to be translated.
        translation_1_chunks (List[str]): The list of initial translations for each source text chunk.
        reflection_chunks (List[str]): The list of reflections on the initial translations.
        country (str): Country specified for target language
    Returns:
        List[str]: The list of improved translations for each source text chunk.
    """

    translation_1_chunks = multichunk_initial_translation(
        source_lang, target_lang, source_text_chunks
    )

    reflection_chunks = multichunk_reflect_on_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        country,
    )

    translation_2_chunks = multichunk_improve_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        reflection_chunks,
    )

    return translation_2_chunks


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    Calculate the chunk size based on the token count and token limit.

    Args:
        token_count (int): The total number of tokens.
        token_limit (int): The maximum number of tokens allowed per chunk.

    Returns:
        int: The calculated chunk size.

    Description:
        This function calculates the chunk size based on the given token count and token limit.
        If the token count is less than or equal to the token limit, the function returns the token count as the chunk size.
        Otherwise, it calculates the number of chunks needed to accommodate all the tokens within the token limit.
        The chunk size is determined by dividing the token limit by the number of chunks.
        If there are remaining tokens after dividing the token count by the token limit,
        the chunk size is adjusted by adding the remaining tokens divided by the number of chunks.

    Example:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """

    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def replace_markdown_links(text):
    # 正则表达式匹配 [![.*?](.*?)](.*?) 的模式
    pattern = re.compile(r"\[(!\[.*?\]\(.*?\))\]\(.*?\)")
    replaced_text = pattern.sub(r"\1", text)

    return replaced_text


def translate(
    source_lang,
    target_lang,
    source_text,
    country,
    curfile,
    max_tokens=MAX_TOKENS_PER_CHUNK,
):
    global textname
    textname = curfile
    """Translate the source_text from source_lang to target_lang."""
    source_text = replace_markdown_links(source_text)
    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)
    ic(MODEL)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        final_translation = one_chunk_translate_text(
            source_lang, target_lang, source_text, country
        )

        return final_translation

    else:
        os.makedirs("cache", exist_ok=True)

        ic("Translating text as multiple chunks")
        if SAVE_CACHE:
            save_cache(
                f"saved_cache/{textname}/{MODEL}/", source_text, "source_text"
            )
        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )
        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        translation_2_chunks = multichunk_translation(
            source_lang, target_lang, source_text_chunks, country
        )
        shutil.rmtree("cache")
        return "".join(translation_2_chunks)
