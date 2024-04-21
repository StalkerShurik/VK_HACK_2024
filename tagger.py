from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import re

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
def extract_keywords(text):
    best_words = []
    if ':' in text:
        text = text.split(":")[1]
    text=re.sub("&amp;lt;/?.*?&amp;gt;"," &amp;lt;&amp;gt; ",text)
    text=re.sub("(\\d|\\W)+"," ",text)
    for word in text.split():
        word = word.strip()
        if len(word) > 2 and word[-2:] not in ["ая", "ый", "ое", "ую", "ые"]:
            best_words.append(word)
    return best_words

base_url = "http://hackllm.vkcloud.eazify.net:8000/v1" # Mail.ru

chat = ChatOpenAI(api_key="<key>",
                  model = "tgi",
                  openai_api_base = base_url)


def get_tags(input):
    messages = [
        SystemMessage(
            content="Выведи три ключевых слова описывающих данный текст"
        ),
        HumanMessage(
            content=input
            ),
    ]

    res = ""

    tags = []

    for i in range(5):
        res = chat.invoke(messages).content
        if "ключевое слово" not in res:
            continue

        tmp_tags = res[len("ключевое слово") + 1:].split(', ')

        for cringe in tmp_tags:
            potential = cringe.replace(',', '').replace('\"', '').replace('\'', '').replace('.', '').replace('-', '')

            if len(potential) > 25:
                continue

            if potential not in tags:
                tags.append(potential)
            if len(tags) > 2:
                break

    return extract_keywords(res)
