from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI

base_url = "http://hackllm.vkcloud.eazify.net:8000/v1" # Mail.ru

chat = ChatOpenAI(api_key="<key>",
                  model = "tgi",
                  openai_api_base = base_url)

input = "В Башкирии в январе—марте 2024 года сельхозорганизациями было произведено 149,4 тыс. тонн молока, это на 4,1% больше, чем за аналогичный период прошлого года, следует из данных Росстата, которые изучил РБК Уфа. Республика оказалась на четвертом месте в Приволжском федеральном округе по объемам производства молока в первом квартале.В целом в ПФО за три месяца было произведено 1,6 млн тонн молока, показатели выросли на 7,1%. Лидером по росту стала"

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

print(tags)
