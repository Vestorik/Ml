import langchain

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub  # или из langchain_openai import OpenAI

# Инициализация модели (пример с Hugging Face)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature": 0.7, "max_length": 100}
)

# Альтернатива с OpenAI:
# from langchain_openai import OpenAI
# llm = OpenAI(model="text-davinci-003", temperature=0.7)

# Шаблон промпта
template = """
Ты — дружелюбный ассистент.
Ответь на вопрос максимально кратко и по делу.

Вопрос: {question}
Ответ:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Создание цепочки
chain = LLMChain(llm=llm, prompt=prompt)

# Запуск цепочки
response = chain.run(question="Какая столица Франции?")
print(response)