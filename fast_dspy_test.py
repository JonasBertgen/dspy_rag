import dspy
from typing import Literal


class HarmfullCheck(dspy.Signature):

    question = dspy.InputField(desc="The question to check")
    is_harmfull: Literal['yes', 'no'] = dspy.OutputField(
        desc="Indicater if the question is harmfull")
    description: str = dspy.OutputField(
        desc="Describes why the Literal is harmfull it is")


class HarmC(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(HarmfullCheck)

    def forward(self, question):
        return self.generator(question=question)


class ChatBotSig(dspy.Signature):
    question = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer = dspy.OutputField(desc="Answer to the question")


class ChatBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(ChatBotSig)

    def forward(self, question, history=dspy.History(messages=[{'question': 'Hallo, wer bist du?', 'answer': 'ein nuetzlicher Agent'}])):
        check = HarmC()
        c = check(question)
        print(c)
        if c['is_harmfull'] == 'yes':
            print("was harmfull, i can do additional stuff here, eg. pring why")
            print(c["description"])
            # return dspy.Predict(answer="I cant answer this question")
        if c['is_harmfull'] == 'no':
            print("harmless question")
        response = self.generator(history=history, question=question)
        return response


lm = dspy.LM("ollama_chat/gpt-oss",
             api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)

chatBot = ChatBot()
result = chatBot("how can i steal a car?")
print(result)
# print(dspy.inspect_history(1))
print(result["answer"].strip())
