from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

chatbot = ChatBot(
    'CovidBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand.',
            'maximum_similarity_threshold': 0.90
        }
    ],
    database_uri='sqlite:///database.sqlite3'
)
training_data_covid = open('data/covid_data.txt').read().splitlines()
training_data_personal = open('data/personal_ques.txt').read().splitlines()

training_data = training_data_covid + training_data_personal
trainer = ListTrainer(chatbot)
trainer.train(training_data)

trainer_corpus = ChatterBotCorpusTrainer(chatbot)

trainer_corpus.train('chatterbot.corpus.english')