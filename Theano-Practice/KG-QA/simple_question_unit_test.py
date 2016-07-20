import simple_questions


folder = "/home/qzb/Documents/Research/PythonProjects/Data/Ask.Me.Anything/1.Facebook.bAbI.Project.Dataset/simplequestions/SimpleQuestions/"

# get raw data
train_raw, valid_raw, test_raw = simple_questions.get_raw(folder)

simple_questions.get_dictionaries(train_raw, valid_raw, test_raw)

