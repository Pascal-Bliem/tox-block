import os

# path handling
PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))
DATA_DIR = os.path.join(PACKAGE_ROOT, 'data')
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "train.csv")
TESTING_DATA_FILE = os.path.join(DATA_DIR, "test.csv")
MERGED_DATA_FILE = os.path.join(DATA_DIR, "merged.csv")
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')
EMBEDDING_FILE= os.path.join(DATA_DIR, "glove.6B.50d.txt")

# this stopword list is based on spaCy's default list
RE_STOPWORD_STRING = "'d\\b|\\b'll\\b|\\b'm\\b|\\b're\\b|\\b's\\b|\\b've\\b|\\ba\\b|\\babout\\b|\\babove\\b|\\bacross\\b|\\bafter\\b|\\bafterwards\\b|\\bagain\\b|\\bagainst\\b|\\ball\\b|\\balmost\\b|\\balone\\b|\\balong\\b|\\balready\\b|\\balso\\b|\\balthough\\b|\\balways\\b|\\bam\\b|\\bamong\\b|\\bamongst\\b|\\bamount\\b|\\ban\\b|\\band\\b|\\banother\\b|\\bany\\b|\\banyhow\\b|\\banyone\\b|\\banything\\b|\\banyway\\b|\\banywhere\\b|\\bare\\b|\\baround\\b|\\bas\\b|\\bat\\b|\\bback\\b|\\bbe\\b|\\bbecame\\b|\\bbecause\\b|\\bbecome\\b|\\bbecomes\\b|\\bbecoming\\b|\\bbeen\\b|\\bbefore\\b|\\bbeforehand\\b|\\bbehind\\b|\\bbeing\\b|\\bbelow\\b|\\bbeside\\b|\\bbesides\\b|\\bbetween\\b|\\bbeyond\\b|\\bboth\\b|\\bbottom\\b|\\bbut\\b|\\bby\\b|\\bca\\b|\\bcall\\b|\\bcan\\b|\\bcannot\\b|\\bcould\\b|\\bdid\\b|\\bdo\\b|\\bdoes\\b|\\bdoing\\b|\\bdone\\b|\\bdown\\b|\\bdue\\b|\\bduring\\b|\\beach\\b|\\beight\\b|\\beither\\b|\\beleven\\b|\\belse\\b|\\belsewhere\\b|\\bempty\\b|\\benough\\b|\\beven\\b|\\bever\\b|\\bevery\\b|\\beveryone\\b|\\beverything\\b|\\beverywhere\\b|\\bexcept\\b|\\bfew\\b|\\bfifteen\\b|\\bfifty\\b|\\bfirst\\b|\\bfive\\b|\\bfor\\b|\\bformer\\b|\\bformerly\\b|\\bforty\\b|\\bfour\\b|\\bfrom\\b|\\bfront\\b|\\bfull\\b|\\bfurther\\b|\\bget\\b|\\bgive\\b|\\bgo\\b|\\bhad\\b|\\bhas\\b|\\bhave\\b|\\bhe\\b|\\bhence\\b|\\bher\\b|\\bhere\\b|\\bhereafter\\b|\\bhereby\\b|\\bherein\\b|\\bhereupon\\b|\\bhers\\b|\\bherself\\b|\\bhim\\b|\\bhimself\\b|\\bhis\\b|\\bhow\\b|\\bhowever\\b|\\bhundred\\b|\\bi\\b|\\bif\\b|\\bin\\b|\\bindeed\\b|\\binto\\b|\\bis\\b|\\bit\\b|\\bits\\b|\\bitself\\b|\\bjust\\b|\\bkeep\\b|\\blast\\b|\\blatter\\b|\\blatterly\\b|\\bleast\\b|\\bless\\b|\\bmade\\b|\\bmake\\b|\\bmany\\b|\\bmay\\b|\\bme\\b|\\bmeanwhile\\b|\\bmight\\b|\\bmine\\b|\\bmore\\b|\\bmoreover\\b|\\bmost\\b|\\bmostly\\b|\\bmove\\b|\\bmuch\\b|\\bmust\\b|\\bmy\\b|\\bmyself\\b|\\bn't\\b|\\bname\\b|\\bnamely\\b|\\bneither\\b|\\bnever\\b|\\bnevertheless\\b|\\bnext\\b|\\bnine\\b|\\bno\\b|\\bnobody\\b|\\bnone\\b|\\bnoone\\b|\\bnor\\b|\\bnot\\b|\\bnothing\\b|\\bnow\\b|\\bnowhere\\b|\\bn‘t\\b|\\bn’t\\b|\\bof\\b|\\boff\\b|\\boften\\b|\\bon\\b|\\bonce\\b|\\bone\\b|\\bonly\\b|\\bonto\\b|\\bor\\b|\\bother\\b|\\bothers\\b|\\botherwise\\b|\\bour\\b|\\bours\\b|\\bourselves\\b|\\bout\\b|\\bover\\b|\\bown\\b|\\bpart\\b|\\bper\\b|\\bperhaps\\b|\\bplease\\b|\\bput\\b|\\bquite\\b|\\brather\\b|\\bre\\b|\\breally\\b|\\bregarding\\b|\\bsame\\b|\\bsay\\b|\\bsee\\b|\\bseem\\b|\\bseemed\\b|\\bseeming\\b|\\bseems\\b|\\bserious\\b|\\bseveral\\b|\\bshe\\b|\\bshould\\b|\\bshow\\b|\\bside\\b|\\bsince\\b|\\bsix\\b|\\bsixty\\b|\\bso\\b|\\bsome\\b|\\bsomehow\\b|\\bsomeone\\b|\\bsomething\\b|\\bsometime\\b|\\bsometimes\\b|\\bsomewhere\\b|\\bstill\\b|\\bsuch\\b|\\btake\\b|\\bten\\b|\\bthan\\b|\\bthat\\b|\\bthe\\b|\\btheir\\b|\\bthem\\b|\\bthemselves\\b|\\bthen\\b|\\bthence\\b|\\bthere\\b|\\bthereafter\\b|\\bthereby\\b|\\btherefore\\b|\\btherein\\b|\\bthereupon\\b|\\bthese\\b|\\bthey\\b|\\bthird\\b|\\bthis\\b|\\bthose\\b|\\bthough\\b|\\bthree\\b|\\bthrough\\b|\\bthroughout\\b|\\bthru\\b|\\bthus\\b|\\bto\\b|\\btogether\\b|\\btoo\\b|\\btop\\b|\\btoward\\b|\\btowards\\b|\\btwelve\\b|\\btwenty\\b|\\btwo\\b|\\bunder\\b|\\bunless\\b|\\buntil\\b|\\bup\\b|\\bupon\\b|\\bus\\b|\\bused\\b|\\busing\\b|\\bvarious\\b|\\bvery\\b|\\bvia\\b|\\bwas\\b|\\bwe\\b|\\bwell\\b|\\bwere\\b|\\bwhat\\b|\\bwhatever\\b|\\bwhen\\b|\\bwhence\\b|\\bwhenever\\b|\\bwhere\\b|\\bwhereafter\\b|\\bwhereas\\b|\\bwhereby\\b|\\bwherein\\b|\\bwhereupon\\b|\\bwherever\\b|\\bwhether\\b|\\bwhich\\b|\\bwhile\\b|\\bwhither\\b|\\bwho\\b|\\bwhoever\\b|\\bwhole\\b|\\bwhom\\b|\\bwhose\\b|\\bwhy\\b|\\bwill\\b|\\bwith\\b|\\bwithin\\b|\\bwithout\\b|\\bwould\\b|\\byet\\b|\\byou\\b|\\byour\\b|\\byours\\b|\\byourself\\b|\\byourselves\\b|\\b‘d\\b|\\b‘ll\\b|\\b‘m\\b|\\b‘re\\b|\\b‘s\\b|\\b‘ve\\b|\\b’d\\b|\\b’ll\\b|\\b’m\\b|\\b’re\\b|\\b’s\\b|\\b’ve"

# model training multi label
LIST_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.1
CLASS_WEIGHT = {0: 0.5565938358935721, 1: 4.917442218798151}
MAX_FEATURES = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_SIZE = 50
DROPOUT = 0.1
NUM_LSTM_UNITS = 50
NUM_DENSE_UNITS = 50
LEARNING_RATE = 0.001

# model persisting
MODEL_NAME = "lstm_model"
ENCODER_NAME = "encoder"
TRAINING_HISTORY_NAME = "lstm_model_training_history"

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()

MODEL_FILE_NAME = f"{MODEL_NAME}_{_version}.h5"
MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL_FILE_NAME)

ENCODER_FILE_NAME = f"{ENCODER_NAME}_{_version}.pkl"
ENCODER_PATH = os.path.join(TRAINED_MODEL_DIR, ENCODER_FILE_NAME)

TRAINING_HISTORY_FILE_NAME = f"{TRAINING_HISTORY_NAME}_{_version}.pkl"
TRAINING_HISTORY_PATH = os.path.join(TRAINED_MODEL_DIR, 
                                     TRAINING_HISTORY_FILE_NAME)

# predicted probability rescaling
RESCALE_PROBA = {"toxic": (0.002116, 0.999858),
                 "severe_toxic": (0.000011, 0.648682),
                 "obscene": (0.000347, 0.997533),
                 "threat": (0.000040, 0.918887),
                 "insult": (0.000373, 0.986303),
                 "identity_hate": (0.000074, 0.901117)}