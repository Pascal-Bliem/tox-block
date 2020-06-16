import os

# path handling
PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))
DATA_DIR = os.path.join(PACKAGE_ROOT, 'data')
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "train.csv")
TESTING_DATA_FILE = os.path.join(DATA_DIR, "test.csv")
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')
EMBEDDING_FILE= os.path.join(DATA_DIR, "glove.6B.50d.txt")

# this stopword list is based on spaCy's default list
RE_STOPWORD_STRING = "'d\\b|\\b'll\\b|\\b'm\\b|\\b're\\b|\\b's\\b|\\b've\\b|\\ba\\b|\\babout\\b|\\babove\\b|\\bacross\\b|\\bafter\\b|\\bafterwards\\b|\\bagain\\b|\\bagainst\\b|\\ball\\b|\\balmost\\b|\\balone\\b|\\balong\\b|\\balready\\b|\\balso\\b|\\balthough\\b|\\balways\\b|\\bam\\b|\\bamong\\b|\\bamongst\\b|\\bamount\\b|\\ban\\b|\\band\\b|\\banother\\b|\\bany\\b|\\banyhow\\b|\\banyone\\b|\\banything\\b|\\banyway\\b|\\banywhere\\b|\\bare\\b|\\baround\\b|\\bas\\b|\\bat\\b|\\bback\\b|\\bbe\\b|\\bbecame\\b|\\bbecause\\b|\\bbecome\\b|\\bbecomes\\b|\\bbecoming\\b|\\bbeen\\b|\\bbefore\\b|\\bbeforehand\\b|\\bbehind\\b|\\bbeing\\b|\\bbelow\\b|\\bbeside\\b|\\bbesides\\b|\\bbetween\\b|\\bbeyond\\b|\\bboth\\b|\\bbottom\\b|\\bbut\\b|\\bby\\b|\\bca\\b|\\bcall\\b|\\bcan\\b|\\bcannot\\b|\\bcould\\b|\\bdid\\b|\\bdo\\b|\\bdoes\\b|\\bdoing\\b|\\bdone\\b|\\bdown\\b|\\bdue\\b|\\bduring\\b|\\beach\\b|\\beight\\b|\\beither\\b|\\beleven\\b|\\belse\\b|\\belsewhere\\b|\\bempty\\b|\\benough\\b|\\beven\\b|\\bever\\b|\\bevery\\b|\\beveryone\\b|\\beverything\\b|\\beverywhere\\b|\\bexcept\\b|\\bfew\\b|\\bfifteen\\b|\\bfifty\\b|\\bfirst\\b|\\bfive\\b|\\bfor\\b|\\bformer\\b|\\bformerly\\b|\\bforty\\b|\\bfour\\b|\\bfrom\\b|\\bfront\\b|\\bfull\\b|\\bfurther\\b|\\bget\\b|\\bgive\\b|\\bgo\\b|\\bhad\\b|\\bhas\\b|\\bhave\\b|\\bhe\\b|\\bhence\\b|\\bher\\b|\\bhere\\b|\\bhereafter\\b|\\bhereby\\b|\\bherein\\b|\\bhereupon\\b|\\bhers\\b|\\bherself\\b|\\bhim\\b|\\bhimself\\b|\\bhis\\b|\\bhow\\b|\\bhowever\\b|\\bhundred\\b|\\bi\\b|\\bif\\b|\\bin\\b|\\bindeed\\b|\\binto\\b|\\bis\\b|\\bit\\b|\\bits\\b|\\bitself\\b|\\bjust\\b|\\bkeep\\b|\\blast\\b|\\blatter\\b|\\blatterly\\b|\\bleast\\b|\\bless\\b|\\bmade\\b|\\bmake\\b|\\bmany\\b|\\bmay\\b|\\bme\\b|\\bmeanwhile\\b|\\bmight\\b|\\bmine\\b|\\bmore\\b|\\bmoreover\\b|\\bmost\\b|\\bmostly\\b|\\bmove\\b|\\bmuch\\b|\\bmust\\b|\\bmy\\b|\\bmyself\\b|\\bn't\\b|\\bname\\b|\\bnamely\\b|\\bneither\\b|\\bnever\\b|\\bnevertheless\\b|\\bnext\\b|\\bnine\\b|\\bno\\b|\\bnobody\\b|\\bnone\\b|\\bnoone\\b|\\bnor\\b|\\bnot\\b|\\bnothing\\b|\\bnow\\b|\\bnowhere\\b|\\bn‘t\\b|\\bn’t\\b|\\bof\\b|\\boff\\b|\\boften\\b|\\bon\\b|\\bonce\\b|\\bone\\b|\\bonly\\b|\\bonto\\b|\\bor\\b|\\bother\\b|\\bothers\\b|\\botherwise\\b|\\bour\\b|\\bours\\b|\\bourselves\\b|\\bout\\b|\\bover\\b|\\bown\\b|\\bpart\\b|\\bper\\b|\\bperhaps\\b|\\bplease\\b|\\bput\\b|\\bquite\\b|\\brather\\b|\\bre\\b|\\breally\\b|\\bregarding\\b|\\bsame\\b|\\bsay\\b|\\bsee\\b|\\bseem\\b|\\bseemed\\b|\\bseeming\\b|\\bseems\\b|\\bserious\\b|\\bseveral\\b|\\bshe\\b|\\bshould\\b|\\bshow\\b|\\bside\\b|\\bsince\\b|\\bsix\\b|\\bsixty\\b|\\bso\\b|\\bsome\\b|\\bsomehow\\b|\\bsomeone\\b|\\bsomething\\b|\\bsometime\\b|\\bsometimes\\b|\\bsomewhere\\b|\\bstill\\b|\\bsuch\\b|\\btake\\b|\\bten\\b|\\bthan\\b|\\bthat\\b|\\bthe\\b|\\btheir\\b|\\bthem\\b|\\bthemselves\\b|\\bthen\\b|\\bthence\\b|\\bthere\\b|\\bthereafter\\b|\\bthereby\\b|\\btherefore\\b|\\btherein\\b|\\bthereupon\\b|\\bthese\\b|\\bthey\\b|\\bthird\\b|\\bthis\\b|\\bthose\\b|\\bthough\\b|\\bthree\\b|\\bthrough\\b|\\bthroughout\\b|\\bthru\\b|\\bthus\\b|\\bto\\b|\\btogether\\b|\\btoo\\b|\\btop\\b|\\btoward\\b|\\btowards\\b|\\btwelve\\b|\\btwenty\\b|\\btwo\\b|\\bunder\\b|\\bunless\\b|\\buntil\\b|\\bup\\b|\\bupon\\b|\\bus\\b|\\bused\\b|\\busing\\b|\\bvarious\\b|\\bvery\\b|\\bvia\\b|\\bwas\\b|\\bwe\\b|\\bwell\\b|\\bwere\\b|\\bwhat\\b|\\bwhatever\\b|\\bwhen\\b|\\bwhence\\b|\\bwhenever\\b|\\bwhere\\b|\\bwhereafter\\b|\\bwhereas\\b|\\bwhereby\\b|\\bwherein\\b|\\bwhereupon\\b|\\bwherever\\b|\\bwhether\\b|\\bwhich\\b|\\bwhile\\b|\\bwhither\\b|\\bwho\\b|\\bwhoever\\b|\\bwhole\\b|\\bwhom\\b|\\bwhose\\b|\\bwhy\\b|\\bwill\\b|\\bwith\\b|\\bwithin\\b|\\bwithout\\b|\\bwould\\b|\\byet\\b|\\byou\\b|\\byour\\b|\\byours\\b|\\byourself\\b|\\byourselves\\b|\\b‘d\\b|\\b‘ll\\b|\\b‘m\\b|\\b‘re\\b|\\b‘s\\b|\\b‘ve\\b|\\b’d\\b|\\b’ll\\b|\\b’m\\b|\\b’re\\b|\\b’s\\b|\\b’ve"

# the following abbreviations will be used BL: binary label, ML: multilabel
LIST_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
BATCH_SIZE = 32
EPOCHS = 1
VALIDATION_SPLIT = 0.1
CLASS_WEIGHT = {0: 0.5565938358935721, 1: 4.917442218798151}
MAX_FEATURES = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_SIZE = 50
# model training binary label
DROPOUT_BL = 0.1
NUM_LSTM_UNITS_BL = 50
NUM_DENSE_UNITS_BL = 50
LEARNING_RATE_BL = 0.001
# model training multi label
DROPOUT_ML = 0.1
NUM_LSTM_UNITS_ML = 50
NUM_DENSE_UNITS_ML = 50
LEARNING_RATE_ML = 0.001

# MODEL PERSISTING
MODEL_NAME_BL = "lstm_binary_label"
ENCODER_NAME_BL = "encoder_binary_label"
MODEL_NAME_ML = "lstm_multi_label"
ENCODER_NAME_ML = "encoder_multi_label"

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()

MODEL_FILE_NAME_BL = f"{MODEL_NAME_BL}_{_version}.h5"
MODEL_PATH_BL = os.path.join(TRAINED_MODEL_DIR, MODEL_FILE_NAME_BL)
MODEL_FILE_NAME_ML = f"{MODEL_NAME_ML}_{_version}.h5"
MODEL_PATH_ML = os.path.join(TRAINED_MODEL_DIR, MODEL_FILE_NAME_ML)

ENCODER_FILE_NAME_BL = f"{ENCODER_NAME_BL}_{_version}.pkl"
ENCODER_PATH_BL = os.path.join(TRAINED_MODEL_DIR, ENCODER_FILE_NAME_BL)
ENCODER_FILE_NAME_ML = f"{ENCODER_NAME_ML}_{_version}.pkl"
ENCODER_PATH_ML = os.path.join(TRAINED_MODEL_DIR, ENCODER_FILE_NAME_ML)