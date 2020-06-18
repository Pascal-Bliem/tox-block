
from tox_block.data_processing import data_handling as dh
from tox_block.prediction import make_predictions, make_single_prediction
from tox_block.config import config

def test_make_predictions():
   
    X_nontoxic, y_nontoxic = dh.load_data_multi_labels("tests/test_data/non_toxic.csv")
    X_toxic, y_toxic = dh.load_data_multi_labels("tests/test_data/toxic.csv")

    pred_nontoxic = make_predictions(X_nontoxic.comment_text.tolist())
    pred_toxic = make_predictions(X_toxic.comment_text.tolist())
    
    # test if types and shapes are correct
    assert pred_nontoxic is not None
    assert pred_toxic is not None
    assert len(pred_nontoxic) == len(X_nontoxic)
    assert len(pred_toxic) == len(X_toxic)
    
    for i, item in pred_nontoxic.items():
        for key in ["text"] + config.LIST_CLASSES:
            if key == "text":
                assert item[key] == X_nontoxic.iloc[i].comment_text
            else:
                assert  isinstance(item[key],float)
                assert  item[key] >= 0 and item[key] < 1

        # assert that average predicted toxicity proability over the 
        # categories for non-toxic samples is below 10%. This limit 
        # is strict because we want to avoid incorrectly blocking a text
        assert 0.1 > sum([item.get(key) for key in config.LIST_CLASSES])/len(config.LIST_CLASSES)
    
    for i, item in pred_toxic.items():
        for key in ["text"] + config.LIST_CLASSES:
            if key == "text":
                assert item[key] == X_toxic.iloc[i].comment_text
            else:
                assert  isinstance(item[key],float)
                assert  item[key] >= 0 and item[key] < 1



def test_make_single_prediction():

    X_nontoxic, y_nontoxic = dh.load_data_multi_labels("tests/test_data/non_toxic.csv")
    X_toxic, y_toxic = dh.load_data_multi_labels("tests/test_data/toxic.csv")

    pred_nontoxic = make_single_prediction(X_nontoxic.comment_text.values[0])
    pred_toxic = make_single_prediction(X_toxic.comment_text.values[0])
    
    # test if types and shapes are correct
    assert pred_nontoxic is not None
    assert pred_toxic is not None
    assert len(pred_nontoxic) == len(config.LIST_CLASSES) + 1
    assert len(pred_toxic) == len(config.LIST_CLASSES) + 1

    for key in ["text"] + config.LIST_CLASSES:
        if key == "text":
            assert pred_nontoxic[key] == X_nontoxic.iloc[0].comment_text
        else:
            assert  isinstance(pred_nontoxic[key],float)
            assert  pred_nontoxic[key] >= 0 and pred_nontoxic[key] < 1

    # assert that average predicted toxicity proability over the 
    # categories for non-toxic samples is below 10%. This limit 
    # is strict because we want to avoid incorrectly blocking a text
    assert 0.1 > sum([pred_nontoxic.get(key) 
                      for key in config.LIST_CLASSES])/len(config.LIST_CLASSES)

    for key in ["text"] + config.LIST_CLASSES:
        if key == "text":
            assert pred_toxic[key] == X_toxic.iloc[0].comment_text
        else:
            assert  isinstance(pred_toxic[key],float)
            assert  pred_toxic[key] >= 0 and pred_toxic[key] < 1