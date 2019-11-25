from keras.models import Model

def display_model_score(model, train, val, test, batch_size):
    train_score = model.evaluate(train[0], train[1], batch_size=batch_size, verbose=1)
    print('Train loss: ', round(train_score[0], 3))
    print('Train accuracy: ', round(train_score[1], 3))
    print('-'*70)
    
    val_score = model.evaluate(val[0], val[1], batch_size=batch_size, verbose=1)
    print('Val loss: ', round(val_score[0], 3))
    print('Val accuracy: ', round(val_score[1], 3))
    print('-'*70)
    
    test_score = model.evaluate(test[0], test[1], batch_size=batch_size, verbose=1)
    print('Test loss: ', round(test_score[0], 3))
    print('Test accuracy: ', round(test_score[1], 3))
