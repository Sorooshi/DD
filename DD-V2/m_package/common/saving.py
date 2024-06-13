import pickle 

def saving_results(results, name):
    with open(f'Results/{name}.txt','wb') as f:
        pickle.dump(results, f)


def save_model(model, name):
    model.save(f'Models/{name}.keras')