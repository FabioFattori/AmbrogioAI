import utilities.DataSetManager as dsm
import classes.AmbrogioResNet50 as ar50
if __name__ == '__main__':
    model = ar50.AmbrogioNet50()
    model.train_model(num_epochs=10)
    model.save_model()
