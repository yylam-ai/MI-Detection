from sklearn.metrics import confusion_matrix
from models import *
import numpy as np
import os
import argparse
import torch
     
ap = argparse.ArgumentParser()
ap.add_argument('-gpu', '--gpu', default='0')
ap.add_argument('-view', '--view', default='multi')
ap.add_argument('-dataPath', '--dataPath', default=os.path.join(os.getcwd(),'DataSplits'))
ap.add_argument('-outputPath', '--outputPath', default=os.path.join(os.getcwd(), 'output'))

args = vars(ap.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']
if not os.path.exists(os.path.join(os.getcwd(),'output', 'matrices')): os.makedirs(os.path.join(os.getcwd(),'output', 'matrices'))

MODEL = ['SVM', 'DT', 'KNN', 'RF', 'MiniRocket', 'MultiRocket', 'CNN', 'dpClassifier', 'dpOptimalCNN']
REFIT= ['AUC']

X_train = np.load(os.path.join(args['dataPath'], 'x_train_' + args['view'] + '.npy'))
X_test = np.load(os.path.join(args['dataPath'], 'x_test_' + args['view'] + '.npy'))
Y_train = np.load(os.path.join(args['dataPath'], 'y_train_' + args['view'] + '.npy'))
Y_test = np.load(os.path.join(args['dataPath'], 'y_test_' + args['view'] + '.npy'))

for f in range(0,5):    
    for i in range(len(MODEL)):           
        for j in range(len(REFIT)):            
            #Shuffle train data
            np.random.seed(seed=3)
            idx = np.random.permutation(len(X_train[f]))
            x_train, y_train = X_train[f][idx], Y_train[f][idx]
            x_test, y_test = X_test[f], Y_test[f]
        
            if MODEL[i] == 'SVM':    
                best_parameters, best_model = SVM_train(x_train, y_train, REFIT[j])
                score = best_model.predict(x_test)
                CM = confusion_matrix(y_test, score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(args['outputPath'], MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                    
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test)
                                
            elif MODEL[i] == 'KNN':
                best_parameters, best_model = KNN_train(x_train, y_train, REFIT[j])
                score = best_model.predict(x_test)
                
                CM = confusion_matrix(y_test, score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(args['outputPath'], MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test)
                                       
            elif MODEL[i] == 'DT':
                best_parameters, best_model = DT_train(x_train, y_train, REFIT[j])
                score = best_model.predict(x_test)
                
                CM = confusion_matrix(y_test, score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(args['outputPath'], MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test)
                                
            elif MODEL[i] == 'RF':
                best_parameters, best_model = RF_train(x_train, y_train, REFIT[j])
                score = best_model.predict(x_test)
                
                CM = confusion_matrix(y_test, score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(args['outputPath'], MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test)
                                
            elif MODEL[i] == 'CNN':      
                x_train = np.expand_dims(x_train, axis = -1)
                x_test = np.expand_dims(x_test, axis = -1)

                X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
                X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

                X_train_tensor = X_train_tensor.permute(0, 2, 1)
                X_test_tensor = X_test_tensor.permute(0, 2, 1)

                best_model, best_parameters = CNN_train(X_train_tensor, y_train_tensor, REFIT[j])    
                score = best_model.predict(X_test_tensor)
                
                CM = confusion_matrix(y_test, score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(args['outputPath'], MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test)
            
            elif MODEL[i] == 'MiniRocket':
                best_parameters, best_model = mini_rocket_classifier_train(x_train, y_train, REFIT[j])
                score = best_model.predict(x_test)
                
                CM = confusion_matrix(y_test, score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(args['outputPath'], MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test)
                                
        
            elif MODEL[i] == 'MultiRocket':
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

                best_parameters, best_model = multi_rocket_classifier_train(x_train, y_train, REFIT[j])
                score = best_model.predict(x_test)
                
                CM = confusion_matrix(y_test, score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(args['outputPath'], MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test)
            
            elif MODEL[i] == 'dpClassifier':      
                x_train = np.expand_dims(x_train, axis = -1)
                x_test = np.expand_dims(x_test, axis = -1)
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

                net = init_dp_classifier_model(x_train)
                base_model = net.classifier
                model = DpClassifierTorchModel(base_model=base_model)

                x_train_emb = net.feature_extractor(torch.tensor(x_train, dtype=torch.float32))
                x_test_emb = net.feature_extractor(torch.tensor(x_test, dtype=torch.float32))
                x_train_emb = x_train_emb.view(x_train_emb.shape[0], -1)
                x_test_emb = x_test_emb.view(x_test_emb.shape[0], -1)

                best_model, best_parameters = dp_classifier_train(model, x_train_emb, y_train, REFIT[j])
                score = best_model.predict(x_test_emb)
                
                CM = confusion_matrix(y_test, score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(args['outputPath'], MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test)
        
            elif MODEL[i] == 'dpOptimalCNN':      
                x_train = np.expand_dims(x_train, axis = -1)
                x_test = np.expand_dims(x_test, axis = -1)
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

                net = init_CNN_opti_model(x_train)
                model = OptiCNNTorchModel(net)

                best_model, best_parameters = CNN_opti_train(model, x_train, y_train, REFIT[j])
                score = best_model.predict(x_test)
                
                CM = confusion_matrix(y_test, score)
                metrics = performance_metrics(CM)
                
                #Save the results
                text_file = open(os.path.join(args['outputPath'], MODEL[i] + '_' + args['view'] + '.txt'), "a")
                text_file.write ("\n\n\n----------FOLD " + str(f) + "-------------\n")
                text_file.write("\n\n\nConfusion Matrix :" + str(CM) + "\n")
                text_file.write ("\nScoring:" + REFIT[j])
                text_file.write ("\nSensitivity:" + str(metrics[0]))
                text_file.write ("\nSpecificity:" + str(metrics[1]))
                text_file.write("\nPrecision:" + str(metrics[2]))
                text_file.write ("\nF1-Score:" + str(metrics[3]))
                text_file.write ("\nF2-Score:" + str(metrics[4]))   
                text_file.write("\nAccuracy:" + str(metrics[5]))
                text_file.write('\nBest paramters:' + str(best_parameters))
                text_file.close()
                
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] +'_score_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), score)
                np.save(os.path.join(os.path.join(os.getcwd(), 'output', 'matrices'), args['view'] + '_y_test_' + MODEL[i] + '_' + REFIT[j] + '_fold' + str(f) + '.npy'), y_test)