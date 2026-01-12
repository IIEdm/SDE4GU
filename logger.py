import torch
import numpy as np

class Logger_ood(object):
    
    def __init__(self, runs, info=None) -> None:
        self.infor = info
        self.runs = runs
        self.results = [[] for _ in range(runs)]
    def add_result(self, run, result):
        self.results[run].append(result)
    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            # argmax1 = result[:, 1].argmin().item()
            argmax1 = result[:, 4].argmax().item()
            argmax = result[:, 6].argmax().item()
            print('ACCURACY')
            print(f'Highest Train: {result[:, 3].max():.2f}')
            print(f'Highest Valid: {result[:, 4].max():.2f}')
            print(f'Highest Test: {result[:, 5].max():.2f}')
            print(f'Chosen epoch: {argmax1}')
            print(f'Final Train: {result[argmax1, 3]:.2f}')
            print(f'Final Test: {result[argmax1, 5]:.2f}')
            self.test_acc=result[argmax1, 2]
            print('Detection Task')
            print(f'AUROC: {result[argmax1, 6]:.2f}')
            print(f'AUPR_in: {result[argmax1, 7]:.2f}')
            print(f'AUPR_out: {result[argmax1, 8]:.2f}')
            print(f'FPR95: {result[argmax1, 9]:.2f}')
            print(f'Detection acc: {result[argmax1, 10]:.2f}')
            
            self.auc = result[argmax1, 6]
            self.aupr_in = result[argmax1, 7]
            self.aupr_out = result[argmax1, 8]
            self.fpr95 = result[argmax1, 9]
            self.detection_acc = result[argmax1, 10]

            
        else:
            best_results = []
            for run in range(self.runs):
                result = 100 * torch.tensor(self.results[run])
                argmax1 = result[:, 4].argmax().item()

                # argmax1 = result[:, 1].argmin().item()
                argmax = result[:, 6].argmax().item()
                best_result = [result[argmax1, 5], \
                               result[argmax1, 6], result[argmax1, 7], \
                                result[argmax1, 8], result[argmax1, 9], result[argmax1, 10]]
                best_results.append(best_result)
            final_result = np.mean(best_results, axis=0)
            print('----------------END RESULT------------------')
            print('ACCURACY')
            
            print(f'END Final Test: {final_result[0]:.2f}')
            self.test_acc=final_result[0]
            print('Detection Task')
            print(f'END  AUROC: {final_result[1]:.2f}')
            print(f'END  AUPR_in: {final_result[2]:.2f}')
            print(f'END  AUPR_out: {final_result[3]:.2f}')
            print(f'END  FPR95: {final_result[4]:.2f}')
            print(f'END  DETECTION acc: {final_result[5]:.2f}')
            
            
            self.auc = final_result[1]
            self.aupr_in = final_result[2]
            self.aupr_out = final_result[3]
            self.fpr95 = final_result[4]
            self.detection_acc = final_result[5]

        return [self.test_acc, \
            self.auc, self.aupr_in, self.aupr_out, self.fpr95, self.detection_acc]


