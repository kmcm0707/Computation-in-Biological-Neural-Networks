# complex,fast 0,0 1,1 2,2 3,3 4,4 5,5 6,12 7,7 8,8 9,9 
import os
import torch

if __name__ == "__main__":
    current_dir = os.getcwd()
    file_directory = ""
    file_directory = current_dir + file_directory
    P_arguments = []
    Q_arguments = []
    P_to_Q_mappings = [ (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 12), (7, 7), (8, 8), (9, 9) ]

    state_dic = torch.load(
                    file_directory + "/UpdateMetaParameters.pth",
                    weights_only=True,
                    map_location="cpu",
                )
    
    state_dic_P_matrix = state_dic["P_matrix"]
    Q_matrix = torch.zeros((state_dic_P_matrix.shape[0], len(Q_arguments)))
    for index in P_arguments:
        P_to_Q_index = P_to_Q_mappings[index]
        if P_to_Q_index[1] in Q_arguments:
            Q_matrix[:, P_to_Q_index[1]] = state_dic_P_matrix[:, index]

    new_state_dic = state_dic
    new_state_dic["Q_matrix"] = Q_matrix
    del new_state_dic["P_matrix"]
    torch.save(
        new_state_dic,
        file_directory + "/UpdateMetaParameters_converted.pth",
    )
    

    