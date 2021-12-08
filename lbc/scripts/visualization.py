import pickle

import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':

    # unpickle file
    print("unpickling started..")
    filename = '/home/kavi/learning_based_control_project/lbc/pickles/sim_results'
    infile = open(filename,'rb')
    sim_results = pickle.load(infile)
    infile.close()
    print("unpickling done!")

    # plot sim_results
    print('plotting results...')
    plt.xlim(0, 11)
    plt.ylim(0, 11)
    matplotlib.use('TkAgg')
    x_coords1 = []
    y_coords1 = []
    x_coords2 = []
    y_coords2 = [] 

    states = sim_results["states"]

    for i_state in states:
        x_coords1.append(i_state[0])
        y_coords1.append(i_state[1])
        x_coords2.append(i_state[21])
        y_coords2.append(i_state[22])   

    # print(x_coords1)
    # print(y_coords1)
    # print(x_coords2)
    # print(y_coords2)

    print(len(states))
    print(((len(states))-1))
    rows = 4
    for i in range(len(states)):
        if (i+1 >= (len(states))-1):
            continue
        plt.subplot(rows, len(states)//rows, i+1)
        plt.plot(x_coords1[0:i+1],y_coords1[0:i+1])
        plt.plot(x_coords2[0:i+1],y_coords2[0:i+1])
        plt.title("t{}".format(i))

    plt.show() 