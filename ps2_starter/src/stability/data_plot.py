import util 
import matplotlib.pyplot as plt


def main():
    # import datasets: 
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=False)
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=False)

    # gen. plot of dataset A: 
    util.plot_points(Xa, Ya)
    plt.savefig('plot_q2b_datasetA.png', bbox_inches='tight')
    
    # clear plot: 
    plt.clf() 
    
    # gen. plot of dataset B:
    util.plot_points(Xb, Yb)
    plt.savefig('plot_q2b_datasetB.png', bbox_inches='tight')
    
if __name__ == '__main__':
    main()