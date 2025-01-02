from a2_3 import Kmeans_header
from a2_4 import GMM_header
from a2_5 import PCA_header
from a2_6 import six_header
from a2_7 import seven_header
from a2_8 import eight_header
from a2_9 import nine_header

def main():
    n = input()
    if n == '3':
        Kmeans_header()
    if n == '4':
        GMM_header()
        
    if n == '5':
        PCA_header()
        
    if n == '6':
        six_header()
        
    if n == '7':
        seven_header()
    
    if n == '8':
        eight_header()
        
    if n == '9':
        nine_header()
        

if __name__ == "__main__":
    main()  