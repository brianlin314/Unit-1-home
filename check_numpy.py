import numpy as np

def view_npy_file(file_path):
    try:
        # Load the .npy file
        data = np.load(file_path)
        
        # Print the data
        # print("Data in", file_path, ":\n", data)
        
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except Exception as e:
        print("An error occurred:", e)
    
    return data

# Example usage
# file_path = "/SSD/p76111262/visual_embedding/DDoS_HOIC.npy"  # Replace with the path to your .npy file
file_path = "/SSD/p76111262/label_embedding/BruteForce-Web.npy" 
n = view_npy_file(file_path)
# print(n.shape)
print(n)