"""
Task: 
Author: Sandaru Jayawardana
"""
import numpy as np
import os

FILE_LOCATION = os.getcwd()

def read_file(file_name = "latent.npy"):
    with np.load(FILE_LOCATION + file_name) as latent_file:
        latents = latent_file['a']

def pre_process_celebA(latent_file_name = "latent.npy", detail_file = "celebA_latents/indexes"):
    print(FILE_LOCATION + "/" + detail_file)
    index_file = open(FILE_LOCATION + "/" + detail_file, "rb")
    index_details = index_file.readline()

    print(index_details)  


pre_process_celebA()