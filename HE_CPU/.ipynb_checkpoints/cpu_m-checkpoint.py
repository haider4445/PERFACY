from dataclasses import dataclass
from typing import Any, Dict
from time import time, process_time, sleep
import pickle
import tenseal as ts
import torch
import sys
#from pympler import asizeof
import statistics
@dataclass
class Results:
    """Class for keeping track of an item in inventory."""
    time: float
    value: Any
    shapes: Any


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def encrypt(weights, context):
    res = {}
    shapes = {}
    # Do encryption
    for key in weights.keys():
        v1 = weights[key]
        shapes[key] = v1.shape
        v1 = v1.view(-1)
        if len(v1) > 8192//2:
            vals = chunks(v1, 8192//2)
            broken = []
            for chunk in vals:
                broken.append(ts.ckks_vector(context, chunk))
            res[key] = broken
        else:
            res[key]= ts.ckks_vector(context, v1)
    return Results(0, res, shapes)

def decrypt(weights, shapes:Dict):
    res = {}
    # Do deencryption
    for key in weights:
        if isinstance(weights[key], list):
            lst = []
            for val in weights[key]:
                lst.extend(val.decrypt())
            res[key] = torch.Tensor(lst).view(shapes[key])
                
        else:
            res[key] = torch.Tensor(weights[key].decrypt())
    return Results(0, res, None)

def write_model(fname: str, model: Dict[str, Any]) -> int:
    with open(fname, "wb") as fptr:
        with io.BytesIO() as buff:
            pickle.dump(model, buff)
            buff.seek(0)
            size = buff.getbuffer().nbytes
            fptr.write((size).to_bytes(32, byteorder="big", signed=False))
            buff.seek(0)
            fptr.write(buff.getbuffer())
    return 32 + size

def read_model(fname)->Dict[str, Any]:
    with open(fname, "rb") as fptr:
        fptr.seek(32)
        data = pickle.load(fptr)
    return data

def get_model():
    return read_model("./data.pickle")

x=get_model()

sleep(100)
bit_list = [[60, 60, 60]]
for bits in bit_list:
    for _ in range(10):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=bits
          )
 
        context.generate_galois_keys()
        context.global_scale = 2**20
        x=get_model()
        sleep(2)
        res = encrypt(x, context) 
        sleep(2)
        res = decrypt(res.value, res.shapes)
        sleep(5)
    sleep(10)
    
