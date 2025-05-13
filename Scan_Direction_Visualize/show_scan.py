import torch


len = 2
cube = [i for i in range(len**3)]

class ShowCubeSeq:
    def __init__(self,len):
        self.len = len
        self.cube_seq = [i for i in range(len**3)]
    def show_unidirectional(self):
        print("show unidirectional!")
        print(f"foward: {self.cube_seq}")
    def show_bidirectional(self):
        print("show bidirectional!")
        print(f"foward: {self.cube_seq}")
        print(f"backward: {self.cube_seq[::-1]}")# [start:end:step]
    def show_tri_oriented(self):
        nslices = self.len
        chunk_len = self.len**3//nslices # 每slice有幾pixel
        inter_slice_seq = []
        
        for j in range(chunk_len):
            for i in range(nslices):
                element_index = i*chunk_len+j
                inter_slice_seq.append(self.cube_seq[element_index])
        
        print("show tri-oriented!")
        print(f"foward: {self.cube_seq}")
        print(f"backward: {self.cube_seq[::-1]}")
        print(f"inter-slice: {inter_slice_seq}")



if __name__ == "__main__":
    show_cube = ShowCubeSeq(len=2)
    show_cube.show_unidirectional()
    show_cube.show_bidirectional()
    show_cube.show_tri_oriented()
    
    
