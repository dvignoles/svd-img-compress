from PIL import Image
import numpy as np

def get_matrices(png_img):
    '''return RGB matrices from png image'''
    img = Image.open(png_img)
    
    mats = {}
    for color,band in {'r':0,'g':1,'b':2}.items():
        mat = np.array(list(img.getdata(band=band)))
        mat.shape = (img.size[1],img.size[0])
        mat = np.matrix(mat)
        mats[color] = mat
    return mats

def svd_matrix(mat):
    '''return the SVD of matrix mat'''
    return np.linalg.svd(mat)

def reconstruct_mat(u,sigma,v,rank):
    '''return matrix from svd decomposition'''
    return np.matrix(u[:, :rank]) * np.diag(sigma[:rank]) * np.matrix(v[:rank, :])

def reconstruct_img(r,g,b,size):
    rgb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b

    img = Image.fromarray(rgb)
    img.save('compressed.png')

def foo(img):
    mats = get_matrices(img)
    
    recon_mats = {}
    for channel,mat in mats.items():
        u,s,vh = svd_matrix(mat)

        recon = reconstruct_mat(u,s,vh,5)
        recon_mats[channel] = recon
    
    reconstruct_img(recon_mats['r'],recon_mats['g'],recon_mats['b'],[2000,2000])

if __name__ == '__main__':
    foo("images/hendrix_final.png")




