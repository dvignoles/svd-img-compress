from PIL import Image, ImageDraw
import numpy as np

def get_matrices(img):
    '''return RGB matrices from png image'''
    
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

def svd_matrix_all(mats):
    '''return the SVD matrix for all rgb channel as dic'''
    channel_svd = {}
    for channel,mat in mats.items():
        u,s,vh = svd_matrix(mat)
        channel_svd[channel] = {'u':u,'s':s,'vh':vh}
    return channel_svd

def reconstruct_mat(u,sigma,v,rank):
    '''return matrix from svd decomposition'''
    return np.matrix(u[:, :rank]) * np.diag(sigma[:rank]) * np.matrix(v[:rank, :])

def reconstruct_img(r,g,b,size):
    rgb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b

    return Image.fromarray(rgb)

def svd_compress(svd_rgb,rank):
    '''construct approximate matrixes from rgb matrix svd decompositions'''
    recon_mats = {}

    #  for channel,mat in mats.items():
    #      u,s,vh = svd_matrix(mat) #Can optimize here (doesn't need to loop for every rank)

    for channel,svd in svd_rgb.items():
        u = svd['u']
        s = svd['s']
        vh = svd['vh']
        recon = reconstruct_mat(u,s,vh,rank)
        recon_mats[channel] = recon
    
    #return reconstruct_img(recon_mats['r'],recon_mats['g'],recon_mats['b'],[size[1],size[0]])
    return recon_mats

def svd_img_save(name,mats,rank):
    '''save compressed image from rgb matrices'''
    size = mats['r'].shape
    svd_rgb= svd_matrix_all(mats)

    recon_mats = svd_compress(svd_rgb,rank)
    img_comp = reconstruct_img(recon_mats['r'],recon_mats['g'],recon_mats['b'],[size[1],size[0]]) 
    img_comp.save('rank{}_{}'.format(str(rank),name+'.png'))


def make_gif(name,mats,top_rank,bottom_rank=1,step=5):
    '''save gif showing rank progression of compressed images via svd'''
    size = mats['r'].shape
    images = []
    svd_rgb= svd_matrix_all(mats)
    for i in range(bottom_rank,top_rank + 1,step):
        #if file not in dir
        recon_mats = svd_compress(svd_rgb,i)
        img_comp = reconstruct_img(recon_mats['r'],recon_mats['g'],recon_mats['b'],[size[1],size[0]]) 
        images.append(img_comp)
        img_comp.save('rank{}_{}'.format(str(rank),name+'.png'))
        #else read from dir
        
        print('rank:',i,'completed')
    
    images[0].save(name + '_{}.gif'.format(str(top_rank)),
               save_all=True,
               append_images=images[1:],
               duration=100,
               loop=0)


if __name__ == '__main__':
    png_img = "images/hendrix_final.png"

    img = Image.open(png_img)
    mats = get_matrices(img)

    #make_gif('hendrix',mats,200)

    svd_img_save('hendrix',mats,500)




