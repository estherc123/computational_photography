import numpy as np
import skimage as sk
import skimage.transform
import skimage.io as skio
import sys
import math

# name of the input file
imname = 'data/cathedral.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)
im = sk.img_as_float(im)

# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(np.int)



# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

#metric: Sum of Squared Differences (SSD)
def SDD(u,v):
    return np.sum((u - v) ** 2)

#metric: Normalized Cross-Correlation (NCC)
def NCC(img1,img2):
    norm_image1 = img1 / np.linalg.norm(img1)
    norm_image2 = img2 / np.linalg.norm(img2)
    return np.sum(norm_image1 * norm_image2)

#this method is only intended for smaller images, as it is slow for large search windows.
def test(u,v):
    img1 = u
    img2 = v
    min_val = SDD(img1, img2)
    # min_val = np.sum((u - v) ** 2)
    result = img1
    for i in range(-20, 20):
        for j in range(-20, 20):
            first = np.roll(np.roll(img1,i,0),j,1)
            c = SDD(first, img2)
            # c = np.sum((img1 - v) ** 2)
            if c < min_val:
                min_val = c
                result = first
    return result

#align two images given their image pyramids.
def align(pym_u, pym_v):
    i,j = best_offset(pym_u, pym_v)
    print("offset:", i,j)
    print("aligned!")
    return np.roll(np.roll(pym_u[0],i,1),j,0)

#crop borders of images to get better alignment results.
def crop_border(img, percentage):
	x_crop = int(percentage*img.shape[0]/2)
	y_crop = int(percentage*img.shape[1]/2)
	return img[x_crop:img.shape[0]-x_crop,y_crop:img.shape[1]-y_crop]

#create an image pyramid from a given image.
def img_pyramid(img):
    pyramid = []
    pyramid.append(np.asarray(img))
	# calculate number of levels with respect to width and height.
    width_dim = int(math.log(img.shape[0],2))
    height_dim = int(math.log(img.shape[1],2))

	#the number of level for this pyramid would be the smallest of the two.
    level = min(width_dim,height_dim)
    for i in range(1,level+1):
    	pyramid.append(sk.transform.rescale(pyramid[i-1],1/2))
    return pyramid

#given two image pyramids, it gives the best offset between two.
def best_offset(pym1, pym2):
    num_level = len(pym1)
    offset_i,offset_j = 0,0

    #going down the pyramid, with coarest scale as the top.
    for i in range(num_level):
        img1 = pym1[num_level-i-1]
        img2 = pym2[num_level-i-1]

        min_val = sys.float_info.max
        best_i,best_j = 0,0
        offset_i *= 2
        offset_j *= 2

        i_window = int(min(10,img1.shape[0]/2))
        j_window = int(min(10,img1.shape[1]/2))

        img1 = np.roll(np.roll(img1, offset_i,1), offset_j, 0)
        img1 = crop_border(img1, 0.1)
        second = crop_border(img2, 0.1)
        for i in range(-i_window, i_window):
            for j in range(-j_window, j_window):
                first = np.roll(np.roll(img1,i,1),j,0)
                c = SDD(first, second)
                if c < min_val:
                    min_val = c
                    best_i,best_j = i,j

        offset_i, offset_j = offset_i+best_i, offset_j+best_j
    return offset_i, offset_j

if __name__ == '__main__':
    #create image pyramids for three channels
    pym_r = img_pyramid(r)
    pym_g = img_pyramid(g)
    pym_b = img_pyramid(b)

    #align the channels. replace align with test for alignment without image pyramid.
    ab = test(b, g)
    ar = test(r, g)
    #ab = align(pym_b, pym_g)
    #ar = align(pym_r, pym_g)

    # create a color image
    im_out = np.dstack([ar, g, ab])

    # save the image
    fname = 'out/lantern.jpg'
    skio.imsave(fname, im_out)

    # display the image
    skio.imshow(im_out)
    skio.show()
