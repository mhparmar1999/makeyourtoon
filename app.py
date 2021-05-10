####### CORE IMPORTS ###############
import cv2
import tensorflow as tf 
import network
import guided_filter
from cv2 import dnn_superres
import streamlit as st
from PIL import Image,ImageEnhance
import time
import numpy as np
import os
import os.path
import random
import string
from datetime import datetime
import base64
from moviepy.editor import *
import threading




    

@st.cache(suppress_st_warning=True)

# LOADING IMAGE
def loadim(img_file):
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img


# CARTOONIZING FUNCTIONS
def cartoonize(image_name):
	# print(img_file.name)
	model_path = "saved_models"
	input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
	network_out = network.unet_generator(input_photo)
	final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

	all_vars = tf.trainable_variables()
	gene_vars = [var for var in all_vars if 'generator' in var.name]
	saver = tf.train.Saver(var_list=gene_vars)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	sess.run(tf.global_variables_initializer())
	saver.restore(sess, tf.train.latest_checkpoint(model_path))


	try:
		image = cv2.imread("test_images/"+image_name)
		image = resize_crop(image)
		batch_image = image.astype(np.float32)/127.5 - 1
		batch_image = np.expand_dims(batch_image, axis=0)
		output = sess.run(final_out, feed_dict={input_photo: batch_image})
		output = (np.squeeze(output)+1)*127.5
		output = np.clip(output, 0, 255).astype(np.uint8)
		cv2.imwrite("cartoonized_images/"+image_name, output)
		return True
	except Exception as e:
		print(e)
		return False

def cartoonize_video(video_name):
	# print(img_file.name)
	model_path = "saved_models"
	input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
	network_out = network.unet_generator(input_photo)
	final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

	all_vars = tf.trainable_variables()
	gene_vars = [var for var in all_vars if 'generator' in var.name]
	saver = tf.train.Saver(var_list=gene_vars)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	sess.run(tf.global_variables_initializer())
	saver.restore(sess, tf.train.latest_checkpoint(model_path))
	length = len(os.listdir("test_videos/"+video_name+"/bitmap/"))
	i=0
	for file in os.listdir("test_videos/"+video_name+"/bitmap/"):
		try:
			i+=1
			print("Frame {}/{}".format(i,length))
			image = cv2.imread("test_videos/"+video_name+"/bitmap/"+file)
			image = resize_crop(image)
			batch_image = image.astype(np.float32)/127.5 - 1
			batch_image = np.expand_dims(batch_image, axis=0)
			output = sess.run(final_out, feed_dict={input_photo: batch_image})
			output = (np.squeeze(output)+1)*127.5
			output = np.clip(output, 0, 255).astype(np.uint8)
			cv2.imwrite("cartoonized_videos/"+video_name+"/bitmap/"+file, output)
		except Exception as e:
			print(e)
			return False

	return True

def cartoon_save(name,img):
    fn ='test_images/'+name
    cv2.imwrite(fn,img)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}" style="text_decoration:none;font_weight:bold;border:2px solid black; color:blue;padding:5px;" >Download {file_label}</a>'
    return href

def extract_audio(video_file):
	my_clip = VideoFileClip(r"test_videos/"+video_file+"/"+video_file)
	my_clip.audio.write_audiofile(r"test_videos/"+video_file+"/result.mp3")
	return True

def merge_audio(video_file):
	video = VideoFileClip(r"cartoonized_videos/"+video_file+"/temp.mp4")
	video.write_videofile("cartoonized_videos/"+video_file+"/output.mp4", audio=r"test_videos/"+video_file+"/result.mp3")
	return True

def my_thread(video_file):
	t1 = threading.Thread(target=cartoonize_video,args=(video_file,))
	t2 = threading.Thread(target=extract_audio,args=(video_file,))
	t1.start()
	t2.start()
	t1.join()
	t2.join()
	return True


#Helper Functions
def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image


def crop(img,x,y):
    y1,x1 = x
    y2,x2 = y
    crop_img = img[y1:y2,x1:x2]
    return crop_img
       
def resize(img,shape):
    img_resized = cv2.resize(img, shape,interpolation = cv2.INTER_CUBIC) 
    return img_resized

def rotate(image,x):
    (h1, w1) = image.shape[:2]
    center = (w1 / 2, h1 / 2)
    Matrix = cv2.getRotationMatrix2D(center, -90 * x, 1.0)
    rotated_image = cv2.warpAffine(image, Matrix, (w1, h1))
    return rotated_image

def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def save(typ,img):
    now = datetime.now()
    time_stamp = now.strftime("%m_%d_%H_%M_%S") 
    fn ='other_images/'+typ+time_stamp+'.png'
    cv2.imwrite(fn,img)

# UPSAMPLING IMAGES
def upScaleEDSR(image):
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()
    path = "saved_models/EDSR_x3.pb"
    sr.readModel(path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 3)
    result = sr.upsample(image)
    return result

def upScaleFSRCNN(image):
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    path = "saved_models/FSRCNN_x3.pb"
    sr.readModel(path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("fsrcnn", 3)
    result = sr.upsample(image)
    return result

def BilinearUpscaling(img,factor = 2):
    img_resized = cv2.resize(img,(0,0), fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    return img_resized
    


# DENOISE FUNCTION
def denoise(img):
    # denoising of image saving it into dst image 
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    return dst
    

    


def main():
    header = """
    <div style='margin-top:auto;padding:8px;font-size:45px;font-weight:bold;text-align:center;font-family:Helvetica, sans-serif'>
    MAKEYOUR <span style='background-color:black;color:white;padding-left:10px; padding-right:20px;'>TOON</span>
    </div>
    <!--
    <div style='margin-top:auto;padding:5px;font-size:20px;;text-align:center;font-family:calibri'>
    App for Cartoonizing Image, GIFs and Videos, AI based Image super resolution for image Upscaling,Denoising and Editing
    </div> -->

    """
    st.set_page_config("MakeYourToon")
    st.markdown(header,unsafe_allow_html=True)
    st.sidebar.subheader("Navigation Links")
    activities = ['Home','Cartoonize Image','Cartoonize Video','Cartoonize GIF','Super Sampling','Denoise','Resize','Rotate','Crop','About']
    sidebar_choice = st.sidebar.radio('Select a feature',activities)

    st.markdown("<hr style='background-color:red'></hr>",unsafe_allow_html=True)

    if sidebar_choice == 'Home':
    	st.image("static/home_title.png",use_column_width=True)
    	st.markdown("<hr style='background-color:gray'>",unsafe_allow_html=True)
    	# WHY MAKEYOURTOON
    	
    	c1,c2 = st.beta_columns((3,1))
    	with c2:
    		st.markdown("<br><br><br>",unsafe_allow_html=True)
	    	st.image("static/questionmark.jpg",use_column_width=True)
    	with c1:
    		html = '''
    	<div style="margin:0px;padding:30px;">
    	<h3 style="margin:0px;font-weight:bold;font-size:35px;font-family:Verdana;color: #cc0099">Why MakeYourToon?</h3>

    	<ul style="padding:10px;font-weight:bold;color:red;font-size:20px;font-family:Verdana">
    	<li>No Need To Download Extra Software</li>
    	<li>Easily Accessible From Anywhere</li>
    	<li>Simple UI</li>
    	<li>Free of cost</li>
    	<li>Audio is not lost while cartoonizing videos</li>
    	</ul>
    	</div>
    	'''
	    	st.markdown(html,unsafe_allow_html=True)

    	# FEATURES
    	st.markdown("<hr style='background-color:gray'>",unsafe_allow_html=True)
    	st.image("static/features.png",use_column_width=True)
    	st.markdown("<hr style='background-color:gray'>",unsafe_allow_html=True)

    	#MEAT OUR TEAM
    	st.markdown("<div style='text-align:center;color:white;background-color:#ff0066;font-size:35px'>OUR TEAM</div><br>",unsafe_allow_html=True)   	
    	col = st.beta_columns(3)
    	with col[0]:
    		st.image("static/mukesh.jpg",caption="Mukesh Parmar")
    	with col[1]:
    		st.image("static/mukesh.jpg",caption="Vedant Prabhu")
    	with col[2]:
    		st.image("static/mukesh.jpg",caption="Omkar Gorhe")

    elif sidebar_choice == 'Cartoonize Image':
    	st.header("Cartoonize Image")
    	img_file = st.file_uploader("Browse your Image or Drag and Drop",type = ['jpg','jpeg','png'])
    	if img_file is None:
    		st.error("Please Select An Image")
    	else:
    		main_img_cv = loadim(img_file)
	    	img_file.name = str(random.randrange(1,100000000000))+".jpg"
	    	
	    	st.image(main_img_cv,channels = 'BGR',use_column_width = True)
	    	if st.button("Click Here To Cartoonize Your Image"):
	    		cartoon_save(img_file.name,main_img_cv)
	    		st.info("File Name Changed To "+img_file.name)
	    		with st.spinner("Cartoonizing Image Please Wait"):
        			ans = cartoonize(img_file.name)
        			if ans:
	        			st.success("Image Cartoonized Successfully")
	        			st.image("cartoonized_images/"+img_file.name)
	        			st.markdown(get_binary_file_downloader_html("cartoonized_images/"+img_file.name), unsafe_allow_html=True)

	        		else:
	        			st.error("Something went wrong.")
	       

    elif sidebar_choice == 'Cartoonize Video':
    	#Current Frame Settings is set to 5
    	st.header("Cartoonize Video")

    	video_file = st.file_uploader("Browse your Video or Drag and Drop (only mp4 supported)",type=['mp4'])
    	if video_file is None:
    		st.error("Select a video")
    	else:

	    	flag=0

	    	if st.button("Click Here To Cartoonize Your Video"):
	    		video_file.name = str(random.randrange(1,100000000000))+".mp4"
	    		if not os.path.isdir("test_videos/"+video_file.name):
		    		os.mkdir("test_videos/"+video_file.name)
		    		os.mkdir("test_videos/"+video_file.name+"/bitmap")
		    		os.mkdir("cartoonized_videos/"+video_file.name)
		    		os.mkdir("cartoonized_videos/"+video_file.name+"/bitmap")
	    		with open(os.path.join("test_videos/"+video_file.name+"/",video_file.name),"wb") as f:
	    			f.write(video_file.getbuffer())
	    		st.info("File Name Changed To "+video_file.name)
		    	# Converting Video into Bitmap
		    	with st.spinner("Cartoonizing in progres...."):
		    		with st.beta_expander("See Details"):
			    		with st.spinner("STEP 1: Converting video into Bitmap Frames....please wait this might take long"):
				    		if not os.system("ffmpeg -i test_videos/"+video_file.name+"/"+video_file.name+" -r 5 test_videos/"+video_file.name+"/bitmap/img%03d.bmp"):
				    			st.success("STEP 1: Video to Bitmap Frame Conversion Successfull")
				    			# Generate Cartoon of Bitmap
				    			with st.spinner("STEP 2: Extracting Audio and Cartoonizing Frames....please wait this might take long"):
						        	if my_thread(video_file.name):
						        		st.success("STEP 2: Cartoonizing Frames Successfull")
						        		# Transform Bitmap Back Into Video
						        		with st.spinner("STEP 3: Forming back your cartoon video....please wait this might take long"):
								        	if not os.system("ffmpeg -r 5 -i cartoonized_videos/"+video_file.name+"/bitmap/img%03d.bmp -c:v libx264 -pix_fmt yuv420p cartoonized_videos/"+video_file.name+"/temp.mp4"):
								        		if merge_audio(video_file.name):
									        		st.success("STEP 3: Conversion Successfull")
									        		flag=1
									        	else:
									        		st.error("Error Merging Audio")
								        	else:
								        		st.error("Error Creating Video")
						        	else:
						        		st.error("Cartoonizing Failed")
			    			else:
			    				st.error("Conversion Failed..refresh the page and try again")
			    	if flag==0:
			    		st.error("Something went wrong...refresh the page and try again.")
			    	else:
				    	st.success("Cartoonization Successfull")
				    	video_file = open("cartoonized_videos/"+video_file.name+"/output.mp4", 'rb')
				    	video_bytes = video_file.read()
				    	st.video(video_bytes)
				    	st.info("File Name : "+video_file.name)

    elif sidebar_choice == 'Cartoonize GIF':
    	st.header("Cartoonize GIF")

    	video_file = st.file_uploader("Browse your GIF or Drag and Drop",type=['.gif'])
    	if video_file is None:
    		st.error("Select a GIF file")
    	else:
    	

    		flag=0

	    	if st.button("Click Here To Cartoonize Your GIF"):
	    		video_file.name = str(random.randrange(1,100000000000))+".gif"
	    		if not os.path.isdir("test_videos/"+video_file.name):
		    		os.mkdir("test_videos/"+video_file.name)
		    		os.mkdir("test_videos/"+video_file.name+"/bitmap")
		    		os.mkdir("cartoonized_videos/"+video_file.name)
		    		os.mkdir("cartoonized_videos/"+video_file.name+"/bitmap")
    			with open(os.path.join("test_videos/"+video_file.name+"/",video_file.name),"wb") as f:
    				f.write(video_file.getbuffer())
    			st.info("File Name Changed To "+video_file.name)
		    	# Converting Video into Bitmap
		    	with st.spinner("Cartoonizing in progres...."):
		    		with st.beta_expander("See Details"):
			    		with st.spinner("STEP 1: Converting GIF into Bitmap Frames....please wait this might take long"):
				    		if not os.system("ffmpeg -i test_videos/"+video_file.name+"/"+video_file.name+" -r 5 test_videos/"+video_file.name+"/bitmap/img%03d.bmp"):
				    			st.success("STEP 1: GIF to Bitmap Frame Conversion Successfull")
				    			# Generate Cartoon of Bitmap
				    			with st.spinner("STEP 2: Cartoonizing Frames....please wait this might take long"):
						        	if cartoonize_video(video_file.name):
						        		st.success("STEP 2: Cartoonizing Frames Successfull")
						        		# Transform Bitmap Back Into Video
						        		with st.spinner("STEP 3: Forming back your cartoon gif....please wait this might take long"):
								        	if not os.system("ffmpeg -r 5 -i cartoonized_videos/"+video_file.name+"/bitmap/img%03d.bmp -c:v libx264 -pix_fmt yuv420p cartoonized_videos/"+video_file.name+"/output.mp4"):
								        		os.rename("cartoonized_videos/"+video_file.name+"/output.mp4","cartoonized_videos/"+video_file.name+"/output.gif")
								        		st.success("STEP 3: Conversion Successfull")
								        		flag=1
								        	else:
								        		st.error("Error Creating Video")
						        	else:
						        		st.error("Cartoonizing Failed")
			    			else:
			    				st.error("Conversion Failed..refresh the page and try again")
			    	if flag==0:
			    		st.error("Something went wrong...refresh the page and try again.")
			    	else:
				    	st.success("Cartoonization Successfull")
				    	video_file = open("cartoonized_videos/"+video_file.name+"/output.gif", 'rb')
				    	video_bytes = video_file.read()
				    	st.video(video_bytes)
				    	st.info("File Name : "+video_file.name)

    elif sidebar_choice == 'Super Sampling':
    	st.header("Image Super Sampling")
    	img_file = st.file_uploader("Browse your Image or Drag and Drop",type = ['jpg','jpeg','png'])
    	if img_file is None:
    		st.error("Please Select An Image")
    	else:
	    	main_img_cv = loadim(img_file)
	    	st.image(main_img_cv,channels = 'BGR',use_column_width = True)
	    	upscale_type = st.selectbox("Select Upsampling Method",['EDSR','FSRCNN','Bilinear'])
	    	if upscale_type == 'EDSR':
	    		st.info('The Enhanced deep residual super sampling method is based on a larger model and will take anywhere from 2 - 8 mins to complete upsampling.')
	    		if st.button("Apply"):
	    			if main_img_cv.shape[0] > 1080 or main_img_cv.shape[1] > 1080:
	    				st.error('Image to Large for upsampling. This image is already above 1080 pixels wide')
	    			else:
	    				with st.spinner("Upscaling... This may take long."):
	    					upscaled_image_cv = upScaleEDSR(main_img_cv)
	    					st.image(upscaled_image_cv, channels="BGR")
	    					st.success("Image has been Upscaled")
		    	if st.button("Apply and Save"):
	                	if main_img_cv.shape[0] > 1080 or main_img_cv.shape[1] > 1080:
	                		st.error('Image too Large for upsampling. This image is already above 1080 pixels wide')
	                	else:
	                		with st.spinner("Upscaling. Please Wait. This may take long."):
	                			upscaled_image_cv = upScaleEDSR(main_img_cv)
	                			save('EDSR_',upscaled_image_cv)

	    	elif upscale_type == 'Bilinear':
	    		st.info('Bilinear is a non AI Image upscaling algorithm based on Bilinear interpolation.')
	    		bilinear_factor = st.slider('Select Upscale Factor',2,4)
	    		if st.button("Apply"):
	    			upscaled_image_cv = BilinearUpscaling(main_img_cv,bilinear_factor)
	    			st.image(upscaled_image_cv, channels="BGR")
	    			st.success("Image has been Upscaled with Bilinear Interpolation")
	    		if st.button("Apply and Save"):
	    			upscaled_image_cv = BilinearUpscaling(main_img_cv,bilinear_factor)
	    			st.image(upscaled_image_cv, channels="BGR")
	    			st.success("Image has been Upscaled with Bilinear Interpolation")
	    			save('bilinear',upscaled_image_cv)
	    			st.success("File has been Saved")

	    	elif upscale_type == 'FSRCNN':
	    		st.info('FSRCNN is small fast model of quickly upscaling images with AI. This model does not produce state of the art accuracy however')
	    		if st.button("Apply"):
	    			if main_img_cv.shape[0] > 1080 or main_img_cv.shape[1] > 1080:
	    				st.error('Image to Large for upsampling. This image is already above 1080 pixels wide')
	    			else:
	    				with st.spinner("Upscaling... This may take long."):
	    					upscaled_image_cv = upScaleFSRCNN(main_img_cv)
	    					st.image(upscaled_image_cv, channels="BGR")
	    					st.success("Image has been Upscaled")
		    	if st.button("Apply and Save"):
	                	if main_img_cv.shape[0] > 1080 or main_img_cv.shape[1] > 1080:
	                		st.error('Image too Large for upsampling. This image is already above 1080 pixels wide')
	                	else:
	                		with st.spinner("Upscaling. Please Wait. This may take long."):
	                			upscaled_image_cv = upScaleFSRCNN(main_img_cv)
	                			save('FSRCNN_',upscaled_image_cv)        	
            

    elif sidebar_choice == 'Denoise':
    	st.header("Image Denoise")
    	img_file = st.file_uploader("Browse your Image or Drag and Drop",type = ['jpg','jpeg','png'])
    	if img_file is None:
    		st.error("Please Select An Image")
    	else:
	    	main_img_cv = loadim(img_file)
	    	st.image(main_img_cv,channels = 'BGR',use_column_width = True)
	    	if st.button("Apply"):
	    		with st.spinner("Denoising.. Please Hold On to your seat belts"):
	    			denoise_image_cv = denoise(main_img_cv)
	    			st.image(denoise_image_cv,channels="BGR",use_column_width = True)
	    			st.success("Image was Denoised")

	    	if st.button("Apply and Save"):
	    		with st.spinner("Denoising.. Please Hold On to your seat belts"):
	    			denoise_image_cv = denoise(main_img_cv)
	    			st.image(denoise_image_cv,channels="BGR",use_column_width = True)
	    			save('Denoise_',denoise_image_cv)
	    			st.success("Image was Denoised and saved")

    elif sidebar_choice == 'Resize':
    	st.header("Image Resize")
    	img_file = st.file_uploader("Browse your Image or Drag and Drop",type = ['jpg','jpeg','png'])
    	if img_file is None:
    		st.error("Please Select An Image")
    	else:
	    	main_img_cv = loadim(img_file)
	    	st.image(main_img_cv,channels = 'BGR',use_column_width = True)
	    	st.markdown("Please Enter the Dimentions you would like to resize the image to.")
	    	dim = st.text_input("Enter Dimentions with a comma",'512,512')
	    	if st.button('Apply'):
	    		dim = dim.split(',')
	    		if len(dim)!=2:
	    			st.error("Incorrect Dimentions")
	    		else:
	    			shape = (int(dim[0]),int(dim[1]))
	    			resize_image = resize(main_img_cv,shape)
	    			st.image(resize_image,channels = 'BGR')
	    			st.success("Image was Resized")
	    	if st.button('Apply and save'):
	    		dim = dim.split(',')
	    		if len(dim)!=2:
	    			st.error("Incorrect Dimentions")
	    		else:
	    			shape = (int(dim[0]),int(dim[1]))
	    			resize_image = resize(main_img_cv,shape)
	    			st.image(resize_image,channels = 'BGR')
	    			save('Resize_',resize_image)
	    			st.success("Image was Resized and saved")

    elif sidebar_choice == 'Rotate':
    	st.header("Image Rotating")
    	img_file = st.file_uploader("Browse your Image or Drag and Drop",type = ['jpg','jpeg','png'])
    	if img_file is None:
    		st.error("Please Select An Image")
    	else:
	    	main_img_cv = loadim(img_file)
	    	st.image(main_img_cv,channels = 'BGR',use_column_width = True)
	    	st.info("Enter the ammount of times u want to rotate image to the Right. Select 3 for one left rotation")
	    	r_times = st.slider('Select number of times to rotate image right',1,3)
	    	if st.button('Apply'):
	    		rotated_image = rotate(main_img_cv,r_times)
	    		st.image(rotated_image,channels="BGR",use_column_width = True)
	    		st.success("Image Rotated.")
	    	if st.button('Apply and save'):
	    		rotated_image = rotate(main_img_cv,r_times)
	    		st.image(rotated_image,channels="BGR",use_column_width = True)
	    		save('rotated_',rotated_image)
	    		st.success("Image Rotated and saved.")

    elif sidebar_choice == 'Crop':
    	st.header("Image Cropper")
    	img_file = st.file_uploader("Browse your Image or Drag and Drop",type = ['jpg','jpeg','png'])
    	if img_file is None:
    		st.error("Please Select An Image")
    	else:
	    	main_img_cv = loadim(img_file)
	    	st.image(main_img_cv,channels = 'BGR',use_column_width = True)
	    	st.info(' Enter The location of upper left and bottom right pixels to crop the image ')
	    	st.write('Your Image Dimentions are',main_img_cv.shape)
	    	x = st.text_input('Enter Upper left pixel location','100,100')
	    	y = st.text_input('Enter Bottom Right Pixel Location','300,300')
	    	if st.button('Apply'):
	    		x = x.split(",")
	    		x = (int(x[0]),int(x[1]))
	    		y = y.split(',')
	    		y = (int(y[0]),int(y[1]))
	    		st.write('Cropping to Dimetions',x,y)
	    		crop_image = crop(main_img_cv,x,y)
	    		st.image(crop_image,channels="BGR",use_column_width = True)
	    		st.success("Image Cropped.")
	    	if st.button('Apply and Save'):
	    		x = x.split(",")
	    		x = (int(x[0]),int(x[1]))
	    		y = y.split(',')
	    		y = (int(y[0]),int(y[1]))
	    		st.write('Cropping to Dimetions',x,y)
	    		crop_image = crop(main_img_cv,x,y)
	    		st.image(crop_image,channels="BGR",use_column_width = True)
	    		save('crop_',crop_image)
	    		st.success("Image Cropped and saved.")

    elif sidebar_choice == 'About':
    	st.header("About")

    	css='''
    	<style>
    	.font { 
    	font-family: Arial; 
    	font-size: 16px;
    	font-weight: 300; 
    	line-height: 32px;
    	padding: 5px 10px;
    	text-align: justify;
    	 }
    	</style>
    	'''
    	st.markdown(css,unsafe_allow_html=True)

    	with st.beta_expander("About MakeYourToon"):
    		data = '''
    		MakeYourToon is an AI based application helping to generate Cartoons of provided Image/GIF/Video developed using GAN Model and OPENCV Library for other Image features. Streamlit Framework is used to develop the UI.
    		'''
    		st.markdown("<p class='font'>"+data+"</p>",unsafe_allow_html=True)
    		st.info("This is a final year project for XIE BE Group 21 for academic year 2020-21")

    	with st.beta_expander("Features of MakeYourToon"):
    		html='''
    	<div style = "padding-top:20px">
    	<ul>
    	<li class='font'>Simple UI</li>
    	<li class='font'>Easy to use</li>
    	<li class='font'>High Quality Output</li>
    	<li class='font'>Audio is retained in video cartoonization</li>
    	<li class='font'>Provides other image features like image super sampling, denoising, rotate and crop</li>
    	</ul>
    	</div>
    	'''
    		st.markdown(html,unsafe_allow_html=True)

    	with st.beta_expander("About Streamlit Framework"):
    		data = '''
    		Streamlit turns data scripts into shareable web apps in minutes.All in Python. 
    		All for free.
    		No front‑end experience required.
    		'''
    		st.markdown("<p class='font' style='text-align=center'>"+data+"</p>",unsafe_allow_html=True)
    		st.write("https://docs.streamlit.io/en/stable/")   			


if __name__ == '__main__':
    main()