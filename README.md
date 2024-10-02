# GPS-PS-Helper
Point + Extended Sources in GPS. Simulated data to Inputs &amp; More: Helper Functions

*model_txt_to_csv_all.py*: convert the text files containing source params to csv files (better readability)

*lat_lon_to_x_y.npy*: reads the csv file, obtain the source locations in lat, long, turn them into pixel coord, save in csv

*fits_to_npy.py*: convert the fits files to npy array and save them

*modify_mask_switch_off.npy*: check through the folder for .csv and mask fits files and randomly switches off 30 percent of the mask with on_off label on csv file. 

*./Example_Patches_JPR/read_fits_check.ipynb*: After running through the previous python files, we visualize some results. 
