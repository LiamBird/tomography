import matplotlib.pyplot as plt
import numpy as np
import os
import glob

from ipywidgets import interact, IntSlider, FloatSlider

from mpl_point_clicker import clicker

import matplotlib.patches as mpatch

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from tqdm import notebook

from random import randint


class ROIGridSelect(object):
    def __init__(self, grid_size, img_stack, roi_x, roi_y, roi_width, roi_height, 
                 save_path=None, reload=True, material_labels=None):
        """
        Subclass for selecting different phases on image stack using mpl_clicker
        
        Inputs from user
        ----------
        grid_size: (int)
            the dimensions of each grid square in pixels (ideally an integer multiple of the total image size)
        save_path: (str)
            directory for saving phase labels corresponding to grid_size (directory only, do not include filename)
        reload: (bool) default = True
            reload from save_path 
        material_labels: (dict) default = None
            if None, material_labels = {"aluminium": 0,
                           "electrode_aluminium": 1,
                           "electrode": 2,
                           "electrode_separator": 3,
                           "separator": 4,}
                           
        Inputs determined by outer PhaseSelect class
        ----------
        img_stack
        roi_x
        roi_y
        roi_width
        roi_height
        
        Methods:
        ----------
        select_positions(idx):
            shows image at position idx in image stack with overlaid grid for selections
        get_positions():
            saves user selections from select_positions
        train_model(labels_to_fit):
            trains the NBGaussian model using the specified number of labels_to_fit        
        test_model():
            tests the NBGaussian model using specified/ all remaining labels not used in train_model        
        apply_model_roi():
            applies the model trained using train_model to the data **for the ROI only** (see PhaseSelect.apply_model for application to whole area)
        view_reconstruction():
            interactive slider view of the reconstruction produced by apply_model_roi
        """
        self.grid_size = grid_size
        self.img_stack = img_stack
        
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height   
        
        x_extent = int(self.roi_width/self.grid_size)
        y_extent = int(self.roi_height/self.grid_size)
        
        roi = img_stack[:,
                self.roi_x:self.roi_x+self.roi_width,
                self.roi_y:self.roi_y+self.roi_height]

        square_arr = []
        for idx in range(roi.shape[0]):
            slice_list = []

            for x in range(x_extent):
                for y in range(y_extent):
                    slice_list.append(roi[idx, x*grid_size:(x+1)*grid_size, y*grid_size:(y+1)*grid_size])

            slice_arr = np.array(slice_list).reshape((x_extent, y_extent, grid_size, grid_size))

            square_arr.append(slice_arr)

        square_arr = np.array(square_arr)
        
        markers = ["v", "o", "*", "s", "^"]
        if material_labels==None:
            self.material_labels = {"aluminium": 0,
                           "electrode_aluminium": 1,
                           "electrode": 2,
                           "electrode_separator": 3,
                           "separator": 4,}
        else:
            self.material_labels = material_labels
            
        self.markers = [markers[m] for m in range(len(self.material_labels))]
        self.marker_colors = ["white"]*len(self.material_labels)
        self.roi = roi
        self.square_arr = square_arr
        self.x_extent = x_extent
        self.y_extent = y_extent
        
        if save_path == None:
            self.labels = {}
            print(save_path+" no path")
        else:
            if os.path.isdir(save_path) == False:
                os.makedirs(save_path)
                print("making save path")
            self.save_path = os.path.join(save_path)
            
            print(save_path)
            if reload == True:
                try:
                    self.labels = np.load(os.path.join(save_path,
                                                       "grid_{}_positions.npy".format(self.grid_size)),
                                          allow_pickle=True).item()
                    self.labelled_slices = np.sort([*self.labels.keys()])
                    self.proportion_labelled = len(self.labels)/self.img_stack.shape[0]

                except:
                    self.labels = {}
            
    def select_positions(self, idx, figsize=(7, 6), 
                 marker_size=10, marker_color="firebrick"):
        self.marker_colors = ["firebrick"]*len(self.material_labels)

        self.idx = idx
        self._marker_size = marker_size
        self._marker_color = marker_color
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.roi[idx], cmap="Greys_r")
        for x in range(self.x_extent):
            ax.axvline(x*self.grid_size, color="k")
        for y in range(self.y_extent):
            ax.axhline(y*self.grid_size, color="k")
        ax.set_xticks([])
        ax.set_yticks([])
        self.klicker = clicker(
                   ax,
                   [*self.material_labels.keys()],
                   markers=self.markers,
                     colors=self.marker_colors,
            markersize=marker_size
                )
        plt.tight_layout()
            
    def get_positions(self, dominant_material="electrode"):
        self.positions_clicker = dict([(keys, self.klicker.get_positions()[keys]) for keys in self.material_labels.keys()])
        container = np.full((self.x_extent, self.y_extent), self.material_labels[dominant_material])

        for keys, values in self.positions_clicker.items():
            for x, y in np.array(values/self.grid_size, dtype=int):
                container[y, x] = self.material_labels[keys]
                
        self.labels.update([(self.idx, container)])
        self.labelled_slices = np.sort([*self.labels.keys()])
        self.proportion_labelled = len(self.labels)/self.img_stack.shape[0]
        
        if "save_path" in vars(self):
            np.save(os.path.join(self.save_path, "grid_{}_positions.npy".format(self.grid_size)),
                    self.labels,
                    allow_pickle=True)
        
    def train_model(self, labels_to_fit, seed=100):
        if type(labels_to_fit) == list:
            labels_to_fit = [i for i in labels_to_fit if i in self.labelled_slices]
        
        elif type(labels_to_fit)==int:
            from random import randint, seed
            seed(seed)
            labels_list = []
            while len(labels_list) < labels_to_fit:
                new_number = randint(1, len(self.labels)-1)
                if new_number not in labels_list:
                    labels_list.append(new_number)
            labels_to_fit = [self.labelled_slices[i] for i in labels_list]
        self.labels_to_fit = labels_to_fit
        
        square_list = []
        square_labels = []
        for labelled_slice in labels_to_fit:
            for xi in range(self.x_extent):
                for yi in range(self.y_extent):
                    square_list.append(self.square_arr[labelled_slice, xi, yi, :, :].flatten())
                    square_labels.append(self.labels[labelled_slice][xi, yi])
                    
        self.X_train = np.array(square_list)
        self.Y_train = np.array(square_labels)
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.Y_train)
        
    def test_model(self, labels_to_test=None, seed=100):
        test_list = []
        test_label = []
        if type(labels_to_test) == int:
            from random import randint, seed
            seed(seed)
            labels_list = []
            while len(labels_list) < labels_to_test:
                new_number = randint(1, len(self.labels)-1)
                if new_number not in labels_list:
                    if new_number not in self.labels_to_fit:
                        if new_number in self.labelled_slices:
                            labels_list.append(new_number)        
            labels_to_test = labels_list
            
        else:
            labels_to_test = [i for i in self.labelled_slices if i not in self.labels_to_fit]

        for labelled_slice in labels_to_test:
            for xi in range(self.x_extent):
                for yi in range(self.y_extent):
                    test_list.append(self.square_arr[labelled_slice, xi, yi, :, :].flatten())
                    test_label.append(self.labels[labelled_slice][xi, yi])
                    
        self.X_test = np.array(test_list)
        self.Y_test = np.array(test_label)
        self.Y_test_predict = self.model.predict(self.X_test)
        
        self.accuracy = accuracy_score(self.Y_test, self.Y_test_predict)
        self.confusion = confusion_matrix(self.Y_test, self.Y_test_predict)
        
    def apply_model_roi(self):
        predict_only_list = []
        labels_to_predict = range(self.roi.shape[0])

        for labelled_slice in labels_to_predict:
            for xi in range(self.x_extent):
                for yi in range(self.y_extent):
                    predict_only_list.append(self.square_arr[labelled_slice, xi, yi, :, :].flatten())
        
        self.X_predict = np.array(predict_only_list)
        self.Y_predict = self.model.predict(self.X_predict)
        
        predict_reshape = self.Y_predict.reshape(len(labels_to_predict), 
                                                 int(self.Y_predict.shape[0]/len(labels_to_predict)))
        predict_reshape = np.array([arr.reshape(self.x_extent, self.y_extent) for arr in predict_reshape])
        
        reconstruct = np.full((self.square_arr.shape[0],
                               self.x_extent*self.grid_size, self.y_extent*self.grid_size), np.nan)

        for nlabel, label in enumerate(labels_to_predict):
            for xi in range(self.x_extent):
                for yi in range(self.y_extent):
                    if predict_reshape[nlabel, xi, yi] == self.material_labels["electrode"]:
                        img = self.square_arr[label, xi, yi]
                        container = np.full((img.shape), np.nan)
                        container=img
                        reconstruct[label, 
                                    xi*self.grid_size:(xi+1)*self.grid_size,
                                    yi*self.grid_size:(yi+1)*self.grid_size] = container
        
        self.reconstruct = reconstruct
        self.predict_reshape = predict_reshape
        
    def view_reconstruction(self, cmap="Greys_r"):
        f, (axes) = plt.subplots(1, 2)
        def update(idx):
            for ax in axes:
                ax.cla()
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0].imshow(self.roi[idx], cmap=cmap)
            axes[1].imshow(self.reconstruct[idx], cmap=cmap)
        interact(update, idx=IntSlider(min=0, max=self.reconstruct.shape[0]-1, step=1))
        
class PhaseSelect(object):
    def __init__(self, image_path, scale=1.625, radius=1500,
                 material_labels=None):
        all_images = glob.glob(os.path.join(image_path, "*.tif"))
        img_stack_full = np.array([plt.imread(img) for img in notebook.tqdm(all_images)])
        square_size = int((radius/scale)*np.cos(np.pi/4))
        min_square = int(img_stack_full.shape[1]/2)-square_size
        max_square = int(img_stack_full.shape[1]/2)+square_size
        
        self.img_stack = img_stack_full[:, min_square:max_square, min_square:max_square]
        
        if material_labels==None:
            self.material_labels = {"aluminium": 0,
                           "electrode_aluminium": 1,
                           "electrode": 2,
                           "electrode_separator": 3,
                           "separator": 4,}
        else:
            self.material_labels = material_labels
        
    def view_img_stack(self, cmap="Greys_r"):
        f, ax = plt.subplots()
        def update(idx, x, y, width, height):
            ax.cla()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(self.img_stack[idx], cmap=cmap)
            rectangle = mpatch.Rectangle((x, y), width, height, facecolor="none", edgecolor="white")
            ax.add_patch(rectangle)
        interact(update, idx=IntSlider(min=0, max=self.img_stack.shape[0]-1),
                         x=IntSlider(min=0, max=self.img_stack.shape[1], step=1, value=0),
                         y=IntSlider(min=0, max=self.img_stack.shape[2], step=1, value=0),
                         width=IntSlider(min=0, max=self.img_stack.shape[1], step=1, value=400),
                         height=IntSlider(min=0, max=self.img_stack.shape[2], step=1, value=400))
        
        return f, ax
    
    def make_grid(self, grid_size, roi_x, roi_y, roi_width, roi_height, save_path=None, material_labels=None):        
        setattr(self, "grid_{}".format(int(grid_size)),                
                ROIGridSelect(grid_size, self.img_stack,
                                                                      roi_x,
                                                                      roi_y, 
                                                                      roi_width,
                                                                      roi_height,
                                                                      save_path=save_path,
                                                                      material_labels=material_labels))
        
    def apply_model(self, grid_size, materials_to_include=["electrode"]):
        model = vars(self)["grid_{}".format(int(grid_size))].model ## placed at top to escape if needed
        
        x_extent_FOV = int(self.img_stack.shape[1]/grid_size)
        y_extent_FOV = int(self.img_stack.shape[2]/grid_size)
        square_arr_FOV = []
        for idx in range(self.img_stack.shape[0]):
            slice_list = []

            for x in range(x_extent_FOV):
                for y in range(y_extent_FOV):
                    slice_list.append(self.img_stack[idx, x*grid_size:(x+1)*grid_size,
                                                          y*grid_size:(y+1)*grid_size])

            slice_arr = np.array(slice_list).reshape((x_extent_FOV, y_extent_FOV, grid_size, grid_size))

            square_arr_FOV.append(slice_arr)

        square_arr_FOV = np.array(square_arr_FOV)
        
        predict_list_FOV = []
        labels_to_predict = range(square_arr_FOV.shape[0])

        for labelled_slice in labels_to_predict:
            for xi in range(x_extent_FOV):
                for yi in range(y_extent_FOV):
                    predict_list_FOV.append(square_arr_FOV[labelled_slice, xi, yi, :, :].flatten())

        X_predict_FOV = np.array(predict_list_FOV)
        
        Y_predict_FOV = model.predict(X_predict_FOV)

        predict_reshape = Y_predict_FOV.reshape(len(labels_to_predict), int(Y_predict_FOV.shape[0]/len(labels_to_predict)))
        predict_reshape = np.array([arr.reshape(x_extent_FOV, y_extent_FOV) for arr in predict_reshape])
        
        reconstruct = np.full((square_arr_FOV.shape[0],
                       x_extent_FOV*grid_size, y_extent_FOV*grid_size), np.nan)

        target_materials = [self.material_labels[material] for material in materials_to_include]
        
        for nlabel, label in enumerate(labels_to_predict):
            for xi in range(x_extent_FOV):
                for yi in range(y_extent_FOV):
                    if predict_reshape[nlabel, xi, yi] in target_materials:
                        img = square_arr_FOV[label, xi, yi]
                        container = np.full((img.shape), np.nan)
                        container=img
                        reconstruct[label, 
                                    xi*grid_size:(xi+1)*grid_size,
                                    yi*grid_size:(yi+1)*grid_size] = container
                        
        self.reconstruct = reconstruct
        self.predict_locations = predict_reshape
                        
    def view_reconstruction(self, cmap="Greys_r"):
        f, (axes) = plt.subplots(1, 2)
        def update(idx):
            for ax in axes:
                ax.cla()
                ax.set_xticks([])
                ax.set_yticks([])
            axes[0].imshow(self.img_stack[idx], cmap=cmap)
            axes[1].imshow(self.reconstruct[idx], cmap=cmap)
        interact(update, idx=IntSlider(min=0, max=self.reconstruct.shape[0]-1, step=1))
