# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 11:56:55 2022

@author: Robert
"""

import numpy as np
from PIL import Image
from scipy import ndimage
import base64
import ast
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from scipy.optimize import curve_fit
import scipy.io
from PIL import Image
import glob
import os
import re
import h5py

PATH = ''

cal = {
#    '17583372': 0.02769, # Rail cam, M=0.1 camera
#    '17571186': 0.00220, # Rail cam, direct, pixel size
#    '18085415': 0.03062, # NF
#    '18085362': 0.0003495  # FF, pixel size
#    'LI20_201': 0.067
    'LI20_201': 0.0618
}

#NF = '18085415'
#FF = '18085362'
#SC = '17583372' # Rail cam, M=0.1 camera
#BC = '17571186' # Rail cam, direct

def set_calibration(cam_name, calibration):
    """ Add/update the passed camera in the calibration list. 
    
    Parameters
    ----------
    cam_name : int, string
        Name of the camera, must match the name in the metadata.
    cal : float
        Pixel calibration in mm/px.
    """
    cal[str(cam_name)] = calibration

def get_filename(instr, dataset, shot):
    """ Get the filename for an instrument, dataset number, and shot number.
    
    Parameters
    ----------
    instr : int, string
        Instrument serial number.
    dataset : int, string
        Dataset number.
    shot : int, string
        Shot number.
    
    Returns
    -------
    fileName : string
        The filename for the shot.
    """
    return "{!s}_{!s}_{:04d}".format(instr, dataset, int(shot))

def get_date_from_dataset(dataset):
    """ Get the year, month, and day from a dataset number.
    
    Parameters
    ----------
    dataset : int, string
        Dataset number.
    
    Returns
    -------
    year : string
        4 character year for the dataset.
    month : string
        2 character month for the dataset.
    day : string
        2 character day for the dataset.
    """
    dataset = str(dataset)
    year = "20{:2s}".format(dataset[0:2])
    month = "{:2s}".format(dataset[2:4])
    day = "{:2s}".format(dataset[4:6])
    return year, month, day

def get_path_from_dataset(root, dataset):
    """ Get the path from the DAQ directory to a given dataset directory.
    
    Parameters
    ----------
    root : string
        The name of the root directory, "META" for example.
    dataSet : int
        Dataset number.
    
    Returns
    -------
    path : string
        A string with the path from the top directory to the dataset.
    """
    year, month, day = get_date_from_dataset(dataset)
    path = PATH + '{:s}/year_{:4s}/month_{:2s}/day_{:2s}/{!s}/'.format(root, year, month, day, dataset)
    return path

def get_path(root, year, month, day):
    return PATH + '{}/year_{:4s}/month_{:2s}/day_{:2s}/'.format(root, year, month, day)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_line(line, rexp):
    """ Regex search against the defined lines, return the first match. """
    for key, rex in rexp.items():
        match = rex.search(line)
        if match:
            return key, match
    return None, None

class IMAGE():
    # Initialization functions ------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, camera, dataset, shot):
        self.dataset = str(dataset)
        self.camera = str(camera)
        self.shot = int(shot)
        self.image = self.load_image()
        self.meta = self.get_image_meta()
        self.data = np.array(self.image, dtype='float')
        if self.camera not in cal:
            set_calibration(self.camera, 1)
            print("Warning, calibration was not defined for camera {!s}, defaulting to 1.".format(self.camera))
        self.cal = cal[self.camera] # In mm/px
        self.gain = self.meta['Gain']
        self.shutter = self.meta['Shutter']
        self.width = self.image.width
        self.height = self.image.height
        self.offset = self.meta['Offset']
        self.xp = np.arange(0, self.width, 1)
        self.yp = np.arange(0, self.height, 1)
        self.x = self.cal*self.xp
        self.y = self.cal*self.yp
        self.center = None
        self.check_image()
    
    def load_image(self):
        """ Load an image from a given data set. 
        
        Returns
        -------
        image : obj
            Pillow image object for the tiff.
        """
        self.path = get_path_from_dataset('IMAGE', self.dataset)
        self.filename = get_filename(self.camera, self.dataset, self.shot) + '.tiff'
        name = self.path + self.filename
        image = Image.open(name)
        return image
    
    def get_image_meta(self):
        """ Return the meta data dictionary from a pillow image object.
        
        Returns
        -------
        meta : dict
            The meta data dictionary contained in the tiff image.
        """
        meta = self.image.tag_v2[270]
        # The second decodes goes from bytes to string
        meta = base64.b64decode(meta).decode()
        # Eval is insecure but this version is a little safer
        # Don't run this on random images from the internet!
        return ast.literal_eval(meta)
    
    def check_image(self):
        """ Verify meta data in the tiff is consistent with filename/image. """
        if str(self.meta['Dataset']) != self.dataset:
            print("Tiff meta data dataset {!s} does not match class dataset {!s}".format(self.meta['Dataset'], self.dataset))
        if int(self.meta['Shot number']) != self.shot:
            print("Tiff meta data shot number {:d} does not match class shot number {:d}".format(self.meta['Shot number'], self.shot))
        if str(self.meta['Serial number']) != self.camera:
            print("Tiff meta data serial {!s} does not match class serial {!s}".format(self.meta['Serial number'], self.camera))
        if self.meta['Pixel'][0] != self.width:
            print("Tiff meta data width {:d} does not match image width {:d}".format(self.meta['Pixel'][0], self.width))
        if self.meta['Pixel'][1] != self.height:
            print("Tiff meta data height {:d} does not match image height {:d}".format(self.meta['Pixel'][0], self.width))
            
    def refresh_calibration(self):
        """ Update the camera calibration if it has been changed. """
        self.cal = cal[self.camera]
        self.x = self.cal*self.xp
        self.y = self.cal*self.yp
    
    # Modification functions --------------------------------------------------
    #--------------------------------------------------------------------------
    def rotate(self, angle):
        """ Rotate the image. 
        
        Parameters
        ----------
        angle : float
            Angle to rotate the image by, in deg.
        """
        self.image = self.image.rotate(angle)
        self.data = np.array(self.image, dtype='float')
        
    def center_image(self, strategy, o, **kwargs):
        """ Center the image by non-uniformaly padding it. Meta will no longer match class parameters.
        
        Parameters
        ----------
        strategy : string
            See calculate_center for available strategies and **kwargs.
        o : int
            Padding on each side of returned array, in px.
        """
        cen_image, center = self.get_center_image(strategy, o, **kwargs)
        self.data = cen_image
        self.image = Image.fromarray(cen_image)
        self.width = self.image.width
        self.height = self.image.height
        self.center = (self.width/2, self.height/2)
        self.xp = np.arange(0, self.width, 1)
        self.yp = np.arange(0, self.height, 1)
        self.x = self.cal*self.xp
        self.y = self.cal*self.yp
        
    # Calculation functions ---------------------------------------------------
    #--------------------------------------------------------------------------
    def calculate_center(self, strategy='cm', threshold=12, f=None, p0=None, center=None):
        """ Calculate the pixel location of the center of the image.
        
        Parameters
        ----------
        strategy : string
            'cm' - the image center is found by taking the cm of the image.
            'mask' - a mask is formed from all pixels with values greather than a threshold.
                The image center is the centroid of the mask.
            'fit' - fit a function to the image to find the center.
            'external' - pass in the location of the mask center.
        threshold : int
            When strategy='mask', threshold is used to create the mask.
        f : function
            When strategy='f', function to fit to the data. The first two free parameters 
            should be the x and y positions of the center. It should accept as the first
            arguments (x, y).
        p0 : array
            When strategy='f', initial guesses for model parameters.
        center : (2) array
            When strategy='external', location of the image center.
        """
        if strategy == 'cm':
            self.center = self.center_cm()
        if strategy == 'mask':
            self.center = self.center_mask(threshold)
        if strategy == 'fit':
            self.fit = self.center_fit(f, p0)
            self.f = f
            self.center = np.array([self.fit[0], self.fit[1]])
        if strategy == 'external':
            self.center = center
    
    def center_cm(self):
        """ Calculate the center of mass of the image. """
        return np.flip(ndimage.center_of_mass(self.data))
    
    def center_mask(self, threshold, plot= False):
        """ Calculate the centroid of a mask of the image. """
        mask = self.data > threshold
        if plot== True:
            plt.figure()
            plt.pcolormesh(self.data)
            plt.figure()
            plt.pcolormesh(mask)
        return np.flip(ndimage.center_of_mass(mask))
    
    def center_fit(self, f, p0):
        """ Fit a function to the data to find the center. """
        X, Y = np.meshgrid(self.xp, self.yp)
        Z = self.data
        xdata = np.vstack((X.ravel(), Y.ravel()))
        ydata = Z.ravel()
        popt, pcov = curve_fit(f, xdata, ydata, p0=p0)
        return popt
    
    def calculate_energy(self):
        """ Calculate the total sum of all the pixels. """
        return np.sum(self.data)
    
    def get_center_image(self, strategy, o, **kwargs):
        """ Return a version of the image that is centered. 
        
        Parameters
        ----------
        strategy : string
            See calculate_center for available strategies and **kwargs.
        o : int
            Padding on each side of returned array, in px.
            
        Returns
        -------
        cen_image : array of floats
            Padded image with the actual image shifted to be centered.
        center : tuple of floats
            The location of the image center in pixel coordinates.
        """
        self.calculate_center(strategy, **kwargs)
        cen = np.array([self.width/2+o, self.height/2+o], dtype='int')
        center = self.center
        if center is None:
            center = (self.width/2, self.height/2)
        shift = np.array(np.rint(cen-center), dtype='int')
        cen_image = np.zeros((self.height+2*o, self.width+2*o))
        start_y = shift[1]
        end_y = start_y+self.height
        start_x = shift[0]
        end_x = start_x+self.width
#        print(start_x, end_x)
        cen_image[start_y:end_y, start_x:end_x] = self.data
        return cen_image, center
    
    # Visualization functions -------------------------------------------------
    #--------------------------------------------------------------------------
    def get_ext(self, cal=True):
        """ Helper function to get the extent for imshow. """
        if cal:
            ext = self.cal*np.array([-0.5, self.width+0.5, self.height+0.5, -0.5])
        else:
            ext = np.array([-0.5, self.width+0.5, self.height+0.5, -0.5])
        return ext
    
    def create_fig_ax(self, cal=True):
        """ Create the figure and ax objects for plotting the image.
        
        Return
        ------
        fig : object
            Matplotlib figure object for the full figure.
        ax : object
            Matplotlib axes object for the image axes.
        ext : (4) array
            Extent of the image for imshow.
        """
        fig = plt.figure(figsize=(4.85, 3), dpi=300)
        ax = plt.subplot()
        if cal:
            ax.set_xlabel(r'$x$ (mm)')
            ax.set_ylabel(r'$y$ (mm)')
            ext = self.cal*np.array([-0.5, self.width+0.5, self.height+0.5, -0.5])
        else:
            ax.set_xlabel(r'$x$ (px)')
            ax.set_ylabel(r'$y$ (px)')
            ext = np.array([-0.5, self.width+0.5, self.height+0.5, -0.5])
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        return fig, ax, ext
    
    def plot_image(self, cal=True, cmap='inferno'):
        """ Convenient plotting code to quickly look at images with meta data.
        
        Parameters
        ----------
        cal : bool
            True to show the calibrated axes or false for the pixel coordinates.
        
        Returns
        -------
        fig : object
            Matplotlib figure object for the full figure.
        ax : object
            Matplotlib axes object for the image axes.
        im : object
            Matplotlib imshow object for the actual image.
        cb : object
            Matplotlib colorbar object.
        ext : (4) array
            Extent of the image for imshow.
        """
        fig, ax, ext = self.create_fig_ax(cal)
        self.plot_dataset_text(ax)
        self.plot_metadata_text(ax)
        im = ax.imshow(self.data, extent=ext, cmap=cmap)
        cb = fig.colorbar(im)
        cb.set_label('Counts')
        ax.tick_params(color='w')
        ax.spines['bottom'].set_color('w')
        ax.spines['top'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.spines['right'].set_color('w')
        return fig, ax, im, cb, ext
    
    def plot_dataset_text(self, ax):
        """ Add text at the top of the figure stating the dataset and shot number. """
        ax.text(0.02, 0.95, r"DS: {!s}".format(self.dataset), color='w', transform=ax.transAxes)
        ax.text(0.77, 0.95, r"Shot: {:04d}".format(self.shot), color='w', transform=ax.transAxes)
    
    def plot_metadata_text(self, ax):
        """ Add text at the bottom of the figure stating image parameters. """
        dy = 0.05
        y = 0.02
        ax.text(0.02, y+dy, r"Size:  {:4d}, {:04d}".format(self.width, self.height), color='w', transform=ax.transAxes)
        ax.text(0.02, y, r"Start: {:4d}, {:4d}".format(self.offset[0], self.offset[1]), color='w', transform=ax.transAxes)
        ax.text(0.66, y+dy, r"Exp:  {:0.2f}ms".format(self.shutter), color='w', transform=ax.transAxes)
        ax.text(0.66, y, r"Gain: {:0.2f}".format(self.gain), color='w', transform=ax.transAxes)
    
    def plot_center(self, radius, cal=True, cmap='inferno'):
        """Plot the image with a beam circle and the 
        
        Parameters
        ----------
        radius : float
            Radius of the beam circle to plot.
        cal : bool
            True to show the calibrated axes or false for the pixel coordinates.
        """
        if self.center is None:
            print('The image center is None, it needs to be calculated before it can be shown.')
        fig, ax, im, cb, ext = self.plot_image(cal, cmap)
        cen = self.center
        if cal:
            cal = self.cal
        else:
            cal = 1.0
        ax.plot([cal*cen[0], cal*cen[0]], [ext[2], ext[3]],  'tab:blue')
        ax.plot([ext[0], ext[1]], [cal*cen[1], cal*cen[1]], 'tab:blue')
        phi = np.linspace(0, 2*np.pi, 1000)
        ax.plot(radius*np.cos(phi)+cal*cen[0], radius*np.sin(phi)+cal*cen[1])
        ax.text(0.02, 0.9, "Beam center:", color='tab:blue', transform=ax.transAxes)
        ax.text(0.02, 0.85, "({:0.3f}, {:0.3f})".format(cal*cen[0], cal*cen[1]), color='tab:blue', transform=ax.transAxes)
        return fig, ax, im, cb, ext
    
    def plot_lineouts(self, cal=True, cmap='inferno'):
        """Plot the image with lineouts through the center. 
        
        Parameters
        ----------
        cal : bool
            True to show the calibrated axes or false for the pixel coordinates.
        """
        if self.center is None:
            print('The image center is None, it needs to be calculated before it can be shown.')
        fig, ax, im, cb, ext = self.plot_image(cal, cmap)
        cen = self.center
        if cal:
            cal = self.cal
        else:
            cal = 1.0
        ax.plot([cal*cen[0], cal*cen[0]], [ext[2], ext[3]],  'tab:blue')
        ax.plot([ext[0], ext[1]], [cal*cen[1], cal*cen[1]], 'tab:blue')
        # TODO implement the lineout bit
        return fig, ax, im, cb, ext

class Dataset():
    rexp = {
        'camera' : re.compile(r'\t\t(?P<cam>[0-9]{8})'),
    }
    
    # Initialization functions ------------------------------------------------
    #--------------------------------------------------------------------------
    def __init__(self, dataset):
        self.dataset = str(dataset)
        self.path = get_path_from_dataset('IMAGE', dataset)
        year, month, day = get_date_from_dataset(dataset)
        self.meta_path = get_path('META', year, month, day)
        self.save_path = get_path_from_dataset('ANALYSIS', dataset)
        make_dir(self.save_path)
        self.parse_dataset_meta()
        self.centers = None
    
    def parse_dataset_meta(self):
        """ Find the cameras used for each dataset and the number of images. """
        self.cameras = []
        self.M = {}
        # XXX M should come from the DAQ, not counting files
        DS = self.dataset
        with open(self.meta_path+'meta_{!s}.txt'.format(DS), 'r') as f:
            for line in f:
                key, match = parse_line(line, self.rexp)
                if key == 'camera':
                    camera = match.group('cam')
                    self.cameras.append(camera)
                    files = glob.glob('{}/{}_{}_*.tiff'.format(self.path, camera, DS))
                    self.M[camera] = len(files)
    
    # Calculation functions ---------------------------------------------------
    #--------------------------------------------------------------------------
    def center_DS(self, camera, strategy, cam_prop, o=500, adapt=0.5, **kwargs):
        """ Load the images from a dataset, center and sum. """
        M = self.M[camera]
        DS =self.dataset
        image = None
        for i in range(M):
            try:
                image = IMAGE(camera, DS, i)
            except:
                print('Failed on intial load of {:s}-{:04d}, trying next image'.format(DS, i))
                continue
            if image is not None:
                break
        energy = image.calculate_energy()
        if strategy == 'external':
            if self.centers is None:
                print("Can not use external centering strategy, centers is None.")
            centers = self.centers
        else:
            centers = np.zeros((M, 2))
        sum_image = np.zeros((image.height+2*o, image.width+2*o))
        
        fails = np.empty(0, dtype='int')
        for i in range(M):
            if i != 0:
                try:
                    image = IMAGE(camera, DS, i)
                except:
                    print("Failed to load {:s}-{:04d}, ignoring image.".format(DS, i))
                    fails = np.append(fails, i)
                    continue
            energy_i = image.calculate_energy()
            if abs((energy_i-energy)/energy) > 0.1:
                print("Pixel total of {:s}-{:04d} differs from reference by more than 10%, ignoring image.".format(str(DS), i))
                fails = np.append(fails, i)
                continue
            if not self.check_image(image, cam_prop): 
                print("Image check failed.")
                fails = np.append(fails, i)
                continue
            if strategy=='adapt_mask':
                high = np.max(image.data)
                level = int(adapt*high)
                cen_image, centers[i, :] = image.get_center_image('mask', o, threshold=level, **kwargs)
            else:
                cen_image, centers[i, :] = image.get_center_image(strategy, o, **kwargs)
            sum_image += cen_image
        
        sum_image *= (M/(M-len(fails)))
        weights = np.ones((M, 2), dtype='int')
        weights[fails, :] = 0
        self.center = np.average(centers, axis=0, weights=weights)
        self.center_std = np.std(centers, axis=0, where=np.array(weights, dtype='bool'))
        self.centers = centers
        self.save_center(camera, sum_image)
        return sum_image, centers, self.center, self.center_std
    
    def check_image(self, image, cam_prop):
        """ Check that the camera properties match the reference. 
        
        Returns
        -------
        status : bool
            True means everything checks out. False means there was a problem.
        """
        if image.height != cam_prop['height']:
            print("Image height of {:d}px doesn't match reference of {:d}px.".format(image.height, cam_prop['height']))
            return False
        if image.width != cam_prop['width']:
            print("Image width of {:d}px doesn't match reference of {:d}px.".format(image.width, cam_prop['width']))
            return False
        if image.gain != cam_prop['gain']:
            print("Image gain of {:0.2f} doesn't match reference of {:0.2f}.".format(image.gain, cam_prop['gain']))
            return False
        if image.height != cam_prop['height']:
            print("Image shutter of {:0.2f}ms doesn't match reference of {:0.2f}ms.".format(image.shutter, cam_prop['shutter']))
            return False
        return True
    
    def save_center(self, camera, sum_image):
        """ Save the calculated image info. """
        name = self.save_path + '{!s}_{!s}_centered.tiff'.format(self.dataset, camera)
        tiff = Image.open(name)
        tiff.write_image(np.array(sum_image, dtype='uint16'))
        tiff.close()
        name = self.save_path + '{!s}_{!s}_centers.txt'.format(self.dataset, camera)
        np.savetxt(name, self.centers)
        name = self.save_path + '{!s}_{!s}_center.txt'.format(self.dataset, camera)
        np.savetxt(name, self.center)
        name = self.save_path + '{!s}_{!s}_center_std.txt'.format(self.dataset, camera)
        np.savetxt(name, self.center_std)
        
    def load_center(self, camera):
        """ Load dataset parameters from a saved file. """
        name = self.save_path + '{!s}_{!s}_centers.txt'.format(self.dataset, camera)
        self.centers = np.loadtxt(name)
        name = self.save_path + '{!s}_{!s}_center.txt'.format(self.dataset, camera)
        self.center = np.loadtxt(name)
        name = self.save_path + '{!s}_{!s}_center_std.txt'.format(self.dataset, camera)
        self.center_std = np.loadtxt(name)
    
    def load_image(self, camera):
        # TODO: Load this as an instance of the IMAGE class rather than a pillow object
        name = self.save_path + '{!s}_{!s}_centered.tiff'.format(self.dataset, camera)
        image = Image.open(name)
        return image


class MatImage(IMAGE):
    def load_image(self):
        """ Load an image from a given data set. 
        
        Returns
        -------
        image : obj
            Pillow image object for the tiff.
        """
        self.path = PATH
        self.filename = "ProfMon-CAMR_{!s}-{!s}.mat".format(self.camera, self.dataset)
        name = self.path + self.filename
        try:
            self.mat = scipy.io.loadmat(name)            
            image = Image.fromarray(self.mat['data'][0][0][1])
        except NotImplementedError:
            with h5py.File(name, 'r') as f:
#                def printname(name):
#                    print(name)
#                f.visit(printname)
                dataset = f['data/img']
                arr = np.array(dataset)
                image = Image.fromarray(arr)
        return image
    
    def get_image_meta(self):
        """ Return the meta data dictionary from a pillow image object.
        
        Returns
        -------
        meta : dict
            The meta data dictionary contained in the tiff image.
        """
        # You can see the name of each field in the mat at mat['data'].dtype.names
        try:
            mat_meta = self.mat['data'][0][0]            
            meta = {}
            meta['Gain'] = 0.0
            meta['Shutter'] = 0.0
            meta['Offset'] = [mat_meta[10][0][0], mat_meta[11][0][0]]
            meta['Dataset'] = self.dataset
            meta['Shot number'] = self.shot
            meta['Serial number'] = self.camera
            meta['Pixel'] = [mat_meta[6][0][0], mat_meta[7][0][0]]
        except:
            meta = {}
            with h5py.File(self.path + self.filename, 'r') as f:
                meta['Pixel'] = \
                [np.array(f['data/nRow'])[0][0], np.array(f['data/nCol'])[0][0]]
            meta['Gain'] = 0.0
            meta['Shutter'] = 0.0
            meta['Offset'] = [0, 0] #I put 0, 0 here because I don't think the new matlab file has offset
            meta['Dataset'] = self.dataset
            meta['Shot number'] = self.shot
            meta['Serial number'] = self.camera
        return meta


class TiffImage(IMAGE):
    def load_image(self):
        """ Load an image from a given data set. 
        
        Returns
        -------
        image : obj
            Pillow image object for the tiff.
        """
        self.path = PATH
        self.filename = self.dataset
        name = self.path + self.filename
        image = Image.open(name)
        return image
    
    def get_image_meta(self):
        """ Return the meta data dictionary from a pillow image object.
        
        Returns
        -------
        meta : dict
            The meta data dictionary contained in the tiff image.
        """
        # You can see the name of each field in the mat at mat['data'].dtype.names
        meta = {}
        meta['Gain'] = 0.0
        meta['Shutter'] = 0.0
        meta['Offset'] = [0, 0]
        meta['Dataset'] = self.dataset
        meta['Shot number'] = self.shot
        meta['Serial number'] = self.camera
        meta['Pixel'] = [self.image.width, self.image.height]
        return meta