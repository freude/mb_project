"""
    The class is for fitting a grid function by a set of gaussian functions
"""

from wf import WF
from IPython.core.debugger import Tracer
import glob
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import Rbf
import math
# -----------------------------------------------------------
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
# -----------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# -----------------------------------------------------------


class GFit(WF):

    def __init__(self, **kw):
        WF.__init__(self,**kw)

        # path to files containing fitting parameters if any (to load instead of
        # fitting)
        self._load = kw.get('pload', '/data/users/mklymenko/work_py/mb_project/')

        # path to save fitting parameters
        self._save = kw.get('psave', '/data/users/mklymenko/work_py/mb_project/')

        self._num_fu = kw.get('num_fu', 55)  # number of Gaussian primitive functions
        self._flag = kw.get('mf', 2)  # choice of the model function

        self._gf = []
        self.error = []  # initialization of sought-for fitting parameters

        self._load = self._load+self._path[-3:-1]+'_'+str(self._qn)+'.dat'

        self.peaks=[]
        self.coords=[]

        # print(self._load)

        initia = kw.get('init', 'fit')

        if initia == 'fit':
            try:
                with open(self._load):
                    self._gf = np.loadtxt(self._load)
                    self._num_fu=len(self._gf)/5
                print 'The fitting parameters are found to be stored on the disk'
                print 'I am just going to read them from'
                print self._load

            except IOError:

                print 'Oh dear.'
                print 'There is no parameters stored on the disk'
                print 'I launch the fitting procedure. It will take time...'
                print 'The fitting is provided for the file'
                print self._path+'ff_'+str(self._sn)+'.dat'
                print '    the quantum state         {}'.format(self._qn)
                print '    the primitive Gaussians  {}'.format(self._num_fu)
                print '---------------------------------------------------------------------\n'

                # self.do_fit_seq(8,1)
                # #doing fitting or loading parameters from disk
                self.do_fit()
    @property
    def gf(self):
        for j in xrange(self._num_fu):
            self._gf[j*5+3] = 1.0/abs(self._gf[j*5+3])
        return np.reshape(self._gf, (self._num_fu, -1))
    # ----------------------------------------------
    def save(self):
        """Save Gaussian functions coefficients to the file"""

        if (self._save != '0'):
            p = self._save+self._path[-3:-1]+'_'+str(self._qn)+'.dat'
            np.savetxt(p, self._gf)
        else:
            sys.exit("Wrong path to save")
    # ---------------------------------------------
    def modelfun1(self, x, *par):
        """ The model function represented by a sum of
        the Gaussian functions with variable positions, widths and
        amplitudes
        """

        g = np.zeros(len(x[0]))

        for j in xrange(len(par)/5):
            x1 = par[j*5]
            x2 = par[j*5+1]
            x3 = par[j*5+2]
            w = par[j*5+3]
            a = par[j*5+4]
            r1 = pow((x[0]-x1), 2)+pow((x[1]-x2), 2)+pow((x[2]-x3), 2)
            g = g+a*np.exp(-r1/abs(w))
            # if ((x1 > -7.0) or (x1 < 7.0)): g=g+300
            # if ((x2 > 0.0) or (x2 < 8.7)): g=g+300
            # if ((x3 > -7.0) or (x3 < 7.0)): g=g+300
            g=g+300*0.5*(np.sign(-abs(x1)+5.0) - 1)
            g=g+300*0.5*(np.sign(-abs(x2-4.0)+4.0) - 1)
            g=g+300*0.5*(np.sign(-abs(x3)+5.0) - 1)
            # if ((a > 1.1) or (a < -1.1)): g=g+300
        return g

# -------------------------------------------------------------------------
# --------------Tool for extracting values of the wave function------------
# -------------------------------------------------------------------------

    def show_func(self, x):
        """Computes the value of the wave function in points stored in
        the vector x using fitting parameters and the model functions.
        """
        if (self._flag == 1):
            g = self.modelfun(x, *self._gf)
        elif (self._flag == 2):
            g = self.modelfun1(x, *self._gf)
        elif ((self._flag == 0) & (self._load != '0')):
            pass
        else:
            # pass
            sys.exit("Wrong flag in do_fit")

        return g
    def show_gf(self, x):
        """Same as show_func(self,x) but returns
        decomposed primitive Gaussian functions
        """
        g = np.zeros((len(x[0]), self._num_fu), dtype=np.float64)
        for j in range(self._num_fu):
            #x1 = self._gf[j*5]
            #x2 = self._gf[j*5+1]
            #x3 = self._gf[j*5+2]
            #w = self._gf[j*5+3]
            #a = self._gf[j*5+4]
            #r1 = pow((x[0]-x1), 2)+pow((x[1]-x2), 2)+pow((x[2]-x3), 2)
            #g[:, j] = a*np.exp(-r1/abs(w))
            g[:, j] = self.modelfun1(x,*self._gf[j:j+5])

        return g
# ----------------------------------------------------------------------
    @staticmethod
    def moments(data):
        """Returns (height, x, y, width_x, width_y)
         the gaussian parameters of a 2D distribution by calculating its
        moments """

        data = np.absolute(data)
        total = data.sum()
        X = np.indices(data.shape)
        x = (X*data).sum()/total
        width = np.sqrt((((X-x)**2)*data).sum()/data.sum())
        m_max = data.max()
        m_min = data.min()
        if np.absolute(m_max) >= np.absolute(m_min):
            height = m_max
        else:
            height = m_min
        return height, x, width
#-----------------------------------------------------------------------
    def do_fit(self):
        """ The function does the fitting procedure"""

        if (self._flag == 1):
            self._gf = [0.2]
            self._gf = self.par*(self._num_fu*len(self._sites)*2)
            x, F = self.read_from_file(
                self._sn, self._qn, self._path)  # read data from the file
            # ,ftol=1.0e-7,xtol=1.0e-8)
            popt, pcov = curve_fit(
                self.modelfun, x, F, p0=self._gf, maxfev=5000)
            self._gf = popt
        elif (self._flag == 2):

            xi, yi, zi = np.mgrid[-6.5:6.5:160j, 4.0:8.9:160j, -7.5:7.5:160j]
            x, y, z = xi.flatten(), yi.flatten(), zi.flatten()
            XX = np.vstack((x, y, z))
            AA = self.invdisttree(XX.T, nnear=130, eps=0, p=1)

            if self.peaks==[]:
                print '\n---------------------------------------------------------------------'
                print 'Detecting maxima and minima of target function...',

                peaks_min, min_coord, peaks_max, max_coord = self.detect_min_max(AA.reshape(xi.shape))
                print 'done'
                print 'Number of the min peaks: {}'.format(len(peaks_min))
                print 'Number of the max peaks: {}'.format(len(peaks_max))
                print '---------------------------------------------------------------------\n'

                if peaks_max==[]:
                    peaks=np.insert(peaks_min, np.arange(len(peaks_max)), peaks_max)
                    coords=np.insert(min_coord, np.arange(max_coord.shape[1]), max_coord, axis=1)
                else:
                    peaks = np.insert(peaks_max, np.arange(len(peaks_min)), peaks_min)
                    coords = np.insert(max_coord, np.arange(min_coord.shape[1]), min_coord, axis=1)

                self.peaks=peaks
                self.coords=coords

            par = [0.0]*(self._num_fu*5)
            j1 = 0
            aaaa = 1
            for j in xrange(self._num_fu):
                if (j > aaaa*self.coords.shape[1]-1):
                    j1 = 0
                    aaaa += 1
                par[j*5] = xi[self.coords[0, j1], self.coords[0, j1], self.coords[0, j1]]
                par[j*5+1] = yi[self.coords[1, j1], self.coords[1, j1], self.coords[1, j1]]
                par[j*5+2] = zi[self.coords[2, j1], self.coords[2, j1], self.coords[2, j1]]
                # par[j*5+3] = 0.1003+0.1000*math.copysign(1, (pow(-1, j)))
                par[j*5+3] = 17.0007

#                if j < 15:
#                    par[j*5+3] = 0.00001
#                else:
#                    par[j*5+3] = 0.0005
                par[j*5+4] = self.peaks[j1]
#                print(coords[0, j1], coords[1, j1], coords[2, j1])
                j1 += 1
            # popt, pcov = curve_fit(self.modelfun1, x[:,1:20000], F[1:20000],p0=par,maxfev=150000,xtol=1e-8,ftol=1e-8)
            popt, pcov = curve_fit(
                self.modelfun1, XX, AA, p0=par, maxfev=50000, xtol=1e-6, ftol=1e-7)
            # popt, pcov = curve_fit(self.modelfun1, XX, AA, p0=par)
            self._gf = popt
#             self.error=np.diagonal(pcov, offset=0)
#             print(pcov)
        else:
            # pass
            sys.exit("Wrong flag in do_fit")

if __name__ == "__main__":

#     try:
#         with open('/data/users/mklymenko/work_py/mb_project/wf0.dat'):
#             os.remove('/data/users/mklymenko/work_py/mb_project/wf0.dat')
#     except IOError:
#         pass

#    filelist = glob.glob('/data/users/mklymenko/work_py/mb_project/wf*')
#    for f in filelist:
#        os.remove(f)

    x = np.linspace(-6.5, 6.5, 300)
    y = np.linspace(4.0, 8.9, 300)

    XXX = np.vstack((x, x*0.0+6.45, x*0.0))

    xi, yi = np.meshgrid(x, y)
    x, y = xi.flatten(), yi.flatten()
    z = x*0.0
    XX = np.vstack((z, y, x))

    wf = GFit(
        init='fit',
        sn=0,
        qn=1,
        mf=2,
        num_fu=12,
        psave='/data/users/mklymenko/work_py/mb_project/',
        pload='/data/users/mklymenko/work_py/mb_project/',
        flag='int')

    wf.save()
    print wf._gf
    # wf.draw_func(x,y,par='2d')
    g = wf.show_func(XX)
    g1 = wf.show_gf(XXX)
    AA = wf.get_value(XX.T)
    fig = plt.figure()
    #for j in range(0,wf._num_fu):
    #     plt.plot(XXX[0,:].T,g1[:,j])

    #plt.plot(xi[150, :].T, g.reshape(xi.shape)[150, :])
    #plt.plot(xi[150, :].T, AA.reshape(xi.shape)[150, :])

    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(xi,yi,g1.reshape(xi.shape), cmap=cm.jet, linewidth=0.2)

    plt.contour(xi, yi, -AA.reshape(xi.shape), colors='red')
    plt.contour(xi, yi, -g.reshape(xi.shape), colors='blue')

    # x=[j for j in xrange(wf.error.__len__())]
    # plt.plot(x,wf.error)

    plt.hold(True)
    plt.show()
