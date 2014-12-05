"""
    The class is for reading from file and interpolating a grid function
"""

import numpy as np
import matplotlib.pyplot as plt
from invdisttree import Invdisttree
from sys import stdout
from skmonaco import mcquad


class WF(object):

    def __init__(self, **kw):

        # the path to files containing wave functions (to fit with)
        self._path = kw.get('p', '/data/users/mklymenko/science/H2_100/programing/dis/v2/')
        self._sn = kw.get('sn', 0)  # id of the system
        self._qn = kw.get('qn', 0)  # id of the wave function

        print '\n--------------------------------------------------------------'
        print 'The wave function is associated with the file'
        print self._path+'ff_'+str(self._sn)+'.dat'
        print '    the number of the quantum state is      {}'.format(self._qn)

        self.flag = kw.get('flag', 'none')

        if self.flag == 'int':
            print("Reading from file and creating the interpolant..."),
            stdout.flush()
            X, F = self.read_from_file(self._sn, self._qn, self._path)
            self.invdisttree = Invdisttree(X.T, F)
            print("Done!")

        print '--------------------------------------------------------------\n'

    @staticmethod
    def read_from_file(jjjj, N, path='/data/users/mklymenko/science/H2_100/programing/dis/v0/'):
        """Read the  wave function of N-th state
        from the file numbered by jjjj"""

        p1 = np.loadtxt(path+'ff_'+str(jjjj)+'.dat')

        a = []
        a = np.where(
            ((p1[:, 1] == 111) & (p1[:, 2] == 111) & (p1[:, 3] == 111)))[0]

        n1 = N
        X = np.array(p1[a[n1]+1:a[n1+1], 0])
        Y = np.array(p1[a[n1]+1:a[n1+1], 1])
        Z = np.array(p1[a[n1]+1:a[n1+1], 2])
        F = np.array(p1[a[n1]+1:a[n1+1], 3])

        return (np.vstack((X, Y, Z)), F)
    @staticmethod
    def detect_min_max(arr):
        """That's a very nice function detecting all local minima and maxima
        and computing their coordinates.
        The method is based on derivatives.
        """

        max_value = max(np.absolute(np.reshape(arr, -1)))
        peaks_max = []
        peaks_min = []
        x_max = []
        y_max = []
        z_max = []
        x_min = []
        y_min = []
        z_min = []

        for j1 in xrange(10, arr.shape[0]-10):
            for j2 in xrange(10, arr.shape[1]-10):
                for j3 in xrange(10, arr.shape[2]-10):
                    if (np.absolute(arr[j1, j2, j3]) > 0.3*max_value):

                        aaaa = [
                            arr[j1, j2, j3 + 1], arr[j1, j2 + 1, j3],
                            arr[j1 + 1, j2, j3], arr[j1, j2, j3 - 1],
                            arr[j1, j2 - 1, j3], arr[j1 - 1, j2, j3],
                            arr[j1 + 1, j2 + 1, j3 + 1],
                            arr[j1 - 1, j2 - 1, j3 - 1],
                            arr[j1 - 1, j2 + 1, j3 + 1], arr[j1, j2 + 1, j3 + 1],
                            arr[j1, j2 - 1, j3 - 1], arr[j1, j2 - 1, j3 + 1],
                            arr[j1, j2 + 1, j3 - 1], arr[j1 + 1, j2, j3 + 1],
                            arr[j1 - 1, j2, j3 - 1], arr[j1 - 1, j2, j3 + 1],
                            arr[j1 + 1, j2, j3 - 1], arr[j1 + 1, j2 + 1, j3],
                            arr[j1 - 1, j2 - 1, j3], arr[j1 + 1, j2 - 1, j3],
                            arr[j1 - 1, j2 + 1, j3], arr
                            [j1 + 1, j2 - 1, j3 + 1], arr
                            [j1 + 1, j2 + 1, j3 - 1], arr
                            [j1 - 1, j2 - 1, j3 + 1], arr
                            [j1 + 1, j2 - 1, j3 - 1], arr
                            [j1 - 1, j2 + 1, j3 - 1]]
                        bbbb = [
                            arr[j1, j2, j3 + 9], arr[j1, j2 + 9, j3],
                            arr[j1 + 9, j2, j3], arr[j1, j2, j3 - 9],
                            arr[j1, j2 - 9, j3], arr[j1 - 9, j2, j3]]

                        if ((arr[j1, j2, j3] > max(aaaa)) and (max(aaaa) > max(bbbb))):
                            peaks_max = np.append(peaks_max, arr[j1, j2, j3])
                            x_max = np.append(x_max, j1)
                            y_max = np.append(y_max, j2)
                            z_max = np.append(z_max, j3)

                        if ((arr[j1, j2, j3] < min(aaaa)) and (min(aaaa) < min(bbbb))):
                            peaks_min = np.append(peaks_min, arr[j1, j2, j3])
                            x_min = np.append(x_min, j1)
                            y_min = np.append(y_min, j2)
                            z_min = np.append(z_min, j3)

        return peaks_min, np.vstack(
            (x_min, y_min, z_min)), peaks_max, np.vstack(
            (x_max, y_max, z_max))
    def get_value(self, x):
        if self.flag != 'int':
            raise ValueError('The interpolant is not defined')
        return self.invdisttree(x, nnear=23, eps=0, p=1)
    def get_matrix(self, x, y, z, flag=None):
        xi, yi, zi = np.meshgrid(x, y, z)
        x, y, z = xi.flatten(), yi.flatten(), zi.flatten()
        XX = np.vstack((x, y, z)).T
        AA = self.get_value(XX)

        if flag == "matrix":
            return zip(xi, yi, zi), AA.reshape(xi.shape)
        else:
            return XX, AA
    def plot2d(self, x, y):
        xi, yi = np.meshgrid(x, y)
        x, y = xi.flatten(), yi.flatten()
        z = x*0.0
        XX = np.vstack((x, y, z))
        AA = self.get_value(XX.T)
        plt.contour(xi, yi, -AA.reshape(xi.shape), colors='red')
        plt.hold(True)
        plt.show()

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

if __name__ == "__main__":

    x = np.linspace(-6.5, 6.5, 300)
    y = np.linspace(4.0, 8.9, 300)

    XXX = np.vstack((x, x*0.0+6.45, 0.0*x))

    xi, yi = np.meshgrid(x, y)
    x, y = xi.flatten(), yi.flatten()
    z = x*0.0
    XX = np.vstack((x, y, z))

    wf = WF(sn=10, qn=0, flag='int')
    AA = wf.get_value(XX.T)

    fig = plt.figure()
    plt.contour(xi, yi, -AA.reshape(xi.shape), colors='red')
    plt.hold(True)
    plt.show()

    def square_mod(x, par):
        wf1 = par.get_value([x[0], x[1], x[2]])
        wf2 = par.get_value([x[0], x[1], x[2]])
        return wf1*wf2

    xl = [-6.5, 0.0, -6.5]
    xu = [6.5, 8.9, 6.5]

    v, a = mcquad(square_mod, xl=xl, xu=xu, npoints=100000, args=[wf])

    print v
    print a

#    def overlap(x1):
#    def two_elec_integrand(x1,x2):
