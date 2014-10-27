#!/usr/bin/python

"""The module of the CoulombInt class."""

import numpy as np
from itertools import combinations_with_replacement as cr
import sys
import os
from wf import WF
from skmonaco import mcquad
import time

class CoulombIntDir(object):

    """Computes Coulomb integrals."""

    def __init__(self, N=2, **kw):

        self.N = N                              # number of basis functions
        self.comb = self.specialCombinations()  # four factor products

        self._load = kw.get('pload', '/data/users/mklymenko/work_py/mb_project/')
        self._save = kw.get('psave', '/data/users/mklymenko/work_py/mb_project/')
        self.cint = []

        if os.path.isfile('/data/users/mklymenko/work_py/mb_project/coint2.dat'):
            os.remove('/data/users/mklymenko/work_py/mb_project/coint2.dat')

        try:               # try to read integrals from the disk
            with open(self._load + "coint2.dat"):
                self.cint = np.loadtxt(self._load + "coint2.dat")
        except IOError:    # there are no integrals on the disk

            wfs_x = []       # envelope functions for X valleys
            wfs_y = []       # envelope functions for Y valleys
            wfs_z = []       # envelope functions for Z valleys

            for j in xrange(self.N):
                #wfs_x.append(WF(qn=j,
                #                p='/data/users/mklymenko/science/H2_100/programing/dis/v0/', flag='int', **kw)) # executing the fitting procedure N-times
                wfs_y.append(WF(qn=j,
                                p='/data/users/mklymenko/science/H2_100/programing/dis/v1/', flag='int', **kw)) # executing the fitting procedure N-times
                #wfs_z.append(WF(qn=j,
                #                p='/data/users/mklymenko/science/H2_100/programing/dis/v2/', flag='int', **kw)) # executing the fitting procedure N-times

            # computing integrals for all possible combinations stored in self.comb
            start_time = time.time()
            self.cint = [self.two_el_int(wfs_y[j[0]],wfs_y[j[1]],wfs_y[j[2]],wfs_y[j[3]]) for j in self.comb]
            # self.cint = self.two_el_int(wfs_y[0],wfs_y[0],wfs_y[0],wfs_y[0])
            # self.cint = [self.overlap_int(wfs_y[j],wfs_y[j]) for j in xrange(self.N)]
            # self.cint = self.overlap_int(wfs_y[0],wfs_y[0])
            print time.time()- start_time
            self.save()

    #-----------------------------------
    def getInt(self, vec):
        i = self.comb.index(tuple(sorted(vec)))
        return self.cint[i]

    #-----------------------------------
    def specialCombinations(self):

        """This function computes the number of possible configurations
        of N basis functions into poducts of four functions."""

        vec = list(xrange(self.N))
        b = []
 #       for a1 in cr(vec, 2):
 #           for a2 in cr(vec, 2):
 #               b.append(a1+a2)
        for a in cr(vec, 4):
            b.append(a)
        return b

    #-----------------------------------

    @staticmethod
    def overlap_int(wf1,wf2):

        def integrand(x, p1, p2):
            wf1 = p1.get_value([x[0], x[1], x[2]])
            wf2 = p2.get_value([x[0], x[1], x[2]])
            return wf1*wf2

        xl = [-6.5, 0.0, -6.5]
        xu = [6.5, 8.9, 6.5]

        v, a = mcquad(integrand, xl=xl, xu=xu, npoints=1000000, args=[wf1, wf2])

        return v, a

    #----------------------------------

    @staticmethod
    def two_el_int(wf1, wf2, wf3, wf4):

        def integrand(x, p1, p2, p3, p4):
            x1=np.array([x[0], x[1], x[2]])
            x2=np.array([x[3], x[4], x[5]])
            wf1 = p1.get_value(x1)
            wf2 = p2.get_value(x1)
            wf3 = p3.get_value(x2)
            wf4 = p4.get_value(x2)
            D=np.sqrt(np.sum((x1-x2)**2))
            return wf1*wf2*(1/D)*wf3*wf4

        xl = [-6.5, 0.0, -6.5, -6.5, 0.0, -6.5]
        xu = [6.5, 8.9, 6.5, 6.5, 8.9, 6.5]

        v, a = mcquad(integrand, xl=xl, xu=xu, npoints=1000000, args=[wf1, wf2, wf3, wf4])

        return v, a

    #-----------------------------------

    def sum2(self, j, i):
        ans = 0
        for k in xrange(self.N):
            ans+=self.getInt([j,i,k,k])

        return ans

    def save(self):     # fixed number of functions, varied amplitudes and widths
        if (self._save != '0'):
            p = self._save + "coint2.dat"
            np.savetxt(p, self.cint)
        else:
            sys.exit("Wrong path to save")

if __name__ == "__main__":

    c=CoulombIntDir()
    print(np.array([c.comb,c.cint]).T)
