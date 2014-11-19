
################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2011 by M. Aichhorn
#
# TRIQS is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# TRIQS. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

from types import *
import numpy
from pytriqs.archive import *
import pytriqs.utility.mpi as mpi
from math import sqrt
from converter_tools import *

class HkConverter(ConverterTools):
    """
    Conversion from general H(k) file to an hdf5 file that can be used as input for the SumKDFT class.
    """

    def __init__(self, hk_file, hdf_file, dft_subgrp = 'dft_input', symmcorr_subgrp = 'dft_symmcorr_input', repacking = False):
        """
        Init of the class.
        """

        assert type(hk_file)==StringType,"hk_file must be a filename"
        self.hdf_file = hdf_file
        self.dft_file = hk_file
        self.dft_subgrp = dft_subgrp
        self.symmcorr_subgrp = symmcorr_subgrp
        self.fortran_to_replace = {'D':'E', '(':' ', ')':' ', ',':' '}

        # Checks if h5 file is there and repacks it if wanted:
        import os.path
        if (os.path.exists(self.hdf_file) and repacking):
            ConverterTools.__repack(self)


    def convert_dmft_input(self, first_real_part_matrix = True, only_upper_triangle = False, weights_in_file = False):
        """
        Reads the input files, and stores the data in the HDFfile
        """
                   
        # Read and write only on the master node
        if not (mpi.is_master_node()): return
        mpi.report("Reading input from %s..."%self.dft_file)

        # R is a generator : each R.Next() will return the next number in the file
        R = ConverterTools.read_fortran_file(self,self.dft_file,self.fortran_to_replace)
        try:
            energy_unit = 1.0                              # the energy conversion factor is 1.0, we assume eV in files
            n_k = int(R.next())                            # read the number of k points
            k_dep_projection = 0                          
            SP = 0                                        # no spin-polarision
            SO = 0                                        # no spin-orbit 
            charge_below = 0.0                            # total charge below energy window is set to 0
            density_required = R.next()                   # density required, for setting the chemical potential
            symm_op = 0                                   # No symmetry groups for the k-sum

            # the information on the non-correlated shells is needed for defining dimension of matrices:
            n_shells = int(R.next())                      # number of shells considered in the Wanniers
                                                          # corresponds to index R in formulas
            # now read the information about the shells:
            shells = [ [ int(R.next()) for i in range(4) ] for icrsh in range(n_shells) ]    # reads iatom, sort, l, dim

            n_corr_shells = int(R.next())                 # number of corr. shells (e.g. Fe d, Ce f) in the unit cell, 
                                                          # corresponds to index R in formulas
            # now read the information about the shells:
            corr_shells = [ [ int(R.next()) for i in range(6) ] for icrsh in range(n_corr_shells) ]    # reads iatom, sort, l, dim, SO flag, irep

            # determine the number of inequivalent correlated shells and maps, needed for further reading
            [n_inequiv_shells, corr_to_inequiv, inequiv_to_corr] = ConverterTools.det_shell_equivalence(self,corr_shells)

            use_rotations = 0
            rot_mat = [numpy.identity(corr_shells[icrsh][3],numpy.complex_) for icrsh in xrange(n_corr_shells)]
            rot_mat_time_inv = [0 for i in range(n_corr_shells)]
            
            # Representative representations are read from file
            n_reps = [1 for i in range(n_inequiv_shells)]
            dim_reps = [0 for i in range(n_inequiv_shells)]
            T = []
            for icrsh in range(n_inequiv_shells):
                n_reps[icrsh] = int(R.next())   # number of representatives ("subsets"), e.g. t2g and eg
                dim_reps[icrsh] = [int(R.next()) for i in range(n_reps[icrsh])]   # dimensions of the subsets
            
                # The transformation matrix:
                # is of dimension 2l+1, it is taken to be standard d (as in Wien2k)
                ll = 2*corr_shells[inequiv_to_corr[icrsh]][2]+1
                lmax = ll * (corr_shells[inequiv_to_corr[icrsh]][4] + 1)
                T.append(numpy.zeros([lmax,lmax],numpy.complex_))
                
                T[icrsh] = numpy.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                                       [1.0/sqrt(2.0), 0.0, 0.0, 0.0, 1.0/sqrt(2.0)],
                                       [-1.0/sqrt(2.0), 0.0, 0.0, 0.0, 1.0/sqrt(2.0)],
                                       [0.0, 1.0/sqrt(2.0), 0.0, -1.0/sqrt(2.0), 0.0],
                                       [0.0, 1.0/sqrt(2.0), 0.0, 1.0/sqrt(2.0), 0.0]])

            # Spin blocks to be read:
            n_spin_blocs = SP + 1 - SO   # number of spins to read for Norbs and Ham, NOT Projectors
        
            # define the number of n_orbitals for all k points: it is the number of total bands and independent of k!
            n_orb = sum([ shells[ish][3] for ish in range(n_shells) ])
            n_orbitals = numpy.ones([n_k,n_spin_blocs],numpy.int) * n_orb

            # Initialise the projectors:
            proj_mat = numpy.zeros([n_k,n_spin_blocs,n_corr_shells,max(numpy.array(corr_shells)[:,3]),max(n_orbitals)],numpy.complex_)

            # Read the projectors from the file:
            for ik in xrange(n_k):
                for icrsh in range(n_corr_shells):
                    for isp in range(n_spin_blocs):

                        # calculate the offset:
                        offset = 0
                        no = 0
                        for i in range(n_shells):
                            if (no==0):
                                if ((shells[i][0]==corr_shells[icrsh][0]) and (shells[i][1]==corr_shells[icrsh][1])):
                                    no = corr_shells[icrsh][3]
                                else:
                                    offset += shells[i][3]

                        proj_mat[ik,isp,icrsh,0:no,offset:offset+no] = numpy.identity(no)
                    
            # now define the arrays for weights and hopping ...
            bz_weights = numpy.ones([n_k],numpy.float_)/ float(n_k)  # w(k_index),  default normalisation 
            hopping = numpy.zeros([n_k,n_spin_blocs,max(n_orbitals),max(n_orbitals)],numpy.complex_)

            if (weights_in_file):
                # weights in the file
                for ik in xrange(n_k) : bz_weights[ik] = R.next()
                
            # if the sum over spins is in the weights, take it out again!!
            sm = sum(bz_weights)
            bz_weights[:] /= sm 

            # Grab the H
            for isp in range(n_spin_blocs):
                for ik in xrange(n_k) :
                    no = n_orbitals[ik,isp]
            # IF TRUE, FIRST READ ALL REAL COMPONENTS OF ONE kPOINT, OTHERWISE TUPLE OF real,im        
                    if (first_real_part_matrix):
                        
                        for i in xrange(no):
                            if (only_upper_triangle):
                                istart = i
                            else:
                                istart = 0
                            for j in xrange(istart,no):
                                hopping[ik,isp,i,j] = R.next()
                
                        for i in xrange(no):
                            if (only_upper_triangle):
                                istart = i
                            else:
                                istart = 0
                            for j in xrange(istart,no):
                                hopping[ik,isp,i,j] += R.next() * 1j
                                if ((only_upper_triangle)and(i!=j)): hopping[ik,isp,j,i] = hopping[ik,isp,i,j].conjugate()
                
                    else:
                    
                        for i in xrange(no):
                            if (only_upper_triangle):
                                istart = i
                            else:
                                istart = 0
                            for j in xrange(istart,no):
                                hopping[ik,isp,i,j] = R.next()
                                hopping[ik,isp,i,j] += R.next() * 1j
                            
                                if ((only_upper_triangle)and(i!=j)): hopping[ik,isp,j,i] = hopping[ik,isp,i,j].conjugate()
            # keep some things that we need for reading parproj:
            things_to_set = ['n_shells','shells','n_corr_shells','corr_shells','n_spin_blocs','n_orbitals','n_k','SO','SP','energy_unit']
            for it in things_to_set: setattr(self,it,locals()[it])
        except StopIteration : # a more explicit error if the file is corrupted.
            raise "HK Converter : reading file dft_file failed!"

        R.close()

        # Save to the HDF5:
        ar = HDFArchive(self.hdf_file,'a')
        if not (self.dft_subgrp in ar): ar.create_group(self.dft_subgrp) 
        things_to_save = ['energy_unit','n_k','k_dep_projection','SP','SO','charge_below','density_required',
                          'symm_op','n_shells','shells','n_corr_shells','corr_shells','use_rotations','rot_mat',
                          'rot_mat_time_inv','n_reps','dim_reps','T','n_orbitals','proj_mat','bz_weights','hopping',
                          'n_inequiv_shells', 'corr_to_inequiv', 'inequiv_to_corr']
        for it in things_to_save: ar[self.dft_subgrp][it] = locals()[it]
        del ar             
