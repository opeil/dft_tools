################################################################################
#
# TRIQS: a Toolbox for Research in Interacting Quantum Systems
#
# Copyright (C) 2011 by M. Aichhorn, L. Pourovskii, V. Vildosola
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
from pytriqs.gf.local import *
import pytriqs.utility.mpi as mpi
from symmetry import *
from sumk_dft import SumkDFT

class SumkDFTTools(SumkDFT):
    """Extends the SumkDFT class with some tools for analysing the data."""


    def __init__(self, hdf_file, h_field = 0.0, use_dft_blocks = False, dft_data = 'dft_input', symmcorr_data = 'dft_symmcorr_input',
                 parproj_data = 'dft_parproj_input', symmpar_data = 'dft_symmpar_input', bands_data = 'dft_bands_input', 
                 transp_data = 'dft_transp_input'):

        SumkDFT.__init__(self, hdf_file=hdf_file, h_field=h_field, use_dft_blocks=use_dft_blocks,
                          dft_data=dft_data, symmcorr_data=symmcorr_data, parproj_data=parproj_data, 
                          symmpar_data=symmpar_data, bands_data=bands_data, transp_data=transp_data)


    def check_input_dos(self, om_min, om_max, n_om, beta=10, broadening=0.01):

        delta_om = (om_max-om_min)/(n_om-1)
        om_mesh = numpy.zeros([n_om],numpy.float_)
        for i in range(n_om): om_mesh[i] = om_min + delta_om * i

        DOS = {}
        for sp in self.spin_block_names[self.SO]:
            DOS[sp] = numpy.zeros([n_om],numpy.float_)

        DOSproj     = [ {} for ish in range(self.n_inequiv_shells) ]
        DOSproj_orb = [ {} for ish in range(self.n_inequiv_shells) ]
        for ish in range(self.n_inequiv_shells):
            for sp in self.spin_block_names[self.corr_shells[self.inequiv_to_corr[ish]]['SO']]:
                dim = self.corr_shells[self.inequiv_to_corr[ish]]['dim']
                DOSproj[ish][sp] = numpy.zeros([n_om],numpy.float_)
                DOSproj_orb[ish][sp] = numpy.zeros([n_om,dim,dim],numpy.float_)

        # init:
        Gloc = []
        for icrsh in range(self.n_corr_shells):
            spn = self.spin_block_names[self.corr_shells[icrsh]['SO']]
            glist = lambda : [ GfReFreq(indices = inner, window = (om_min,om_max), n_points = n_om) for block,inner in self.gf_struct_sumk[icrsh]]
            Gloc.append(BlockGf(name_list = spn, block_list = glist(),make_copies=False))
        for icrsh in range(self.n_corr_shells): Gloc[icrsh].zero()                        # initialize to zero

        for ik in range(self.n_k):

            G_latt_w=self.lattice_gf(ik=ik,mu=self.chemical_potential,iw_or_w="w",broadening=broadening,mesh=(om_min,om_max,n_om),with_Sigma=False)
            G_latt_w *= self.bz_weights[ik]

            # non-projected DOS
            for iom in range(n_om):
                for bname,gf in G_latt_w:
                    asd = gf.data[iom,:,:].imag.trace()/(-3.1415926535)
                    DOS[bname][iom] += asd

            for icrsh in range(self.n_corr_shells):
                tmp = Gloc[icrsh].copy()
                for bname,gf in tmp: tmp[bname] << self.downfold(ik,icrsh,bname,G_latt_w[bname],gf) # downfolding G
                Gloc[icrsh] += tmp

        if self.symm_op != 0: Gloc = self.symmcorr.symmetrize(Gloc)

        if self.use_rotations:
            for icrsh in range(self.n_corr_shells):
                for bname,gf in Gloc[icrsh]: Gloc[icrsh][bname] << self.rotloc(icrsh,gf,direction='toLocal')

        # Gloc can now also be used to look at orbitally resolved quantities
        for ish in range(self.n_inequiv_shells):
            for bname,gf in Gloc[self.inequiv_to_corr[ish]]: # loop over spins
                for iom in range(n_om): DOSproj[ish][bname][iom] += gf.data[iom,:,:].imag.trace()/(-3.1415926535)

                DOSproj_orb[ish][bname][:,:,:] += gf.data[:,:,:].imag/(-3.1415926535)

        # output:
        if mpi.is_master_node():
            for sp in self.spin_block_names[self.SO]:
                f=open('DOS%s.dat'%sp, 'w')
                for i in range(n_om): f.write("%s    %s\n"%(om_mesh[i],DOS[sp][i]))
                f.close()

                for ish in range(self.n_inequiv_shells):
                    f=open('DOS%s_proj%s.dat'%(sp,ish),'w')
                    for i in range(n_om): f.write("%s    %s\n"%(om_mesh[i],DOSproj[ish][sp][i]))
                    f.close()

                    for i in range(self.corr_shells[self.inequiv_to_corr[ish]]['dim']):
                        for j in range(i,self.corr_shells[self.inequiv_to_corr[ish]]['dim']):
                            Fname = 'DOS'+sp+'_proj'+str(ish)+'_'+str(i)+'_'+str(j)+'.dat'
                            f=open(Fname,'w')
                            for iom in range(n_om): f.write("%s    %s\n"%(om_mesh[iom],DOSproj_orb[ish][sp][iom,i,j]))
                            f.close()




    def read_parproj_input_from_hdf(self):
        """
        Reads the data for the partial projectors from the HDF file
        """

        things_to_read = ['dens_mat_below','n_parproj','proj_mat_all','rot_mat_all','rot_mat_all_time_inv']
        value_read = self.read_input_from_hdf(subgrp=self.parproj_data,things_to_read = things_to_read)
        return value_read



    def dos_partial(self,broadening=0.01):
        """calculates the orbitally-resolved DOS"""

        assert hasattr(self,"Sigma_imp_w"), "dos_partial: Set Sigma_imp_w first."

        value_read = self.read_parproj_input_from_hdf()
        if not value_read: return value_read
        if self.symm_op: self.symmpar = Symmetry(self.hdf_file,subgroup=self.symmpar_data)

        mu = self.chemical_potential

        gf_struct_proj = [ [ (sp, range(self.shells[i]['dim'])) for sp in self.spin_block_names[self.SO] ]  for i in range(self.n_shells) ]
        Gproj = [BlockGf(name_block_generator = [ (block,GfReFreq(indices = inner, mesh = self.Sigma_imp_w[0].mesh)) 
                               for block,inner in gf_struct_proj[ish] ], make_copies = False ) for ish in range(self.n_shells)]
        for ish in range(self.n_shells): Gproj[ish].zero()

        mesh = [x.real for x in self.Sigma_imp_w[0].mesh]
        n_om = len(mesh)

        DOS = {}
        for sp in self.spin_block_names[self.SO]:
            DOS[sp] = numpy.zeros([n_om],numpy.float_)

        DOSproj     = [ {} for ish in range(self.n_shells) ]
        DOSproj_orb = [ {} for ish in range(self.n_shells) ]
        for ish in range(self.n_shells):
            for sp in self.spin_block_names[self.SO]:
                dim = self.shells[ish]['dim']
                DOSproj[ish][sp] = numpy.zeros([n_om],numpy.float_)
                DOSproj_orb[ish][sp] = numpy.zeros([n_om,dim,dim],numpy.float_)

        ikarray=numpy.array(range(self.n_k))

        for ik in mpi.slice_array(ikarray):

            G_latt_w = self.lattice_gf(ik=ik,mu=mu,iw_or_w="w",broadening=broadening)
            G_latt_w *= self.bz_weights[ik]

            # non-projected DOS
            for iom in range(n_om):
                for bname,gf in G_latt_w: DOS[bname][iom] += gf.data[iom,:,:].imag.trace()/(-3.1415926535)

            #projected DOS:
            for ish in range(self.n_shells):
                tmp = Gproj[ish].copy()
                for ir in range(self.n_parproj[ish]):
                    for bname,gf in tmp: tmp[bname] << self.downfold(ik,ish,bname,G_latt_w[bname],gf,shells='all',ir=ir)
                    Gproj[ish] += tmp

        # collect data from mpi:
        for bname in DOS:
            DOS[bname] = mpi.all_reduce(mpi.world, DOS[bname], lambda x,y : x+y)
        for ish in range(self.n_shells):
            Gproj[ish] << mpi.all_reduce(mpi.world, Gproj[ish], lambda x,y : x+y)
        mpi.barrier()

        if self.symm_op != 0: Gproj = self.symmpar.symmetrize(Gproj)

        # rotation to local coord. system:
        if self.use_rotations:
            for ish in range(self.n_shells):
                for bname,gf in Gproj[ish]: Gproj[ish][bname] << self.rotloc(ish,gf,direction='toLocal',shells='all')

        for ish in range(self.n_shells):
            for bname,gf in Gproj[ish]:
                for iom in range(n_om): DOSproj[ish][bname][iom] += gf.data[iom,:,:].imag.trace()/(-3.1415926535)
                DOSproj_orb[ish][bname][:,:,:] += gf.data[:,:,:].imag / (-3.1415926535)


        if mpi.is_master_node():
            # output to files
            for sp in self.spin_block_names[self.SO]:
                f=open('./DOScorr%s.dat'%sp, 'w')
                for i in range(n_om): f.write("%s    %s\n"%(mesh[i],DOS[sp][i]))
                f.close()

                # partial
                for ish in range(self.n_shells):
                    f=open('DOScorr%s_proj%s.dat'%(sp,ish),'w')
                    for i in range(n_om): f.write("%s    %s\n"%(mesh[i],DOSproj[ish][sp][i]))
                    f.close()

                    for i in range(self.shells[ish]['dim']):
                        for j in range(i,self.shells[ish]['dim']):
                            Fname = './DOScorr'+sp+'_proj'+str(ish)+'_'+str(i)+'_'+str(j)+'.dat'
                            f=open(Fname,'w')
                            for iom in range(n_om): f.write("%s    %s\n"%(mesh[iom],DOSproj_orb[ish][sp][iom,i,j]))
                            f.close()




    def spaghettis(self,broadening,shift=0.0,plot_range=None, ishell=None, invert_Akw=False, fermi_surface=False):
        """ Calculates the correlated band structure with a real-frequency self energy.
            ATTENTION: Many things from the original input file are overwritten!!!"""

        assert hasattr(self,"Sigma_imp_w"), "spaghettis: Set Sigma_imp_w first."
        things_to_read = ['n_k','n_orbitals','proj_mat','hopping','n_parproj','proj_mat_all']
        value_read = self.read_input_from_hdf(subgrp=self.bands_data,things_to_read=things_to_read)
        if not value_read: return value_read

        if fermi_surface: ishell=None

        # FIXME CAN REMOVE?
        # print hamiltonian for checks:
        if self.SP == 1 and self.SO == 0:
            f1=open('hamup.dat','w')
            f2=open('hamdn.dat','w')

            for ik in range(self.n_k):
                for i in range(self.n_orbitals[ik,0]):
                    f1.write('%s    %s\n'%(ik,self.hopping[ik,0,i,i].real))
                for i in range(self.n_orbitals[ik,1]):
                    f2.write('%s    %s\n'%(ik,self.hopping[ik,1,i,i].real))
                f1.write('\n')
                f2.write('\n')
            f1.close()
            f2.close()
        else:
            f=open('ham.dat','w')
            for ik in range(self.n_k):
                for i in range(self.n_orbitals[ik,0]):
                    f.write('%s    %s\n'%(ik,self.hopping[ik,0,i,i].real))
                f.write('\n')
            f.close()


        #=========================================
        # calculate A(k,w):

        mu = self.chemical_potential
        spn = self.spin_block_names[self.SO]

        # init DOS:
        mesh = [x.real for x in self.Sigma_imp_w[0].mesh]
        n_om = len(mesh)

        if plot_range is None:
            om_minplot = mesh[0]-0.001
            om_maxplot = mesh[n_om-1] + 0.001
        else:
            om_minplot = plot_range[0]
            om_maxplot = plot_range[1]

        if ishell is None:
            Akw = {}
            for sp in spn: Akw[sp] = numpy.zeros([self.n_k, n_om ],numpy.float_)
        else:
            Akw = {}
            for sp in spn: Akw[sp] = numpy.zeros([self.shells[ishell]['dim'],self.n_k, n_om ],numpy.float_)

        if fermi_surface:
            om_minplot = -2.0*broadening
            om_maxplot =  2.0*broadening
            Akw = {}
            for sp in spn: Akw[sp] = numpy.zeros([self.n_k,1],numpy.float_)

        if not ishell is None:
            GFStruct_proj =  [ (sp, range(self.shells[ishell]['dim'])) for sp in spn ]
            Gproj = BlockGf(name_block_generator = [ (block,GfReFreq(indices = inner, mesh = self.Sigma_imp_w[0].mesh)) for block,inner in GFStruct_proj ], make_copies = False)
            Gproj.zero()

        for ik in range(self.n_k):

            G_latt_w = self.lattice_gf(ik=ik,mu=mu,iw_or_w="w",broadening=broadening)
            if ishell is None:
                # non-projected A(k,w)
                for iom in range(n_om):
                    if (mesh[iom] > om_minplot) and (mesh[iom] < om_maxplot):
                        if fermi_surface:
                            for bname,gf in G_latt_w: Akw[bname][ik,0] += gf.data[iom,:,:].imag.trace()/(-3.1415926535) * (mesh[1]-mesh[0])
                        else:
                            for bname,gf in G_latt_w: Akw[bname][ik,iom] += gf.data[iom,:,:].imag.trace()/(-3.1415926535)
                            Akw[bname][ik,iom] += ik*shift                       # shift Akw for plotting in xmgrace -- REMOVE


            else:
                # projected A(k,w):
                Gproj.zero()
                tmp = Gproj.copy()
                for ir in range(self.n_parproj[ishell]):
                    for bname,gf in tmp: tmp[bname] << self.downfold(ik,ishell,bname,G_latt_w[bname],gf,shells='all',ir=ir)
                    Gproj += tmp

                # FIXME NEED TO READ IN ROTMAT_ALL FROM PARPROJ SUBGROUP, REPLACE ROTLOC WITH ROTLOC_ALL
                # TO BE FIXED:
                # rotate to local frame
                #if (self.use_rotations):
                #    for bname,gf in Gproj: Gproj[bname] << self.rotloc(0,gf,direction='toLocal')

                for iom in range(n_om):
                    if (mesh[iom] > om_minplot) and (mesh[iom] < om_maxplot):
                        for ish in range(self.shells[ishell]['dim']):
                            for ibn in spn:
                                Akw[ibn][ish,ik,iom] = Gproj[ibn].data[iom,ish,ish].imag/(-3.1415926535)


        # END k-LOOP
        if mpi.is_master_node():
            if ishell is None:

                for ibn in spn:
                    # loop over GF blocs:

                    if invert_Akw:
                        maxAkw=Akw[ibn].max()
                        minAkw=Akw[ibn].min()


                    # open file for storage:
                    if fermi_surface:
                        f=open('FS_'+ibn+'.dat','w')
                    else:
                        f=open('Akw_'+ibn+'.dat','w')

                    for ik in range(self.n_k):
                        if fermi_surface:
                            if invert_Akw:
                                Akw[ibn][ik,0] = 1.0/(minAkw-maxAkw)*(Akw[ibn][ik,0] - maxAkw)
                            f.write('%s    %s\n'%(ik,Akw[ibn][ik,0]))
                        else:
                            for iom in range(n_om):
                                if (mesh[iom] > om_minplot) and (mesh[iom] < om_maxplot):
                                    if invert_Akw:
                                        Akw[ibn][ik,iom] = 1.0/(minAkw-maxAkw)*(Akw[ibn][ik,iom] - maxAkw)
                                    if shift > 0.0001:
                                        f.write('%s      %s\n'%(mesh[iom],Akw[ibn][ik,iom]))
                                    else:
                                        f.write('%s     %s      %s\n'%(ik,mesh[iom],Akw[ibn][ik,iom]))

                            f.write('\n')

                    f.close()

            else:
                for ibn in spn:
                    for ish in range(self.shells[ishell]['dim']):

                        if invert_Akw:
                            maxAkw=Akw[ibn][ish,:,:].max()
                            minAkw=Akw[ibn][ish,:,:].min()

                        f=open('Akw_'+ibn+'_proj'+str(ish)+'.dat','w')

                        for ik in range(self.n_k):
                            for iom in range(n_om):
                                if (mesh[iom] > om_minplot) and (mesh[iom] < om_maxplot):
                                    if invert_Akw:
                                        Akw[ibn][ish,ik,iom] = 1.0/(minAkw-maxAkw)*(Akw[ibn][ish,ik,iom] - maxAkw)
                                    if shift > 0.0001:
                                        f.write('%s      %s\n'%(mesh[iom],Akw[ibn][ish,ik,iom]))
                                    else:
                                        f.write('%s     %s      %s\n'%(ik,mesh[iom],Akw[ibn][ish,ik,iom]))

                            f.write('\n')

                        f.close()


    def partial_charges(self,beta=40):
        """Calculates the orbitally-resolved density matrix for all the orbitals considered in the input.
           The theta-projectors are used, hence case.parproj data is necessary"""

        value_read = self.read_parproj_input_from_hdf()
        if not value_read: return value_read
        if self.symm_op: self.symmpar = Symmetry(self.hdf_file,subgroup=self.symmpar_data)

        # Density matrix in the window
        spn = self.spin_block_names[self.SO]
        ntoi = self.spin_names_to_ind[self.SO]
        self.dens_mat_window = [ [numpy.zeros([self.shells[ish]['dim'],self.shells[ish]['dim']],numpy.complex_) for ish in range(self.n_shells)]
                                 for isp in range(len(spn)) ]    # init the density matrix

        mu = self.chemical_potential
        GFStruct_proj = [ [ (sp, range(self.shells[i]['dim'])) for sp in spn ]  for i in range(self.n_shells) ]
        if hasattr(self,"Sigma_imp_iw"):
            Gproj = [BlockGf(name_block_generator = [ (block,GfImFreq(indices = inner, mesh = self.Sigma_imp_iw[0].mesh)) for block,inner in GFStruct_proj[ish] ], make_copies = False)
                     for ish in range(self.n_shells)]
            beta = self.Sigma_imp_iw[0].mesh.beta
        else:
            Gproj = [BlockGf(name_block_generator = [ (block,GfImFreq(indices = inner, beta = beta)) for block,inner in GFStruct_proj[ish] ], make_copies = False)
                     for ish in range(self.n_shells)]

        for ish in range(self.n_shells): Gproj[ish].zero()

        ikarray=numpy.array(range(self.n_k))

        for ik in mpi.slice_array(ikarray):
            G_latt_iw = self.lattice_gf(ik=ik,mu=mu,iw_or_w="iw",beta=beta)
            G_latt_iw *= self.bz_weights[ik]

            for ish in range(self.n_shells):
                tmp = Gproj[ish].copy()
                for ir in range(self.n_parproj[ish]):
                    for bname,gf in tmp: tmp[bname] << self.downfold(ik,ish,bname,G_latt_iw[bname],gf,shells='all',ir=ir)
                    Gproj[ish] += tmp

        #collect data from mpi:
        for ish in range(self.n_shells):
            Gproj[ish] << mpi.all_reduce(mpi.world, Gproj[ish], lambda x,y : x+y)
        mpi.barrier()


        # Symmetrisation:
        if self.symm_op != 0: Gproj = self.symmpar.symmetrize(Gproj)

        for ish in range(self.n_shells):

            # Rotation to local:
            if self.use_rotations:
                for bname,gf in Gproj[ish]: Gproj[ish][bname] << self.rotloc(ish,gf,direction='toLocal',shells='all')

            isp = 0
            for bname,gf in Gproj[ish]: #dmg.append(Gproj[ish].density()[bname])
                self.dens_mat_window[isp][ish] = Gproj[ish].density()[bname]
                isp+=1

        # add Density matrices to get the total:
        dens_mat = [ [ self.dens_mat_below[ntoi[spn[isp]]][ish]+self.dens_mat_window[isp][ish] for ish in range(self.n_shells)]
                     for isp in range(len(spn)) ]

        return dens_mat

# ----------------- transport -----------------------

    def read_transport_input_from_hdf(self):
        """
        Reads the data for transport calculations from the HDF file
        """
        thingstoread = ['band_window','band_window_optics','velocities_k','lattice_angles','lattice_constants','lattice_type','n_symmetries','rot_symmetries']
        retval = self.read_input_from_hdf(subgrp=self.transp_data,things_to_read = thingstoread)
        return retval
    
    
    def cellvolume(self, lattice_type, lattice_constants, latticeangle):
        """
        Calculate cell volume: volumecc conventional cell, volumepc, primitive cell.
        """
        a = lattice_constants[0]
        b = lattice_constants[1]
        c = lattice_constants[2]
        c_al = numpy.cos(latticeangle[0])
        c_be = numpy.cos(latticeangle[1])
        c_ga = numpy.cos(latticeangle[2])
        volumecc = a * b * c * numpy.sqrt(1 + 2 * c_al * c_be * c_ga - c_al ** 2 - c_be * 82 - c_ga ** 2)
      
        det = {"P":1, "F":4, "B":2, "R":3, "H":1, "CXY":2, "CYZ":2, "CXZ":2}
        volumepc = volumecc / det[lattice_type]
      
        return volumecc, volumepc


    def transport_distribution(self, directions=['xx'], energy_window=None, Om_mesh=[0.0], beta=40, with_Sigma=False, n_om=None, broadening=0.01):
        """
        calculate Tr A(k,w) v(k) A(k, w+Om) v(k).
        energy_window: regime for omega integral
        Om_mesh: mesh for optic conductivitity. Om_mesh is repinned to the self-energy mesh!
        directions: list of directions: xx,yy,zz,xy,yz,zx. 
        with_Sigma: Use Sigma_w = 0 if False (In this case it is necessary to specifiy the energywindow (energy_window),
        the number of omega points (n_om) in the window and the broadening (broadening)).
        """
       
        # Check if wien converter was called and read transport subgroup form hdf file
        if mpi.is_master_node():
            ar = HDFArchive(self.hdf_file, 'a')
            if not (self.transp_data in ar): raise IOError, "transport_distribution: No %s subgroup in hdf file found! Call convert_transp_input first." %self.transp_data
        self.read_transport_input_from_hdf()
        
        n_inequiv_spin_blocks = self.SP + 1 - self.SO  # up and down are equivalent if SP = 0
        self.directions = directions
	dir_to_int = {'x':0, 'y':1, 'z':2}
            
        # k-dependent-projections.
        assert self.k_dep_projection == 1, "transport_distribution: k dependent projection is not implemented!"
        
        # calculate A(k,w)
        #######################################
        
        # Define mesh for Greens function and in the specified energy window
        if (with_Sigma == True):
            self.omega = numpy.array([round(x.real,12) for x in self.Sigma_imp_w[0].mesh])
            mesh = None
            mu = self.chemical_potential
            n_om = len(self.omega)
            print "Using omega mesh provided by Sigma!"

            if energy_window is not None:
                # Find according window in Sigma  mesh
                ioffset = numpy.sum(self.omega < energy_window[0])
                self.omega = self.omega[numpy.logical_and(self.omega >= energy_window[0], self.omega <= energy_window[1])]
                n_om = len(self.omega)
                
                # Truncate Sigma to given omega window
		# In the future there should be an option in gf to manipulate the mesh (e.g. truncate) directly.
		# For we stick with this: 
                for icrsh in range(self.n_corr_shells):
                    Sigma_save = self.Sigma_imp_w[icrsh].copy()
                    spn = self.spin_block_names[self.corr_shells[icrsh]['SO']]
                    glist = lambda : [ GfReFreq(indices = inner, window=(self.omega[0], self.omega[-1]),n_points=n_om) for block, inner in self.gf_struct_sumk[icrsh]]
                    self.Sigma_imp_w[icrsh] = BlockGf(name_list = spn, block_list = glist(),make_copies=False)
                    for i,g in self.Sigma_imp_w[icrsh]:
                        for iL in g.indices:
                            for iR in g.indices:
                                for iom in xrange(n_om):
                                    g.data[iom,iL,iR] = Sigma_save[i].data[ioffset+iom,iL,iR]             
        else:
            assert n_om is not None, "transport_distribution: Number of omega points (n_om) needed to calculate transport distribution!"
            assert energy_window is not None, "transport_distribution: Energy window needed to calculate transport distribution!"
	    assert broadening != 0.0 and broadening is not None, "transport_distribution: Broadening necessary to calculate transport distribution!" 
	    self.omega = numpy.linspace(energy_window[0],energy_window[1],n_om)
            mesh = [energy_window[0], energy_window[1], n_om]
	    mu = 0.0

        # Check if energy_window is sufficiently large
        if (abs(self.fermi_dis(self.omega[0]*beta)*self.fermi_dis(-self.omega[0]*beta)) > 1e-5
            or abs(self.fermi_dis(self.omega[-1]*beta)*self.fermi_dis(-self.omega[-1]*beta)) > 1e-5):
                print "\n####################################################################"
                print "transport_distribution: WARNING - energy window might be too narrow!"
                print "####################################################################\n"

        # Define mesh for optic conductivity
        d_omega = round(numpy.abs(self.omega[0] - self.omega[1]), 12)
        iOm_mesh = numpy.array([int(Om / d_omega) for Om in Om_mesh])
        self.Om_mesh = iOm_mesh * d_omega

        if mpi.is_master_node():
            print "Chemical potential: ", mu
            print "Using n_om = %s points in the energy_window [%s,%s]"%(n_om, self.omega[0], self.omega[-1]),
            print "where the omega vector is:"
            print self.omega
            print "Calculation requested for Omega mesh:   ", numpy.array(Om_mesh)
            print "Omega mesh automatically repinned to:  ", self.Om_mesh
        
        self.Gamma_w = {direction: numpy.zeros((len(self.Om_mesh), n_om), dtype=numpy.float_) for direction in self.directions}
        
        # Sum over all k-points
        ikarray = numpy.array(range(self.n_k))
        for ik in mpi.slice_array(ikarray):
            # Calculate G_w  for ik and initialize A_kw
            G_w = self.lattice_gf(ik, mu, iw_or_w="w", beta=beta, broadening=broadening, mesh=mesh, with_Sigma=with_Sigma)
            A_kw = [numpy.zeros((self.n_orbitals[ik][isp], self.n_orbitals[ik][isp], n_om), dtype=numpy.complex_) 
				for isp in range(n_inequiv_spin_blocks)]
            
	    for isp in range(n_inequiv_spin_blocks):
                # Obtain A_kw from G_w (swapaxes is used to have omega in the 3rd dimension)
                A_kw[isp].real = -copy.deepcopy(G_w[self.spin_block_names[self.SO][isp]].data.swapaxes(0,1).swapaxes(1,2)).imag / numpy.pi 
                b_min = max(self.band_window[isp][ik, 0], self.band_window_optics[isp][ik, 0])
                b_max = min(self.band_window[isp][ik, 1], self.band_window_optics[isp][ik, 1])
                A_i = slice(b_min - self.band_window[isp][ik, 0], b_max - self.band_window[isp][ik, 0] + 1)
                v_i = slice(b_min - self.band_window_optics[isp][ik, 0], b_max - self.band_window_optics[isp][ik, 0] + 1)
                
                # loop over all symmetries
                for R in self.rot_symmetries:
                    # get transformed velocity under symmetry R
                    vel_R = copy.deepcopy(self.velocities_k[isp][ik])
                    for nu1 in range(self.band_window_optics[isp][ik, 1] - self.band_window_optics[isp][ik, 0] + 1):
                        for nu2 in range(self.band_window_optics[isp][ik, 1] - self.band_window_optics[isp][ik, 0] + 1):
                            vel_R[nu1][nu2][:] = numpy.dot(R, vel_R[nu1][nu2][:])
                    
                    # calculate Gamma_w for each direction from the velocities vel_R and the spectral function A_kw
		    for direction in self.directions:
                        for iw in xrange(n_om):
                            for iq in range(len(self.Om_mesh)):
                                if(iw + iOm_mesh[iq] >= n_om): continue
                                self.Gamma_w[direction][iq, iw] += (numpy.dot(numpy.dot(numpy.dot(vel_R[v_i, v_i, dir_to_int[direction[0]]], 
                                                                    A_kw[isp][A_i, A_i, iw]), vel_R[v_i, v_i, dir_to_int[direction[1]]]), 
                                                                    A_kw[isp][A_i, A_i, iw + iOm_mesh[iq]]).trace().real * self.bz_weights[ik])
        
	for direction in self.directions: 
            self.Gamma_w[direction] = (mpi.all_reduce(mpi.world, self.Gamma_w[direction], lambda x, y : x + y) 
					/ self.cellvolume(self.lattice_type, self.lattice_constants, self.lattice_angles)[1] / self.n_symmetries)

        
    def transport_coefficient(self, direction, iq=0, n=0, beta=40):
        """
        calculates the transport coefficients A_n in a given direction and for a given Omega. (see documentation)
        A_1 is set to nan if requested for Omega != 0.0
        iq: index of Omega point in Om_mesh
        direction: 'xx','yy','zz','xy','xz','yz'
        """
        if not (mpi.is_master_node()): return
        
        assert hasattr(self,'Gamma_w'), "transport_coefficient: Run transport_distribution first or load data from h5!"
        A = 0.0
        omegaT = self.omega * beta
        d_omega = self.omega[1] - self.omega[0]
        if (self.Om_mesh[iq] == 0.0):
            for iw in xrange(self.Gamma_w[direction].shape[1]):
                A += self.Gamma_w[direction][iq, iw] * self.fermi_dis(omegaT[iw]) * self.fermi_dis(-omegaT[iw]) * numpy.float(omegaT[iw])**n * d_omega
        elif (n == 0.0):
            for iw in xrange(self.Gamma_w[direction].shape[1]):
                A += (self.Gamma_w[direction][iq, iw] * (self.fermi_dis(omegaT[iw]) - self.fermi_dis(omegaT[iw] + self.Om_mesh[iq] * beta)) 
                     / (self.Om_mesh[iq] * beta) * d_omega)
        else:
                A = numpy.nan
        return A *  numpy.pi * (2.0-self.SP)

    
    def conductivity_and_seebeck(self, beta=40):
        """
        Calculates the Seebeck coefficient and the conductivity for a given Gamma_w
        """    
        if not (mpi.is_master_node()): return
    
        assert hasattr(self,'Gamma_w'), "conductivity_and_seebeck: Run transport_distribution first or load data from h5!"
        n_q = self.Gamma_w[self.directions[0]].shape[0]
        
        A0 = {direction: numpy.full((n_q,),numpy.nan) for direction in self.directions}              
        A1 = {direction: numpy.full((n_q,),numpy.nan) for direction in self.directions}
        self.seebeck = {direction: numpy.nan for direction in self.directions}
        self.optic_cond = {direction: numpy.full((n_q,),numpy.nan) for direction in self.directions}
        
        for direction in self.directions:
            for iq in xrange(n_q):
                A0[direction][iq] = self.transport_coefficient(direction, iq=iq, n=0, beta=beta)
                A1[direction][iq] = self.transport_coefficient(direction, iq=iq, n=1, beta=beta)
                print "A_0 in direction %s for Omega = %.2f    %e a.u." % (direction, self.Om_mesh[iq], A0[direction][iq])
                print "A_1 in direction %s for Omega = %.2f    %e a.u." % (direction, self.Om_mesh[iq], A1[direction][iq])
                if ~numpy.isnan(A1[direction][iq]):
                    # Seebeck is overwritten if there is more than one Omega = 0 in Om_mesh
                    self.seebeck[direction] = - A1[direction][iq] / A0[direction][iq] * 86.17
            self.optic_cond[direction] = A0[direction] * 10700.0
            for iq in xrange(n_q):
               print "Conductivity in direction %s for Omega = %.2f       %f  x 10^4 Ohm^-1 cm^-1" % (direction, self.Om_mesh[iq], self.optic_cond[direction][iq])
               if not (numpy.isnan(A1[direction][iq])):
                    print "Seebeck in direction      %s for Omega = 0.00      %f  x 10^(-6) V/K" % (direction, self.seebeck[direction])
          

    def fermi_dis(self, x):
        """
        fermi distribution at x = omega * beta
        """
        return 1.0/(numpy.exp(x)+1)
    
