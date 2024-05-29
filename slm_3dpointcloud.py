import numpy
import os
import time
import matplotlib.pyplot as plt
import pycuda.driver as cuda_driver
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
from pycuda import gpuarray
import pycuda.autoinit # https://teratail.com/questions/128032

cuda_code = """ 
        texture<unsigned char, 2> tex;
        __global__ void project_to_slm_comp(float*spots_params, float*pup_xc, float*pup_yc, float*hologram_real,
         float*hologram_imag, float*float_pars, int*int_pars)
        {
            int tx = threadIdx.x+blockDim.x * blockIdx.x;
            int PUP_NP_COMP=int_pars[3];
            int PUP_INDEX=int_pars[4];
            tx+=PUP_INDEX;
            if (tx<PUP_INDEX+PUP_NP_COMP)
            {
                int NSPOTS=int_pars[1];
                float xc=pup_xc[tx];
                float yc=pup_yc[tx];
                float rc2=xc*xc+yc*yc;
                float tempphase;
                float tempden;
                float pist;
                float w_int;
                float COEFFXY=float_pars[0];
                float COEFFZ=float_pars[1];
                hologram_real[tx]=0.0;
                hologram_imag[tx]=0.0;
                for (int spot=0; spot < NSPOTS; spot++)
                {
                    pist=spots_params[7*spot+5];
                    w_int=spots_params[7*spot+3]*spots_params[7*spot+4];
                    tempphase=COEFFXY * (
                    xc*spots_params[7*spot]+yc*spots_params[7*spot+1])+ COEFFZ *rc2/1000*spots_params[7*spot+2];
                    hologram_real[tx] += w_int*cos(tempphase+pist);
                    hologram_imag[tx] += w_int*sin(tempphase+pist);                    
                }
            tempden=sqrt(hologram_real[tx]*hologram_real[tx]+hologram_imag[tx]*hologram_imag[tx]);
            hologram_real[tx] =hologram_real[tx]/tempden;
            hologram_imag[tx] =hologram_imag[tx]/tempden;             
            }
        }
        __global__ void project_to_spots_setup_comp(float*spots_params, float*pup_xc, float*pup_yc, float*hologram_real,
         float*hologram_imag, float*temp_real, float*temp_imag, float*pupil_int, float*float_pars, int*int_pars)
        {
            int tx = threadIdx.x+blockDim.x * blockIdx.x;
            int ty = threadIdx.y+blockDim.y * blockIdx.y;
            int NSPOTS=int_pars[1];
            int PUP_NP_COMP=int_pars[3];
            int PUP_INDEX=int_pars[4];
            int NPIX=blockDim.x;            
            float COEFFXY=float_pars[0];
            float COEFFZ=float_pars[1];
            float tempphase;
            if (ty<NSPOTS)
            {
                temp_real[NPIX*ty+tx]=0.0;
                temp_imag[NPIX*ty+tx]=0.0;
                for (int x=tx+PUP_INDEX; x<PUP_INDEX+PUP_NP_COMP; x+=NPIX)
                {
                    float xc=pup_xc[x];
                    float yc=pup_yc[x];
                    float rc2=xc*xc+yc*yc;
                    float hologram_phase=atan2(hologram_imag[x],hologram_real[x]);
                    tempphase=COEFFXY*(
                    xc*spots_params[7*ty]+yc*spots_params[7*ty+1])+ COEFFZ*rc2/1000*spots_params[7*ty+2];   
                    tempphase=hologram_phase-tempphase;
                    temp_real[NPIX*ty+tx] += pupil_int[x]*cos(tempphase);
                    temp_imag[NPIX*ty+tx] += pupil_int[x]*sin(tempphase);                    
                }
            }
        }
        __global__ void project_to_spots_end_comp(float*spots_params, float*temp_real, float*temp_imag, int*int_pars)
        {
            int tx = threadIdx.x+blockDim.x * blockIdx.x;
            int PUP_NP=int_pars[0];
            int NSPOTS = int_pars[1]; 
            int NPIX = int_pars[2];
            int PUP_NP_COMP=int_pars[3];
            float out_real = 0.0;
            float out_imag = 0.0;
            if (tx<NSPOTS)
            {
                for (int x=0; x<NPIX; x++)
                {
                    out_real+=temp_real[NPIX*tx+x];
                    out_imag+=temp_imag[NPIX*tx+x];                    
                }
            spots_params[7*tx+5]=atan2(out_imag, out_real);
            spots_params[7*tx+6]=(out_imag*out_imag + out_real*out_real)/spots_params[7*tx+4]/NSPOTS;
            }
        }
        __global__ void project_to_slm(float*spots_params, float*pup_xc, float*pup_yc, float*hologram_real,
         float*hologram_imag, float*float_pars, int*int_pars)
        {
            int tx = threadIdx.x+blockDim.x * blockIdx.x;
            int PUP_NP=int_pars[0];
            if (tx<PUP_NP)
            {
                int NSPOTS=int_pars[1];
                float xc=pup_xc[tx];
                float yc=pup_yc[tx];
                float rc2=xc*xc+yc*yc;
                float tempphase;
                float tempden;
                float pist;
                float w_int;
                float COEFFXY=float_pars[0];
                float COEFFZ=float_pars[1];
                hologram_real[tx]=0.0;
                hologram_imag[tx]=0.0;
                for (int spot=0; spot < NSPOTS; spot++)
                {
                    pist=spots_params[7*spot+5];
                    w_int=spots_params[7*spot+3]*spots_params[7*spot+4];
                    tempphase=COEFFXY * (
                    xc*spots_params[7*spot]+yc*spots_params[7*spot+1])+ COEFFZ *rc2/1000*spots_params[7*spot+2];
                    hologram_real[tx] += w_int*cos(tempphase+pist);
                    hologram_imag[tx] += w_int*sin(tempphase+pist);                    
                }
            tempden=sqrt(hologram_real[tx]*hologram_real[tx]+hologram_imag[tx]*hologram_imag[tx]);
            hologram_real[tx] =hologram_real[tx]/tempden;
            hologram_imag[tx] =hologram_imag[tx]/tempden;             
            }
        }
        __global__ void project_to_spots_setup(float*spots_params, float*pup_xc, float*pup_yc, float*hologram_real,
         float*hologram_imag, float*temp_real, float*temp_imag, float*pupil_int, float*float_pars, int*int_pars)
        {
            int tx = threadIdx.x+blockDim.x * blockIdx.x;
            int ty = threadIdx.y+blockDim.y * blockIdx.y;
            int PUP_NP=int_pars[0];
            int NSPOTS=int_pars[1];
            int NPIX=blockDim.x;            
            float COEFFXY=float_pars[0];
            float COEFFZ=float_pars[1];
            float tempphase;
            if (ty<NSPOTS)
            {
                temp_real[NPIX*ty+tx]=0.0;
                temp_imag[NPIX*ty+tx]=0.0;
                for (int x=tx; x<PUP_NP; x+=NPIX)
                {
                    float xc=pup_xc[x];
                    float yc=pup_yc[x];
                    float rc2=xc*xc+yc*yc;
                    float hologram_phase=atan2(hologram_imag[x],hologram_real[x]);
                    tempphase=COEFFXY*(
                    xc*spots_params[7*ty]+yc*spots_params[7*ty+1])+ COEFFZ*rc2/1000*spots_params[7*ty+2];   
                    tempphase=hologram_phase-tempphase;
                    temp_real[NPIX*ty+tx] += pupil_int[x]*cos(tempphase);
                    temp_imag[NPIX*ty+tx] += pupil_int[x]*sin(tempphase);                    
                }
            }
        }
        __global__ void project_to_spots_end(float*spots_params, float*temp_real, float*temp_imag, int*int_pars)
        {
            int tx = threadIdx.x+blockDim.x * blockIdx.x;
            int NSPOTS = int_pars[1]; 
            int NPIX = int_pars[2];
            float out_real = 0.0;
            float out_imag = 0.0;
            if (tx<NSPOTS)
            {
                for (int x=0; x<NPIX; x++)
                {
                    out_real+=temp_real[NPIX*tx+x];
                    out_imag+=temp_imag[NPIX*tx+x];                    
                }
            spots_params[7*tx+5]=atan2(out_imag, out_real);
            spots_params[7*tx+6]=(out_imag*out_imag + out_real*out_real)/spots_params[7*tx+4]/NSPOTS;
            }
        }
        __global__ void update_weights(float*spots_params, int*int_pars)
        {
            int tx = threadIdx.x+blockDim.x * blockIdx.x;
            int NSPOTS = int_pars[1];
            float mean=0.0;
            for(int i=0;i<NSPOTS;i++)
            {
                mean+=sqrt(spots_params[7*i+6]);
            }
            mean=mean/NSPOTS;
            if (tx<NSPOTS)
            {
                spots_params[7*tx+3]=spots_params[7*tx+3]*mean/sqrt(spots_params[7*tx+6]);
            }
        }
        __global__ void fill_screen_output(float*hologram_real, float*hologram_imag, int*screenpars, int*pixelscoordsx,
         int*pixelscoordsy, unsigned char* screenoutput)
        {
            int tx = threadIdx.x+blockDim.x * blockIdx.x;
            int PUP_NP=screenpars[0];
            int screenwidth=screenpars[1];
            int lutlow=screenpars[2];
            int luthigh=screenpars[3];
            if (tx<PUP_NP)
            {
            screenoutput[screenwidth*pixelscoordsy[tx]+pixelscoordsx[tx]]=lutlow+(
            atan2(hologram_imag[tx],hologram_real[tx])+3.141592653)/(2*3.141592654)*(luthigh-lutlow);
            }            
        }        
        """


def gauss(coords,I,off,x0,y0,w0_x,w0_y):
    x, y = coords
    return I*(numpy.exp(-(x-x0)**2/w0_x**2-(y-y0)**2/w0_y**2))+off


class SlmControl:
    def __init__(self,  wavelength_nm, pixel_size_um, focal_mm, beam_radius_mm= None,
                 screenID=None, active_area_coords=None, lut_edges=[0, 255]):
        print("""Thanks for using SLM-3dPointCloud.
Library openly available for non-commercial use at https://github.com/ppozzi/SLM-3dPointCloud.
If used for academic purposes, please consider citing the appropriate literature (https://doi.org/10.3389/fncel.2021.609505, https://doi.org/10.3390/mps2010002))""")

        self.screenID = None
        if active_area_coords is None:
            self.res = 512
            self.position = (0, 0)
        else:
            self.res = active_area_coords[2]
            self.position = (active_area_coords[0], active_area_coords[1])
        self.aperture_position = (0, 0)
        self.screenresolution = (self.res, self.res)

        self.lut_edges = lut_edges

        self.mod = SourceModule(cuda_code, options=DEFAULT_NVCC_FLAGS, keep=True)
        self.project_to_slm = self.mod.get_function("project_to_slm")
        self.project_to_spots_setup = self.mod.get_function("project_to_spots_setup")
        self.project_to_spots_end = self.mod.get_function("project_to_spots_end")
        self.project_to_slm_comp = self.mod.get_function("project_to_slm_comp")
        self.project_to_spots_setup_comp = self.mod.get_function("project_to_spots_setup_comp")
        self.project_to_spots_end_comp = self.mod.get_function("project_to_spots_end_comp")
        self.update_weights = self.mod.get_function("update_weights")
        self.fill_screen_output = self.mod.get_function("fill_screen_output")

        self.lam = wavelength_nm*0.001
        self.pix_size = pixel_size_um
        self.f = focal_mm*1000.0
        self.blocksize_forward = 512
        XC, YC = numpy.meshgrid(numpy.linspace(-self.pix_size * self.res / 2, self.pix_size * self.res / 2, self.res),
                                numpy.linspace(-self.pix_size * self.res / 2, self.pix_size * self.res / 2, self.res))
        RC2 = XC ** 2 + YC ** 2
        pupil_coords = numpy.where(numpy.sqrt(RC2) <= self.pix_size * self.res / 2)
        indexes = numpy.asarray(range(pupil_coords[0].shape[0]))
        numpy.random.shuffle(indexes)
        self.pupil_coords = (pupil_coords[0][indexes], pupil_coords[1][indexes])
        self.PUP_NP = self.pupil_coords[0].shape[0]

        # XC_unit, YC_unit = numpy.meshgrid(numpy.linspace(-1.0, 1.0, self.res),
        #                         numpy.linspace(-1.0, 1.0, self.res))
        # pupil_int = gauss((XC_unit, YC_unit), 1.0, 0.0, 0.00851336, -0.02336506,  0.48547321,  0.50274484)**2

        if beam_radius_mm == None:
            pupil_int=numpy.ones((self.res,self.res))
        else:
            pupil_int = numpy.exp(-(XC**2+YC**2)/(1000.0*beam_radius_mm)**2)
        pupil_int = pupil_int[self.pupil_coords]
        pupil_int = (pupil_int/numpy.sum(pupil_int)).astype("float32")
        self.PUP_INT_gpu = gpuarray.to_gpu(pupil_int)
        self.holo_real_gpu = gpuarray.to_gpu(numpy.zeros(self.PUP_NP, dtype="float32"))
        self.holo_imag_gpu = gpuarray.to_gpu(numpy.zeros(self.PUP_NP, dtype="float32"))
        self.XC_gpu = gpuarray.to_gpu(XC[self.pupil_coords].astype("float32"))
        self.YC_gpu = gpuarray.to_gpu(YC[self.pupil_coords].astype("float32"))
        self.float_pars_gpu = gpuarray.to_gpu(numpy.asarray([2.0 * numpy.pi / (self.lam * self.f),
                                                             numpy.pi / (self.lam * self.f ** 2) * 10 ** 3]
                                                            ).astype("float32"))
        self.screen_pup_coords_y_gpu = gpuarray.to_gpu((self.pupil_coords[0]+self.aperture_position[0]).astype("int32"))
        self.screen_pup_coords_x_gpu = gpuarray.to_gpu((self.pupil_coords[1]+self.aperture_position[1]).astype("int32"))
        self.screenpars_gpu = gpuarray.to_gpu(numpy.asarray([self.PUP_NP,self.screenresolution[1], self.lut_edges[0], self.lut_edges[1]]).astype("int32"))

    def rs(self, spots_coords, spots_ints, get_perf=False):
        t = time.perf_counter()
        SPOTS_N = spots_coords.shape[0]
        spots_parameters=numpy.zeros((SPOTS_N, 7))
        spots_parameters[:, 0:3] = spots_coords
        spots_parameters[:, 3] = 1.0
        spots_parameters[:, 4] = spots_ints
        spots_parameters[:, 5] = numpy.random.random(SPOTS_N)*2*numpy.pi
        spots_parameters[:, 6] = 0.0
        spots_parameters[:, 4] = spots_parameters[:, 4]/numpy.sum(spots_parameters[:, 4])
        spots_params_gpu = gpuarray.to_gpu(spots_parameters.astype("float32"))
        SPOTS_N=spots_parameters.shape[0]
        int_pars_gpu = gpuarray.to_gpu(
            numpy.asarray([self.PUP_NP, SPOTS_N, self.blocksize_forward, 0, 0]).astype("int32"))
        self.project_to_slm(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                            self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                            block=(1024, 1, 1), grid=(int(self.PUP_NP/1024)+1, 1, 1))
        if get_perf:
            result = int_pars_gpu.get()
            T = time.perf_counter()-t
            temp_real_gpu = gpuarray.to_gpu(numpy.zeros((self.blocksize_forward, SPOTS_N)).astype("float32"))
            temp_imag_gpu = gpuarray.to_gpu(numpy.zeros((self.blocksize_forward, SPOTS_N)).astype("float32"))
            self.project_to_spots_setup(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                        self.holo_imag_gpu, temp_real_gpu, temp_imag_gpu, self.PUP_INT_gpu,
                                        self.float_pars_gpu, int_pars_gpu,
                                        block=(self.blocksize_forward, int(1024 / self.blocksize_forward), 1),
                                        grid=(1, int(SPOTS_N / (1024 / self.blocksize_forward)) + 1, 1))
            self.project_to_spots_end(spots_params_gpu, temp_real_gpu, temp_imag_gpu, int_pars_gpu,
                                      block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
            spots_ints = spots_params_gpu.get()[:,6]
            e=numpy.sum(spots_ints/spots_parameters[:, 4]/SPOTS_N)
            u=1-(numpy.amax(spots_ints)-numpy.amin(spots_ints))/(
                            numpy.amax(spots_ints)+numpy.amin(spots_ints))
            m=numpy.mean(spots_ints)
            v=numpy.mean((spots_ints/m-1)**2)
            return {"Time": T, "Efficiency": e,"Uniformity": u, "Variance": v}

    def gs(self, spots_coords, spots_ints, iterations, get_perf=False):
        t=time.perf_counter()
        SPOTS_N = spots_coords.shape[0]
        spots_parameters=numpy.zeros((SPOTS_N, 7))
        spots_parameters[:, 0:3] = spots_coords
        spots_parameters[:, 3] = 1.0
        spots_parameters[:, 4] = spots_ints
        spots_parameters[:, 5] = numpy.random.random(SPOTS_N)*2*numpy.pi
        spots_parameters[:, 6] = 0.0
        spots_parameters[:, 4] = spots_parameters[:, 4]/numpy.sum(spots_parameters[:, 4])
        spots_params_gpu = gpuarray.to_gpu(spots_parameters.astype("float32"))
        SPOTS_N = spots_parameters.shape[0]
        temp_real_gpu = gpuarray.to_gpu(numpy.zeros((self.blocksize_forward, SPOTS_N)).astype("float32"))
        temp_imag_gpu = gpuarray.to_gpu(numpy.zeros((self.blocksize_forward, SPOTS_N)).astype("float32"))
        int_pars_gpu = gpuarray.to_gpu(
            numpy.asarray([self.PUP_NP, spots_parameters.shape[0], self.blocksize_forward, 0, 0]).astype("int32"))
        self.project_to_slm(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                            self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                            block=(1024,1,1),grid=(int(self.PUP_NP/1024)+1, 1, 1))
        for i in range(iterations):
            self.project_to_spots_setup(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                        self.holo_imag_gpu, temp_real_gpu, temp_imag_gpu, self.PUP_INT_gpu,
                                        self.float_pars_gpu, int_pars_gpu,
                                        block=(self.blocksize_forward, int(1024 / self.blocksize_forward), 1),
                                        grid=(1, int(SPOTS_N / (1024 / self.blocksize_forward)) + 1, 1))
            self.project_to_spots_end(spots_params_gpu, temp_real_gpu, temp_imag_gpu, int_pars_gpu,
                                      block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
            self.project_to_slm(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                                block=(1024, 1, 1), grid=(int(self.PUP_NP/1024)+1, 1, 1))
        if get_perf:
            result = int_pars_gpu.get()
            T = time.perf_counter()-t
            self.project_to_spots_setup(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                        self.holo_imag_gpu, temp_real_gpu, temp_imag_gpu, self.PUP_INT_gpu,
                                        self.float_pars_gpu, int_pars_gpu,
                                        block=(self.blocksize_forward, int(1024 / self.blocksize_forward), 1),
                                        grid=(1, int(SPOTS_N / (1024 / self.blocksize_forward)) + 1, 1))
            self.project_to_spots_end(spots_params_gpu, temp_real_gpu, temp_imag_gpu, int_pars_gpu,
                                      block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
            spots_ints = spots_params_gpu.get()[:, 6]
            e = numpy.sum(spots_ints * spots_parameters[:, 4]) * SPOTS_N
            u = 1 - (numpy.amax(spots_ints) - numpy.amin(spots_ints)) / (
                    numpy.amax(spots_ints) + numpy.amin(spots_ints))
            m=numpy.mean(spots_ints)
            v=numpy.mean((spots_ints/m-1)**2)
            return {"Time": T, "Efficiency": e,"Uniformity": u, "Variance": v}

    def wgs(self, spots_coords, spots_ints, iterations, get_perf=False):
        t = time.perf_counter()
        SPOTS_N = spots_coords.shape[0]
        spots_parameters=numpy.zeros((SPOTS_N, 7))
        spots_parameters[:, 0:3] = spots_coords
        spots_parameters[:, 3] = 1.0
        spots_parameters[:, 4] = spots_ints
        spots_parameters[:, 5] = numpy.random.random(SPOTS_N)*2*numpy.pi
        spots_parameters[:, 6] = 0.0
        spots_parameters[:, 4] = spots_parameters[:, 4]/numpy.sum(spots_parameters[:, 4])
        spots_params_gpu = gpuarray.to_gpu(spots_parameters.astype("float32"))
        SPOTS_N = spots_parameters.shape[0]
        temp_real_gpu = gpuarray.to_gpu(numpy.zeros((self.blocksize_forward, SPOTS_N)).astype("float32"))
        temp_imag_gpu = gpuarray.to_gpu(numpy.zeros((self.blocksize_forward, SPOTS_N)).astype("float32"))
        int_pars_gpu = gpuarray.to_gpu(
            numpy.asarray([self.PUP_NP, spots_parameters.shape[0], self.blocksize_forward, 0, 0]).astype("int32"))
        self.project_to_slm(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                            self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                            block=(1024, 1, 1), grid=(int(self.PUP_NP/1024)+1, 1, 1))
        for i in range(iterations):
            self.project_to_spots_setup(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                        self.holo_imag_gpu, temp_real_gpu, temp_imag_gpu, self.PUP_INT_gpu,
                                        self.float_pars_gpu, int_pars_gpu,
                                        block=(self.blocksize_forward, int(1024 / self.blocksize_forward), 1),
                                        grid=(1, int(SPOTS_N / (1024 / self.blocksize_forward)) + 1, 1))
            self.project_to_spots_end(spots_params_gpu, temp_real_gpu, temp_imag_gpu, int_pars_gpu,
                                      block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
            self.update_weights(spots_params_gpu, int_pars_gpu,
                                block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
            self.project_to_slm(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                                block=(1024, 1, 1), grid=(int(self.PUP_NP/1024)+1, 1, 1))
        if get_perf:
            result = int_pars_gpu.get()
            T = time.perf_counter()-t
            self.project_to_spots_setup(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                        self.holo_imag_gpu, temp_real_gpu, temp_imag_gpu, self.PUP_INT_gpu,
                                        self.float_pars_gpu, int_pars_gpu,
                                        block=(self.blocksize_forward, int(1024 / self.blocksize_forward), 1),
                                        grid=(1, int(SPOTS_N / (1024 / self.blocksize_forward)) + 1, 1))
            self.project_to_spots_end(spots_params_gpu, temp_real_gpu, temp_imag_gpu, int_pars_gpu,
                                      block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
            spots_ints = spots_params_gpu.get()[:, 6]
            e = numpy.sum(spots_ints * spots_parameters[:, 4]) * SPOTS_N
            u = 1 - (numpy.amax(spots_ints) - numpy.amin(spots_ints)) / (
                    numpy.amax(spots_ints) + numpy.amin(spots_ints))
            m=numpy.mean(spots_ints)
            v=numpy.mean((spots_ints/m-1)**2)
            return {"Time": T, "Efficiency": e,"Uniformity": u, "Variance": v}

    def cs_gs(self, spots_coords, spots_ints, iterations, comp, get_perf=False):
        t = time.perf_counter()
        SPOTS_N = spots_coords.shape[0]
        spots_parameters=numpy.zeros((SPOTS_N, 7))
        spots_parameters[:, 0:3] = spots_coords
        spots_parameters[:, 3] = 1.0
        spots_parameters[:, 4] = spots_ints
        spots_parameters[:, 5] = numpy.random.random(SPOTS_N)*2*numpy.pi
        spots_parameters[:, 6] = 0.0
        spots_parameters[:, 4] = spots_parameters[:, 4]/numpy.sum(spots_parameters[:, 4])
        spots_params_gpu = gpuarray.to_gpu(spots_parameters.astype("float32"))
        SPOTS_N = spots_parameters.shape[0]
        PUP_NP_COMP = int(comp*self.PUP_NP)
        temp_real_gpu = gpuarray.to_gpu(numpy.zeros((self.blocksize_forward, SPOTS_N)).astype("float32"))
        temp_imag_gpu = gpuarray.to_gpu(numpy.zeros((self.blocksize_forward, SPOTS_N)).astype("float32"))
        int_pars_gpu = gpuarray.to_gpu(
            numpy.asarray([self.PUP_NP, spots_parameters.shape[0], self.blocksize_forward, PUP_NP_COMP, 0]
                          ).astype("int32"))
        self.project_to_slm_comp(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                            self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                            block=(1024,1,1),grid=(int(PUP_NP_COMP/1024)+1, 1, 1))
        for i in range(iterations):
            self.project_to_spots_setup_comp(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                             self.holo_imag_gpu, temp_real_gpu, temp_imag_gpu, self.PUP_INT_gpu,
                                             self.float_pars_gpu, int_pars_gpu,
                                             block=(self.blocksize_forward, int(1024 / self.blocksize_forward), 1),
                                             grid=(1, int(SPOTS_N / (1024 / self.blocksize_forward)) + 1, 1))
            self.project_to_spots_end_comp(spots_params_gpu, temp_real_gpu, temp_imag_gpu, int_pars_gpu,
                                           block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
            int_pars_gpu = gpuarray.to_gpu(
                numpy.asarray([self.PUP_NP, spots_parameters.shape[0], self.blocksize_forward, PUP_NP_COMP,
                               int(i*PUP_NP_COMP/2)%(self.PUP_NP-PUP_NP_COMP)]).astype("int32"))
            if i < iterations-1:
                self.project_to_slm_comp(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                         self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                                         block=(1024, 1, 1), grid=(int(PUP_NP_COMP/1024)+1, 1, 1))
            else:
                self.project_to_slm(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                    self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                                    block=(1024, 1, 1), grid=(int(self.PUP_NP / 1024) + 1, 1, 1))
        if get_perf:
            result = int_pars_gpu.get()
            T = time.perf_counter()-t
            self.project_to_spots_setup(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                        self.holo_imag_gpu, temp_real_gpu, temp_imag_gpu, self.PUP_INT_gpu,
                                        self.float_pars_gpu, int_pars_gpu,
                                        block=(self.blocksize_forward, int(1024 / self.blocksize_forward), 1),
                                        grid=(1, int(SPOTS_N / (1024 / self.blocksize_forward)) + 1, 1))
            self.project_to_spots_end(spots_params_gpu, temp_real_gpu, temp_imag_gpu, int_pars_gpu,
                                      block=(int(SPOTS_N / 2 + 1), 1, 1), grid=(2, 1, 1))
            spots_ints = spots_params_gpu.get()[:, 6]
            e = numpy.sum(spots_ints * spots_parameters[:, 4]) * SPOTS_N
            u = 1 - (numpy.amax(spots_ints) - numpy.amin(spots_ints)) / (
                    numpy.amax(spots_ints) + numpy.amin(spots_ints))
            m=numpy.mean(spots_ints)
            v=numpy.mean((spots_ints/m-1)**2)
            return {"Time": T, "Efficiency": e,"Uniformity": u, "Variance": v}

    def cs_wgs(self, spots_coords, spots_ints, iterations, comp, get_perf=False):
        t = time.perf_counter()
        SPOTS_N = spots_coords.shape[0]
        spots_parameters=numpy.zeros((SPOTS_N, 7))
        spots_parameters[:, 0:3] = spots_coords
        spots_parameters[:, 3] = 1.0
        spots_parameters[:, 4] = spots_ints
        spots_parameters[:, 5] = numpy.random.random(SPOTS_N)*2*numpy.pi
        spots_parameters[:, 6] = 0.0
        spots_parameters[:, 4] = spots_parameters[:, 4]/numpy.sum(spots_parameters[:, 4])
        spots_params_gpu = gpuarray.to_gpu(spots_parameters.astype("float32"))
        SPOTS_N = spots_parameters.shape[0]
        PUP_NP_COMP = int(comp*self.PUP_NP)
        temp_real_gpu = gpuarray.to_gpu(numpy.zeros((self.blocksize_forward, SPOTS_N)).astype("float32"))
        temp_imag_gpu = gpuarray.to_gpu(numpy.zeros((self.blocksize_forward, SPOTS_N)).astype("float32"))
        int_pars_gpu = gpuarray.to_gpu(
            numpy.asarray([self.PUP_NP, spots_parameters.shape[0], self.blocksize_forward, PUP_NP_COMP, 0]
                          ).astype("int32"))
        self.project_to_slm_comp(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                            self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                            block=(1024,1,1),grid=(int(PUP_NP_COMP/1024)+1, 1, 1))
        for i in range(iterations-1):
            self.project_to_spots_setup_comp(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                             self.holo_imag_gpu, temp_real_gpu, temp_imag_gpu, self.PUP_INT_gpu,
                                             self.float_pars_gpu, int_pars_gpu,
                                             block=(self.blocksize_forward, int(1024 / self.blocksize_forward), 1),
                                             grid=(1, int(SPOTS_N / (1024 / self.blocksize_forward)) + 1, 1))
            self.project_to_spots_end_comp(spots_params_gpu, temp_real_gpu, temp_imag_gpu, int_pars_gpu,
                                           block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
            int_pars_gpu = gpuarray.to_gpu(
                numpy.asarray([self.PUP_NP, spots_parameters.shape[0], self.blocksize_forward, PUP_NP_COMP,
                               int(i*PUP_NP_COMP/2)%(self.PUP_NP-PUP_NP_COMP)]).astype("int32"))
            self.update_weights(spots_params_gpu, int_pars_gpu,
                                block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
            if i < iterations-2:
                self.project_to_slm_comp(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                         self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                                         block=(1024, 1, 1), grid=(int(PUP_NP_COMP/1024)+1, 1, 1))
            else:
                self.project_to_slm(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                    self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                                    block=(1024, 1, 1), grid=(int(self.PUP_NP / 1024) + 1, 1, 1))
                self.project_to_spots_setup(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                            self.holo_imag_gpu, temp_real_gpu, temp_imag_gpu, self.PUP_INT_gpu,
                                            self.float_pars_gpu, int_pars_gpu,
                                            block=(self.blocksize_forward, int(1024 / self.blocksize_forward), 1),
                                            grid=(1, int(SPOTS_N / (1024 / self.blocksize_forward)) + 1, 1))
                self.project_to_spots_end(spots_params_gpu, temp_real_gpu, temp_imag_gpu, int_pars_gpu,
                                          block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
                self.update_weights(spots_params_gpu, int_pars_gpu,
                                    block=(int(SPOTS_N / 2+1), 1, 1), grid=(2, 1, 1))
                self.project_to_slm(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                    self.holo_imag_gpu, self.float_pars_gpu, int_pars_gpu,
                                    block=(1024, 1, 1), grid=(int(self.PUP_NP / 1024) + 1, 1, 1))
        if get_perf:
            result = int_pars_gpu.get()
            T = time.perf_counter()-t
            self.project_to_spots_setup(spots_params_gpu, self.XC_gpu, self.YC_gpu, self.holo_real_gpu,
                                        self.holo_imag_gpu, temp_real_gpu, temp_imag_gpu, self.PUP_INT_gpu,
                                        self.float_pars_gpu, int_pars_gpu,
                                        block=(self.blocksize_forward, int(1024 / self.blocksize_forward), 1),
                                        grid=(1, int(SPOTS_N / (1024 / self.blocksize_forward)) + 1, 1))
            self.project_to_spots_end(spots_params_gpu, temp_real_gpu, temp_imag_gpu, int_pars_gpu,
                                      block=(int(SPOTS_N/2+1), 1, 1), grid=(2, 1, 1))
            spots_ints = spots_params_gpu.get()[:,6]
            e=numpy.sum(spots_ints*spots_parameters[:, 4])*SPOTS_N
            u=1 - (numpy.amax(spots_ints) - numpy.amin(spots_ints)) / (
                    numpy.amax(spots_ints) + numpy.amin(spots_ints))
            m=numpy.mean(spots_ints)
            v=numpy.mean((spots_ints/m-1)**2)
            return {"Time": T, "Efficiency": e,"Uniformity": u, "Variance": v}

    def wait_gpu(self):
        self.float_pars_gpu.get()

    def get_phase(self):
        outarr = numpy.zeros((self.res,self.res))
        outarr[self.pupil_coords] = numpy.arctan2(self.holo_imag_gpu.get(), self.holo_real_gpu.get())
        return outarr

    def set_phase(self,phase):
        phase = phase[self.pupil_coords]
        self.holo_real_gpu = gpuarray.to_gpu(numpy.cos(phase).astype("float32"))
        self.holo_imag_gpu = gpuarray.to_gpu(numpy.sin(phase).astype("float32"))
