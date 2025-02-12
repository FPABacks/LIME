import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys
sys.path.append(os.environ['MFORCE_DIR']+'/py_src/')
import Fortran_namelist as fnml


if np.size(sys.argv[1:]) < 1:
    print('Please provide the name of input parameter file', flush = True)
    quit()
elif np.size(sys.argv[1:]) == 1:
    GridSetup_File = sys.argv[1] #'Input_Sun.par'
    print('Unsing Grid Setup File ./' + GridSetup_File , flush = True)
else:
    print('Too many intun arguments!')
    quit()


# read the namelist form Grid Setup files 
GridSetup_Lsn = 'setup'
GridSetup_Nml = {
    'ParamFile'  : 0,
    'PlotOutput' : 0,
    'Output_DIR' : 0,
    'DMC_DIR'    : 0,
    'PlotBack'   : 0,
    'al_min'     : 0,
    'al_max'     : 0,
    'N_al'       : 0,
    'Q0_min'     : 0,
    'Q0_max'     : 0,
    'N_Q0'       : 0}
GridSetup_Mask = ['s','L','s','s','s','f','f','i','f','f','i']
# GridSetup_Nml = fnml.Read_namelist(GridSetup_File,GridSetup_Lsn,GridSetup_Nml,GridSetup_Mask)

try:
    GridSetup_Nml = fnml.Read_namelist(GridSetup_File,GridSetup_Lsn,GridSetup_Nml,GridSetup_Mask)
except Exception as Exc:
    print(Exc)
    quit()
else:
    print('Accessing LTE input parameter file ./' +  GridSetup_Nml['ParamFile'],flush=True)

# here are some internal controlls 
# 1) to create fitting plots set plot_Mt = True
plot_Mt = GridSetup_Nml['PlotOutput']
# 2) To output plots in png set backend to 'agg' or to 'ps' for ps, comment out fo rinteractive window
plt.switch_backend(GridSetup_Nml['PlotBack'])
# 3) set the output folder as '/foldername' for the plots
DMC = GridSetup_Nml['DMC_DIR']
# 4) Set the output folder as '/foldername' for tables
Output = GridSetup_Nml['Output_DIR']

# define the fitting function 
def M(x, a, q0):
    return ( (np.float128(1.) + np.float128(x*q0) )**(np.float128(1.) -np.float128(a) ) - np.float128(1.) )/( (np.float128(1.) - np.float128(a) )*np.float128(x*q0) ) 

# read the namelist form fortran input files 
File =  GridSetup_Nml['ParamFile']
Listname = 'init_param'
Namelist = {
    'DIR'    : 0,
    'lgTmin' : 0,
    'lgTmax' : 0,
    'N_lgT'  : 0,
    'lgDmin' : 0,
    'lgDmax' : 0,
    'N_lgD'  : 0,
    'lgttmin': 0,
    'lgttmax': 0,
    'N_tt'   : 0,
    'Ke_norm': 0}
Mask = ['s','f','f','i','f','f','i','f','f','i','f']

try:
    Namelist = fnml.Read_namelist(File,Listname,Namelist,Mask)
except Exception as Exc:
    print(Exc)
    quit()
else:
    print('Accessing data input file ./' +  Namelist['DIR'],flush=True)
        

if Output.strip():
    # define the name of the directory to be created    
    path = './'+ Namelist['DIR']+Output
    DMC = Output + DMC
    try:
        os.mkdir(path)
    except  OSError as Err:
        if  Err.__class__  == FileExistsError:
            print ("Output directory %s already exists" % path,flush=True)
        else:
            print(Err)
            quit()
    else:
        print ("Successfully created the output directory %s " % path,flush=True)
else:
    print('Using /' +Namelist['DIR']+' as output directly')

if plot_Mt:
    # define the name of the directory to be created    
    path = './'+ Namelist['DIR']+DMC
    try:
        os.mkdir(path)
    except  OSError as Err:
        if  Err.__class__  == FileExistsError:
            print ("Plot directory %s already exists" % path,flush=True)
        else:
            print(Err)
            quit()
    else:
        print ("Successfully created the plot directory %s " % path,flush=True)
else:
    print('Using blined mode')

# create corresponding to the namlist range of T and D
lgT_list = np.linspace(Namelist['lgTmin'], Namelist['lgTmax'], num=Namelist['N_lgT'])
lgD_list = np.linspace(Namelist['lgDmin'], Namelist['lgDmax'], num=Namelist['N_lgD'])
lgtt_grd = np.linspace(Namelist['lgttmin'], Namelist['lgttmax'], num=Namelist['N_tt'])

# create storage for the input variables 
al = np.zeros([Namelist['N_lgD'],Namelist['N_lgT']])
Q0 = np.zeros([Namelist['N_lgD'],Namelist['N_lgT']])
Mt_data = np.zeros([Namelist['N_tt'],Namelist['N_lgD'],Namelist['N_lgT']])
chi2_map = np.zeros([Namelist['N_lgD'],Namelist['N_lgT']])


# Setup the grid of models in [Q0 al]
Q0_grid = 10.0**np.linspace(GridSetup_Nml['Q0_min'],GridSetup_Nml['Q0_max'],num=GridSetup_Nml['N_Q0'])  # 10.0**np.linspace(-1, 8.0, num=500) 
al_grid = np.linspace(GridSetup_Nml['al_min'],GridSetup_Nml['al_max'],num=GridSetup_Nml['N_al']) # np.linspace(0.0e-10,1.-10e-5, num=100)
lgtt_grid = np.linspace(Namelist['lgttmin'], Namelist['lgttmax'], num=Namelist['N_tt'])
Mn_grid = np.zeros([np.size(Q0_grid),np.size(al_grid),np.size(lgtt_grid)])
chi2_su = np.zeros([np.size(Q0_grid),np.size(al_grid)])
# print(np.shape(chi2_su))

print('creating the grid', flush = True)
for indQ in range(np.size(Q0_grid)):
    for inda in range(np.size(al_grid)):
        for indtt in range(np.size(lgtt_grid)):
            Mn_grid[indQ,inda,indtt] = np.max([M(10.**lgtt_grid[indtt], al_grid[inda], Q0_grid[indQ]), np.float128(1.e-20)])

print('Starting fitting', flush = True)
# cycle over the range of D and T 
for indD in range(Namelist['N_lgD']):
    for indT in range(Namelist['N_lgT']):

        # indD = 10
        # indT = 10
        # Conpose the filenames 
        lgT = lgT_list[indT]
        lgD = lgD_list[indD]
        filename = './' + Namelist['DIR'] + '/Mt_%4.2f_%5.1f' % (lgT, lgD)
        print(filename, flush = True)

        # load the inputfle
        try:
            inp = np.loadtxt(filename)
        except Exception as Exc:
            print(Exc)
            quit()

        # Parse input form file into variables 
        lgt = inp[:,0]
        Mt_data[:,indD,indT] = inp[:,1]

        # get bar Q
        Q = np.max( Mt_data[:,indD,indT])

        # normalize the Mt for the fitting 
        Mt =  Mt_data[:,indD,indT]
        Mn = Mt_data[:,indD,indT]/Q

        # set the weight by shofting the fitting region 


        for indQ in range(np.size(Q0_grid)):
            for inda in range(np.size(al_grid)):
                # chi2 = np.sum( (Mn[:] - Mn_grid[indQ,inda,:])**2. 
                #                 / Mn_grid[indQ,inda,:] )
                chi2 = np.sum( (np.log10(Mn[:]) - np.log10(Mn_grid[indQ,inda,:]))**2.) 
                #                / ( np.log10(Mn_grid[indQ,inda,:]) + 50 ) )
                chi2_su[indQ,inda] = chi2

       
        minchi2 = np.unravel_index(np.argmin(chi2_su), chi2_su.shape)
        chi2_map[indD,indT] = np.min(chi2_su)
        al[indD,indT]  = al_grid[minchi2[1]] 
        Q0[indD,indT] = Q0_grid[minchi2[0]] 

        if plot_Mt:
            chi2_su[(chi2_su > 5.0e9)] = 5.0e9
            fig1 = plt.figure(1,figsize=(12,8))
            plt.clf()
            plt.contourf(al_grid,np.log10(Q0_grid),np.log10(chi2_su))
            plt.plot(al[indD,indT],np.log10(Q0[indD,indT]),'wX')
            # plt.clim(-3,9)  # identical to caxis([-4,4]) in MATLAB
            plt.colorbar()
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.title('lg(chi^2)   lgT =%4.2f lgD =%5.1f.png' % (lgT, lgD), fontsize=25)
            plt.xlabel('a', fontsize=25)
            plt.ylabel('lg Q_0', fontsize=25)
            plt.savefig('./'+ Namelist['DIR']+DMC+'/lgT=%4.2f_lgD=%5.1f_chi2.png' % (lgT, lgD))
            
            fig1 = plt.figure(2,figsize=(12,8))
            plt.clf()
            plt.plot(lgt,np.log10(Mt))
            plt.plot(lgt,np.log10(Q*Mn_grid[minchi2[0],minchi2[1],:]),
                    label='al = %4.2f Q0 =%5.0f Qb =%5.0f,' % (al_grid[minchi2[1]], Q0_grid[minchi2[0]],Q))
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            plt.legend(loc=3, fontsize=25)
            plt.title('lgT =%4.2f lgD =%5.1f.png' % (lgT, lgD), fontsize=25)
            plt.xlabel('lg t', fontsize=25)
            plt.ylabel('lg M', fontsize=25)
            plt.savefig('./'+ Namelist['DIR']+DMC+'/lgT=%4.2f_lgD=%5.1f_fit.png' % (lgT, lgD))
            

        # print()
        # plt.show()
        # quit()

# open output file 
filehand = open('./'+ Namelist['DIR']+Output.strip()+'/al_TD', 'w')
# wtire the temperature range
filehand.write('% 10.8e' % (0.0))
for x in lgT_list:
    filehand.write(' % 10.8e' % (x))

# write density range and append values 
for indD in range(len(lgD_list)):
    filehand.write('\n')
    filehand.write('% 10.8e' % (lgD_list[indD]))
    for indT in range(len(lgT_list)):
        filehand.write(' % 10.8e' % (al[indD,indT]))

# close output file 
filehand.close()

# open output file 
filehand = open('./'+ Namelist['DIR']+Output.strip()+'/Q0_TD', 'w')
# wtire the temperature range
filehand.write('% 10.8e' % (0.0))
for x in lgT_list:
    filehand.write(' % 10.8e' % (x))

# write density range and append values 
for indD in range(len(lgD_list)):
    filehand.write('\n')
    filehand.write('% 10.8e' % (lgD_list[indD]))
    for indT in range(len(lgT_list)):
        filehand.write(' % 10.8e' % (Q0[indD,indT]))

# close output file 
filehand.close()

# open output file 
filehand = open('./'+ Namelist['DIR']+Output.strip()+'/chi2_TD', 'w')
# wtire the temperature range
filehand.write('% 10.8e' % (0.0))
for x in lgT_list:
    filehand.write(' % 10.8e' % (x))

# write density range and append values 
for indD in range(len(lgD_list)):
    filehand.write('\n')
    filehand.write('% 10.8e' % (lgD_list[indD]))
    for indT in range(len(lgT_list)):
        filehand.write(' % 10.8e' % (chi2_map[indD,indT]))

# close output file 
filehand.close()


# open output file 
filehand = open('./'+ Namelist['DIR']+Output.strip()+'/info.txt', 'w')
filehand.write('% 3i    ! number of T points\n' % (Namelist['N_lgT']))
filehand.write('% 3i    ! nimber of D points\n' % (Namelist['N_lgT']))
filehand.write('% 4.3f ! fidutial normalisation \n' % (Namelist['Ke_norm']))
filehand.close()
print('Finished', flush=True)
