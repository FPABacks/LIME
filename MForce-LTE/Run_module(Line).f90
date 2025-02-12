PROGRAM main
   USE f90getopt
   USE LTE_Line_module
   USE CGS_constants
   IMPLICIT NONE


   ! setup Fortran IO variables 
   INTEGER, PARAMETER :: stdin  = 5
   INTEGER, PARAMETER :: stdout = 6
   INTEGER, PARAMETER :: stderr = 0

   REAL(DP), DIMENSION(:), ALLOCATABLE :: lgT_list
   REAL(DP), DIMENSION(:), ALLOCATABLE :: lgD_list

   REAL(DP), DIMENSION(:), ALLOCATABLE :: occupN_list

   CHARACTER(20) PAR_File
   REAL(DP) :: lgTmin,lgTmax,lgDmin, lgDmax
   REAL(DP) :: X_mass, Y_mass, Z_mass 
   INTEGER(I4B) :: N_lgT, N_lgD, SpecL,SpecI, SpecZ
   CHARACTER(20) DIR
   LOGICAL :: ver

   INTEGER(I4B) :: T_gird_counter,D_gird_counter

   REAL(DP) :: D, T, Lam, StatWeight, gf
   INTEGER(I4B) :: Ll,Lu, I, Z, NumOfMatchingLines
   
   ! INTEGER(I4B) :: exclude = 23
   INTEGER(I4B) :: Nlines(30,9) = 29

   INTEGER(I4B) :: ind1,ind2
   CHARACTER(10) str_3
   CHARACTER(10) str_2
   CHARACTER(10) str_1

   ! Define the output identefiers
   INTEGER(I2B), PARAMETER :: ID_NameList = 11
   INTEGER(I2B), PARAMETER :: ID_Tmp1 = 12  !  Occupation numbers
   INTEGER(I2B), PARAMETER :: ID_Tmp2 = 13  !  Line data

   !Define types

   TYPE(ATOMIC_DATA_TYPE) :: ATOM_DATA
   TYPE(LINE_DATA_TYPE)   :: LINE_DATA
   TYPE(OCCUPATION_NUMBER_TYPE):: NUMB_DENS !(30)

   ! Input argument type
   TYPE(option_s):: opts(2)

   ! ... namelist
   NAMELIST / init_param / lgTmin, lgTmax, N_lgT, &
                           lgDmin, lgDmax, N_lgD, &
                           X_mass, Z_mass, &
                           SpecZ, SpecI, SpecL, &
                           ver, DIR
   !

   CALL LoGo()

   ! Parse the input argument
   opts(1) = option_s( "input", .TRUE., 'i' )

   ! If no options were committed
   IF (command_argument_count() .eq. 0 ) THEN
      STOP 'Please use option --input or -i Input_Parameter_file_name !'
   END IF

   ! Process options one by one
   DO 
      SELECT CASE( getopt( "i:", opts ) ) ! opts is optional (for longopts only)
      CASE( char(0) )
         EXIT
      CASE( 'i' )
         read (optarg, '(A)') PAR_File
         WRITE(*,*) 'Using input parameter file: ',TRIM(PAR_File)
         WRITE(*,*) " "
         CALL  flush(stdout) 
      END SELECT
   END DO


   OPEN(ID_NameList, FILE=TRIM(PAR_File), STATUS='OLD', FORM='FORMATTED')
   READ(ID_NameList, NML=init_param)
   CLOSE(ID_NameList)

   WRITE(*,*) "----- Input Setup -----"
   WRITE(*,*)'Setting Chemical composition to X=',X_mass,'Z=',Z_mass
   WRITE(*,*) " "
   WRITE(*,*)'         Min                       Max                       N'
   WRITE(*,*)'Lg T  ',lgTmin, lgTmax, N_lgT
   WRITE(*,*)'Lg D  ',lgDmin, lgDmax, N_lgD
   WRITE(*,*)'SpecZ =',SpecZ
   WRITE(*,*)'SpecI =',SpecI
   WRITE(*,*)'SpecL =',SpecL
   WRITE(*,*)'Verbose = ',ver
   WRITE(*,*)'Output destionation -',DIR
   WRITE(*,*) " "
   CALL  flush(stdout) 

   WRITE(*,*)'Initialise Grid'
   CALL  flush(stdout) 
   
   ALLOCATE(occupN_list(N_lgT))


   ALLOCATE(lgT_list(N_lgT))
   CALL linspace(lgTmin, lgTmax,lgT_list)
   ALLOCATE(lgD_list(N_lgD))
   CALL linspace(lgDmin, lgDmax,lgD_list)

   IF(ANY(ISNAN(lgT_list))) STOP '---- lgT is nan ----'
   IF(ANY(ISNAN(lgD_list))) STOP '---- lgD is nan ----'
   IF(SIZE(lgT_list).NE.N_lgT) STOP '---- lgT Length mismatch ----'
   IF(SIZE(lgD_list).NE.N_lgD) STOP '---- lgD Length mismatch ----'
   WRITE(*,*)'Grid - dine'
   WRITE(*,*) " "
   CALL  flush(stdout) 
  
   
   WRITE(*,*)'Creating output destionation - ./',TRIM(DIR)
   CALL EXECUTE_COMMAND_LINE('mkdir -p '//TRIM(DIR), WAIT=.TRUE.)

   ! create occparion number dencity temp outpute file
   OPEN (unit = ID_Tmp1 , file = './'//TRIM(DIR)//'/Tmp1 ', FORM='formatted',STATUS='unknown', ACTION='write')
   WRITE(ID_Tmp1 ,*) 0.0d0, lgT_list

   ! create occparion transition strength temp outpute file
   OPEN (unit = ID_Tmp2 , file = './'//TRIM(DIR)//'/Tmp2 ', FORM='formatted',STATUS='unknown', ACTION='write')

   WRITE(*,*)'Creating output destionation - done'
   WRITE(*,*) " "
   CALL  flush(stdout) 

   ! Step 1) Get atomic data [done], aboundence data [done], Line data [done] (use X = 1 for H no Y|=0 !!)
   WRITE(*,*)'Initialise ATOM and NUMB'
   CALL ATOM_DATA%Initialise(verbose  = ver)
   IF (X_mass.GT.-1) THEN
      WRITE(*,*) "  >Using input composition"
      Y_mass = (1.0d0 - X_mass - Z_mass)
      CALL NUMB_DENS%Initialise(ATOM_DATA, X_frac=X_mass, Y_frac = Y_mass, verbose  = ver)
   ELSE 
      WRITE(*,*) "  >Using Solar composition"
      CALL NUMB_DENS%Initialise(ATOM_DATA, verbose  = ver)
   END IF
   WRITE(*,*) 'X=',NUMB_DENS%X_frac,'Y=',NUMB_DENS%Y_frac,'Z=',1.0d0 - NUMB_DENS%x_frac - NUMB_DENS%Y_frac
   WRITE(*,*)'ATOM and NUMB  - done'
   WRITE(*,*) " "
   CALL  flush(stdout) 

   ! Get Line data
   WRITE(*,*)'Initialise Line Data'
   CALL LINE_DATA%Initialise(verbose  = ver)

   WRITE(*,*)'  > Number of lines =',LINE_DATA%Total_line_numb
   WRITE(*,*)'Line Data - dine'
   WRITE(*,*) " "
   CALL  flush(stdout) 


   ! clear the line number counter
   NumOfMatchingLines = 0
   WRITE(*,*)'Line temp file'
         
   ! cycle over each line in the list
   DO ind1 = 1, LINE_DATA%Total_line_numb

      ! get transition identifiers for given line
      Ll = LINE_DATA%ID(ind1,Ll_)
      Lu = LINE_DATA%ID(ind1,Lu_)
      I = LINE_DATA%ID(ind1,I_)
      Z = LINE_DATA%ID(ind1,Z_)
      Lam = LINE_DATA%Lambda(ind1)
      gf = LINE_DATA%gf_val(ind1)

      ! IF(Z.EQ.exclude) CYCLE
      Nlines(Z,I) = Nlines(Z,I)+1

      ! check if the given level is available in atomic data
      IF (Ll.GT.ATOM_DATA%List_L(I,Z)) STOP '---- exceeding the available level ----'

      IF ((Ll.EQ.SpecL).AND.(I.EQ.SpecI).AND.(Z.EQ.SpecZ)) THEN
         ! RTIDE line data ro file 
         WRITE(ID_Tmp2,*) Ll,Lu,gf,Lam
         NumOfMatchingLines = NumOfMatchingLines + 1
      ENDIF
      
   END DO
   WRITE(*,*) " > Nlines = ", NumOfMatchingLines
   CLOSE(ID_Tmp2)
   CALL  flush(stdout) 

   WRITE(*,*)'Ocupparion temp file'
   ! Get Degeneracy of Species of interest 
   StatWeight = ATOM_DATA%Degeneracy(SpecL,SpecI,SpecZ)

   DO D_gird_counter = 1,N_lgD
      D = 10.0d0**lgD_list(D_gird_counter)

      ! celar the occuparion number list
      occupN_list = 0.0d0
      DO T_gird_counter = 1,N_lgT
         T = 10.0d0**lgT_list(T_gird_counter)

         WRITE(*,*)'Step ',(D_gird_counter - 1)*N_lgD + T_gird_counter,'/',N_lgD*N_lgT
         WRITE(*,*)'Set lgT =',lgT_list(T_gird_counter),' lgD =',lgD_list(D_gird_counter)
         CALL  flush(stdout)

         CALL NUMB_DENS%Set(rho = D, T = T, verbose = ver)

         occupN_list(T_gird_counter) = LOG10(NUMB_DENS%Occupation(SpecL,SpecI,SpecZ))

      END DO !T_gird_counter

      ! write dencity and all temperatures point of Q_0
      WRITE(ID_Tmp1,*)lgD_list(D_gird_counter), occupN_list

   END DO !D_gird_counter

   WRITE(ID_Tmp1,*) ATOM_DATA%Names(SpecZ), SpecI, SpecL
   CLOSE(ID_Tmp1)
   WRITE(*,*)'Write occupation numbers - done'
   CALL  flush(stdout) 

   CALL CombineFiles()

   WRITE(*,*)'Program - done'
   CALL  flush(stdout) 

CONTAINS

   SUBROUTINE Ilumination_finction(T_io,nu_i0_io,W_i_io)
      REAL(DP), INTENT(IN)  :: T_io
      REAL(DP), INTENT(IN)  :: nu_i0_io(:)
      REAL(DP), INTENT(OUT) :: W_i_io(:)

      REAL(DP), PARAMETER :: C1 = 2.0d0*pi*plnk_const/lght_speed**2.0d0
      REAL(DP), PARAMETER :: C2 = plnk_const/bolz_const
      REAL(DP) :: F

      IF(ANY( nu_i0_io.EQ.0)) STOP '---- Nu = 0 ----'

      F = sigma_stef*T**4.0d0

      W_i_io(:) = C1 * nu_i0_io(:)**4.0d0 / ( EXP( C2*nu_i0_io(:)/T_io ) - 1.0d0) / F
      ! print*,C1, C2, F
   END SUBROUTINE Ilumination_finction


   ! Generates evenly spaced numbers from `from` to `to` (inclusive).
   !
   ! Inputs:
   ! -------
   !
   ! from, to : the lower and upper boundaries of the numbers to generate
   !
  ! Outputs:
   ! -------
   !
   ! array : Array of evenly spaced numbers
   !
   SUBROUTINE linspace(from, to, array)
      REAL(dp), INTENT(in) :: from, to
      REAL(dp), INTENT(out) :: array(:)
      REAL(dp) :: range
      INTEGER :: n, ind
      n = SIZE(array)
      range = to - from

      IF (n == 0) RETURN

      IF (n == 1) THEN
         array(1) = from
         RETURN
      END IF


      DO ind=1, n
         array(ind) = from + range * (ind - 1) / (n - 1)
      END DO
   END SUBROUTINE linspace


   ! this subroutine collects Temp 1 and Temp 2 files into a uniform output data file 
   ! description of data file is as follows 
   !
   ! NameOfElement(char)  IonisationStage(int)  LowerTransitionLevel(int)
   ! Aw(double) '! Atomic weight of the lelment (in m_p)'
   ! N_L(int) '! Numder Of Line Transitions'
   ! N_T(int)  N_D(int) '! Number Of Temparature Points   ----  Number Of Dencity Point'
   !
   !         Ll          Lu                        gf                  Lam
   ! LowerLevel1(int)  UpperLevel1(int)  gfValue1(double)  Trans.Wavelenth[Ang]1(double)
   ! LowerLevel2   ...   ....   ....  
   ! ...           ...   ....   ....
   ! LowerLevelN_Line ...  ....  ....
   !
   ! StatWeight(double)  ! Statistical Weight of the Lower Level
   ! 00000   lgT1     lgT2    ... lgTN_T
   ! lgD1    lgN(11)  lgN(12) ...
   ! lgD2    ...      lgN(22) ...
   ! ...     ...    ...   ... 
   ! lgDN_D
   SUBROUTINE CombineFiles()
      ! REAL(dp),    INTENT(in) :: StatWeight
      ! INTEGER(dp), INTENT(in) :: NumOfMatchingLines,N_lgT,N_lgD
      ! CHARACTER(20), INTENT(in) :: DIR

      INTEGER(I2B), PARAMETER :: ID_outp = 14  !  Full output file 
      INTEGER(I2B)            :: dummy_ind
      CHARACTER(256)          :: dummi_line
      ! create occparion number dencity temp outpute file
      OPEN (unit = ID_Tmp1 , file = './'//TRIM(DIR)//'/Tmp1 ', FORM='formatted',STATUS='unknown', ACTION='read')

      ! create occparion transition strength temp outpute file
      OPEN (unit = ID_Tmp2 , file = './'//TRIM(DIR)//'/Tmp2 ', FORM='formatted',STATUS='unknown', ACTION='read')

      ! Generate the  output name
      WRITE(str_1,'(I2.2)') SpecZ
      WRITE(str_2,'(I2.2)') SpecI
      WRITE(str_3,'(I2.2)') SpecL

      ! PRINT*,'./'//TRIM(DIR)//'/'//TRIM(ADJUSTL(str_1))//'_'//TRIM(ADJUSTL(str_2))//'_'//TRIM(ADJUSTL(str_3))//'.txt'

      OPEN (unit = ID_Outp , &
            file = './'//TRIM(DIR)//'/'//TRIM(ADJUSTL(str_1))//'_'//TRIM(ADJUSTL(str_2))//'_'//TRIM(ADJUSTL(str_3))//'.txt', &
            FORM='formatted',STATUS='unknown', ACTION='write')

      WRITE(ID_Outp, *) TRIM(ATOM_DATA%Names(SpecZ))//TRIM(ADJUSTL(str_2)),' ', TRIM(str_3)
      WRITE(ID_Outp, *) NUMB_DENS%Atomic_weight(SpecZ), '! Atomic weight of the lelment (in m_p)'
      WRITE(ID_Outp, *) NumOfMatchingLines, '! Numder Of Line Transitions'
      WRITE(ID_Outp, *) N_lgT, '! Number Of Temparature Points'
      WRITE(ID_Outp, *) N_lgD, '! Number Of Dencity Point'
      WRITE(ID_Outp, *) StatWeight, '! Statistical Weight of the Lower Level'
      WRITE(ID_Outp, *) ''
      WRITE(ID_Outp, *) '         Ll          Lu                        gf                  Lam' 
   
      DO dummy_ind = 1,NumOfMatchingLines
         READ(ID_Tmp2,'(A)') dummi_line
         WRITE(ID_Outp,*) TRIM(dummi_line)
      END DO

      WRITE(ID_Outp, *) ''
      WRITE(ID_Outp, *) '! Row(1 ) 0000 lgT1   lgT2 ...'
      WRITE(ID_Outp, *) '! Row(2-) lgD1 lgN11  lgN12  ...'

      DO dummy_ind = 1,N_lgD
         READ(ID_Tmp1,'(A)') dummi_line
         WRITE(ID_Outp,*) TRIM(dummi_line)
      END DO

      CLOSE(ID_Tmp2)
      CLOSE(ID_Tmp1)
      CLOSE(ID_Outp)

      CALL EXECUTE_COMMAND_LINE('rm ./'//TRIM(DIR)//'/Tmp1 ', WAIT=.TRUE.)
      CALL EXECUTE_COMMAND_LINE('rm ./'//TRIM(DIR)//'/Tmp2 ', WAIT=.TRUE.)

   END SUBROUTINE CombineFiles
   
   SUBROUTINE LoGo()
   
      CALL EXECUTE_COMMAND_LINE('clear')
      WRITE(*,*) " "
      WRITE(*,*) "     ___           ___           ___           ___           ___           ___       &
      &.             ___       ___           ___     "
      WRITE(*,*) "    /\__\         /\  \         /\  \         /\  \         /\  \         /\  \      &
      &.            /\__\     /\  \         /\  \    "
      WRITE(*,*) "   /::|  |       /::\  \       /::\  \       /::\  \       /::\  \       /::\  \     &
      &.           /:/  /     \:\  \       /::\  \   "
      WRITE(*,*) "  /:|:|  |      /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/\:\  \    &
      &.          /:/  /       \:\  \     /:/\:\  \  "
      WRITE(*,*) " /:/|:|__|__   /::\~\:\  \   /:/  \:\  \   /::\~\:\  \   /:/  \:\  \   /::\~\:\  \   &
      &.         /:/  /        /::\  \   /::\~\:\  \ "
      WRITE(*,*) "/:/ |::::\__\ /:/\:\ \:\__\ /:/__/ \:\__\ /:/\:\ \:\__\ /:/__/ \:\__\ /:/\:\ \:\__\  &
      &.        /:/__/        /:/\:\__\ /:/\:\ \:\__\"
      WRITE(*,*) "\/__/~~/:/  / \/__\:\ \/__/ \:\  \ /:/  / \/_|::\/:/  / \:\  \  \/__/ \:\~\:\ \/__/  &
      &.        \:\  \       /:/  \/__/ \:\~\:\ \/__/"
      WRITE(*,*) "      /:/  /       \:\__\    \:\  /:/  /     |:|::/  /   \:\  \        \:\ \:\__\    &
      &.         \:\  \     /:/  /       \:\ \:\__\  "
      WRITE(*,*) "     /:/  /         \/__/     \:\/:/  /      |:|\/__/     \:\  \        \:\ \/__/    &
      &.          \:\  \    \/__/         \:\ \/__/  "
      WRITE(*,*) "    /:/  /                     \::/  /       |:|  |        \:\__\        \:\__\      &
      &.           \:\__\                  \:\__\    "
      WRITE(*,*) "    \/__/                       \/__/         \|__|         \/__/         \/__/      &
      &.            \/__/                   \/__/    "
      WRITE(*,*) " "
      ! WRITE(*,*) ",_       _   .______                                          ,__      ,_________  ,_______ " 
      ! WRITE(*,*) "| \     / |  | _____|                                         | |      |___  ,___| |  _____|"
      ! WRITE(*,*) "|  \   /  |  | |       ____    _  ___.   ____     _____       | |          | |     | |      "  
      ! WRITE(*,*) "| \ \ /   |  | '--,   / __ \  | |/ __|  / ___\   / ___ \ ==== | |          | |     | '---,  "
      ! WRITE(*,*) "| |\ / /| |  | ,--'  | /  \ | | / /    | /      | /__/_/      | |          | |     | ,---'  "
      ! WRITE(*,*) "| | v_/ | |  | |     | \__/ | |  /     | \___   | \___        | |____      | |     | |_____ "
      ! WRITE(*,*) "|_|     |_|  |_|      \____/  |_|       \____/   \____/       |,_____|     |_|     |_______|"
      ! WRITE(*,*) " "
      CALL  flush(stdout) 
   END SUBROUTINE LoGo

END PROGRAM main
