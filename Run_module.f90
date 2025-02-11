PROGRAM main
   USE f90getopt
   USE LTE_Line_module
   USE CGS_constants
   IMPLICIT NONE


   ! setup Fortran IO variables 
   INTEGER, PARAMETER :: stdin  = 5
   INTEGER, PARAMETER :: stdout = 6
   INTEGER, PARAMETER :: stderr = 0

   ! seting up the computationsl variables 
   REAL(DP), DIMENSION(:), ALLOCATABLE :: W_i
   REAL(DP), DIMENSION(:), ALLOCATABLE :: q_i
   REAL(DP), DIMENSION(:), ALLOCATABLE :: nu_i0

   REAL(DP), DIMENSION(:), ALLOCATABLE :: tt_list
   REAL(DP), DIMENSION(:), ALLOCATABLE :: lgT_list
   REAL(DP), DIMENSION(:), ALLOCATABLE :: lgD_list

   REAL(DP), DIMENSION(:), ALLOCATABLE :: kape_list
   REAL(DP), DIMENSION(:), ALLOCATABLE :: barQ_list
   ! REAL(DP), DIMENSION(:), ALLOCATABLE :: notQ_list

   CHARACTER(20) PAR_File
   REAL(DP) :: lgTmin,lgTmax,lgDmin, lgDmax, lgttmin, lgttmax
   REAL(DP) :: Ke_norm, X_mass, Y_mass, Z_mass 
   INTEGER(I4B) :: N_tt, N_lgT, N_lgD
   CHARACTER(32) DIR
   LOGICAL :: ver

   INTEGER(I4B) :: T_gird_counter,D_gird_counter

   REAL(DP) :: D, T, Lam, gl, nl,  chi_i
   REAL(DP) :: tau_s, Mt
   INTEGER(I4B) :: Ll, I, Z
   
   ! INTEGER(I4B) :: exclude = 23
   INTEGER(I4B) :: Nlines(30,9) = 29

   INTEGER(I4B) :: ind1,ind2
   CHARACTER(10) str_2
   CHARACTER(10) str_1

   ! Define the output identefiers
   INTEGER(I2B), PARAMETER :: ID_NameList = 11
   ! INTEGER(I2B), PARAMETER :: ID_notQ = 12
   INTEGER(I2B), PARAMETER :: ID_barQ = 13
   INTEGER(I2B), PARAMETER :: ID_Ke = 14
   INTEGER(I2B), PARAMETER :: ID_Mt = 15
   !Define types
   ! Mass of hydrogen
   REAL(DP) :: mH = 1.67372360d-24

   TYPE(ATOMIC_DATA_TYPE) :: ATOM_DATA
   TYPE(LINE_DATA_TYPE)   :: LINE_DATA
   TYPE(OCCUPATION_NUMBER_TYPE):: NUMB_DENS !(30)

   ! Input argument type
   TYPE(option_s):: opts(2)

   ! ... namelist
   NAMELIST / init_param / lgTmin, lgTmax, N_lgT, &
                           lgDmin, lgDmax, N_lgD, &
                           lgttmin, lgttmax, N_tt, &
                           Ke_norm, X_mass, Z_mass, &
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
   !WRITE(*,*)'Setting Chemical composition to X=',X_mass,'Z=',Z_mass
   WRITE(*,*) " "
   WRITE(*,*)'         Min                       Max                       N'
   WRITE(*,*)'Lg T  ',lgTmin, lgTmax, N_lgT
   WRITE(*,*)'Lg D  ',lgDmin, lgDmax, N_lgD
   WRITE(*,*)'Lg t  ',lgttmin, lgttmax, N_tt
   IF (Ke_norm .LE. 0.0d0) THEN
      WRITE(*,*)'Normalisation set to actual Ke' ! if input is set to 0 normalize to actual value of Kappa_e
   ELSE
      WRITE(*,*)'Normalisation set to fiducioal Ke = ', Ke_norm
   ENDIF
   WRITE(*,*)'Verbose = ',ver
   WRITE(*,*)'Output destionation -',DIR
   WRITE(*,*) " "
   CALL  flush(stdout) 

   WRITE(*,*)'Initialise Grid'
   CALL  flush(stdout) 
   
   ALLOCATE(barQ_list(N_lgT))
   ! ALLOCATE(notQ_list(N_lgT))
   ALLOCATE(kape_list(N_lgT))
   
   ALLOCATE(lgT_list(N_lgT))
   CALL linspace(lgTmin, lgTmax,lgT_list)
   ALLOCATE(lgD_list(N_lgD))
   CALL linspace(lgDmin, lgDmax,lgD_list)
   ALLOCATE(tt_list(N_tt))
   CALL linspace(lgttmin, lgttmax,tt_list)

   IF(ANY(ISNAN(tt_list))) STOP '---- tt is nan ----'
   IF(ANY(ISNAN(lgT_list))) STOP '---- lgT is nan ----'
   IF(ANY(ISNAN(lgD_list))) STOP '---- lgD is nan ----'
   IF(SIZE(tt_list).NE.N_tt) STOP '---- tt Length mismatch ----'
   IF(SIZE(lgT_list).NE.N_lgT) STOP '---- lgT Length mismatch ----'
   IF(SIZE(lgD_list).NE.N_lgD) STOP '---- lgD Length mismatch ----'
   WRITE(*,*)'Grid - dine'
   WRITE(*,*) " "
   CALL  flush(stdout) 
  
   
   WRITE(*,*)'Creating output destionation - ./',TRIM(DIR)
   CALL EXECUTE_COMMAND_LINE('mkdir -p '//TRIM(DIR), WAIT=.TRUE.)

   ! ! create Q_0 outpute file
   ! OPEN (unit = ID_notQ, file = './'//TRIM(DIR)//'/Q0_TD', FORM='formatted',STATUS='unknown', ACTION='write')
   ! WRITE(ID_notQ,*) 0.0d0, lgT_list

   ! create Qbar outpute file
   OPEN (unit = ID_barQ, file = './'//TRIM(DIR)//'/Qb_TD', FORM='formatted',STATUS='unknown', ACTION='write')
   WRITE(ID_barQ,*) 0.0d0, lgT_list

   ! create Ke outpute file
   OPEN (unit = ID_Ke, file = './'//TRIM(DIR)//'/Ke_TD', FORM='formatted',STATUS='unknown', ACTION='write')
   WRITE(ID_Ke,*) 0.0d0, lgT_list

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

   ALLOCATE(q_i(LINE_DATA%Total_line_numb))
   ALLOCATE(nu_i0(LINE_DATA%Total_line_numb))
   ALLOCATE(W_i(LINE_DATA%Total_line_numb))

   nu_i0(:) = lght_speed /(LINE_DATA%Lambda(:) * Aa2cgs)

   WRITE(*,*)'Line Data - dine'
   WRITE(*,*) " "
   CALL  flush(stdout) 


   DO D_gird_counter = 1,N_lgD
      D = 10.0d0**lgD_list(D_gird_counter)

      DO T_gird_counter = 1,N_lgT
         T = 10.0d0**lgT_list(T_gird_counter)

         WRITE(*,*)'Step ',(D_gird_counter - 1)*N_lgD + T_gird_counter,'/',N_lgD*N_lgT
         WRITE(*,*)'Set lgT =',lgT_list(T_gird_counter),' lgD =',lgD_list(D_gird_counter)
         CALL  flush(stdout)

         CALL Ilumination_finction(T,nu_i0,W_i)
         IF(ANY(ISNAN(W_i))) STOP '---- Wi is nan ----'
         
         CALL NUMB_DENS%Clear()
         CALL NUMB_DENS%Set(rho = D, T = T, verbose = ver)
         ! Dec 17 - test - DD
         kape_list(T_gird_counter) = NUMB_DENS%Electron * sigma_Thom / D
         WRITE(*,*)' > Kappa_e =', kape_list(T_gird_counter)

         WRITE(*,*)'Line strength'
         barQ_list(T_gird_counter) = 0.0d0
         ! notQ_list(T_gird_counter) = 0.0d0

         ! cycle over each line in the list
         DO ind1 = 1, LINE_DATA%Total_line_numb

            ! get transition identifiers for given line
            Ll = LINE_DATA%ID(ind1,Ll_)
            I = LINE_DATA%ID(ind1,I_)
            Z = LINE_DATA%ID(ind1,Z_)
            Lam = LINE_DATA%Lambda(ind1)* Aa2cgs

            ! IF(Z.EQ.exclude) CYCLE
            Nlines(Z,I) = Nlines(Z,I)+1

            ! check if the given level is available in atomic data
            IF (Ll.GT.ATOM_DATA%List_L(I,Z)) STOP '---- exceeding the available level ----'
            gl = ATOM_DATA%Degeneracy(Ll,I,Z)

            nl = NUMB_DENS%Occupation(Ll,I,Z)
            T =  NUMB_DENS%T

            ! compute the linstrengch 
            chi_i = sigma_clas * nl * LINE_DATA%gf_val(ind1) / gl  &
               *(1.0d0 - EXP(-NUMB_DENS%Bolz_norm/(Lam * T)))

            ! Do normalisation according to the input Ke
            IF (Ke_norm .LE. 0.0d0) THEN
               ! If input is set to 0 normalize to actual value of Kappa_e
               q_i(ind1) = chi_i/(sigma_Thom * NUMB_DENS%Electron) * Lam/lght_speed ! normalization to real electrone number density
            ELSE
               ! If input is more then 0 normalize to an input fiducial value of Kappa_e
               q_i(ind1) = chi_i/(Ke_norm * NUMB_DENS%rho) * Lam/lght_speed ! normalisation to fiducial kappa_e and actual density
            ENDIF 
            
            ! Check that the line strength is computed correctly 
            IF(ISNAN(q_i(ind1)).OR.(q_i(ind1).LT.0.0d0)) THEN
               WRITE(*,*) chi_i,nl,T,gl,Lam
               STOP '---- qi is nan ----'
            ENDIF

            barQ_list(T_gird_counter) = barQ_list(T_gird_counter) + q_i(ind1)*W_i(ind1)
            ! notQ_list(T_gird_counter) = notQ_list(T_gird_counter) + q_i(ind1)*q_i(ind1)*W_i(ind1)
         END DO
         WRITE(*,*)'  > \barQ =',barQ_list(T_gird_counter)

         ! notQ_list(T_gird_counter) = notQ_list(T_gird_counter)/barQ_list(T_gird_counter) ! final normalizationa Q0 = (sum wi qi^2)/Qb
         ! WRITE(*,*)'  > Q_not =',notQ_list(T_gird_counter)

         WRITE(*,*)'Compute Mt'

         ! Generate the  output name
         WRITE(str_1,'(F4.2)') LOG10(T)
         WRITE(str_2,'(F5.1)') LOG10(D)

         !create MT_logT_logD outpute file
         OPEN (unit = ID_Mt, file = './'//TRIM(DIR)//'/Mt_'//TRIM(str_1)//'_'//TRIM(str_2),& 
            FORM='formatted',STATUS='unknown', ACTION='write')

         ! fore each tt
         DO ind1 = 1,N_tt
            ! clean Mt an <tau>
            Mt = 0.0d0

            ! For each line
            DO ind2 = 1, LINE_DATA%Total_line_numb
               tau_s = q_i(ind2) * 10**tt_list(ind1)

               ! check if sobolev optical depth is 0, if yes take limit 
               IF(tau_s .EQ. 0.0d0) THEN
                  Mt = Mt + q_i(ind2)*W_i(ind2)
               ELSE
                  Mt = Mt + q_i(ind2)*W_i(ind2) * (1.0d0 - EXP(-tau_s))/tau_s
               END IF
            END DO
            IF(ISNAN(Mt)) STOP '---- Mt is nan ----'
            
            ! write the CAK t and M(t)
            WRITE(ID_Mt,*) tt_list(ind1), Mt
            
         END DO

         CLOSE(ID_Mt)

         WRITE(*,*)'WRITE Mt - done'
         WRITE(*,*) " "
         CALL  flush(stdout) 

      END DO !T_gird_counter

      ! ! write dencity and all temperatures point of Q_0
      ! WRITE(ID_notQ,*)lgD_list(D_gird_counter), notQ_list 

      ! write dencity and all temperatures point of bar Q
      WRITE(ID_barQ,*)lgD_list(D_gird_counter), barQ_list 

      ! write dencity and all temperatures point of kappa electrone
      WRITE(ID_Ke,*)lgD_list(D_gird_counter), kape_list 
   END DO !D_gird_counter

   ! CLOSE(ID_notQ)
   ! WRITE(*,*)'Write Q not - done'

   CLOSE(ID_barQ)
   WRITE(*,*)'Write bar Q - done'

   CLOSE(ID_Ke)
   WRITE(*,*)'Write kappa e - done'

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

   SUBROUTINE LoGo()
   
      !CALL EXECUTE_COMMAND_LINE('clear')
      !WRITE(*,*) " "
      !WRITE(*,*) "     ___           ___           ___           ___           ___           ___       &
      !&.             ___       ___           ___     "
      !WRITE(*,*) "    /\__\         /\  \         /\  \         /\  \         /\  \         /\  \      &
      !&.            /\__\     /\  \         /\  \    "
      !WRITE(*,*) "   /::|  |       /::\  \       /::\  \       /::\  \       /::\  \       /::\  \     &
      !&.           /:/  /     \:\  \       /::\  \   "
      !WRITE(*,*) "  /:|:|  |      /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/\:\  \    &
      !&.          /:/  /       \:\  \     /:/\:\  \  "
      !WRITE(*,*) " /:/|:|__|__   /::\~\:\  \   /:/  \:\  \   /::\~\:\  \   /:/  \:\  \   /::\~\:\  \   &
      !&.         /:/  /        /::\  \   /::\~\:\  \ "
      !WRITE(*,*) "/:/ |::::\__\ /:/\:\ \:\__\ /:/__/ \:\__\ /:/\:\ \:\__\ /:/__/ \:\__\ /:/\:\ \:\__\  &
      !&.        /:/__/        /:/\:\__\ /:/\:\ \:\__\"
      !WRITE(*,*) "\/__/~~/:/  / \/__\:\ \/__/ \:\  \ /:/  / \/_|::\/:/  / \:\  \  \/__/ \:\~\:\ \/__/  &
      !&.        \:\  \       /:/  \/__/ \:\~\:\ \/__/"
      !WRITE(*,*) "      /:/  /       \:\__\    \:\  /:/  /     |:|::/  /   \:\  \        \:\ \:\__\    &
      !&.         \:\  \     /:/  /       \:\ \:\__\  "
      !WRITE(*,*) "     /:/  /         \/__/     \:\/:/  /      |:|\/__/     \:\  \        \:\ \/__/    &
      !&.          \:\  \    \/__/         \:\ \/__/  "
      !WRITE(*,*) "    /:/  /                     \::/  /       |:|  |        \:\__\        \:\__\      &
      !&.           \:\__\                  \:\__\    "
      !WRITE(*,*) "    \/__/                       \/__/         \|__|         \/__/         \/__/      &
      !&.            \/__/                   \/__/    "
       WRITE(*,*) " "
       WRITE(*,*) ",_       _  .______                                     ,__    ,_________ ,_______ " 
       WRITE(*,*) "| \     / | | _____|                                    | |    |___  ,___||  _____|"
       WRITE(*,*) "|  \   /  | | |       ____   _  ___.  ____   _____      | |        | |    | |      "  
       WRITE(*,*) "| \ \ /   | | '--,   / __ \ | |/ __| / ___\ / ___ \ === | |        | |    | '---,  "
       WRITE(*,*) "| |\ / /| | | ,--'  | /  \ || / /   | /    | /__/_/     | |        | |    | ,---'  "
       WRITE(*,*) "| | v_/ | | | |     | \__/ ||  /    | \___ | \___       | |____    | |    | |_____ "
       WRITE(*,*) "|_|     |_| |_|      \____/ |_|      \____/ \____/      |,_____    |_|    |_______|"
       WRITE(*,*) " "
      CALL  flush(stdout) 
   END SUBROUTINE LoGo
END PROGRAM main
