MODULE LTE_Line_module
   IMPLICIT NONE
   ! Module for computations of LTE number densities and line strengs using the
   ! atomic and line data files from FASTWIND.
   !
   ! The Module was developed by Luka Poniatowski in 2020
   !   and contains routines ratially based on original routins form FASTWIND nlte_approx.f90
   !
   ! This module contains the definitions of variables and variable types
   ! used to allocate the atomic, line data, partition functions, occupation numbers, etc.
   !
   ! version: 1.2 
   ! (07/07/21)
   ! -- in TYPE(OCCUPATION_NUMBER_TYPE)
   ! REAL(SP), DIMENSION(:), ALLOCATABLE :: Aboundance
   !      |
   !      V
   ! REAL(DP), DIMENSION(:), ALLOCATABLE :: Aboundance
   ! -- in Read_Composition
   ! REAL(SP), DIMENSION(:) ,ALLOCATABLE :: Aboundance
   !      |
   !      V
   ! REAL(DP), DIMENSION(:) ,ALLOCATABLE :: Aboundance
   ! -- in Read_Composition
   ! REAL(DP) :: Z_frac  = 0.0d0
   !      |
   !      V
   ! REAL(DP) :: Z_frac  = 0.0d0
   ! ----
   !
   ! Identifiers used in the description for definitions of dimensions and variable types:
   ! UI - Unsingen INTEGER
   ! DP - DOUBLE PRECISION
   ! 1D/2D/3D - 1/2/3 dimentional array
   ! AT - ATOMIC_DATA_TYPE
   ! LT - LINE_DATA_TYPE
   ! OT - OCCUPATION_NUMBER_TYPE
   ! ----
   !
   ! Within this module variable names are reserd as follows:
   ! Z - Nuclear charge of the atom :: UI
   ! I - Ionisation state   :: UI
   ! L - Energy level  :: UI
   ! Ll / Lu - Lower / Upper Energy level :: UI
   ! Xg - Energy of the gorund level  :: DP
   ! Xl - Excitation energy of a level  :: DP
   ! XI - Ionisaation energy from the ground level of a ion :: DP
   ! gg - Degeneracy of the groung level  :: DP
   ! gL - Degenerecy of a level :: DP
   ! ATOMIC_DATA - Structure containing atomic data :: AT
   ! Number_denc - Structure containing number dencities
   ! LINE_DATA - structure containing line transition data :: LT
   ! ---
 
   ! Hard coding the 4 byte size of integers
   INTEGER, PARAMETER :: I4B = SELECTED_INT_KIND(9)
   INTEGER, PARAMETER :: I2B = SELECTED_INT_KIND(2)
   INTEGER, PARAMETER :: SP = KIND(1.0)
   INTEGER, PARAMETER :: DP = KIND(1.d0)
   ! REAL(DP) :: N_ele, dL, dg, EI, EL, Eg
   ! INTEGER(I2B) :: Ll, Lu, L, I, Z
 
 
   ! Fixed parameters for accesing the line ID Z,I,Ll,Lu values
   INTEGER(I2B) :: Z_ = 1, I_ = 2, Ll_ = 3, Lu_ = 4
   !
   ! Trd
   ! rho
   ! Z_frac
   ! Y_frac
   ! X_frac
   !
 
   TYPE ATOMIC_DATA_TYPE
      ! This variable type is used to allocate the atomic data
 
      ! The variables are as follows:
 
      ! Maximum Nuclear charge Z, Ionisation state I,
      !  and Energy level L available in the atomic table
      INTEGER(I4B) :: MaxZ, MaxI, MaxL
 
      ! List of maximum ionisation states of each Z
      ! The list has to be 1D array of dimension (MaxZ)
      INTEGER(I4B), DIMENSION(:), ALLOCATABLE :: List_I
 
      ! List of maximum energy level for ion I of each Z
      ! The list has to be 2D array of dimensions (MaxI, MaxZ)
      INTEGER(I4B), DIMENSION(:,:), ALLOCATABLE :: List_L
 
      ! Lists of degeneracies, excitation energies
      !  for a sfecific ionisation saste I of an atom Z
      ! Lists must be 3D array of dimensions (MaxL,MaxI,MaxZ)
      REAL(DP), DIMENSION(:,:,:), ALLOCATABLE :: Degeneracy
      REAL(DP), DIMENSION(:,:,:), ALLOCATABLE :: Lev_energy
      
      REAL(DP), DIMENSION(:,:,:), ALLOCATABLE :: Lev_counter
 
      ! List of ionisation eneries of a specific ion I of atom Z
      ! List must be 2D array of dimensions (MaxI,MaxZ)
      REAL(DP), DIMENSION(:,:), ALLOCATABLE :: Ion_energy
 
      CHARACTER (len=3) :: Names(30)
 
    CONTAINS
      PROCEDURE, PASS :: Initialise => Read_Atom_Data
      PROCEDURE, PASS :: Info       =>  Print_ATOMIC_DATA_TYPE_info
   END TYPE ATOMIC_DATA_TYPE
 
   TYPE LINE_DATA_TYPE
      ! This variable type is used to allocate the line transition data
 
      ! The variables are as follows:
 
      ! Total number of transitions avvailable
      INTEGER(I4B) :: Total_line_numb
 
      ! List of line ftransition Identifiers
      ! List must be 2D array of dimention (Total_line_numb,4)
      ! where:
      ! (Total_line_numb,1) - Nuclear charge Z of the element
      ! (Total_line_numb,2) - Ionisation stage I of the element
      ! (Total_line_numb,3) - Lower energy level
      ! (Total_line_numb,4) - Upper energy level
      INTEGER, DIMENSION(:,:), ALLOCATABLE :: ID
 
      ! Lists of transition wavelength (Angstrom)
      !  and oscilatior strength of specific transition with identifier ID
      ! Lists must be 1D array of dimention (Total_line_numb)
      REAL(SP), DIMENSION(:), ALLOCATABLE :: Lambda
      REAL(SP), DIMENSION(:), ALLOCATABLE :: gf_val
 
      ! THE Index variable stores the indexes founr by Find routine
      ! It is allocated every time it is called
      INTEGER, DIMENSION(:), ALLOCATABLE :: Index
 
    CONTAINS
      PROCEDURE, PASS :: Initialise => Read_Line_Data_ascii
      PROCEDURE, PASS :: Find       => Find_line_index
      PROCEDURE, PASS :: Info       => Print_LINE_DATA_TYPE_info
   END TYPE LINE_DATA_TYPE
 
   TYPE OCCUPATION_NUMBER_TYPE
      ! This variable type is used to allocate the number densities
 
      ! The variables are as follows:
 
      ! Number aboundences of elements with nuclear charge Z realtive to He
      REAL(DP), DIMENSION(:), ALLOCATABLE :: Aboundance
 
      ! Number dencity of nucleis with nuclear charge Z = rho*Aboundance*Y/(He_mass)
      REAL(DP), DIMENSION(:), ALLOCATABLE :: Nuclei
 
      ! Partition function of ionisation stage I of element Z (I,Z)
      REAL(DP), DIMENSION(:,:), ALLOCATABLE :: Partition
 
      ! Number dencity of ions I of element Z (I,Z)
      REAL(DP), DIMENSION(:,:), ALLOCATABLE :: Ions
 
      ! Occupation number dencities of level L of ion I of element Z (L,I,Z)
      REAL(DP), DIMENSION(:,:,:), ALLOCATABLE :: Occupation
 
      ! Number dencity of electrones
      REAL(DP) :: Electron
 
      ! Mass fractions of H and He
      REAL(DP) :: X_frac
      REAL(DP) :: Y_frac
 
      ! Temperature assigned to the type
      REAL(DP) :: T
      ! Mass dencity assigned to the type
      REAL(DP) :: rho
 
      ! Atomic weights from fastwind
      REAL(DP) :: Atomic_weight(30) = &
           [ 1.008d0,  4.003d0,  6.941d0,  9.012d0, 10.811d0, 12.011d0,&
           14.007d0, 16.000d0, 18.998d0, 20.180d0, 22.990d0, 24.305d0,&
           26.982d0, 28.085d0, 30.974d0, 32.066d0, 35.453d0, 39.948d0,&
           39.098d0, 40.078d0, 44.956d0, 47.88d0,  50.941d0, 51.996d0,&
           54.938d0, 55.847d0, 58.933d0, 58.69d0,  63.546d0, 65.39d0]
 
      ! h_plank*cc/kb as addopted from fastwind
      REAL(DP) :: Bolz_norm = 1.43883549673335d0
 
      ! Saha normalization factor (2pi m_e k_B/h^2)^(3/2) addopted from FW
      REAL(DP) :: Saha_norm = 2.414535115067086e+15
 
      ! Mass of hydrogen
      REAL(DP) :: mH = 1.67372360d-24
 
      ! Pointer to the Atomic data the occupation numbers are initiated with
      TYPE(ATOMIC_DATA_TYPE), POINTER :: ATOMIC_DATA
 
    CONTAINS
      PROCEDURE, PASS :: Initialise => Read_Composition
      PROCEDURE, PASS :: Set        => Comput_Occupation_Numbers
      PROCEDURE, PASS :: Clear      => Clear_OCCUPATION_NUMBER_TYPE
      PROCEDURE, PASS :: Info       => Print_OCCUPATION_NUMBER_TYPE_info
   END TYPE OCCUPATION_NUMBER_TYPE
 
 
   PRIVATE :: Read_Atom_Data, Print_ATOMIC_DATA_TYPE_info, Read_Line_Data_ascii, &
        Find_line_index,Print_LINE_DATA_TYPE_info,Read_Composition,Comput_Occupation_Numbers, &
        Comput_Partition_Functions, Comput_Ionisation_Fractions,Print_OCCUPATION_NUMBER_TYPE_info
 
   !=============================================================================
   !=============================================================================
   !=============================================================================
   !=============================================================================
 
 CONTAINS
 
   SUBROUTINE Read_Atom_Data(OBJ,verbose)
     ! In-Out and optional variables
     CLASS(ATOMIC_DATA_TYPE), INTENT(INOUT) ::  OBJ
     LOGICAL, OPTIONAL :: verbose
     LOGICAL :: ver
 
     ! Tepmroray variables used in reading and sortind tables
     INTEGER, PARAMETER  :: Lines_genr_info = 149
     INTEGER :: Core_cahrge(Lines_genr_info) = 0.d0 ! list of nuclear charges Z available
     INTEGER :: Ionis_stage(Lines_genr_info) = 0.d0 ! list of ionisation stages I for each Z
     INTEGER :: Numbr_level(Lines_genr_info) = 0.d0 ! list of number of energy levels E for each I in each Z
     INTEGER :: MaxL = 0, MaxI = 0, MaxZ = 0, Z = 0, I = 0, L = 0, N_L = 0, ind = 0, IO_status = 0, iduma(50) = 0
 
     ! Dummy variables for variout uses
     DOUBLE PRECISION :: rdum = 0.d0
     INTEGER :: idum = 0
     CHARACTER(LEN=10) :: cdum = '          '
     
     ! String containing the dectination to the project directiory
     CHARACTER(256) DATA_DIR
 
     ! Get the destination ot the data location (check if the enviroment was set)
     CALL get_environment_variable("MFORCE_DIR", DATA_DIR)
     IF (LEN_TRIM(DATA_DIR).EQ.0) STOP '---- Error: MFORCE_DIR not set -----'
     DATA_DIR = TRIM(DATA_DIR)//'/DATA'
 
     ! Pars the input optianal variables
     ! If verbose = .true. specified give datailed status
     ver = .FALSE.
     IF(PRESENT(verbose)) ver = verbose
 
     IF(ver) PRINT*,' '
     IF(ver) PRINT*,'Initialising the Atomic Data Type:'
 
     ! Chech if atomic data is already Initialised
     IF(ALLOCATED(OBJ%List_I)) THEN
        IF(ver) PRINT*,'  - Already Initialised, Nothing to do here'
        RETURN
     END IF
 
     ! create the temproraly storage for the data from the generalinfo file containing
     ! the  complit list of available energy level (E) for aveilable inoisation stage (I) of
     ! given element with nuclear charge (Z). Organisation of file as follows:
     !
     ! Nu  Nu  Nu  Nu  Nu  Nu  Nu  Nu  Nu  Nu  Nu
     ! v   |   |   x   x   x   |   x   x   |   x
     ! Line counter C.         |           |
     !     |   |               |           |
     !     v   |               |           v
     !     Nuclear chagre Z.   v           Number of available energy levels E.
     !         v               Number of available lines for I in Z.
     !         Ionisation stage I of Z.
 
     ! open and read the generalinfo
     OPEN (UNIT=100, FILE=TRIM(DATA_DIR)//'/generalinfo', STATUS='OLD',IOSTAT=IO_status)
     IF(IO_status .NE. 0) THEN
        PRINT*,'---- Error Opening the '//TRIM(DATA_DIR)//'/generalinfo file STATUS ', &
             IO_status,' -----'
        STOP
     ENDIF
 
     DO  ind = 1,Lines_genr_info
        READ(100,*)idum,Core_cahrge(ind),Ionis_stage(ind),&
             idum,idum,idum,idum,idum,idum,Numbr_level(ind),idum
     END DO
     CLOSE(100)
     IF(ver)PRINT*,'  - General information read'
     ! generalinfo file read and closed
 
     ! Create output storage for maximum available E and I for each Z
     OBJ%MaxZ = MAXVAL(Core_cahrge)
     OBJ%MaxI = MAXVAL(Ionis_stage) + 1
     MaxZ = OBJ%MaxZ
     MaxI = OBJ%MaxI
 
     ALLOCATE(OBJ%List_I(MaxZ))
     OBJ%List_I = 0
     OBJ%MaxL = MAXVAL(Numbr_level)
     MaxL = OBJ%MaxL
 
     ALLOCATE(OBJ%List_L(MaxI,MaxZ))
     OBJ%List_L = 0
 
     IF(ver)PRINT*,'    > Maximum nuclear charge:  ', OBJ%MaxZ
     IF(ver)PRINT*,'    > Maximum ionisation stage:', OBJ%MaxI
     IF(ver)PRINT*,'    > Maximum energy level:    ', OBJ%MaxL
 
     ! Create output storage for the data from the atomnl2_meta_v2.0 file containing the
     ! the lists ionisation energies and degenerecies of every energy level E
     ! for every ionisation state I of spicies with nuclear charge Z.
     ! Orgatisation of file as follows:
     !
     ! non indented lines
     !
     ! Nu  Nu  Nu  Nu     Nu  Nu  Nu
     ! v   |   |   |      |   x   x
     ! Counter corresponding to C.
     !     v   |   |      |
     !     Nuclear charge Z.
     !         v   |      |
     !         Ionisation stage I  of Z.
     !             v      |
     !             Number of energy levels E available for I of Z.
     !                    v
     !                    Ionisation energy of the  gorund lavel Xg (cm^-1).
     !
     ! indented lines
     ! Nu  Nu  Nu  Nu  Nu  St  Nu  Nu  Nu  Nu  Nu  Nu  Nu  Nu
     ! v   x   x   |   |   |   x   |   x   x   x   x   x   x
     ! Counter of the energy level E.
     !             v   |   |       |
     !             Gorund to level E excitation energy Xl (cm^-1).
     !              Ionisation energy of the level E is then -hc*(Xg - Xl).
     !                 v   |       |
     !                 Degeneracy of the level E.
     !                     v       |
     !                     String identifire of
     !                     matastabil 'm' and subordinant's'  levels.
     !                             v
     !                             Transition to ground energy (cm^-1).
 
     ALLOCATE(OBJ%Degeneracy(MaxL,MaxI,MaxZ))
     OBJ%Degeneracy = 0.0d0
     ALLOCATE(OBJ%Lev_energy(MaxL,MaxI,MaxZ))
     OBJ%Lev_energy = 0.0d0
     ALLOCATE(OBJ%Lev_counter(MaxL,MaxI,MaxZ))
     OBJ%Lev_counter = 0.0d0
     ALLOCATE(OBJ%Ion_energy(MaxI,MaxZ))
     OBJ%Ion_energy = 0.0d0
 
     ! open and read atomnl2_meta_v2.0
     OPEN (UNIT=100, FILE=TRIM(DATA_DIR)//'/atomnl2_meta_v2.0', STATUS='OLD',IOSTAT=IO_status)
     IF(IO_status .NE. 0) THEN
        PRINT*,'---- Error Opening the '//TRIM(DATA_DIR)//'/atomnl2_meta_v2.0 file STATUS ', &
             IO_status,' -----'
        STOP
     ENDIF
 
     DO ind = 1,Lines_genr_info
        ! get info from temproraly lists from Generatlinfo
        Z = Core_cahrge(ind) ! nuclar charge of the element we are reading
        I = Ionis_stage(ind) ! ionisation state of the element
        N_L = Numbr_level(ind) ! number of energy levels available for given spicies
        OBJ%List_L(I,Z) = N_L ! save the number of available energy levels for each spicies
 
        ! retriev data for the ground state ionisation
        READ(100,*)idum, idum, idum, idum, OBJ%Ion_energy(I,Z), rdum, rdum
 
        !------------   for testing  -----------
        ! IF(Z .EQ. 26) THEN ! fortesting
        !     PRINT*,OBJ%Ion_energy(I,Z)
        ! END IF
        !---------------------------------------
 
        ! contenue reading indented lines for oll the other levels
        DO L = 1,N_L
           READ(100,*)OBJ%Lev_counter(L,I,Z), idum, idum, OBJ%Lev_energy(L,I,Z), OBJ%Degeneracy(L,I,Z), &
                cdum, rdum, rdum, rdum, rdum, rdum, rdum, rdum, cdum
           ! the data in Lev_energy(E,I,Z) and Degeneracy(L,I,Z) is organized for each level E of
           ! given ionisation state I for every element Z as
        END DO
     END DO
     CLOSE(100)
     IF(ver)PRINT*,'  - Atomic data read '
     ! atomnl2_meta_v2.0 read and closed
 
 
 
     ! open and read the partit file fontaining the degeneracies of 1+
     ! ionisation state of the maximum available ionisation state
     ! file containing the list of Max[I(Z)] and degeneracy for ionisation state Max[I(Z)] + 1
 
     OPEN (UNIT=100, FILE=TRIM(DATA_DIR)//'/partit', STATUS='OLD',IOSTAT=IO_status)
     IF(IO_status .NE. 0) THEN
        PRINT*,'---- Error Opening the '//TRIM(DATA_DIR)//'/partit file STATUS  ', &
             IO_status,' -----'
        STOP
     ENDIF
 
     READ(100,*) (iduma(ind),ind=1,MaxZ)
     DO Z = 1,MaxZ
        OBJ%List_I(Z) = iduma(Z) + 1
     END DO
 
     READ(100,*) (iduma(ind),ind=1,MaxZ)
     DO Z = 1,MaxZ
        I = OBJ%List_I(Z)
        OBJ%Degeneracy(1,I,Z) = iduma(Z)
     END DO
     CLOSE(100)
     IF(ver)PRINT*,'  - "partit" read'
     ! partit read and closed
 
     OBJ%Names = ['H ', 'He', 'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', &
          & 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar', &
          & 'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', &
          & 'Ni', 'Cu', 'Zn']
 
     IF(ver) THEN
        PRINT*,'  > Elements Elements:'
        PRINT*,OBJ%Names(1:MaxZ)
        PRINT*,'  - Atomic Data Type READY'
     END IF
   END SUBROUTINE Read_Atom_Data
   !=============================================================================
 
   SUBROUTINE Print_ATOMIC_DATA_TYPE_info(OBJ)
     CLASS(ATOMIC_DATA_TYPE), INTENT(IN) ::  OBJ
     LOGICAL :: Initialised = .FALSE.
 
     PRINT*,''
     PRINT*,'Information on ATOMIC_DATA_TYPE requested'
 
     IF(.NOT.ALLOCATED(OBJ%List_I)) THEN
        PRINT*,'  - ATOMIC_DATA_TYPE seems NOT Initialised '
        PRINT*,'  > For Initialisation CALL ATOMIC_DATA_TYPE_NAME%Initialise()'
        PRINT*,'  > Printing ONLY General information'
     ELSE
        PRINT*,'  - ATOMIC_DATA_TYPE IS Initialised '
        PRINT*,'  > Printing Full information '
        Initialised = .TRUE.
     END IF
 
     PRINT*,''
     PRINT*,'  - Available variables:'
     PRINT*,'  > MaxZ, MaxI, MaxL - Maximum number of Z,I,L '
     PRINT*,'  > List_I(Z) - Maximum available ionisation stage for element Z '
     PRINT*,'  > List_L(I,Z) - Maximum level for ionisation stage I of Z'
     PRINT*,'  > Degeneracy(L,I,Z) - Degenerecy of level L of I,Z'
     PRINT*,'  > Lev_energy(L,I,Z) - Excitation energy of level L of I,Z'
     PRINT*,'  > Lev_counter(L,I,Z) - Level counter for level L of I,Z'
     PRINT*,'  > Ion_energy(I,Z) - Ionisation energy from ground level of I,Z'
     PRINT*,'  > Names(Z) - Name of element Z'
     PRINT*,''
     PRINT*,'  - Available PROCEDURE:'
     PRINT*,'  > Initialise(verbose) - Initialize the Atomic data, read files and assigne values'
     PRINT*,'    > verbose - (OPTIONAL) enable/desable output '
     PRINT*,'  > Info() - give Information about the Type/object '
 
     IF(Initialised) THEN
        PRINT*,'  - Allocated data'
        PRINT*,'  > MaxZ:', OBJ%MaxZ
        PRINT*,'  > MaxI:', OBJ%MaxI
        PRINT*,'  > MaxL:', OBJ%MaxL
        PRINT*,'  > Available elements:'
        PRINT*,OBJ%Names(1:OBJ%MaxZ)
     END IF
 
     PRINT*,'This PROCEDURE is under constraction'
   END SUBROUTINE Print_ATOMIC_DATA_TYPE_info
   !=============================================================================
   !=============================================================================
   !=============================================================================
   !=============================================================================
 
 
 
   SUBROUTINE Read_Line_Data_ascii(OBJ,verbose)
     ! In-Out and optional variables
     CLASS(LINE_DATA_TYPE), INTENT(INOUT) :: OBJ
     LOGICAL, OPTIONAL :: verbose
     LOGICAL :: ver
 
     ! Temproraly variables
     INTEGER(I4B) :: Total_line_numb = 0, IO_status = 0, ind = 0, ind1 = 0, ind2 = 0, ind3 = 0
     INTEGER(I4B),  DIMENSION(:), ALLOCATABLE ::  Line_ID
     REAL(SP),  DIMENSION(:), ALLOCATABLE :: Lambda
     REAL(SP),  DIMENSION(:), ALLOCATABLE :: gf_val
 
     ! Dummy variables
     INTEGER(I4B) :: idum  = 0
 
     ! String containing the dectination to the project directiory
     CHARACTER(256) DATA_DIR
 
     ! Get the destination ot the data location (check if the enviroment was set)
     CALL get_environment_variable("MFORCE_DIR", DATA_DIR)
     IF (LEN_TRIM(DATA_DIR).EQ.0) STOP '---- Error: MFORCE_DIR not set -----'
     DATA_DIR = TRIM(DATA_DIR)//'/DATA'
 
     ! Pars the input optianal variables
     ! If verbose = .true. specified give datailed status
     ver = .FALSE.
     IF(PRESENT(verbose)) ver = verbose
 
     IF(ver) PRINT*,' '
     IF(ver) PRINT*,'Initialising the Line Data Type:'
 
 
     ! Chech if atomic data is already Initialised
     IF(ALLOCATED(OBJ%ID)) THEN
        IF(ver) PRINT*,'  - Already Initialised, Nothing to do here'
        RETURN
     END IF
 
     ! Open the line list info file
     OPEN (UNIT=100, FILE=TRIM(DATA_DIR)//'/nl3info', STATUS='OLD',IOSTAT=IO_status)
     ! Check if opening was sucssesful
     IF(IO_status .NE. 0) THEN
        PRINT*,'---- Error Opening the '//TRIM(DATA_DIR)//'/nl3info file STATUS ', &
             IO_status, ' ----'
        STOP
     ENDIF
 
     ! Read the total number of lines
     READ(100,*)OBJ%Total_line_numb,idum,idum
     Total_line_numb = OBJ%Total_line_numb
     CLOSE(100)
 
     IF(ver)PRINT*,'  - General info read'
     IF(ver)PRINT*,'    > Total number of available transition', OBJ%Total_line_numb
 
     ! Allocating the variables
     ALLOCATE(Line_ID(Total_line_numb))
     Line_ID = 0
     ALLOCATE(Lambda(Total_line_numb))
     Lambda = 0.0d0
     ALLOCATE(gf_val(Total_line_numb))
     gf_val = 0.0d0
     ALLOCATE(OBJ%ID(Total_line_numb,4))
     OBJ%ID = 0
     ALLOCATE(OBJ%Lambda(Total_line_numb))
     OBJ%Lambda = 0.0d0
     ALLOCATE(OBJ%gf_val(Total_line_numb))
     OBJ%gf_val = 0.0d0
 
     ! OPEN line identifiers
     OPEN (UNIT=100, FILE=TRIM(DATA_DIR)//'/nl3i_all_ascii', FORM='formatted', STATUS='OLD',IOSTAT=IO_status)
     ! Check if opening was sucssesful
     IF(IO_status .NE. 0) THEN
        PRINT*,'---- Error Opening the '//TRIM(DATA_DIR)//'/nl3i_all_ascii file STATUS ', &
             IO_status, ' ----'
        STOP
     ENDIF
 
     ! OPEN Transition wavelength
     OPEN (UNIT=101, FILE=TRIM(DATA_DIR)//'/nl3a_all_ascii', FORM='formatted', STATUS='OLD',IOSTAT=IO_status)
     ! Check if opening was sucssesful
     IF(IO_status .NE. 0) THEN
        PRINT*,'---- Error Opening the '//TRIM(DATA_DIR)//'/nl3a_all_ascii file STATUS ', &
             IO_status, ' ----'
        STOP
     ENDIF
 
     ! OPEN Transition wavelength
     OPEN (UNIT=102, FILE=TRIM(DATA_DIR)//'/nl3g_all_ascii', FORM='formatted', STATUS='OLD',IOSTAT=IO_status)
     ! Check if opening was sucssesful
     IF(IO_status .NE. 0) THEN
        PRINT*,'---- Error Opening the '//TRIM(DATA_DIR)//'/nl3g_all_ascii file STATUS ', &
             IO_status, ' ----'
        STOP
     ENDIF
 
     !READ
     ! DO ind = 1,83
     !   ind2 = (ind-1)*50000 + 1
     !   ind3 = ind2 + 50000 - 1
     !
     !   READ(100)(Line_ID(ind1),ind1=ind2,ind3)
     !   READ(101)(Lambda(ind1),ind1=ind2,ind3)
     !   READ(102)(gf_val(ind1),ind1=ind2,ind3)
     ! END DO
     ! PRINT*,ind3,Total_line_numb - 7108
     ! READ(100)(Line_ID(ind1),ind1=ind3+1,Total_line_numb)
     ! READ(101)(Lambda(ind1),ind1=ind3+1,Total_line_numb)
     ! READ(102)(gf_val(ind1),ind1=ind3+1,Total_line_numb)
 
     READ(100,*)(Line_ID(ind),ind=1,Total_line_numb)
     READ(101,*)(Lambda(ind),ind=1,Total_line_numb)
     READ(102,*)(gf_val(ind),ind=1,Total_line_numb)
     CLOSE(100)
     CLOSE(101)
     CLOSE(102)
 
 
     ! PARSE The Line ID of following format:
     ! zZI0LlLu  (z - is not fixed and can appear or disapperar
     ! depending on length,  0-has a fixd location and is not used)
     ! zZ - Nuclear charge Z
     ! I  - ionisation
     ! 0  - Not used
     ! Ll - Lower level
     ! Lu - Upper level
     ! Examples:
     !  1100102: Z = 1(Hydrogen), I = 1 (Neutral), 0, 01 = ground level, 02 = First excited
     ! 19500317: Z = 19(K), I = 5(4x Ionised), 0, 03 = 3rd excited, 17 = 17th excited
     ! indexis for storing  Z_ = 1, I_ = 2, Ll_ = 3, Lu = 4
     ! for testinf
     !Total_line_numb = 1
     ! ZZIILiLu
     
     !DO ind = 1,Total_line_numb
        ! bunch of integer arithmetics to parts the IDs
     !   OBJ%ID(ind,1) = Line_ID(ind)/1000000
     !   OBJ%ID(ind,2) = (Line_ID(ind) - OBJ%ID(ind,1)*1000000)/100000
     !   OBJ%ID(ind,3) = MOD(Line_ID(ind),10000)/100
     !   OBJ%ID(ind,4) = MOD(Line_ID(ind),100)
 
        ! check each translation
     !   IF( (OBJ%ID(ind,1)*1000000 + OBJ%ID(ind,2)*100000 + OBJ%ID(ind,3)*100 &
     !        + OBJ%ID(ind,4)).NE.Line_ID(ind) ) &
     !        STOP '---- InconsistencyIn reading the line ID ----'
     !END DO
     
     DO ind = 1,Total_line_numb
        ! bunch of integer arithmetics to parts the IDs
        OBJ%ID(ind,1) = Line_ID(ind)/1000000
        OBJ%ID(ind,2) = (Line_ID(ind) - OBJ%ID(ind,1)*1000000)/100000
        OBJ%ID(ind,3) = MOD(Line_ID(ind),10000)/100
        OBJ%ID(ind,4) = MOD(Line_ID(ind),100)
 
        ! check each translation
        IF( (OBJ%ID(ind,1)*1000000 + OBJ%ID(ind,2)*100000 + OBJ%ID(ind,3)*100 &
             + OBJ%ID(ind,4)).NE.Line_ID(ind) ) &
             STOP '---- InconsistencyIn reading the line ID ----'
     END DO
     IF(ver)PRINT*,'  - Line IDs read'
 
     ! Thech that the last entry of the file is correctly read
     !  by comparing it to manually enterd value (Mast be changed id the list is changed)
     !IF((Lambda(Total_line_numb)/3.7174720d+07 - 1.0d0).GT.1.0d-8) THEN
     !   PRINT*,'---- Inconsistency in the last wavelength of the list ----'
     !   PRINT*,'  > Expected value:',3.7174720d+07
     !   PRINT*,'  > Read value:    ',Lambda(Total_line_numb)
     !   STOP
     !END IF
 
     ! If read was sucssesful store the wavelength array
     OBJ%Lambda(:) = Lambda(:)
     IF(ver)PRINT*,'  - Line wavelength read'
 
 
     ! Thech that the last entry of the file is correctly read
     !  by comparing it to manually enterd value (Mast be changed id the list is changed)
     !IF((gf_val(Total_line_numb)/1000.0 - 1.0d0).GT.1.0d-8) THEN
     !   PRINT*,'---- Inconsistency in the last wavelength of the list ----'
     !   PRINT*,'  > Expected value:',1000.0
     !   PRINT*,'  > Read value:    ',gf_val(Total_line_numb)
     !   STOP
     !END IF
 
     ! If read was sucssesful store the gf values array
     OBJ%gf_val(:) = gf_val(:)
     IF(ver)PRINT*,'  - Line oscilator strength read' 
 
     IF(ver)PRINT*,'  - Line Data Type READY'
   END SUBROUTINE Read_Line_Data_ascii
   !=============================================================================
 
   SUBROUTINE Find_line_index(OBJ,Z,I,Ll,Lu,Lambda,Toler,verbose)
     ! In-Out and optional variables
     CLASS(LINE_DATA_TYPE), INTENT(INOUT) ::  OBJ
     ! INTEGER(I4B), INTENT(OUT) :: INDEX
     INTEGER(I4B), OPTIONAL :: Z
     INTEGER(I4B), OPTIONAL :: I
     INTEGER(I4B), OPTIONAL :: Ll
     INTEGER(I4B), OPTIONAL :: Lu
     REAL(SP), OPTIONAL :: Lambda
     REAL(SP), OPTIONAL :: Toler
     LOGICAL, OPTIONAL :: verbose
     LOGICAL :: ver
 
     INTEGER(I4B) :: ind,ind1, Total_line_numb, ID
     INTEGER, DIMENSION(:), ALLOCATABLE :: ID_Index
 
 
     ! Pars the input optianal variables
     ! If verbose = .true. specified give datailed status
     ver = .FALSE.
     IF(PRESENT(verbose)) ver = verbose
 
     IF(ver) PRINT*,' '
     IF(ver) PRINT*,'Find index called'
 
     IF(.NOT.ALLOCATED(OBJ%Lambda)) STOP '---- Terminating: Line Data Not Initialized ----'
 
     ! Deallocate the OBJ%Index array for allocation of new matches
     IF(ALLOCATED(OBJ%Index))DEALLOCATE(OBJ%Index)
 
     ! copy the number of lines for coding simplisity
     Total_line_numb = OBJ%Total_line_numb
     ALLOCATE(ID_Index(Total_line_numb))
 
     ! Check whot inputs are provided fo lookup
     IF(PRESENT(Lambda).AND.PRESENT(Toler)) THEN
        ! Find by lambda
        IF(ver) PRINT*,'  - Search index by input wavelength'
 
        ! start counting the number of matches
        ind1 = 0
 
        ! go trhough each line index
        DO ind=1,Total_line_numb
 
           ! If the
           IF(ABS(OBJ%Lambda(ind)/Lambda - 1) .LE. Toler) THEN
              ! increase the counter of found matches by 1 (a)
              ind1 = ind1 + 1
 
              ! if the id match save the index of the line
              ID_Index(ind1) = ind
           END IF
        END DO
 
     ELSE IF( PRESENT(Z).AND.PRESENT(I).AND.PRESENT(Ll).AND.PRESENT(Lu) ) THEN
        ! Find by ID
        IF(ver) PRINT*,'  - Search index by input transition ID (Z,I,Ll,Lu)'
 
        ! convert input into single integer
        ID = Z*1000000 + I*100000 + Ll*100 + Lu
        IF(ver) PRINT*,'   > Looking for:', ID
 
        ! start counting the number of matches
        ind1 = 0
 
        ! go trhough each line index
        DO ind=1,Total_line_numb
 
           ! Chech if the ID on the line matches the input ID
           IF( (OBJ%ID(ind,1)*1000000 + OBJ%ID(ind,2)*100000 + &
                OBJ%ID(ind,3)*100 + OBJ%ID(ind,4)) .EQ. ID)  THEN
 
              ! increase the counter of found matches by 1 (a)
              ind1 = ind1 + 1
 
              ! if the id match save the index of the line
              ID_Index(ind1) = ind
           END IF
        END DO
     ELSE
        ! No input provided - abbort
        IF(ver) PRINT*,'  - No imput provided'
        ! allocate and store 0
        ALLOCATE(OBJ%Index(1))
        OBJ%Index = 0
     END IF
 
 
     IF(ver) PRINT*,'  > Found matches:', ind1
 
     ! if any match is found allocate corresponding number of OBJ%Index
     ! and store the matchin indexes. If no match  found allocate and store 0
     IF(ind1.GT.0) THEN
        ALLOCATE(OBJ%Index(ind1))
        OBJ%Index(1:ind1) = ID_Index(1:ind1)
     ELSE
        ALLOCATE(OBJ%Index(1))
        OBJ%Index = 0
     END IF
   END SUBROUTINE Find_line_index
   !=============================================================================
 
   SUBROUTINE Print_LINE_DATA_TYPE_info(OBJ)
     CLASS(LINE_DATA_TYPE), INTENT(IN) ::  OBJ
     LOGICAL :: Initialised = .FALSE.
 
     PRINT*,' '
     PRINT*,'Information on LINE_DATA_TYPE requested'
 
     IF(.NOT.ALLOCATED(OBJ%ID)) THEN
        PRINT*,'  - LINE_DATA_TYPE seems NOT Initialised '
        PRINT*,'  > For Initialisation CALL LINE_DATA_TYPE_NAME%Initialise()'
        PRINT*,'  > Printing ONLY General information'
     ELSE
        PRINT*,'  - LINE_DATA_TYPE IS Initialised '
        PRINT*,'  > Printing Full information '
        Initialised = .TRUE.
     END IF
 
     PRINT*,'  - Available variables:'
     PRINT*,'  > Total_line_numb - Number of available lines'
     PRINT*,'  > ID - Transition identifier (output)'
     PRINT*,'  > Lambda - Transition wavelength (output)'
     PRINT*,'  > gf_val - Transition gf value (output)'
     PRINT*,'  > Index - List of line index found (output) '
     PRINT*,'  - Available PROCEDURE'
     PRINT*,'  > Initialise(verbose)'
     PRINT*,'    > verbose - (OPTIONAL) enable/desable output '
     PRINT*,'  > Find(Z,I,Ll,Lu,Lambda,Toler,verbose) - '
     PRINT*,'    > Z - (OPTIONAL) nuclear change Z of element of interest'
     PRINT*,'    > I - (OPTIONAL) ionisation stage I of lelemnt of interest '
     PRINT*,'    > Ll - (OPTIONAL) lower level of transition'
     PRINT*,'    > Lu - (OPTIONAL) upper level of transition'
     PRINT*,'    > Lambda - (OPTIONAL)'
     PRINT*,'    > Toler - (OPTIONAL)'
     PRINT*,'    > verbose - (OPTIONAL) enable/desable output '
     PRINT*,'  > Info() - give Information about the Type/object'
 
     IF(Initialised) THEN
        PRINT*,'  - Allocated data'
     END IF
 
     PRINT*,'This PROCEDURE is under constraction'
   END SUBROUTINE Print_LINE_DATA_TYPE_info
   !=============================================================================
   !=============================================================================
   !=============================================================================
   !=============================================================================
 
 
 
 
   SUBROUTINE Read_Composition(OBJ,ATOMIC_DATA,X_frac,Y_frac,verbose)
     ! In-Out and optional variables
     CLASS(OCCUPATION_NUMBER_TYPE), INTENT(INOUT) :: OBJ
     CLASS(ATOMIC_DATA_TYPE),TARGET, INTENT(IN)   :: ATOMIC_DATA
     REAL(DP), OPTIONAL :: X_frac
     REAL(DP), OPTIONAL :: Y_frac
     LOGICAL, OPTIONAL :: verbose
     LOGICAL :: ver, USE_solar_composition, USE_H_normalisation
 
     REAL(DP), DIMENSION(:) ,ALLOCATABLE :: Mass_fractions
     REAL(DP), DIMENSION(:) ,ALLOCATABLE :: Aboundance
     REAL(DP) :: Z_frac  = 0.0d0
     INTEGER(I4B) :: MaxZ = 0, MaxI = 0, MaxL = 0, Z = 0, ind = 0, IO_status = 0
 
 
     INTEGER(I4B) :: idum, i
     CHARACTER(LEN=10) :: cdum = '          '
 
     ! String containing the dectination to the project directiory
     CHARACTER(256) DATA_DIR
 
     ! Get the destination ot the data location (check if the enviroment was set)
     CALL get_environment_variable("MFORCE_DIR", DATA_DIR)
     IF (LEN_TRIM(DATA_DIR).EQ.0) STOP '---- Error: MFORCE_DIR not set -----'
     DATA_DIR = TRIM(DATA_DIR)//'/DATA'
     
 
     ! Pars the input optianal variables
     ! If verbose = .true. specified give datailed status
     ver = .FALSE.
     IF(PRESENT(verbose)) ver = verbose
 
 
     ! create the linc to atomic data for the later use
     OBJ%ATOMIC_DATA => ATOMIC_DATA
 
 
     ! Chech if ATOMIC_DATA is allocated - if not Stop
     IF(.NOT.ALLOCATED(ATOMIC_DATA%List_I)) STOP '---- Terminating: Atomic data not Initialised ----'
 
 
     ! create short-hands
     MaxZ = ATOMIC_DATA%MaxZ
     MaxI = ATOMIC_DATA%MaxI
     MaxL = ATOMIC_DATA%MaxL
 
     IF(ver)PRINT*,' '
     IF(ver) PRINT*,'Initialising the Occupation number Type:'
 
     ! Chech if atomic data is already Initialised
     IF(ALLOCATED(OBJ%Aboundance)) THEN
        IF(ver) PRINT*,'  - Already Initialised, Nothing to do here'
        RETURN
     END IF
 
     ! Check the input X_frac + Y_frac < 1 and Y_frac |= 0
     IF ((PRESENT(X_frac).AND.PRESENT(Y_frac)) ) THEN
        IF( (X_frac + Y_frac) .LE. 1.0d0) THEN
           IF(ver) PRINT*, '  - Input mass fractions provided'
           USE_solar_composition = .FALSE.
           IF(X_frac.NE.0.0d0) THEN
              IF(ver) PRINT*, '  - Scaling aboundances to H'
              USE_H_normalisation = .TRUE.
           ELSE
              IF(ver) PRINT*, '  - Scaling aboundances to He'
              USE_H_normalisation = .FALSE.
           END IF
        ELSE
           STOP '---- Error in input X and Y mass fractions (X+Y > 1 or Y = 0) ----'
        END IF
     ELSE
        USE_solar_composition = .TRUE.
        USE_H_normalisation = .TRUE.
        IF(ver) PRINT*, '  - No input mass fractions provided: Using solar values'
     END IF
 
     ! Allocate and zero out the occupation number arraies
     ALLOCATE(OBJ%Aboundance(MaxZ))
     OBJ%Aboundance = 0.0d0
     ALLOCATE(OBJ%Partition(MaxI,MaxZ))
     OBJ%Partition = 0.0d0
     ALLOCATE(OBJ%Ions(MaxI,MaxZ))
     OBJ%Ions = 0.0d0
     ALLOCATE(OBJ%Nuclei(MaxZ))
     OBJ%Nuclei = 0.0d0
     ALLOCATE(OBJ%Occupation(MaxL,MaxI,MaxZ))
     OBJ%Occupation = 0.0d0
 
     ! Allocate temp variables
     ALLOCATE(Mass_fractions(MaxZ))
     Mass_fractions = 0.0d0
     ALLOCATE(Aboundance(MaxZ))
     Aboundance = 0.0d0
 
     ! Clear the rho and T by settiong them to negative values to net
     OBJ%rho  = -1.d0
     OBJ%T    = -1.d0
 
     ! Open the line list info file
     !OPEN (UNIT=100, FILE=TRIM(DATA_DIR)//'/asplund_numabun_2009', STATUS='OLD',IOSTAT=IO_status)
     OPEN (UNIT=100, FILE=TRIM(DATA_DIR)//'/number_abundance', STATUS='OLD',IOSTAT=IO_status)
     ! Check if opening was sucssesful
     IF(IO_status .NE. 0) THEN
        PRINT*,'---- Error Opening the '//TRIM(DATA_DIR)//'/number_abundance file STATUS ', &
             IO_status, ' ----'
        STOP
     ENDIF
 
     DO Z = 1,MaxZ
        READ(100,*)idum,cdum,OBJ%Aboundance(Z)
     END DO
     CLOSE(100)
     IF(ver) PRINT*, '  - Aboundances READ'
 
     ! here we normalise abandences to H or He depending if there is Y aboundance and compute mass fractions for solar mixture.
     !IF (USE_H_normalisation) THEN
     !   OBJ%Aboundance(:) = 10.0d0 ** (OBJ%Aboundance(:) - OBJ%Aboundance(1))
     !ELSE
     !   OBJ%Aboundance(:) = 10.0d0 ** (OBJ%Aboundance(:) - OBJ%Aboundance(2))
     !END IF
     
     ! fix to read in abundances rather than 10**12 dex - DD

     !JS-ADD DEC 24:  always normalised to H here      
     OBJ%Aboundance(:) = OBJ%Aboundance(:) !/OBJ%Aboundance(1)  
        
     Mass_fractions(:) = OBJ%Aboundance(:) * OBJ%Atomic_weight(:) ! compute direct masses
     Mass_fractions(:) = Mass_fractions(:)/SUM(Mass_fractions(:)) ! normalize to compute mass fractions
     
     print*, Mass_fractions(:)

     stop
     ! DD change reading mass fractions 
     !Mass_fractions(:) = Mass_fractions(:)

     
    
     !IF (.NOT.USE_solar_composition) THEN
     !   Mass_fractions(1) = X_frac ! set the H mass fraction  to X
     !   Mass_fractions(2) = Y_frac ! set the He mass fraction to Y
     !   Z_frac = MAX(0.0d0,1.0d0 - Mass_fractions(1) - Mass_fractions(2))
     !   IF(ver) PRINT*,'  > Z fraction ',Z_frac
     !   Mass_fractions(3:MaxZ) =  Mass_fractions(3:MaxZ)/SUM(Mass_fractions(3:MaxZ))*Z_frac
 
 
     !   OBJ%X_frac = X_frac
     !   OBJ%Y_frac = Y_frac
 
     !   OBJ%Aboundance(:) = Mass_fractions(:)/OBJ%Atomic_weight(:)
     !   IF (USE_H_normalisation) THEN
     !      OBJ%Aboundance(:) = OBJ%Aboundance(:)/OBJ%Aboundance(1)
     !   ELSE
     !      OBJ%Aboundance(:) = OBJ%Aboundance(:)/OBJ%Aboundance(2)
     !   END IF
     !ELSE
     !   OBJ%X_frac = Mass_fractions(1)
     !   OBJ%Y_frac = Mass_fractions(2)
     !END IF
     
     !>  DD 
     
     !X_frac = Mass_fractions(1)
     Y_frac = Mass_fractions(2)
     Z_frac = SUM(Mass_fractions(3:MaxZ))
     X_frac = MAX(0.0d0,1.0d0 - Y_frac - Z_frac) 
     !MAX(0.0d0,1.0d0 - Mass_fractions(1) - Mass_fractions(2))
     
     !MAX(0.0d0,1.0d0 - Mass_fractions(1) - Mass_fractions(2))
     
     ! Allocate the input X and Y mass fractions
     OBJ%X_frac = X_frac
     OBJ%Y_frac = Y_frac
     Mass_fractions(3:MaxZ) = Mass_fractions(3:MaxZ)/SUM(Mass_fractions(3:MaxZ))*Z_frac
     
     OBJ%Aboundance(:) = OBJ%Aboundance(:)
     
     
     
     !print*, X_frac, Y_frac, Z_frac
     !stop
     !JS-TEST: Dec 24 
     !print*,'Mass and number fractions (H,He...):'
     !do i=1,MaxZ
     !   print*, Mass_fractions(i),OBJ%Aboundance(i) 
     !enddo
     !stop 'tee...'
     
     IF(ver) PRINT*,'  - Occupation Number Type READY'
   END SUBROUTINE Read_Composition
   !=============================================================================
 
 
   SUBROUTINE Comput_Occupation_Numbers(OBJ,rho,T,N_electron,verbose)
     CLASS(OCCUPATION_NUMBER_TYPE), INTENT(INOUT) :: OBJ
     REAL(DP), INTENT(IN) :: rho
     REAL(DP), INTENT(IN) :: T
     REAL(DP), OPTIONAL :: N_electron
     LOGICAL, OPTIONAL :: verbose
     CLASS(ATOMIC_DATA_TYPE),POINTER    :: ATOMIC_DATA
     LOGICAL :: ver, iterative
 
     ! Temprorary variables
     REAL(DP), DIMENSION(:),ALLOCATABLE :: Excitation_numb_frac
     REAL(DP) :: gg = 0.0d0, Xg = 0.0d0, gl = 0.0d0, Xl = 0.0d0, SUM_Excitation_numb_frac = 0.0d0
     REAL(DP) :: Bolz_fact = 0.0d0, N_electron_old = 0.0d0, N_electron_new = 0.0d0
     INTEGER(I4B) :: L = 0, I = 0, Z = 0, ind = 0
 
     ! Pars the input optianal variables
     ! If trerative = true recompute the electrone number density
     iterative = .TRUE.
     IF(PRESENT(N_electron)) iterative = .FALSE.
     ! If verbose = .true. specified give datailed status
     ver = .FALSE.
     IF(PRESENT(verbose)) ver = verbose
 
 
     ! Chech if the OCCUPATION_NUMBER_TYPE is allocated
     IF(.NOT.ALLOCATED(OBJ%Aboundance)) STOP '---- Terminating: Occupation Numbre Type not Initialised ----'
 
     ! Check input dencity and temperature
     IF(.NOT.(rho.GT.0)) STOP '---- Terminating: Dencity not specified ---- '
     IF(.NOT.(T.GT.0)) STOP '---- Terminating: Temperature not specified ---- '
 
     IF(ver) PRINT*,' '
     IF(ver) PRINT*, 'Set Temperature and Dencity CALLED'
     IF(ver) PRINT*, '  - Computing Occupation numbers'
 
 
     ! create the assosiation(Short-hand) to use the previous link to atomic data
     ATOMIC_DATA => OBJ%ATOMIC_DATA
 
     ! Store the inpude ,ass density and temperature
     OBJ%rho  = rho
     OBJ%T    = T
 
     ! To compute the Occupation numbers do the following:
     ! 1) For  given Rho(D) and Aboundance Compute the numper density of each spicies
     ! 2) If not given Compute the approximate numberdencity of electronse assuming fully ionised H and 1/2 rest
     ! 3) For a given T_rad(T) and Atomic Data compute partition functions for each ion
     ! 4) Using the Atomic Data and partition functions compute the Ion number dencity (solve SAHA)
     ! 5) If N_electon was not specified iterate 4) to compute the N_electon
     ! 6a) Using the Atomic Data and Ion number dencity compute the occupation numbers (solve BOLTZMANN)
     ! 6b) Sum up occupation numbers to check the mass conservation
 
     ! ----- Executing 1) !!!!
     ! compute the output total number densities for each spicies H or He depending if there is Y aboundance
     
     
     IF (OBJ%X_frac .NE. 0.0d0) THEN
        IF(ver) PRINT*, '  - Using the X mass fraction for computations'
        OBJ%Nuclei(:) = rho*OBJ%Aboundance(:)*OBJ%X_frac/OBJ%mH
     ELSE IF (OBJ%X_frac .EQ. 0.0d0 .AND. OBJ%Y_frac .NE. 0.0d0) THEN
        IF(ver) PRINT*, '  - Using the Y mass fraction for computations'
        OBJ%Nuclei(:) = rho*OBJ%Aboundance(:)*OBJ%Y_frac/(4.0d0*OBJ%mH)
     ELSE
        IF(ver) PRINT*, '  - Using the Z mass fraction for computations'
        OBJ%Nuclei(:) = rho*OBJ%Aboundance(:)*(1 - OBJ%X_frac - OBJ%Y_frac)/(12.0d0*OBJ%mH)    
     END IF
     
     ! DO Z = 1,30
     !    PRINT*,OBJ%Nuclei(Z)
     ! END DO
     IF(ANY(ISNAN(OBJ%Nuclei(:)))) STOP '---- Nuclei Is NaN ----'
     IF(ver) PRINT*, '  - Number densities of each Nuclei computed'
 
     ! ------ Executing 2)
     ! Here if N_electon was not given compute the Approximate
     IF(iterative) THEN
        ! here compute the initial guess of electron number dencity by assuming
        ! that hydrogen is fully ionised and elements form He and up contributes 1/2 of their electrons
        OBJ%Electron = OBJ%Nuclei(1)
        DO Z = 2,ATOMIC_DATA%MaxZ
           OBJ%Electron = OBJ%Electron + Z/2.0d0 * OBJ%Nuclei(Z)
        END DO
 
        IF(ver) PRINT*, '   - No Input Electron Number Dencity found'
        IF(ver) PRINT*, '   > Using iterative scheme, with initial guess:', OBJ%Electron
     ELSE
        OBJ%Electron = N_electron
        IF(ver) PRINT*, '   - Input Electron Number Dencity found',OBJ%Electron
     END IF
 
 
     ! ----- Executing 3)
     CALL Comput_Partition_Functions(OBJ,ATOMIC_DATA,T,verbose = ver)
     IF(ANY(ISNAN(OBJ%Partition))) STOP '---- Partition functuon Is NaN ----'
     IF(ver) PRINT*, '  - Partition Functions READY'
 
 
     ! ----- Executing 4 % 5)
     N_electron_old = OBJ%Electron
     N_electron_new = OBJ%Electron
 
     ! If input electrone dencity was not specified do iteration
     IF(iterative) THEN
 
        DO ind = 1,100 ! set the maximum number of iterations to 100
 
           ! update the electrone numbe dencities by Ne_new = sqrt(Ne_new * Ne_old)
           ! see Lattimer & Cranmer 2020
           OBJ%Electron = SQRT(N_electron_old*N_electron_new)
           N_electron_old = OBJ%Electron
 
           CALL Comput_Ionisation_Fractions(OBJ,ATOMIC_DATA,T,iterative,verbose = ver)
           IF(ISNAN(OBJ%Electron).OR.(OBJ%Electron.EQ.0.0d0)) STOP '---- Electrons Is NaN/Zero ----'
           N_electron_new = OBJ%Electron
 
           ! chech if the change in the numberdencity is less then 0.1%
           ! is yes Ne is converged - exit the loop -- otherwithe continue
           IF(ABS(N_electron_old/N_electron_new - 1) .LT. 1.0d-3) THEN
              IF(ver) PRINT*,'   > Electron Number Dencity converged',OBJ%Electron
              EXIT
           ELSE
              IF(ver) PRINT*,'   > Iteration:',ind,' Electron Number Dencity:',OBJ%Electron
           END IF
        END DO
 
        ! if the maximum number of iterations was reached terminate
        IF(ind.GT.99) STOP '---- Electron Number Dencity NOT converged ----'
     ELSE
 
        ! compute the ionisation structure using the input electrone number density
        CALL Comput_Ionisation_Fractions(OBJ,ATOMIC_DATA,T,iterative,verbose = ver)
     END IF
 
     !
     ! DO Z = 1,30
     !   PRINT*, Z
     !   DO I = 1,ATOMIC_DATA%List_I(Z)
     !    PRINT*,I, OBJ%Ions(I,Z)
     !  END DO
     ! END DO
     IF(ANY(ISNAN(OBJ%Ions(:,:)))) STOP '---- Ions Is NaN ----'
     IF(ver) PRINT*, '  - Ionisation Fructions READY'
 
 
     ! ----- Executing 6a)
 
     ! compute the occupation number solving
     IF(ver) PRINT*,'   - Computing the occupation numbers'
     ALLOCATE(Excitation_numb_frac (ATOMIC_DATA%MaxL) )
     ! Z = 1; I = 1;
     DO Z = 1,ATOMIC_DATA%MaxZ ! for each element
 
        DO I = 1,ATOMIC_DATA%List_I(Z)-1 ! for each ion of element
 
           Excitation_numb_frac = 0.0d0
           Excitation_numb_frac(1) = 1.0d0 ! set the fraction of groung state to 1
           gg = ATOMIC_DATA%Degeneracy(1,I,Z) ! copy the statistical weight of ground level
           Xg = ATOMIC_DATA%Lev_energy(1,I,Z) ! copy the excitation energy of ground level
 
           SUM_Excitation_numb_frac = 1.0d0
 
           DO L = 2,ATOMIC_DATA%List_L(I,Z) ! for every level higher then gorund
 
              gl = ATOMIC_DATA%Degeneracy(L,I,Z) ! copy the statistical weight of the level
              Xl = ATOMIC_DATA%Lev_energy(L,I,Z) ! copy the excitation energy of the level
 
              Bolz_fact = EXP( -OBJ%Bolz_norm * (Xl-Xg) / T ) ! compute the Boltzman factor
              Excitation_numb_frac(L) =  gl/gg * Bolz_fact  ! compute the number fraction of the level to ground
 
              SUM_Excitation_numb_frac = SUM_Excitation_numb_frac + Excitation_numb_frac(L)
 
              IF(ISNAN(Excitation_numb_frac(L))) STOP '---- Ef is nan ----'
              IF(ISNAN(SUM_Excitation_numb_frac)) STOP '---- SUM Ef is nan ----'
           END DO
 
           ! --------------------------------------------------------
           ! for some reason the SUM(Excitation_numb_frac) becoes nan
           ! otherwithe here we use the dummy_SUM
           ! --------------------------------------------------------
 
           ! IF(ISNAN(SUM(Excitation_numb_frac)))  THEN
           !   PRINT*,SUM(Excitation_numb_frac),dummy_SUM
           !    DO L = 1,ATOMIC_DATA%List_L(I,Z)
           !       PRINT*,L,Excitation_numb_frac(L)
           !    END DO
           !    STOP '---- Sum(Ef) is NaN ----'
           ! END IF
           ! IF(SUM(Excitation_numb_frac).EQ.0.0d0) &
           !      STOP '---- Sum(Ef) is Zero ----'
 
           ! Compute the occupation number of each level using the number density of ion and fractions
           OBJ%Occupation(1:ATOMIC_DATA%List_L(I,Z),I,Z) = &
                OBJ%Ions(I,Z)*Excitation_numb_frac(1:ATOMIC_DATA%List_L(I,Z))/SUM_Excitation_numb_frac
 
           IF(ver.AND.(Z.EQ.26).AND.(I.EQ.1) ) THEN
              PRINT*,'   - Number fractions of Fe I relative to ground '
              DO L = 1,ATOMIC_DATA%List_L(1,26)
                 PRINT*, '    > ', L, OBJ%Occupation(L,1,26)
              END DO
           END IF
 
        END DO! for each ion of element
     END DO! for each element
 
     ! here the Occup_Number_denc(:,I,Z)/Atomic_data{:,I,Z}(2) should be
     ! equivalent of occng(i,:,radius), where i is record number from gen.info.
     ! for testing run:
     ! Z = 26; I = 1;
     IF(ver.AND.(Z.EQ.26) ) THEN
        PRINT*,'   - Occupation number densities of Fe I '
        DO L = 1,ATOMIC_DATA%List_L(1,26)
           PRINT*, '    > ', L, OBJ%Occupation(L,1,26)
        END DO
     END IF
 
 
     ! ----- Executing 6b)
     ! cheking the total number dencity
     DO Z = 1,ATOMIC_DATA%MaxZ
        DO I = 1,ATOMIC_DATA%List_I(Z)-1
           IF ( SUM(OBJ%Occupation(:,I,Z))/OBJ%Ions(I,Z) - 1 .GT. 1.0d-6 ) THEN
              STOP '---- Number Dencities NOT conserved ----'
           END IF
        END DO
     END DO
 
    !  DEALLOCATE(OBJ%Partition) this is a bug PArtition function shoud not be deallocated 
     IF(ver) PRINT*,'  - Occupation Numbers READY'
   END SUBROUTINE Comput_Occupation_Numbers
   !=============================================================================
 
 
   SUBROUTINE Comput_Partition_Functions(OBJ,ATOMIC_DATA,T,verbose)
     CLASS(OCCUPATION_NUMBER_TYPE), INTENT(INOUT) :: OBJ
     CLASS(ATOMIC_DATA_TYPE),       INTENT(IN)    :: ATOMIC_DATA
     REAL(DP), INTENT(IN) :: T
     LOGICAL, OPTIONAL :: verbose
     LOGICAL :: ver
 
     ! Temprorary variables
     INTEGER(I4B) :: Z = 0, I = 0, L = 0
     REAL(DP) :: Bolz_fact = 0.0d0, Xg = 0.0d0, gg = 0.0d0, Xl = 0.0d0, gl = 0.0d0
     ! Pars the input optianal variables
     ! If verbose = .true. specified give datailed status
     ver = .FALSE.
     IF(PRESENT(verbose)) ver = verbose
 
     IF(ver) PRINT*,'  - Computing Partition Functions'
     IF(ver) PRINT*,'   - Partition fanctions of Fe:'
     ! compute the partition FUNCTION of a single ionisation state I of a single
     ! atom Z
     ! Partition_functuon = zeros(MaxI,MaxZ) is replaces by OBJ%Partition
 
     ! For every available nuclear charge Z from 1 to Z_max
     DO Z = 1,ATOMIC_DATA%MaxZ ! Do for each Z
 
        ! % For every available ionization stage I of Z
        DO I = 1,ATOMIC_DATA%List_I(Z) ! Do for each Ionisation
 
           Xg = ATOMIC_DATA%Lev_energy(1,I,Z) ! energy of the ground level
           gg = ATOMIC_DATA%Degeneracy(1,I,Z) ! degenerecy of the ground level
           OBJ%Partition(I,Z) = gg ! Partition FUNCTION of (I,Z) - ground level contributes degenerecy
 
           ! For energy level E of Z WITH I
           DO L = 2,ATOMIC_DATA%List_L(I,Z) ! Do for each level
 
              Xl = ATOMIC_DATA%Lev_energy(L,I,Z) ! excitation energy of level E
              gl = ATOMIC_DATA%Degeneracy(L,I,Z) ! degenerecy of the level E
              Bolz_fact = EXP( - OBJ%Bolz_norm * (Xl-Xg)/T ) ! Boltzmann factor EXP( -x/kT)
              OBJ%Partition(I,Z) = OBJ%Partition(I,Z) + gl * Bolz_fact ! Partition FUNCTION of (I,Z)
 
           END DO ! Do for each level
 
           IF(ver .AND. (Z.EQ.26) ) PRINT*,'   > ',I, OBJ%Partition(I,Z)
        END DO ! Do for each Ionisation
     END DO! Do for each Z
   END SUBROUTINE Comput_Partition_Functions
   !=============================================================================
 
   SUBROUTINE Comput_Ionisation_Fractions(OBJ,ATOMIC_DATA,T,iterative,verbose)
     CLASS(OCCUPATION_NUMBER_TYPE), INTENT(INOUT) :: OBJ
     CLASS(ATOMIC_DATA_TYPE),       INTENT(IN)    :: ATOMIC_DATA
     REAL(DP), INTENT(IN) :: T
     LOGICAL, OPTIONAL :: verbose
     LOGICAL :: iterative
     LOGICAL :: ver
 
     ! Temprorary variables
     INTEGER(I4B) :: Z = 0, I = 0, L = 0, k = 0
     REAL(DP) :: Bolz_fact = 0.0d0, Saha_fact = 0.0d0, Xi = 0.0d0, Upart1 = 0.0d0, Upart2 = 0.0d0
     REAL(DP), DIMENSION(:,:), ALLOCATABLE :: S
     REAL(DP), DIMENSION(:,:), ALLOCATABLE :: Fractions_to_nutral
     ! REAL(DP), DIMENSION(:,:), ALLOCATABLE :: Number_denc
     REAL(DP), DIMENSION(:,:), ALLOCATABLE :: Number_frac
 
     ! Pars the input optianal variables
     ver = .FALSE.
     IF(PRESENT(verbose)) ver = verbose
 
     IF(ver) PRINT*,'   - Computing Ionisation Fractions'
 
     ALLOCATE(S(ATOMIC_DATA%MaxI,ATOMIC_DATA%MaxZ))
     S = 0.0d0
     ! For every available nuclear charge Z from 1 to Z_max
     ! compute the Saha fraction n_e * n(I+1)/n(I) and store in S(I,Z)
     DO Z = 1,ATOMIC_DATA%MaxZ
        ! Z = 26   ! for testing
 
        ! For every available ionization stage I of Z
        DO I = 1,ATOMIC_DATA%List_I(Z)-1
 
           Upart1 = OBJ%Partition(I,Z) ! partition function of the ionization state I
           Upart2 = OBJ%Partition(I+1,Z) ! partition function of the ionization state I+1
           Xi = ATOMIC_DATA%Ion_energy(I,Z) ! ionization energy of the gorund level of the ionization state I
           Saha_fact = EXP( -OBJ%Bolz_norm*(Xi)/T ) ! Saha exponentia factor
           S(I,Z) = 2.0d0 *Upart2/Upart1 * T**1.5d0 * OBJ%Saha_norm * Saha_fact ! Saha fraction where S(I) = n_e * n(I+1)/n(I)
 
        END  DO! For every available ionization stage
     END DO ! For every  nucleus
 
     IF(ver) THEN
        PRINT*,'    - The S(I) = n_e * n(I+1)/n(I) of Fe:'
        DO I = 1,ATOMIC_DATA%List_I(26)
           PRINT*,'     > ',I,S(I,26)
        END DO
     END IF
 
     ! %%%%%% % for testing  % %%%%%%%%
     ! % psi = log(S/n_elect); % is equvalent to psi in fastwind
     ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
     ! initiate temprorary memory
     ALLOCATE(Fractions_to_nutral(ATOMIC_DATA%MaxI,ATOMIC_DATA%MaxZ)) ! ionisation fractions with respect to neutral Ion n_{I}/n_{1} of Z
     Fractions_to_nutral = 0.0d0
     ALLOCATE(Number_frac(ATOMIC_DATA%MaxI,ATOMIC_DATA%MaxZ)) ! output number fraction of ions in with respect to the total number
     Number_frac = 0.0d0
     !  Here we conpute the number density of each ionisation state for everiy spicies Z by solving:
     ! \[ n_{z,\,1}\left(1 + \sum\limits_{k=1}^{I_{max}} \prod\limits_{i=1}^{k}
     !  \frac{S_{z,\,i}}{n_e}\right) = N_{z,\,tot} \]
     !  For that we first compute the fractions on ion number dencity
     !  with respec to nutral ions as:
     !  \[f_{z,\, i} = \frac{ n_{z,\, i} }{ n_{z,\, 1} } \Rightarrow
     !  \prod\limits_{i=1}^{I} \frac{S_{z,\,i}}{n_e} \]
     !  Then find number density of neutral ions
     ! \[ n_{z,\,1} = \frac{ N_{z,\,tot} }{ \sum\limits_{k=1}^{I_{max}}f_{z,\,i} }\]
     !  And finally we compute number density for each ionisation state as:
     ! \[ n_{z,\,i} = n_{z,\,1} f_{z,\,i} \]
     !  see the notes for definition
 
     ! cycle through all the spicies
     DO Z = 1,ATOMIC_DATA%MaxZ
        !  Z = 26;
 
        Fractions_to_nutral(1,Z) = 1.0d0
 
        ! do the  sum over every ionisation state
        DO I = 2,ATOMIC_DATA%List_I(Z)
           Fractions_to_nutral(I,Z) = 1.0d0
 
           ! do the multiplication for all the
           DO k = 1,I-1
              Fractions_to_nutral(I,Z) = Fractions_to_nutral(I,Z)*S(k,Z)/OBJ%Electron
              ! here the Fractions_to_nutral(i,z) is equivalent of
              ! exp(prod(i)) form Fastwind and sum(Fractions_to_nutral(:,Z))
              ! = summ for each atom
           END DO ! do the multiplication
        END DO ! do the  sum
 
        Number_frac(:,Z) = Fractions_to_nutral(:,Z)/SUM(Fractions_to_nutral(:,Z))
 
        ! here Number_frac(:,Z) is equicalent to fkl(Z,I,?)  in fastwind
        OBJ%Ions(:,Z) = OBJ%Nuclei(Z)*Number_frac(:,Z)
 
     END DO ! cycle through all the spicies
 
     IF(ver) THEN
        PRINT*,'    - Ion number fractions of Fe relative to ground '
        DO I = 1,ATOMIC_DATA%List_I(26)
           PRINT*,'     > ',I,Fractions_to_nutral(I,26)
        END DO
     END IF
 
     IF(iterative) THEN
        ! here we compute the total numberdencity of electrons by computing number of electrons given by each ion
        OBJ%Electron = 0
        DO Z = 1,ATOMIC_DATA%MaxZ
           DO I = 1,ATOMIC_DATA%List_I(Z)
              OBJ%Electron = OBJ%Electron + OBJ%Ions(I,Z)*(I-1)
           END DO
        END DO
     END IF
     IF(ISNAN(OBJ%Electron)) STOP '---- Error With Electron number density (NAN) ----'
   END SUBROUTINE Comput_Ionisation_Fractions
   !=============================================================================
 
 
   SUBROUTINE Clear_OCCUPATION_NUMBER_TYPE(OBJ)
     CLASS(OCCUPATION_NUMBER_TYPE), INTENT(INOUT) :: OBJ
     CLASS(ATOMIC_DATA_TYPE),POINTER    :: ATOMIC_DATA
 
     ! create the assosiation(Short-hand) to use the previous link to atomic data
     ATOMIC_DATA => OBJ%ATOMIC_DATA
 
     DEALLOCATE(OBJ%Partition)
     DEALLOCATE(OBJ%Ions)
     DEALLOCATE(OBJ%Nuclei)
     DEALLOCATE(OBJ%Occupation)
 
     ALLOCATE(OBJ%Partition(ATOMIC_DATA%MaxI,ATOMIC_DATA%MaxZ))
     OBJ%Partition = 0.0d0
     ALLOCATE(OBJ%Ions(ATOMIC_DATA%MaxI,ATOMIC_DATA%MaxZ))
     OBJ%Ions = 0.0d0
     ALLOCATE(OBJ%Nuclei(ATOMIC_DATA%MaxZ))
     OBJ%Nuclei = 0.0d0
     ALLOCATE(OBJ%Occupation(ATOMIC_DATA%MaxL,ATOMIC_DATA%MaxI,ATOMIC_DATA%MaxZ))
     OBJ%Occupation = 0.0d0
 
     ! Number dencity of electrones
     OBJ%Electron = 0.0d0
 
     ! Temperature assigned to the type
     OBJ%T = -1.0d0
     ! Mass dencity assigned to the type
     OBJ%rho = -1.0d0
   END SUBROUTINE Clear_OCCUPATION_NUMBER_TYPE
   !=============================================================================
 
 
   SUBROUTINE Print_OCCUPATION_NUMBER_TYPE_info(OBJ)
     CLASS(OCCUPATION_NUMBER_TYPE), INTENT(IN) ::  OBJ
     LOGICAL :: Initialised = .FALSE.
     LOGICAL :: Set = .FALSE.
     PRINT*,' '
     PRINT*,'Information on OCCUPATION_NUMBER_TYPE requested'
 
     IF(.NOT.ALLOCATED(OBJ%Aboundance)) THEN
        PRINT*,'  - OCCUPATION_NUMBER_TYPE seems NOT Initialised '
        PRINT*,'  > For Initialisation CALL OCCUPATION_NUMBER_TYPE_NAME%Initialise()'
        PRINT*,'  > Printing ONLY General information'
     ELSE
        PRINT*,'  - OCCUPATION_NUMBER_TYPE IS Initialised '
        Initialised = .TRUE.
        IF(OBJ%rho.GT.0) THEN
           PRINT*,'  - OCCUPATION_NUMBER_TYPE Temperature and Density are Set'
           PRINT*,'  > Printing Full information '
           Set = .TRUE.
        ELSE
           PRINT*,'  - OCCUPATION_NUMBER_TYPE Temperature and Density are NOT Set'
           PRINT*,'  > For Set T and rho CALL OCCUPATION_NUMBER_TYPE_NAME%Set()'
           PRINT*,'  > Printing ONLY General information'
        END IF
     END IF
 
     PRINT*,''
     PRINT*,'  - Available variables:'
     PRINT*,'  > Aboundance(z) - Number aboundance of element Z'
     PRINT*,'  > Nuclei(Z) - Number density of elemet Z'
     PRINT*,'  > Ions(I,Z) - Number density of nucleis Z in ionisation stage I'
     PRINT*,'  > Occupation(L,I,Z) - Number density of Z I in excitation state L'
     PRINT*,'  > Electron - Number density of electrons (input see Set()/output)'
     PRINT*,'  > X_frac - Hydrogen mass fraction (input see Initialise()/output)'
     PRINT*,'  > Y_frac - Helium mass fraction (input see Initialise()/output)'
     PRINT*,'  > T - Input temperature in K (input)'
     PRINT*,'  > rho - Input mass dencity in CGS (input)'
     PRINT*,'  > Bolz_norm - h_plank*cc/kb (constant, can be manually overridden)'
     PRINT*,'  > Saha_norm - (2pi m_e k_B/h^2)^(3/2) (constant, can be manually overridden)'
     PRINT*,'  > mH - mass of Hydrogen (constant, can be manually overridden)'
     PRINT*,'  > Atomic_weight(Z) - Atomic weight of element Z (output)'
     PRINT*,''
     PRINT*,'  - Available PROCEDURE'
     PRINT*,'  > Initialise(ATOMIC_DATA,X_frac,Y_frac,verbose) - Initialise the occupation numbers '
     PRINT*,'    > ATOMIC_DATA - input atomic data'
     PRINT*,'    > X_frac - (OPTIONAL) input hydrogen mass fraction'
     PRINT*,'    > Y_frac - (OPTIONAL) input helium mass fraction'
     PRINT*,'    > verbose - (OPTIONAL) enable/desable output '
     PRINT*,'  > Set(rho,T,N_electron,verbose) - '
     PRINT*,'      Set temperature and density to initiate the computation of occupation numbers'
     PRINT*,'    > rho - input density '
     PRINT*,'    > T - input temperature '
     PRINT*,'    > N_electron - (OPTIONAL) input electrone number Density'
     PRINT*,'       If no value provided N_electron is computed internaly'
     PRINT*,'    > verbose - (OPTIONAL) enable/desable output '
     PRINT*,'  > Info() - give Information about the Type/object'
 
     IF(Initialised) THEN
        PRINT*,'  - Allocated data'
        PRINT*,'This PROCEDURE is under constraction'
     END IF
   END SUBROUTINE Print_OCCUPATION_NUMBER_TYPE_info
   !=============================================================================
 END MODULE LTE_Line_module
 
