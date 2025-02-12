def Read_namelist(File,Listname, Namelist, Mask):
    #  READ_NAMELIST reads the Fortran formated namelist files
    #  File      - destination+name of the file to be read
    #  Listname  - Name of the list to be read 
    #  Namelist  - Dictionary containing Name of the namelist to be  read
    #  Mask      - List of format specifiers corresponding to Name-list
    #              F or f - Numbers
    #              S / s  - String
    #              L / l  - Logical

    def get_keys(dict):
        return [*dict]


    if len(Namelist) != len(Mask):
        raise Exception('Namelist and Mask must be same length')

    # Prapet names
    Listname = '&' + Listname
    List_of_names = get_keys(Namelist)
    number_of_names = len(Namelist)

    #  open the file
    try:
        fin = open(File)
    except IOError as Err:
        raise Exception(Err)

    for ind in range(number_of_names):

        str = '---'
        str2 = str.split()
        # Navigate to correct namelist
        while ( ( str2[0].find(Listname) == -1 ) and (str2[0].find('!') == -1) ):
            str = fin.readline()
            if str == '':
                # print('No list correponding to name ' +Listname+' found')
                raise Exception('No list correponding to name ' +Listname+' found')
            elif len(str.strip()) == 0:
                str = '---'
                str2 = str.split()
            else:
                str2 = str.split()

        str = '---'
        str2 = str.split()  
        while ( (str.find(List_of_names[ind]) == -1) and (str2[-1] != '/') ) or (str2[0].find('!') > -1) :
            str = fin.readline()
            if str == '':
                raise Exception('No enrty corresponding to name ' +List_of_names[ind]+' found' )
            elif len(str.strip()) == 0:
                str = '---'
                str2 = str.split()
            else:
                str2 = str.split()

        # print('looking for  ' + List_of_names[ind])
        # print('input string ' + str)
        # print('last chunk   ' + str2[-1])
        if  ( (str2[-1]=='/')  and not (str.find(List_of_names[ind])>-1) ):
            raise Exception('No enrty corresponding to name ' +List_of_names[ind]+' found' ) 
        elif (str2[0].find('!')>-1) :
            raise Exception('Error detected: reading-'+str2+'while looking - ' +List_of_names[ind] ) 
        else : 
            # locate the '/' and Remove it from the end 
            value_loc = str.find('/')
            value = str[:value_loc]
            value = value.strip()
            
            # locate the '=' signe and get attachet striped string
            value_loc = value.find('=') + 1
            value = value[value_loc:]
            value = value.strip()
                
            # locate the '!' signe and get attachet striped string
            value_loc = value.find('!')
            if value_loc != -1:
                value = value[:value_loc]
                value = value.strip()

            # print('parsed   -'+ value)
            # Store the value in Namelist
            if Mask[ind] == 'I' or Mask[ind] == 'i':
                Namelist[List_of_names[ind]] = int(value)

            if Mask[ind] == 'F' or Mask[ind] == 'f':
                value = value.replace('d', 'e')
                value = value.replace('D', 'e')
                Namelist[List_of_names[ind]] = float(value)

            if Mask[ind] == 'S' or Mask[ind] == 's':
                loc = value.find("'")
                value = value[loc:]
                value = value.strip("'")
                Namelist[List_of_names[ind]] = value

            if Mask[ind] == 'L' or Mask[ind] == 'l':
                if value == 'T' or value == 't':
                    Namelist[List_of_names[ind]] = True
                elif value == 'F' or value == 'f':
                    Namelist[List_of_names[ind]] = False
                else:
                    loc = value.find(".")
                    value = value[loc:]
                    value = value.strip(".")
                    if value == 'TRUE' or value == 'true':
                        Namelist[List_of_names[ind]] = True
                    else:
                        Namelist[List_of_names[ind]] = False

        fin.seek(0, 0)
        # print(Namelist[List_of_names[ind]])

    fin.close()

    return Namelist
