class DateRegex(object):
    
    @staticmethod
    def __get_day_attributes():
        # day of the month with optional suffix, such as 7th, 22nd
        dd = '(([0-2]?[0-9]|3[01])(\s*)([sS][tT]|[nN][dD]|[rR][dD]|[tT][hH])?)'
        # two-digit numeric day of the month
        DD = '(0[0-9]|[1-2][0-9]|3[01])'
        
        return dd, DD
    
    @staticmethod
    def __get_month_attributes():

        m = \
        '([jJ][aA][nN]([uU][aA][rR][yY])?|'+\
        '[fF][eE][bB]([rR][uU][aA][rR][yY])?|'+\
        '[mM][aA][rR]([cC][hH])?|'+\
        '[aA][pP][rR]([iI][lL])?|'+\
        '[mM][aA][yY]|'+\
        '[jJ][uU][nN]([eE])?|'+\
        '[jJ][uU][lL]([yY])?|'+\
        '[aA][uU][gG]([uU][sS][tT])?|'+\
        '[sS][eE][pP]([tT][eE][mM][bB][eE][rR])?|'+\
        '[oO][cC][tT]([oO][bB][eE][rR])?|'+\
        '[nN][oO][vV]([eE][mM][bB][eE][rR])?|'+\
        '[dD][eE][cC]([eE][mM][bB][eE][rR])?)'
        M = m

        # numeric month
        mm = '(0?[0-9]|1[0-2]|' + m + ')'

        # two digit month
        MM = '(0[0-9]|1[0-2]|' + m + ')'
        
        return m, M, mm, MM

    @staticmethod
    def __get_year_attributes():
        
        # two or four digit year
        y = '([0-9]{4}|[0-9]{2})'

        # two digit year
        yy = '([0-9]{2})'

        # four digit year
        YY = '([0-9]{4})'
        
        return y, yy, YY
    
    @staticmethod
    def __get_sep_attributes():

        date_sep = '[-./]'
        date_sep_optional = '[-./]*'
        date_sep_no_full = '[-/]'
        
        return date_sep, date_sep_optional, date_sep_no_full
    
    #def get_week_attributes():
    #    w = \
    #    '([mM][oO][nN]([dD][aA][yY])?|'+\
    #    '[tT][uU][eE]([sS][dD][aA][yY])?|'+\
    #    '[wW][eE][dD]([nN][eE][sS][dD][aA][yY])?|'+\
    #    '[tT][hH][uU][gG]([uU][sS][tT])?|'+\
    #    '[sS][eE][pP]([tT][eE][mM][bB][eE][rR])?|'+\
    #    '[oO][cC][tT]([oO][bB][eE][rR])?|'+\
    #    '[nN][oO][vV]([eE][mM][bB][eE][rR])?|'+\
    #    '[dD][eE][cC]([eE][mM][bB][eE][rR])?)'
    
    @staticmethod
    def get_infixes():
        
        dd, DD = DateRegex.__get_day_attributes()
        m, M, mm, MM = DateRegex.__get_month_attributes()
        y, yy, YY = DateRegex.__get_year_attributes()
        date_sep, date_sep_optional, date_sep_no_full = DateRegex.__get_sep_attributes()
        
        date_1 = y + '/' + mm + '/' + dd + '(?!([/]+|\d+))'
        date_2 = y + '/' + dd + '/' + mm + '(?!([/]+|\d+))'
        date_3 = dd + '/' + mm + '/' + y + '(?!([/]+|\d+))'
        date_4 = mm + '/' + dd + '/' + y + '(?!([/]+|\d+))'
        #Do I make this optional (date_sep_optional) - need to check
        date_5 = y + date_sep + m + date_sep + dd + '(?!\d)'
        date_6 = y + date_sep + dd + date_sep + m
        date_7 = dd + date_sep + m + date_sep + y 
        date_8 = m + date_sep + dd + date_sep + y
        date_9 =  y + date_sep + m
        date_10 = m + date_sep + y
        date_11 = dd + date_sep + m
        date_12 = m + date_sep + dd
        date_13 = '(?<!([/]|\d))' + y + '/' + dd + '(?!([/]+|\d+))'
        date_14 = '(?<!([/]|\d))' + y + '/' + dd + '(?!([/]+|\d+))'
        date_15 = '(?<!([/]|\d))' + dd + '/' + y + '(?!([/]+|\d+))'
        date_16 = '(?<!([/]|\d))' + mm + '/' + y + '(?!([/]+|\d+))'
        date_17 = '(?<!([/]|\d))' + dd + '/' + mm + '(?!([/]+|\d+))'
        date_18 = '(?<!([/]|\d))' + mm + '/' + dd + '(?!([/]+|\d+))'
        
        date_infixes = [date_1, date_2, date_3, date_4, date_5, date_6,\
                date_7, date_8, date_9, date_10, date_11, date_12,\
                date_13, date_14, date_15, date_16, date_17, date_18]
        
        return date_infixes