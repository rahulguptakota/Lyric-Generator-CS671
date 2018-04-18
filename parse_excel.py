import re
import xlrd
book = xlrd.open_workbook('PERC_mendelly.xls')
sh = book.sheet_by_index(0)
nrows = sh.nrows
for i in range(nrows):
    print(sh.cell_value(rowx=i, colx=0))
    print()
