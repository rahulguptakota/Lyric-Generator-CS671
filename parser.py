import re
import csv
with open('all.csv', 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	i = 0
	for row in spamreader:
		print(row[1])
		i+=1
		# print("-------------------------------------------------------")
		# if i > 5:
		# 	break