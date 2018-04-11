import re
import csv
with open('songdata.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	i = 0
	for row in spamreader:
		print(row[3])
		i+=1
		print("-------------------------------------------------------")
		if i > 5:
			break