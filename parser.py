import re
import csv
with open('songdata.csv', 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	i = 0
	for row in spamreader:
		print(row[0], row[1], row[2], row[3][:100])
		i+=1
		print("-------------------------------------------------------")
		if i > 5:
			break