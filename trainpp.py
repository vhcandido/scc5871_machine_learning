import csv
import json
import urllib2
import datetime

path = '../../data/'
csvFile = open(path+'train.csv','rb')
outFile = open('trainpp.csv','wt')

reader = csv.reader(csvFile)
writer = csv.writer(outFile)

header = []

def getGenderFromName(name):
	#data = json.load(urllib2.urlopen("https://gender-api.com/get?name=%s"%(name)))
	return "Unknown" #data["gender"][0].upper()+data["gender"][1:]

def strDateToSeconds(strDate):
	inputDate=datetime.datetime.strptime(strDate,"%Y-%m-%d %H:%M:%S")
	epoch = datetime.datetime.utcfromtimestamp(0)
	return (inputDate-epoch).total_seconds()

def strAgeToDays(age):
	age = age if len(age)>0 else "0 days"
	theAge = age.split(' ')
	return int(theAge[0])*(365 if theAge[1][0]=='y' else 30 if theAge[1][0]=='m' else 7 if theAge[1][0]=='w' else 1)

for row in reader:
	oldheader=row
	for i in range(len(row)+2):
		if row[i] =='Color':
			row[i]='ColorA'
			row.insert(i+1,'ColorB')
		#elif row[i]=='Breed':
		#	row[i]='BreedA'
		#	row.insert(i+1,'BreedB')
		elif row[i]=='SexuponOutcome':
			row[i]='Gender'
			row.insert(i+1,'IsIntact')
	header=row
	writer.writerow(row)
	break

for row in reader:
	r = row
	for i in range(len(header)):
		if header[i]=='DateTime':
			row[i]=strDateToSeconds(row[i])
		elif header[i]=='Gender':
			row[i]=row[i].split(' ')
			row.insert(i+1, "True" if row[i][0]=='Intact' else "Unknown" if row[i][0]=="Unknown" else "False")
			if len(row[i])>1:
				row[i]=row[i][1]
			else:
				if len(row[1]) > 0:
					row[i]=getGenderFromName(row[1])
				else:
					row[i]='Unknown'
		elif header[i]=='AgeuponOutcome':
			row[i] = strAgeToDays(row[i])
		elif header[i]=='Breed':
			try:
				row[i].index('Mix')
				row[i] = "Mix"
			except ValueError:
				row[i] = "Pure"
		elif header[i]=='ColorA':
			row[i]=row[i].split('/')

			for j in range(len(row[i])):
				row[i][j]=row[i][j].split(' ')[0]

			if len(row[i]) > 1:
				row.insert(i+1,row[i][1])
			else:
				row.insert(i+1,'Unknown')
			row[i]=row[i][0]
		"""elif header[i]=='BreedA' or header[i]=='ColorA':
			row[i]=row[i].split('/')

			if header[i]=='ColorA':
				for j in range(len(row[i])):
					row[i][j]=row[i][j].split(' ')[0]

			if len(row[i]) > 1:
				row.insert(i+1,row[i][1].split(' Mix')[0])
			else:
				row.insert(i+1,'Unknown')
			row[i]=row[i][0].split(' Mix')[0]"""
	writer.writerow(row)

csvFile.close()
outFile.close()

