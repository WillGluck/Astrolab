import requests
import shutil
import csv

included_cols = [0, 5, 6, 8]
url = 'http://casjobs.sdss.org/ImgCutoutDR7/getjpeg.aspx?ra={ra}&dec={dec}&width=512&height=512&scale=0.5'
count = 2

with open('zoo2MainSpecz.csv', newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		content = None
		if spamreader.line_num > 2 :
			content = list(row[i] for i in included_cols)
				
		if content:
			formatted_url = url.format(ra=content[1], dec=content[2])			
			response = requests.get(formatted_url, stream=True)		
			if response.status_code == 200:
				with open(str(count) + '.jpg', 'wb') as f:
					response.raw.decode_content = True
					shutil.copyfileobj(response.raw, f)
					count += 1					
