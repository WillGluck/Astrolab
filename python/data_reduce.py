import requests
import shutil
import csv
# import time

included_cols = [0, 5, 6, 8]
url = 'http://casjobs.sdss.org/ImgCutoutDR7/getjpeg.aspx?ra={ra}&dec={dec}&width=512&height=512&scale={scale}'
scale = 0.5

with open('zoo2MainSpecz.csv', newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		if spamreader.line_num - 1 >= 2 :
			if spamreader.line_num % 500 = 0:
				scale -= 0.05
			content = list(row[i] for i in included_cols)
			if content:
				formatted_url = url.format(ra=content[1], dec=content[2], scale=)
				while True:
					try:
						response = requests.get(formatted_url, stream=True, timeout=1)
						if response.status_code == 200:
							with open(str(spamreader.line_num - 1) + '.jpg', 'wb') as f:
								response.raw.decode_content = True
								shutil.copyfileobj(response.raw, f)
								break
						else:
							print("Retorno não foi sucesso, tentando novamente")
					except:
						print("Exceção ocorreu, tentando novamente")
