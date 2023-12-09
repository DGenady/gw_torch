import json
import urllib

dataset = 'o3a_16khz_r1'
gpsstart = 1238166018   # start of s5
gpsend   = 1253977218   # end of s5
detector = 'h1'
save_file = f'./{dataset}_{detector}_{gpsstart}_{gpsend}_files.txt'

urlformat = 'https://gwosc.org/archive/links/{0}/{1}/{2}/{3}/json/'
url = urlformat.format(dataset, detector, GPSstart, GPSend)
print("Tile catalog URL is ", url)

r = urllib.request.urlopen(url).read()    # get the list of files
files = json.loads(r)             # parse the json

output_list = []
for file in files['strain']:
    if file['format'] == 'hdf5':
        output_list.append(file['url'])
with open(save_file, "w") as f:
    for url in output_list:
        f.write(url+"\n")
