import json, urllib

dataset = 'O3a_16KHZ_R1'
GPSstart = 1238166018   # start of S5
GPSend   = 1253977218   # end of S5
detector = 'H1'

urlformat = 'https://gwosc.org/archive/links/{0}/{1}/{2}/{3}/json/'
url = urlformat.format(dataset, detector, GPSstart, GPSend)
print("Tile catalog URL is ", url)

r = urllib.request.urlopen(url).read()    # get the list of files
tiles = json.loads(r)             # parse the json

output_list = []
for file in tiles['strain']:
    if file['format'] == 'hdf5':
        output_list.append(file['url'])

output_list