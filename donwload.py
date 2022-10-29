import wget
url = 'http://physionet.org/files/taichidb/1.0.2/Single-task/S0088_ST_V1.hea'
output = 'D:\Dataset'
filename = wget.download(url,out=output)
filename
#python -m wget -o D:\Dataset http://physionet.org/files/taichidb/1.0.2/Single-task/S0088_ST_V1.hea
# python -m wget -r -N -c -np http://physionet.org/files/taichidb/1.0.2/Single-task/
# python -m wget https://physionet.org/files/taichidb/1.0.2/Single-task/
# python -m wget -r -np -nH -R index.html http://physionet.org/files/taichidb/1.0.2/Single-task/
# python -m wget -r http://physionet.org/files/taichidb/1.0.2/Single-task/
#
# -OutFile "s1.hea"