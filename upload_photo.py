import http.server, cgi, os
PORT = 9999
DIR = os.path.expanduser('~/pyqt_intrusion/uploads')
os.makedirs(DIR, exist_ok=True)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.end_headers()
        self.wfile.write(b'<html><body><h2>Upload Photo</h2><form method=POST enctype=multipart/form-data><input type=file name=file accept=image/*><br><br><input type=submit value=Upload></form></body></html>')
    def do_POST(self):
        f=cgi.FieldStorage(fp=self.rfile,headers=self.headers,environ={'REQUEST_METHOD':'POST'})['file']
        p=os.path.join(DIR,f.filename)
        with open(p,'wb') as o: o.write(f.file.read())
        self.send_response(200);self.send_header('Content-Type','text/html;charset=utf-8');self.end_headers()
        self.wfile.write(f'<h3>OK: {f.filename}</h3>'.encode())
http.server.HTTPServer(('0.0.0.0',PORT),H).serve_forever()
