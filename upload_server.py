import http.server, cgi, os, sys
PORT = 9999
DIR = os.path.expanduser('~/pyqt_intrusion/uploads')
os.makedirs(DIR, exist_ok=True)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(b'''<html><body><h2>Upload Video</h2>
<form method="POST" enctype="multipart/form-data">
<input type="file" name="file" accept="video/*"><br><br>
<input type="submit" value="Upload"></form></body></html>''')
    def do_POST(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST'})
        f = form['file']
        path = os.path.join(DIR, f.filename)
        with open(path, 'wb') as out: out.write(f.file.read())
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(f'<h3>OK: {f.filename}</h3><a href="/">Back</a>'.encode())

http.server.HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
