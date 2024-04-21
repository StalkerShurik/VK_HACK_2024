from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import json
import inference_classifier
import load_bert_model
import tagger

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        f = open('./page.html', mode='rb')
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(f.read())
        f.close()
        return

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        json_str = body.decode('utf-8')
        data = json.loads(json_str)
        print(data)

        text = data['text']
        topic = inference_classifier.classify([text])
        
        title = data['title']
        description = data['description']
        tags = [load_bert_model.predict_keywords(title, description, "", top_keywords=5)[0]]
        tagger_tags = tagger.get_tags(text)
        tags += tagger_tags

        result = json.dumps({ 'topic': topic, 'tags': ' '.join(tags) }, ensure_ascii=False)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        response = BytesIO()
        response.write(result.encode('utf-8'))
        self.wfile.write(response.getvalue())
        return


httpd = HTTPServer(('', 8000), SimpleHTTPRequestHandler)
httpd.serve_forever()
