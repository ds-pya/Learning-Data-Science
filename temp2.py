+ import base64
+ import io
+ import pandas as pd
+ from dash import ctx, dcc

...

layout = html.Div([
    ...
+   dcc.Store(id='stored-data'),  # ⬅️ 파싱된 데이터 저장소
    ...
])

+ @dash.callback(
+     Output('stored-data', 'data'),
+     Input('upload-data', 'contents'),
+     State('upload-data', 'filename'),
+ )
+ def parse_and_store(contents, filename):
+     if contents is None:
+         return None
+     content_type, content_string = contents.split(',')
+     decoded = base64.b64decode(content_string)
+     try:
+         if filename.endswith('.csv'):
+             df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
+         elif filename.endswith('.xlsx'):
+             df = pd.read_excel(io.BytesIO(decoded))
+         else:
+             return None
+         return df.to_dict('records')  # dict format for Store
+     except Exception as e:
+         return None