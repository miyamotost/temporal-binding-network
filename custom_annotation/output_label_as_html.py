import pandas as pd
import matplotlib.pyplot as plt

f_con = open('./annotations/label.txt', 'r', encoding='UTF-8')
index = []
name = []
for line in f_con:
    line = line.replace('\n', '')
    line = line.split(', ')
    index.append(line[1])
    name.append(line[2])
f_con.close()

df = pd.DataFrame({
    'ラベル名': index,
    '動作': name
})

html_template = """
<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
  </head>
  <body>
    <script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
    <div class="container" style="margin: 30px;">
        {table}
    </div>
  </body>
</html>
"""

table = df.to_html(classes=["table", "table-bordered", "table-hover"])
html = html_template.format(table=table)

with open("./annotations/label.html", "w") as f:
    f.write(html)
