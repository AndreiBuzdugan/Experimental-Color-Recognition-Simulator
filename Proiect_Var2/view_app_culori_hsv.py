from flask import Flask, render_template_string
import pandas as pd

app = Flask(__name__)

csv_file_path = r'C:\Users\Andrei\Desktop\RN\Proiect\Proiect_Var2\date_antrenament.csv'

df = pd.read_csv(csv_file_path)

df['H'] = (df['H'] / 180) * 360
df['S'] = (df['S'] / 255) * 100
df['V'] = (df['V'] / 255) * 100


@app.route('/')
def index():
    html_table = df.to_html(classes='table table-striped', index=False)

    html_template = """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Culori HSV</title>
        <!-- Includeți stilurile Bootstrap pentru a formata tabelul frumos -->
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
      </head>
      <body>
        <div class="container mt-5">
          <h2>Graficul codurilor de culoare HSV transformate</h2>
          <!-- Afișează tabelul HTML generat din DataFrame -->
          {{ html_table | safe }}
        </div>
      </body>
    </html>
    """
    return render_template_string(html_template, html_table=html_table)

if __name__ == '__main__':
    app.run(debug=True)
