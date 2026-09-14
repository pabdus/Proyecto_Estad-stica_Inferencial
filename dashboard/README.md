# ¿Qué factores marcan la diferencia en el rendimiento académico?

Tablero interactivo de decisión basado en inferencia estadística.
Proyecto de Estadística Inferencial · Ciencia de Datos · Fundación Universitaria Compensar, 2026.

Pablo Alberto Duque Marín · Geymer Duvan Useche Ruiz · Lorena Osorio Olaya

## Qué hace
Ejecuta en vivo el procedimiento integrado del proyecto sobre el conjunto
*Student Performance* (Cortez y Silva, 2008; UCI, n = 649):

1. **Prueba t para dos poblaciones** (Levene → Student o Welch, IC, d de Cohen)
2. **ANOVA de un factor** (tabla ANOVA, η², Tukey HSD con letras)
3. **Correlación y regresión múltiple** (coeficientes estandarizados)
4. **Priorización**: jerarquía de factores ordenada por magnitud del efecto

El usuario puede cambiar los factores, el nivel de significancia y subir su propio CSV.

## Ejecutar en local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit Community Cloud (gratis)
1. Subir esta carpeta a un repositorio público de GitHub (app.py, requirements.txt,
   student-por.csv y la carpeta .streamlit/).
2. Entrar a https://share.streamlit.io con la cuenta de GitHub.
3. "New app" → elegir el repositorio, rama `main` y archivo `app.py` → "Deploy".
4. En dos o tres minutos la app queda en una URL pública del tipo
   `https://<usuario>-<repo>.streamlit.app`.

## Datos
Cortez, P., y Silva, A. (2008). Using data mining to predict secondary school student
performance. EUROSIS. https://archive.ics.uci.edu/dataset/320/student+performance
# Proyecto_Estad-stica_Inferencial
