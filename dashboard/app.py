# -*- coding: utf-8 -*-
"""
¿Qué factores marcan la diferencia?
Tablero de decisión basado en inferencia estadística sobre el rendimiento académico.

Proyecto de Estadística Inferencial · Fundación Universitaria Compensar · 2026
Pablo Alberto Duque Marín · Geymer Duvan Useche Ruiz · Lorena Osorio Olaya

Datos: UCI Student Performance (Cortez & Silva, 2008)
https://archive.ics.uci.edu/dataset/320/student+performance
"""
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ──────────────────────────────────────────────────────────────────────
# Configuración e identidad visual
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="¿Qué factores marcan la diferencia?",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK, SUB = "#1f2933", "#5b6773"
BLUE, BLUE_D = "#3a6ea5", "#2b4f79"
CLAY = "#c1654f"
PAPER, MIST, LINE = "#fbfaf7", "#eef1f4", "#dfe3e8"
PALETTE = [BLUE, CLAY, "#6b8fb5", "#d9a679", "#4a7c7c", "#8c6d9e"]

st.markdown(f"""
<style>
.titulo {{ font-size: 2.05rem; font-weight: 600; line-height: 1.15; letter-spacing: -0.01em; margin: 0.2rem 0 0.3rem 0; max-width: 34ch; }}
.subtitulo {{ color: {SUB}; font-size: 1.02rem; max-width: 70ch; margin-bottom: 1.1rem; }}
.veredicto {{ border-left: 4px solid {CLAY}; background: #fff; border-radius: 0 0.4rem 0.4rem 0;
             padding: 0.9rem 1.2rem; margin: 0.6rem 0 1.2rem 0; font-size: 1.08rem; line-height: 1.5; max-width: 90ch; }}
.veredicto b {{ color: {CLAY}; font-weight: 600; }}
.paso {{ border-top: 1px solid {LINE}; padding-top: 0.6rem; margin-top: 1.1rem; }}
.paso h4 {{ margin: 0 0 0.35rem 0; font-size: 1.02rem; font-weight: 600; }}
.nota {{ color: {SUB}; font-size: 0.88rem; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────
ETIQUETAS = {
    "higher": "Aspira a educación superior", "address": "Zona de residencia", "sex": "Sexo",
    "schoolsup": "Apoyo educativo extra", "paid": "Clases particulares pagadas",
    "internet": "Acceso a internet en casa", "famsup": "Apoyo educativo familiar",
    "romantic": "Relación sentimental", "activities": "Actividades extracurriculares",
    "Medu": "Nivel educativo de la madre", "Fedu": "Nivel educativo del padre",
    "studytime": "Tiempo semanal de estudio", "Mjob": "Ocupación de la madre",
    "Fjob": "Ocupación del padre", "reason": "Motivo de elección del colegio",
    "failures": "Reprobaciones previas", "absences": "Ausencias", "age": "Edad",
    "Dalc": "Consumo de alcohol entre semana", "Walc": "Consumo de alcohol en fin de semana",
    "goout": "Salidas con amigos", "freetime": "Tiempo libre", "health": "Estado de salud",
    "traveltime": "Tiempo de desplazamiento", "famrel": "Calidad de la relación familiar",
}
NIVELES = {
    "Medu": {0: "Sin educación", 1: "Primaria", 2: "5º a 9º", 3: "Secundaria", 4: "Superior"},
    "Fedu": {0: "Sin educación", 1: "Primaria", 2: "5º a 9º", 3: "Secundaria", 4: "Superior"},
    "studytime": {1: "Menos de 2 h", 2: "2 a 5 h", 3: "5 a 10 h", 4: "Más de 10 h"},
    "higher": {"yes": "Sí", "no": "No"}, "address": {"U": "Urbana", "R": "Rural"},
    "sex": {"F": "Femenino", "M": "Masculino"}, "schoolsup": {"yes": "Sí", "no": "No"},
    "paid": {"yes": "Sí", "no": "No"}, "internet": {"yes": "Sí", "no": "No"},
    "famsup": {"yes": "Sí", "no": "No"}, "romantic": {"yes": "Sí", "no": "No"},
    "activities": {"yes": "Sí", "no": "No"},
}
def et(v): return ETIQUETAS.get(v, v)
def niv(var, x): return NIVELES.get(var, {}).get(x, str(x))
def f(x, d=2): return f"{x:.{d}f}".replace(".", ",")
def pf(p): return "< 0,001" if p < 0.001 else f"= {f(p, 3)}"

UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

@st.cache_data(show_spinner="Descargando el conjunto de datos de UCI…")
def _descargar_uci():
    """Trae student-por.csv del repositorio de UCI si no está junto a la app."""
    import io as _io, zipfile, ssl, urllib.request
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        z = zipfile.ZipFile(_io.BytesIO(r.read()))
    with z.open("student-por.csv") as fh:
        return pd.read_csv(fh, sep=";")

@st.cache_data
def cargar_datos(archivo=None):
    if archivo is not None:
        return pd.read_csv(archivo, sep=None, engine="python")
    local = Path(__file__).parent / "student-por.csv"
    if local.exists():
        return pd.read_csv(local, sep=";")
    return _descargar_uci()

def ic_media(s, alpha):
    n = len(s); m = s.mean(); se = s.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return m, m - t * se, m + t * se

def prueba_t(a, b, alpha):
    W, pl = stats.levene(a, b, center="median")
    iguales = pl >= alpha
    t, p = stats.ttest_ind(a, b, equal_var=iguales)
    na, nb = len(a), len(b)
    if iguales:
        sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
        se = np.sqrt(sp2 * (1 / na + 1 / nb)); gl = na + nb - 2
    else:
        va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
        se = np.sqrt(va + vb); gl = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    tc = stats.t.ppf(1 - alpha / 2, gl)
    diff = a.mean() - b.mean()
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return dict(W=W, pl=pl, iguales=iguales, t=t, p=p, gl=gl, tc=tc, diff=diff,
                ic=(diff - tc * se, diff + tc * se), d=diff / sp, rechaza=p < alpha)

def magnitud_d(d):
    d = abs(d)
    return "insignificante" if d < 0.2 else "pequeño" if d < 0.5 else "mediano" if d < 0.8 else "grande"

def magnitud_eta(e):
    return "pequeño" if e < 0.06 else "mediano" if e < 0.14 else "grande"

def letras_tukey(orden, difiere):
    """Letras compactas: grupos que comparten letra no difieren."""
    conjuntos = []
    for g in orden:
        colocado = False
        for c in conjuntos:
            if all(not difiere(g, h) for h in c):
                c.append(g); colocado = True
        if not colocado:
            conjuntos.append([g])
    abc = "abcdefgh"
    return {g: "".join(abc[i] for i, c in enumerate(conjuntos) if g in c) for g in orden}

def fig_medias(labels, medias, los, his, titulo, anot=None, letras=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=medias, mode="markers+lines",
        marker=dict(size=13, color=BLUE, line=dict(width=0)),
        line=dict(color=BLUE, width=1.5, dash="dot"),
        error_y=dict(type="data", symmetric=False, array=his, arrayminus=los,
                     color=BLUE_D, thickness=2, width=10),
        hovertemplate="%{x}<br>media = %{y:.2f}<extra></extra>", name=""))
    for i, (x, m) in enumerate(zip(labels, medias)):
        fig.add_annotation(x=x, y=m, text=f"{m:.2f}", xshift=34, showarrow=False,
                           font=dict(size=12, color=INK))
        if letras:
            fig.add_annotation(x=x, y=m + his[i], yshift=16, text=letras[i], showarrow=False,
                               font=dict(size=14, color=CLAY, family="IBM Plex Sans"))
    fig.update_layout(
        title=dict(text=titulo, x=0, font=dict(size=16, color=INK)),
        paper_bgcolor="white", plot_bgcolor="white", height=380,
        margin=dict(l=40, r=40, t=60, b=40), showlegend=False,
        yaxis=dict(title="Nota final media (G3)", gridcolor=LINE, zeroline=False),
        xaxis=dict(showgrid=False), font=dict(family="IBM Plex Sans", color=INK))
    if anot:
        fig.add_annotation(xref="paper", yref="paper", x=0, y=1.08, text=anot, showarrow=False,
                           font=dict(size=12, color=CLAY), xanchor="left")
    return fig

# ──────────────────────────────────────────────────────────────────────
# Sidebar: datos y parámetros
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Datos")
    fuente = st.radio("Conjunto de datos", ["Student Performance (UCI, n = 649)", "Subir mi propio archivo"],
                      label_visibility="collapsed")
    archivo = None
    if fuente.startswith("Subir"):
        archivo = st.file_uploader("Archivo CSV con la misma estructura (separador ; o ,)", type=["csv"])
        st.markdown('<p class="nota">Debe contener la columna G3 y los factores que quiera analizar.</p>',
                    unsafe_allow_html=True)
    try:
        df = cargar_datos(archivo)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}"); st.stop()
    if "G3" not in df.columns:
        st.error("El archivo no tiene la columna G3 (nota final)."); st.stop()

    st.markdown("### Nivel de significancia")
    alpha = st.select_slider("α", options=[0.01, 0.05, 0.10], value=0.05,
                             format_func=lambda x: f"{x:.2f}")

    st.markdown("### Factor de dos grupos")
    dicot = [c for c in df.columns if c != "G3" and df[c].nunique() == 2]
    _pref2 = ["higher", "address", "sex", "schoolsup", "paid", "internet", "famsup"]
    dicot = sorted(dicot, key=lambda c: (_pref2.index(c) if c in _pref2 else 99, c))
    fac2 = st.selectbox("Comparar la nota entre", dicot, index=0, format_func=et)

    st.markdown("### Factor de tres o más grupos")
    multi = [c for c in df.columns if c != "G3" and 3 <= df[c].nunique() <= 6]
    _pref3 = ["Medu", "Fedu", "studytime", "Mjob", "Fjob", "reason"]
    multi = sorted(multi, key=lambda c: (_pref3.index(c) if c in _pref3 else 99, c))
    fac3 = st.selectbox("Comparar la nota entre", multi, index=0, format_func=et, key="fac3")
    minimo = st.number_input("Tamaño mínimo por grupo (los menores se unen al vecino)", 5, 100, 20)

    st.markdown("### Predictores de la regresión")
    cand = [c for c in ["failures", "studytime", "absences", "age", "Dalc", "Walc", "goout", "freetime",
                        "health", "traveltime", "famrel", "Medu", "Fedu"] if c in df.columns]
    preds = st.multiselect("Variables numéricas y ordinales", cand,
                           default=[c for c in ["failures", "studytime", "absences", "Dalc", "Medu"] if c in cand],
                           format_func=et)
    st.markdown("---")
    st.markdown('<p class="nota">Estadística Inferencial · Ciencia de Datos<br>Fundación Universitaria Compensar, 2026<br>'
                'P. Duque · G. Useche · L. Osorio</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Cálculos centrales (se reutilizan en varias pestañas)
# ──────────────────────────────────────────────────────────────────────
lv2 = sorted(df[fac2].dropna().unique(), key=lambda x: str(x))
# grupo "positivo" primero cuando aplique (yes, U)
if set(map(str, lv2)) == {"yes", "no"}: lv2 = ["yes", "no"]
if set(map(str, lv2)) == {"U", "R"}: lv2 = ["U", "R"]
A = df.loc[df[fac2] == lv2[0], "G3"]; B = df.loc[df[fac2] == lv2[1], "G3"]
T = prueba_t(A, B, alpha)

# Agrupar niveles pequeños del factor multinivel
serie = df[fac3].copy()
conteo = serie.value_counts()
mapa_g = {}
if pd.api.types.is_numeric_dtype(serie):
    niveles = sorted(conteo.index)
    for n_ in niveles:
        mapa_g[n_] = n_
    # une hacia el vecino superior mientras el grupo sea pequeño
    for n_ in niveles:
        if conteo[n_] < minimo:
            vecinos = [m_ for m_ in niveles if m_ > n_ and conteo[m_] >= minimo]
            if vecinos: mapa_g[n_] = vecinos[0]
            else:
                menores = [m_ for m_ in niveles if m_ < n_]
                if menores: mapa_g[n_] = mapa_g[menores[-1]]
    serie_g = serie.map(mapa_g)
    def nombre_grupo(k):
        fuentes = sorted([n_ for n_, m_ in mapa_g.items() if m_ == k])
        if len(fuentes) == 1: return niv(fac3, k)
        return f"{niv(fac3, fuentes[0])} a {niv(fac3, fuentes[-1])}"
    grupos_k = sorted(serie_g.unique())
    etiquetas_g = {k: nombre_grupo(k) for k in grupos_k}
else:
    serie_g = serie
    grupos_k = sorted(serie_g.unique(), key=lambda x: df.loc[serie_g == x, "G3"].mean())
    etiquetas_g = {k: niv(fac3, k) for k in grupos_k}
grupos = [df.loc[serie_g == k, "G3"] for k in grupos_k]
W3, pl3 = stats.levene(*grupos, center="median")
F, pF = stats.f_oneway(*grupos)
gm = df["G3"].mean()
SCE = sum(len(g) * (g.mean() - gm) ** 2 for g in grupos)
SCD = sum(((g - g.mean()) ** 2).sum() for g in grupos)
k = len(grupos); N = len(df)
CME, CMD = SCE / (k - 1), SCD / (N - k)
Fc = stats.f.ppf(1 - alpha, k - 1, N - k)
eta2 = SCE / (SCE + SCD)
tk = pairwise_tukeyhsd(df["G3"], serie_g.map(etiquetas_g), alpha=alpha)
tk_df = pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0])
def difiere(g, h):
    r = tk_df[((tk_df.group1 == g) & (tk_df.group2 == h)) | ((tk_df.group1 == h) & (tk_df.group2 == g))]
    return bool(r.iloc[0]["reject"]) if len(r) else False
orden_medias = sorted([etiquetas_g[k_] for k_ in grupos_k], key=lambda e: df.loc[serie_g.map(etiquetas_g) == e, "G3"].mean())
letras = letras_tukey(orden_medias, difiere)

# Regresión múltiple (fase 3)
dfr = df.copy()
formula_terms = []
for c in dicot:
    if c in ("higher", "address"):
        dfr[c + "_b"] = (dfr[c] == lv2[0] if c == fac2 else dfr[c].isin(["yes", "U"])).astype(int)
        formula_terms.append(c + "_b")
formula_terms += [p_ for p_ in preds if pd.api.types.is_numeric_dtype(dfr[p_])]
modelo = None
if len(formula_terms) >= 1:
    modelo = smf.ols("G3 ~ " + " + ".join(formula_terms), data=dfr).fit()
    betas = {v: modelo.params[v] * dfr[v].std(ddof=1) / dfr["G3"].std(ddof=1) for v in formula_terms}

# Resultados canónicos del proyecto: siempre los tres factores declarados como
# preguntas de investigación, sin importar qué elija el usuario en la barra lateral.
CANON = {}
try:
    if {"higher", "address", "Medu"} <= set(df.columns):
        CANON["higher"] = prueba_t(df.loc[df["higher"] == "yes", "G3"], df.loc[df["higher"] == "no", "G3"], 0.05)
        CANON["address"] = prueba_t(df.loc[df["address"] == "U", "G3"], df.loc[df["address"] == "R", "G3"], 0.05)
        _mg = df["Medu"].replace({0: 1})
        _gs = [df.loc[_mg == kk, "G3"] for kk in sorted(_mg.unique())]
        _F, _p = stats.f_oneway(*_gs)
        _gm = df["G3"].mean()
        _SCE = sum(len(g) * (g.mean() - _gm) ** 2 for g in _gs)
        _SCD = sum(((g - g.mean()) ** 2).sum() for g in _gs)
        CANON["medu"] = dict(F=_F, p=_p, eta2=_SCE / (_SCE + _SCD), k=len(_gs), N=len(df),
                             dif=max(g.mean() for g in _gs) - min(g.mean() for g in _gs))
except Exception:
    CANON = {}

# ──────────────────────────────────────────────────────────────────────
# Encabezado y veredicto
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="titulo">¿Qué factores marcan la diferencia en el rendimiento académico?</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitulo">Un tablero para decidir con evidencia dónde intervenir. Cada resultado '
            'muestra la hipótesis, la prueba, la decisión y, sobre todo, el tamaño del efecto.</div>',
            unsafe_allow_html=True)

top = "el" if T["diff"] > 0 else "el segundo"
lider_reg = ""
if modelo is not None:
    v_lider = max(betas, key=lambda v: abs(betas[v]))
    lider_reg = f" Al considerar todos los factores a la vez, el de mayor peso es <b>{et(v_lider.replace('_b',''))}</b>."
st.markdown(
    f'<div class="veredicto">Con α = {f(alpha)}, la nota media {"<b>sí difiere</b>" if T["rechaza"] else "<b>no difiere</b>"} '
    f'según <b>{et(fac2).lower()}</b>: la brecha es de <b>{f(abs(T["diff"]))} puntos</b> sobre 20 '
    f'(efecto {magnitud_d(T["d"])}, d = {f(abs(T["d"]))}). '
    f'Según <b>{et(fac3).lower()}</b> la nota {"<b>sí difiere</b>" if pF < alpha else "<b>no difiere</b>"} entre grupos '
    f'(η² = {f(eta2, 3)}, efecto {magnitud_eta(eta2)}).{lider_reg}</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudiantes en la muestra", f"{len(df)}", border=True)
c2.metric("Nota final media (0 a 20)", f(df["G3"].mean()), border=True)
c3.metric(f"Brecha según {et(fac2).lower()}", f"{f(abs(T['diff']))} pts",
          delta=f"efecto {magnitud_d(T['d'])}", delta_color="off", border=True)
c4.metric(f"Variabilidad explicada por {et(fac3).lower()}", f"{f(eta2*100,1)} %",
          delta=f"efecto {magnitud_eta(eta2)}", delta_color="off", border=True)

st.markdown("")
tabs = st.tabs(["Problema y datos", "Descriptivo", "Fase 1 · Prueba t", "Fase 2 · ANOVA",
                "Fase 3 · Regresión", "Priorización", "Marco teórico", "Conclusiones"])

# ──────────────────────────────────────────────────────────────────────
# Pestaña 1: problema y datos
# ──────────────────────────────────────────────────────────────────────
with tabs[0]:
    izq, der = st.columns([1.3, 1])
    with izq:
        st.markdown("#### El problema")
        st.markdown(
            "Una institución educativa que quiera mejorar el rendimiento de sus estudiantes no puede actuar "
            "sobre todos los factores a la vez: necesita priorizar. Pero no se sabe con claridad cuáles de los "
            "factores personales, familiares y del entorno tienen un **efecto estadísticamente significativo** "
            "sobre la nota final, ni de qué tamaño es ese efecto. Sin esa evidencia, los programas de apoyo se "
            "diseñan sobre supuestos.")
        st.markdown("#### La solución")
        st.markdown(
            "Un procedimiento integrado en tres fases que clasifica cada factor según su naturaleza estadística "
            "y le aplica la prueba que le corresponde:\n\n"
            "1. **Prueba t para dos poblaciones** en los factores de dos grupos (aspiración, zona, sexo…).\n"
            "2. **Análisis de varianza y Tukey** en los factores de tres o más niveles (educación de los padres, tiempo de estudio…).\n"
            "3. **Correlación y regresión múltiple** para estimar el peso de todos los factores a la vez.\n\n"
            "El resultado no es una lista de valores p, sino una **jerarquía de factores ordenada por la magnitud de su efecto**, "
            "que es lo que permite decidir. Este tablero ejecuta las tres fases en vivo y deja cambiar los factores y el nivel "
            "de significancia.")
        st.markdown("#### Población objetivo")
        st.markdown("Estudiantes de educación secundaria de contextos comparables al de la muestra: sistema "
                    "público, 15 a 22 años, condiciones socioeconómicas similares. Los resultados establecen "
                    "asociación, no causalidad.")
    with der:
        st.markdown("#### Los datos")
        st.markdown(
            "Conjunto **Student Performance** (Cortez y Silva, 2008): 649 estudiantes de dos escuelas públicas "
            "portuguesas, 33 variables, sin valores faltantes. Acceso abierto en el repositorio de la Universidad "
            "de California, Irvine:  \n"
            "[archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)")
        dicc = pd.DataFrame([
            ("G3", "Nota final del curso", "Cuantitativa discreta", "Respuesta"),
            ("higher", "Aspira a educación superior", "Cualitativa nominal dicotómica", "Prueba t"),
            ("address", "Zona de residencia", "Cualitativa nominal dicotómica", "Prueba t"),
            ("Medu", "Nivel educativo de la madre", "Cualitativa ordinal (5 niveles)", "ANOVA"),
            ("studytime", "Tiempo semanal de estudio", "Cualitativa ordinal (4 niveles)", "ANOVA"),
            ("failures", "Reprobaciones previas", "Cuantitativa discreta", "Regresión"),
            ("Dalc", "Consumo de alcohol entre semana", "Cualitativa ordinal (5 niveles)", "Regresión"),
        ], columns=["Variable", "Descripción", "Naturaleza y nivel", "Técnica"])
        st.dataframe(dicc, hide_index=True, width="stretch")
        st.markdown('<p class="nota">Las variables Medu, Fedu, studytime y Dalc están codificadas con números, pero son '
                    'cualitativas ordinales: el código expresa orden, no cantidad. G3 se trata como cuasi-continua para la '
                    'inferencia sobre la media, con apoyo en n = 649 y el Teorema del Límite Central.</p>',
                    unsafe_allow_html=True)
        with st.expander("Ver las primeras filas del archivo"):
            st.dataframe(df.head(12), width="stretch")

# ──────────────────────────────────────────────────────────────────────
# Pestaña 2: descriptivo
# ──────────────────────────────────────────────────────────────────────
with tabs[1]:
    a1, a2 = st.columns([1.2, 1])
    with a1:
        fig = go.Figure(go.Histogram(x=df["G3"], xbins=dict(start=-0.5, end=20.5, size=1),
                                     marker=dict(color=BLUE, line=dict(color="white", width=1)),
                                     hovertemplate="G3 = %{x}<br>n = %{y}<extra></extra>"))
        fig.add_vline(x=df["G3"].mean(), line=dict(color=CLAY, width=2, dash="dash"))
        fig.add_annotation(x=df["G3"].mean(), y=1, yref="paper", text=f"media {f(df['G3'].mean(),1)}",
                           showarrow=False, xshift=40, font=dict(color=CLAY))
        fig.update_layout(title=dict(text="Distribución de la nota final", x=0, font=dict(size=16)),
                          paper_bgcolor="white", plot_bgcolor="white", height=360,
                          margin=dict(l=40, r=20, t=60, b=40), bargap=0.05,
                          xaxis=dict(title="G3", showgrid=False), yaxis=dict(title="Estudiantes", gridcolor=LINE),
                          font=dict(family="IBM Plex Sans", color=INK))
        st.plotly_chart(fig, width="stretch")
    with a2:
        s = df["G3"]
        desc = pd.DataFrame({
            "Estadístico": ["n", "Media", "Mediana", "Desviación estándar", "Mínimo", "Máximo", "Asimetría", "Curtosis (exceso)"],
            "Valor": [str(len(s)), f(s.mean()), f(s.median(), 1), f(s.std(ddof=1)), str(int(s.min())), str(int(s.max())),
                      f(stats.skew(s), 3), f(stats.kurtosis(s), 3)]})
        st.markdown("#### Medidas de resumen de G3")
        st.dataframe(desc, hide_index=True, width="stretch")
        st.markdown(f'<p class="nota">La asimetría negativa se debe a un grupo de estudiantes con nota muy baja. '
                    f'El tamaño de la muestra respalda el uso del Teorema del Límite Central para las pruebas sobre medias.</p>',
                    unsafe_allow_html=True)
    st.markdown("#### Distribución por grupos")
    b1, b2 = st.columns(2)
    for col, var, s_ in ((b1, fac2, df[fac2]), (b2, fac3, serie_g.map(etiquetas_g))):
        fig = go.Figure()
        cats = lv2 if var == fac2 else [etiquetas_g[k_] for k_ in grupos_k]
        for i, cat in enumerate(cats):
            fig.add_trace(go.Box(y=df.loc[s_ == cat, "G3"], name=niv(var, cat) if var == fac2 else cat,
                                 marker_color=PALETTE[i % len(PALETTE)], boxmean=True,
                                 line=dict(width=1.2), fillcolor=PALETTE[i % len(PALETTE)], opacity=0.85))
        fig.update_layout(title=dict(text=f"Nota final según {et(var).lower()}", x=0, font=dict(size=15)),
                          paper_bgcolor="white", plot_bgcolor="white", height=340, showlegend=False,
                          margin=dict(l=40, r=20, t=50, b=40), yaxis=dict(title="G3", gridcolor=LINE),
                          font=dict(family="IBM Plex Sans", color=INK))
        col.plotly_chart(fig, width="stretch")

# ──────────────────────────────────────────────────────────────────────
# Pestaña 3: prueba t
# ──────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown(f"### ¿La nota media difiere según {et(fac2).lower()}?")
    n1, n2 = niv(fac2, lv2[0]), niv(fac2, lv2[1])
    izq, der = st.columns([1, 1.1])
    with izq:
        st.markdown('<div class="paso"><h4>1. Hipótesis</h4></div>', unsafe_allow_html=True)
        st.latex(r"H_0:\ \mu_1 = \mu_2 \qquad H_1:\ \mu_1 \neq \mu_2")
        st.markdown(f"μ₁ = nota media poblacional del grupo **{n1}** (n = {len(A)}); μ₂ = del grupo **{n2}** (n = {len(B)}). "
                    f"Prueba de dos colas, α = {f(alpha)}.")
        st.markdown('<div class="paso"><h4>2. Homogeneidad de varianzas (Levene)</h4></div>', unsafe_allow_html=True)
        st.markdown(f"W = {f(T['W'],3)}, p {pf(T['pl'])} → "
                    + ("varianzas iguales: se usa la **t de Student** con varianza combinada."
                       if T["iguales"] else "varianzas distintas: se usa la **corrección de Welch**."))
        st.latex(r"t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}, \qquad "
                 r"s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}")
        st.markdown('<div class="paso"><h4>3. Decisión</h4></div>', unsafe_allow_html=True)
        st.markdown(f"t = **{f(T['t'],3)}** con {f(T['gl'],0)} gl; t crítico = ±{f(T['tc'],3)}; p **{pf(T['p'])}**.")
        if T["rechaza"]: st.badge("Se rechaza H₀: las medias difieren", icon=":material/check_circle:", color="red")
        else: st.badge("No se rechaza H₀: sin evidencia de diferencia", icon=":material/info:", color="blue")
        st.markdown('<div class="paso"><h4>4. Magnitud del efecto</h4></div>', unsafe_allow_html=True)
        st.latex(r"IC_{1-\alpha}:\ (\bar{x}_1-\bar{x}_2) \pm t_{\alpha/2,\,gl}\cdot EE, \qquad d = \frac{\bar{x}_1-\bar{x}_2}{s_p}")
        st.markdown(f"Diferencia = **{f(T['diff'])}** puntos (IC {int((1-alpha)*100)} %: {f(T['ic'][0])} a {f(T['ic'][1])}); "
                    f"d de Cohen = **{f(T['d'])}** → efecto **{magnitud_d(T['d'])}**.")
    with der:
        m1, l1, h1 = ic_media(A, alpha); m2, l2, h2 = ic_media(B, alpha)
        st.plotly_chart(fig_medias([n1, n2], [m1, m2], [m1 - l1, m2 - l2], [h1 - m1, h2 - m2],
                                   f"Media de G3 con IC {int((1-alpha)*100)} % por grupo",
                                   anot=f"t = {f(T['t'])}, p {pf(T['p'])}, d = {f(T['d'])}"),
                        width="stretch")
        # distribución t con región crítica
        x = np.linspace(-5, max(6, abs(T["t"]) + 1.5), 600); y = stats.t.pdf(x, T["gl"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=BLUE, width=2), name="t bajo H₀", hoverinfo="skip"))
        xr = x[(x <= -T["tc"]) | (x >= T["tc"])]
        fig.add_trace(go.Scatter(x=np.concatenate([[-T["tc"]], x[x <= -T["tc"]], [-T["tc"]]]),
                                 y=np.concatenate([[0], y[x <= -T["tc"]], [0]]), fill="toself",
                                 fillcolor="rgba(193,101,79,0.35)", line=dict(width=0), name="región de rechazo", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=np.concatenate([[T["tc"]], x[x >= T["tc"]], [T["tc"]]]),
                                 y=np.concatenate([[0], y[x >= T["tc"]], [0]]), fill="toself",
                                 fillcolor="rgba(193,101,79,0.35)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_vline(x=T["t"], line=dict(color=CLAY, width=2, dash="dash"))
        fig.add_annotation(x=T["t"], y=max(y) * 0.6, text=f"t observado = {f(T['t'])}", showarrow=False,
                           xshift=-6, xanchor="right", font=dict(color=CLAY))
        fig.update_layout(title=dict(text="Región de rechazo y estadístico observado", x=0, font=dict(size=15)),
                          paper_bgcolor="white", plot_bgcolor="white", height=280, showlegend=True,
                          legend=dict(orientation="h", y=-0.25), margin=dict(l=40, r=20, t=50, b=40),
                          xaxis=dict(title="t", showgrid=False), yaxis=dict(showticklabels=False, showgrid=False),
                          font=dict(family="IBM Plex Sans", color=INK))
        st.plotly_chart(fig, width="stretch")
    with st.expander("Código que produce este resultado"):
        st.code(f'''a = df.loc[df["{fac2}"]=="{lv2[0]}", "G3"]; b = df.loc[df["{fac2}"]=="{lv2[1]}", "G3"]
W, p_levene = stats.levene(a, b, center="median")          # W = {f(T['W'],3)}, p = {f(T['pl'],3)}
t, p = stats.ttest_ind(a, b, equal_var={T['iguales']})       # t = {f(T['t'],3)}, p = {T['p']:.2e}
t_crit = stats.t.ppf(1 - {alpha}/2, df={f(T['gl'],0)})      # {f(T['tc'],3)}''', language="python")

# ──────────────────────────────────────────────────────────────────────
# Pestaña 4: ANOVA
# ──────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown(f"### ¿La nota media difiere según {et(fac3).lower()}?")
    izq, der = st.columns([1, 1.1])
    with izq:
        st.markdown('<div class="paso"><h4>1. Hipótesis</h4></div>', unsafe_allow_html=True)
        st.latex(r"H_0:\ \mu_1 = \mu_2 = \cdots = \mu_k \qquad H_1:\ \exists\, i \neq j:\ \mu_i \neq \mu_j")
        st.markdown(f"k = {k} grupos: " + "; ".join(f"**{etiquetas_g[k_]}** (n = {len(g)})" for k_, g in zip(grupos_k, grupos)) + ".")
        if pd.api.types.is_numeric_dtype(serie) and any(mapa_g[n_] != n_ for n_ in mapa_g):
            st.markdown(f'<p class="nota">Los niveles con menos de {minimo} observaciones se unieron al nivel vecino para '
                        f'estimar medias y varianzas con precisión.</p>', unsafe_allow_html=True)
        st.markdown('<div class="paso"><h4>2. Supuestos</h4></div>', unsafe_allow_html=True)
        st.markdown(f"Levene: W = {f(W3,3)}, p {pf(pl3)} → " + ("varianzas homogéneas." if pl3 >= alpha
                    else "varianzas no homogéneas: interpretar con cautela (Welch-ANOVA sería la alternativa)."))
        st.markdown('<div class="paso"><h4>3. Estadístico F</h4></div>', unsafe_allow_html=True)
        st.latex(r"F = \frac{CME}{CMD} = \frac{SCE/(k-1)}{SCD/(N-k)}")
        tabla = pd.DataFrame({
            "Fuente": ["Entre grupos", "Dentro de grupos", "Total"],
            "Suma de cuadrados": [f(SCE), f(SCD), f(SCE + SCD)],
            "gl": [k - 1, N - k, N - 1],
            "Cuadrado medio": [f(CME, 3), f(CMD, 3), ""],
            "F": [f(F, 3), "", ""], "F crítico": [f(Fc, 3), "", ""], "p": [pf(pF).replace("= ", ""), "", ""]})
        st.dataframe(tabla, hide_index=True, width="stretch")
        if pF < alpha: st.badge("Se rechaza H₀: al menos un grupo difiere", icon=":material/check_circle:", color="red")
        else: st.badge("No se rechaza H₀", icon=":material/info:", color="blue")
        st.markdown('<div class="paso"><h4>4. Tamaño del efecto</h4></div>', unsafe_allow_html=True)
        st.latex(r"\eta^2 = \frac{SCE}{SCE + SCD}")
        st.markdown(f"η² = **{f(eta2,3)}**: el factor explica el {f(eta2*100,1)} % de la variabilidad → efecto **{magnitud_eta(eta2)}**.")
    with der:
        labs = [etiquetas_g[k_] for k_ in grupos_k]
        ms, los, his = [], [], []
        for g in grupos:
            m, lo, hi = ic_media(g, alpha); ms.append(m); los.append(m - lo); his.append(hi - m)
        st.plotly_chart(fig_medias(labs, ms, los, his, f"Media de G3 con IC {int((1-alpha)*100)} % por grupo",
                                   anot=f"F({k-1}, {N-k}) = {f(F)}, p {pf(pF)}, η² = {f(eta2,3)}",
                                   letras=[letras[l] for l in labs]), width="stretch")
        st.markdown('<p class="nota">Grupos que comparten una letra no difieren significativamente (Tukey HSD).</p>',
                    unsafe_allow_html=True)
        st.markdown("#### Comparaciones de Tukey")
        st.latex(r"HSD = q_{\alpha,\,k,\,N-k}\sqrt{\frac{CMD}{n_h}}")
        tk_show = tk_df.copy()
        tk_show = tk_show.rename(columns={"group1": "Grupo 1", "group2": "Grupo 2", "meandiff": "Diferencia",
                                          "p-adj": "p ajustado", "lower": "IC inferior", "upper": "IC superior", "reject": "¿Difieren?"})
        for c in ("Diferencia", "IC inferior", "IC superior"): tk_show[c] = tk_show[c].astype(float).map(lambda v: f(v))
        tk_show["p ajustado"] = tk_show["p ajustado"].astype(float).map(lambda v: pf(v).replace("= ", ""))
        tk_show["¿Difieren?"] = tk_show["¿Difieren?"].map(lambda v: "Sí" if str(v) == "True" else "No")
        st.dataframe(tk_show[["Grupo 1", "Grupo 2", "Diferencia", "IC inferior", "IC superior", "p ajustado", "¿Difieren?"]],
                     hide_index=True, width="stretch")
    with st.expander("Código que produce este resultado"):
        st.code(f'''grupos = [df.loc[df["{fac3}_g"]==k, "G3"] for k in {list(grupos_k)}]
W, p_levene = stats.levene(*grupos, center="median")      # W = {f(W3,3)}, p = {f(pl3,3)}
F, p = stats.f_oneway(*grupos)                             # F = {f(F,3)}, p = {pF:.2e}
tukey = pairwise_tukeyhsd(df["G3"], df["{fac3}_g"], alpha={alpha})''', language="python")

# ──────────────────────────────────────────────────────────────────────
# Pestaña 5: regresión (fase 3)
# ──────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### ¿Qué factor pesa más cuando se consideran todos a la vez?")
    st.markdown("Las fases 1 y 2 responden si un factor incide; esta fase estima cuánto pesa cada uno **controlando por los demás**, "
                "con un modelo de regresión lineal múltiple. Los coeficientes estandarizados (β) permiten comparar factores "
                "medidos en escalas distintas.")
    st.latex(r"G3_i = \beta_0 + \beta_1 x_{1i} + \cdots + \beta_p x_{pi} + \varepsilon_i")
    if modelo is None:
        st.warning("Selecciona al menos un predictor en la barra lateral.")
    else:
        izq, der = st.columns([1, 1.1])
        with izq:
            st.markdown("#### Correlación de cada predictor con la nota")
            corr_rows = []
            for v in formula_terms:
                r, p = stats.pearsonr(dfr[v], dfr["G3"])
                corr_rows.append((et(v.replace("_b", "")), r, p))
            corr_rows.sort(key=lambda t_: -abs(t_[1]))
            fig = go.Figure(go.Bar(x=[r for _, r, _ in corr_rows], y=[n for n, _, _ in corr_rows], orientation="h",
                                   marker=dict(color=[CLAY if r < 0 else BLUE for _, r, _ in corr_rows]),
                                   text=[f"{r:+.2f}" for _, r, _ in corr_rows], textposition="outside",
                                   hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>"))
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=60 + 42 * len(corr_rows),
                              margin=dict(l=10, r=40, t=10, b=30), xaxis=dict(title="r de Pearson", range=[-0.6, 0.6], gridcolor=LINE, zeroline=True),
                              yaxis=dict(autorange="reversed"), font=dict(family="IBM Plex Sans", color=INK))
            st.plotly_chart(fig, width="stretch")
            st.latex(r"H_0:\ \rho = 0 \qquad H_1:\ \rho \neq 0")
        with der:
            st.markdown("#### Modelo de regresión múltiple")
            filas = []
            for v in formula_terms:
                lo, hi = modelo.conf_int().loc[v]
                filas.append((et(v.replace("_b", "")), f(modelo.params[v], 3), f"[{f(lo)}; {f(hi)}]",
                              f(betas[v], 3), pf(modelo.pvalues[v]).replace("= ", ""),
                              "Sí" if modelo.pvalues[v] < alpha else "No"))
            filas.sort(key=lambda t_: -abs(float(t_[3].replace(",", "."))))
            st.dataframe(pd.DataFrame(filas, columns=["Predictor", "b", f"IC {int((1-alpha)*100)} %", "β estandarizado", "p", "Significativo"]),
                         hide_index=True, width="stretch")
            st.markdown(f"R² = **{f(modelo.rsquared,3)}** (ajustado {f(modelo.rsquared_adj,3)}); "
                        f"F = {f(modelo.fvalue)}, p {pf(modelo.f_pvalue)}. El modelo explica el "
                        f"**{f(modelo.rsquared*100,1)} %** de la variabilidad de la nota.")
            st.markdown('<p class="nota">Las notas de los periodos previos (G1, G2) se excluyen a propósito: miden el mismo '
                        'rendimiento en un momento anterior y no son factores contextuales.</p>', unsafe_allow_html=True)
        with st.expander("Código que produce este resultado"):
            st.code(f'''modelo = smf.ols("G3 ~ {' + '.join(formula_terms)}", data=df).fit()
beta = {{v: modelo.params[v] * df[v].std() / df["G3"].std() for v in predictores}}''', language="python")

# ──────────────────────────────────────────────────────────────────────
# Pestaña 6: priorización
# ──────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("### Jerarquía de factores: dónde intervenir primero")
    st.markdown("La salida del procedimiento integrado. Cada fila reúne la prueba aplicada, la decisión y la magnitud del efecto; "
                "el orden lo da la magnitud, no el valor p.")
    filas = [dict(Factor=et(fac2), Técnica="Prueba t", Estadístico=f"t = {f(T['t'])}", p=pf(T["p"]).replace("= ", ""),
                  Efecto=f"d = {f(abs(T['d']))}", Magnitud=magnitud_d(T["d"]), orden=abs(T["d"]) / 0.8,
                  Lectura=f"{f(abs(T['diff']))} puntos de brecha entre {niv(fac2, lv2[0]).lower()} y {niv(fac2, lv2[1]).lower()}")]
    filas.append(dict(Factor=et(fac3), Técnica="ANOVA", Estadístico=f"F = {f(F)}", p=pf(pF).replace("= ", ""),
                      Efecto=f"η² = {f(eta2,3)}", Magnitud=magnitud_eta(eta2), orden=eta2 / 0.14,
                      Lectura=f"hasta {f(max(ms) - min(ms))} puntos entre el grupo más bajo y el más alto"))
    if modelo is not None:
        for v in sorted(formula_terms, key=lambda v: -abs(betas[v])):
            nombre = et(v.replace("_b", ""))
            if nombre in (filas[0]["Factor"], filas[1]["Factor"]): continue
            filas.append(dict(Factor=nombre, Técnica="Regresión", Estadístico=f"b = {f(modelo.params[v])}",
                              p=pf(modelo.pvalues[v]).replace("= ", ""), Efecto=f"β = {f(betas[v])}",
                              Magnitud=("no significativo" if modelo.pvalues[v] >= alpha else magnitud_d(betas[v] * 2)),
                              orden=abs(betas[v]) / 0.4 if modelo.pvalues[v] < alpha else 0,
                              Lectura=f"{f(abs(modelo.params[v]))} puntos por cada unidad, controlando los demás factores"))
    filas.sort(key=lambda r: -r["orden"])
    for i, r in enumerate(filas, 1): r["Orden"] = i
    tabla = pd.DataFrame(filas)[["Orden", "Factor", "Técnica", "Estadístico", "p", "Efecto", "Magnitud", "Lectura"]]
    st.dataframe(tabla, hide_index=True, width="stretch")
    fig = go.Figure(go.Bar(y=[r["Factor"] for r in filas][::-1], x=[r["orden"] for r in filas][::-1], orientation="h",
                           marker=dict(color=[CLAY if i == 0 else BLUE for i in range(len(filas))][::-1]),
                           text=[r["Efecto"] for r in filas][::-1], textposition="outside", hoverinfo="skip"))
    fig.update_layout(title=dict(text="Magnitud relativa del efecto (escala comparable: 1 = efecto grande)", x=0, font=dict(size=15)),
                      paper_bgcolor="white", plot_bgcolor="white", height=80 + 40 * len(filas),
                      margin=dict(l=10, r=60, t=50, b=30), xaxis=dict(gridcolor=LINE, range=[0, max(1.5, max(r["orden"] for r in filas) * 1.25)]),
                      font=dict(family="IBM Plex Sans", color=INK))
    st.plotly_chart(fig, width="stretch")
    st.markdown("#### Qué implica para una institución")
    st.markdown(
        "- **Actuar primero sobre lo que pesa más y se puede modificar.** La aspiración a continuar estudios y las reprobaciones "
        "previas encabezan la jerarquía y admiten intervención directa: orientación vocacional, vinculación con la educación "
        "superior y acompañamiento académico temprano a quienes ya reprobaron.\n"
        "- **Usar los factores que no se pueden cambiar para focalizar.** El nivel educativo de la madre no se modifica, pero "
        "identifica a los estudiantes que más se benefician de apoyo adicional.\n"
        "- **No sobredimensionar lo que resulta pequeño.** La zona de residencia es significativa, pero su efecto es menor: la "
        "ubicación importa menos que el entorno familiar y escolar.")
    st.markdown('<p class="nota">Los resultados establecen asociación, no causalidad: el diseño es observacional. La inferencia se '
                'restringe a la población objetivo definida.</p>', unsafe_allow_html=True)
    buf = io.StringIO(); tabla.to_csv(buf, index=False, sep=";")
    st.download_button("Descargar la jerarquía (CSV)", buf.getvalue(), "jerarquia_factores.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────────
# Pestaña 7: marco teórico
# ──────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown("### La teoría que sostiene cada decisión")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Prueba de hipótesis.** Se contrasta una hipótesis nula, que niega el efecto, con una alternativa que lo afirma, "
                    "y se decide con un riesgo de error fijado de antemano: α es la probabilidad de rechazar una H₀ verdadera "
                    "(error de tipo I). H₀ nunca se demuestra; solo se rechaza o no (Llinás Solano, 2017; Walpole et al., 2011).")
        st.markdown("**Valor p y valor crítico.** El valor p es la probabilidad de un resultado al menos tan extremo como el observado "
                    "si H₀ fuera cierta; no es la probabilidad de que H₀ sea cierta (Casella y Berger, 2017). Comparar p con α o el "
                    "estadístico con el valor crítico lleva a la misma decisión.")
        st.markdown("**Teorema del Límite Central.** La distribución muestral de la media se aproxima a la normal cuando la muestra crece, "
                    "sin importar la forma de la población. Con grupos mayores de 30 observaciones, las pruebas sobre medias son válidas "
                    "aunque G3 sea asimétrica (Vladimirovna Panteleeva y Gutiérrez González, 2016).")
        st.markdown("**Prueba t para dos muestras independientes.** Compara dos medias con el error estándar de su diferencia; Levene decide "
                    "si se usa la varianza combinada (Student) o la corrección de Welch (Proaño Rivera, 2020).")
    with c2:
        st.markdown("**Análisis de varianza.** Con tres o más grupos, un único contraste global evita inflar el error de tipo I; descompone la "
                    "variabilidad total en la parte entre grupos y la parte dentro de grupos (Hernández Ripalda et al., 2019; Díaz Rodríguez, 2022).")
        st.markdown("**Comparaciones múltiples.** Tukey HSD localiza qué pares difieren controlando la tasa de error del conjunto.")
        st.markdown("**Regresión múltiple.** Estima el efecto de cada predictor manteniendo constantes los demás; los coeficientes "
                    "estandarizados permiten jerarquizar predictores en escalas distintas (Montgomery, 2017).")
        st.markdown("**Tamaño del efecto.** Significativo no es lo mismo que importante: d de Cohen y η² dicen cuánto, y son lo que permite "
                    "priorizar entre factores que resultan todos significativos.")
    st.markdown("---")
    st.markdown("**Fuentes.** Cortez, P., y Silva, A. (2008). *Using data mining to predict secondary school student performance*. EUROSIS. · "
                "Casella, G., y Berger, R. L. (2017). *Estadística. Probabilidad e inferencia estadística*. Cengage. · "
                "Díaz Rodríguez, M. (2022). *Estadística inferencial aplicada* (2a ed.). Universidad del Norte. · "
                "Hernández Ripalda, M. D., Tapia Esquivias, M., y Hernández González, S. (2019). *Estadística inferencial 2*. Patria. · "
                "Llinás Solano, H. (2017). *Estadística inferencial*. Universidad del Norte. · "
                "Montgomery, D. C. (2017). *Diseño y análisis de experimentos*. Limusa. · "
                "Proaño Rivera, W. B. (2020). *Estadística descriptiva e inferencial*. Universidad del Azuay. · "
                "Vladimirovna Panteleeva, O., y Gutiérrez González, E. (2016). *Estadística inferencial 1*. Patria. · "
                "Walpole, R. E., Myers, R. H., Myers, S. L., y Ye, K. (2011). *Probabilidad y estadística para ingeniería y ciencias*. Pearson.")

# ──────────────────────────────────────────────────────────────────────
# Pestaña 8: conclusiones
# ──────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.markdown("### Conclusiones")

    if CANON:
        h, a, m = CANON["higher"], CANON["address"], CANON["medu"]
        st.markdown("#### Respuesta a las preguntas de investigación")
        st.markdown("Las tres preguntas que el proyecto se planteó desde la etapa de contextualización, "
                    "contrastadas con α = 0,05.")
        st.markdown(
            f"**1. ¿La nota media difiere entre quienes aspiran a la educación superior y quienes no?**  \n"
            f"Sí. Quienes aspiran obtienen **{f(abs(h['diff']))} puntos más** sobre 20 "
            f"(IC 95 %: {f(h['ic'][0])} a {f(h['ic'][1])}; t = {f(h['t'])}, p {pf(h['p'])}), un efecto "
            f"**{magnitud_d(h['d'])}** (d = {f(abs(h['d']))}). Es la brecha más amplia del estudio.")
        st.markdown(
            f"**2. ¿La nota media difiere entre estudiantes de zona urbana y rural?**  \n"
            f"Sí, pero la brecha es mucho menor: **{f(abs(a['diff']))} puntos** a favor de los urbanos "
            f"(IC 95 %: {f(a['ic'][0])} a {f(a['ic'][1])}; t = {f(a['t'])}, p {pf(a['p'])}), un efecto "
            f"**{magnitud_d(a['d'])}** (d = {f(abs(a['d']))}). Significativa, pero de poca magnitud práctica.")
        st.markdown(
            f"**3. ¿La nota media difiere según el nivel educativo de la madre y, si es así, entre cuáles niveles?**  \n"
            f"Sí. El factor explica el **{f(m['eta2']*100,1)} %** de la variabilidad "
            f"(F({m['k']-1}, {m['N']-m['k']}) = {f(m['F'])}, p {pf(m['p'])}), un efecto "
            f"**{magnitud_eta(m['eta2'])}**. Las comparaciones de Tukey muestran que la diferencia no es un "
            f"escalón parejo: se concentra en los extremos, con una distancia de **{f(m['dif'])} puntos** "
            f"entre el grupo más bajo y el más alto, mientras los niveles intermedios no difieren entre sí.")
        if modelo is not None:
            v_top = max(betas, key=lambda v: abs(betas[v]))
            st.markdown(
                f"**Pregunta añadida en la etapa de transferencia: ¿cuál pesa más al considerarlos todos a la vez?**  \n"
                f"La regresión múltiple, que no formaba parte de las preguntas iniciales, aporta el criterio "
                f"que faltaba: **{et(v_top.replace('_b',''))}** encabeza la jerarquía (β = {f(betas[v_top],3)}) "
                f"y el modelo explica el **{f(modelo.rsquared*100,1)} %** de la variabilidad "
                f"(F = {f(modelo.fvalue)}, p {pf(modelo.f_pvalue)}).")
        st.divider()

    st.markdown("#### Conclusiones generales del proyecto")
    st.markdown(
        "**La respuesta al problema es una jerarquía, no una lista.** Los factores analizados resultan "
        "significativos, de modo que un análisis que se detuviera en el valor p los pondría a todos al mismo "
        "nivel. El tamaño del efecto es lo que los separa y los ordena, y esa ordenación es justamente lo "
        "que permite a una institución decidir dónde concentrar sus recursos.")
    st.markdown(
        "**Ninguna técnica sola bastaba.** La prueba t solo alcanza a los factores de dos grupos; el "
        "análisis de varianza, a los de tres o más; la regresión pondera todo a la vez pero no compara "
        "grupos. La articulación de las tres en un procedimiento escalonado, que clasifica cada factor "
        "según su naturaleza estadística y le aplica la prueba que le corresponde, es el aporte del trabajo.")
    st.markdown(
        "**El rigor está en los supuestos, no solo en el resultado.** Cada contraste verificó la "
        "homogeneidad de varianzas con Levene antes de elegir entre Student y Welch, y sostuvo la "
        "aproximación a la normalidad en el tamaño de los grupos y el Teorema del Límite Central. Un "
        "resultado sin esa verificación previa no sería defendible.")
    st.markdown(
        "**Del dato a la decisión.** El proyecto recorrió las tres etapas completas: describir los datos, "
        "contrastarlos formalmente y convertir los hallazgos en un instrumento que otros pueden usar con su "
        "propia información. Ese recorrido, y no cada prueba por separado, es lo que convierte el análisis "
        "en una herramienta de decisión.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Implicaciones prácticas")
        st.markdown(
            "- **Intervenir donde pesa y se puede cambiar.** Las reprobaciones previas y la aspiración a "
            "continuar estudios encabezan la jerarquía y admiten acción directa: acompañamiento académico "
            "temprano a quienes ya reprobaron y orientación vocacional para quienes no proyectan seguir "
            "estudiando.\n"
            "- **Usar lo que no se puede cambiar para focalizar.** El nivel educativo de la madre no se "
            "modifica, pero señala a los estudiantes que más se benefician de apoyo adicional.\n"
            "- **No sobredimensionar lo pequeño.** La zona de residencia es significativa, pero con un "
            "efecto pequeño: la ubicación importa menos que el entorno familiar y escolar.")
    with c2:
        st.markdown("#### Alcance y limitaciones")
        st.markdown(
            "- **Asociación, no causalidad.** El diseño es observacional: no permite afirmar que actuar "
            "sobre un factor cause una mejora. Que quienes aspiran a la universidad rindan más es "
            "compatible con varias explicaciones, incluida la inversa.\n"
            "- **Normalidad aproximada.** La nota final tiene asimetría negativa; la validez de las pruebas "
            "se apoya en el tamaño de los grupos y en la homogeneidad de varianzas verificada.\n"
            "- **Población acotada.** La inferencia se restringe a estudiantes de secundaria de contextos "
            "comparables al de la muestra, no a cualquier sistema educativo.")

    st.markdown("#### El paso siguiente")
    st.markdown(
        "La limitación central marca la recomendación. Si una institución implementa un programa sobre "
        "alguno de los factores que encabezan la jerarquía, conviene hacerlo como un **diseño completamente "
        "aleatorizado con grupo de control**, y evaluar su efecto con la misma prueba t o el mismo análisis "
        "de varianza que se aplican aquí. Este tablero dice **dónde vale la pena experimentar**; el "
        "experimento dirá si la intervención funciona.")

    with st.expander("Resultados de la configuración actual del tablero"):
        st.markdown(
            f"Con los factores y el nivel de significancia seleccionados en este momento "
            f"(α = {f(alpha)}): la nota media **{'sí' if T['rechaza'] else 'no'} difiere** según "
            f"{et(fac2).lower()} (t = {f(T['t'])}, p {pf(T['p'])}, d = {f(abs(T['d']))}), y "
            f"**{'sí' if pF < alpha else 'no'} difiere** según {et(fac3).lower()} "
            f"(F = {f(F)}, p {pf(pF)}, η² = {f(eta2,3)}). Estos valores cambian al modificar la "
            f"configuración; las conclusiones de arriba corresponden a los factores y al α que el proyecto "
            f"fijó desde su planteamiento.")

    st.divider()
    st.markdown('<p class="nota">Proyecto de Estadística Inferencial · Ciencia de Datos · Fundación '
                'Universitaria Compensar, 2026 · Pablo Alberto Duque Marín, Geymer Duvan Useche Ruiz y '
                'Lorena Osorio Olaya · Datos: Cortez y Silva (2008).</p>', unsafe_allow_html=True)
