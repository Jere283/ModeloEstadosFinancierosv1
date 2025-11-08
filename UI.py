
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from matplotlib.backends.backend_pdf import PdfPages
import io

st.set_page_config(page_title="Simulación Financiera Restaurante", layout="wide")

# --- Parámetros base ---
insumos_base = {
    "Arroz": (900, 80),
    "Papa": (300, 250),
    "Carne": (6000, 100),
    "Otros": (2500, 75)
}
mano_obra = {
    "Cocinero": (1, 750000),
    "Ayudantes": (2, 310000),
}

personal = {
    "Meseros": (3, 150000),
    "Administrador": (1, 200000)
}

# --- Funciones ---
def calcular_estado(precio_almuerzo, almuerzos_diarios, dias, precio_carne,
                    servicio_por_almuerzo, impuesto_industria,
                    arrendamiento=850000, depreciacion=350000, factor_prestacional=0.52):

    ventas = precio_almuerzo * almuerzos_diarios * dias

    # Mano de obra
    mano_obra_base = sum([cant * sueldo for cant, sueldo in mano_obra.values()])
    mano_obra_totales = mano_obra_base * (1 + factor_prestacional)

    # Costo materia prima
    costo_mp = 0.0
    for nombre, (precio_kg, gramos) in insumos_base.items():
        if nombre == "Carne":
            precio_actual = precio_carne
        else:
            # En esta linea calculo el precio actual de cada mp usando distribucion uniforme
            # En donde le doy 2 opciones de fluctuacion entre el 90% y el 110 % +-10%
            precio_actual = np.random.uniform(precio_kg*0.9, precio_kg*1.1)
        costo_mp += (precio_actual * (gramos / 1000.0)) * almuerzos_diarios * dias

    servicios_publicos = servicio_por_almuerzo * almuerzos_diarios * dias

    costo_de_ventas = mano_obra_totales + costo_mp + servicios_publicos

    utilidad_bruta_en_ventas = ventas - costo_de_ventas

    gasto_en_personal = sum([cant * sueldo for cant,sueldo in personal.values()])

    comisiones_meseros = ventas * 0.01 * personal["Meseros"][0]
    comisiones_admin = ventas * 0.075 * personal["Administrador"][0]
    comisiones_totales = comisiones_admin + comisiones_meseros

    impuestos = ventas * impuesto_industria

    gastos = gasto_en_personal + comisiones_totales + arrendamiento + depreciacion

    utilidad_antes_de_impuesto = utilidad_bruta_en_ventas - gastos

    utilidad_despues_de_impuesto = utilidad_antes_de_impuesto - impuestos

    return {"ventas": ventas, "mano_de_obra":mano_obra_totales,
            "materia_prima": costo_mp, "costos_indirectos":servicios_publicos,
            "utilidad_bruta_en_ventas": utilidad_bruta_en_ventas, "costo_de_ventas": costo_de_ventas,
            "personal": gasto_en_personal,  "comisiones": comisiones_totales, "arrendamiento": arrendamiento,
            "depreciaciones": depreciacion, "gastos": gastos, "utilidad_bruta": utilidad_antes_de_impuesto,
            "utilidad_neta": utilidad_despues_de_impuesto
            }


def monte_carlo(N, dias_range, ventas_range, precio_range, carne_range):
    rows = []
    for _ in range(N):
        dias = np.random.uniform(*dias_range)
        alm = np.random.uniform(*ventas_range)
        precio_almuerzo = np.random.uniform(*precio_range)
        precio_carne = np.random.uniform(*carne_range)
        res = calcular_estado(precio_almuerzo, alm, dias, precio_carne, 75, 0.005)
        res.update({
            "dias": dias, "almuerzos": alm, "precio_almuerzo": precio_almuerzo, "precio_carne": precio_carne
        })
        rows.append(res)
    return pd.DataFrame(rows)


def guardar_db(df):
    conn = sqlite3.connect("simulaciones2.db")
    df.to_sql("resultados", conn, if_exists="append", index=False)
    conn.close()


def generar_pdf(df, media, std):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df["utilidad_neta"], bins=40, kde=True, ax=ax)
        ax.axvline(media, color="red", linestyle="--", label=f"Media ${media:,.0f}")
        ax.set_title("Distribución de utilidades simuladas")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.boxplot(x=df["utilidad_neta"], ax=ax)
        ax.set_title("Boxplot de utilidades")
        pdf.savefig(fig)
        plt.close(fig)
    buf.seek(0)
    return buf


# --- UI ---
st.title("🍽️ Simulación Financiera de Restaurante")

st.sidebar.header("Configuración de Simulación")
N = st.sidebar.number_input("Número de simulaciones", 100, 10000, 1000, step=100)
dias_range = st.sidebar.slider("Días trabajados", 22, 25, (22,25))
ventas_range = st.sidebar.slider("Almuerzos diarios", 80, 115, (80,115))
precio_range = st.sidebar.slider("Precio del almuerzo ($)", 3000, 4500, (3500,4000))
carne_range = st.sidebar.slider("Precio de la carne (kg)", 5000, 7000, (5500,6500))

if st.sidebar.button("Ejecutar simulación"):
    with st.spinner("Ejecutando simulaciones..."):
        df = monte_carlo(N, dias_range, ventas_range, precio_range, carne_range)
        media = df["utilidad_neta"].mean()
        std = df["utilidad_neta"].std()
        prob_perdida = (df["utilidad_neta"] < 0).mean() * 100

        guardar_db(df)

    st.success("✅ Simulación completada")

    col1, col2, col3 = st.columns(3)
    col1.metric("Media utilidad", f"${media:,.0f}")
    col2.metric("Desv. estándar", f"${std:,.0f}")
    col3.metric("Probabilidad de pérdida", f"{prob_perdida:.1f}%")

    # --- Gráficos ---
    st.subheader("Distribución de utilidades simuladas")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(df["utilidad_neta"], bins=40, kde=True, ax=ax)
    ax.axvline(media, color="red", linestyle="--", label=f"Media = ${media:,.0f}")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Boxplot de utilidades")
    fig, ax = plt.subplots(figsize=(10,2))
    sns.boxplot(x=df["utilidad_neta"], ax=ax)
    st.pyplot(fig)

    # --- Exportar ---
    st.subheader("📤 Exportar resultados")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", data=csv, file_name="resultados_simulacion.csv", mime="text/csv")

    pdf_buf = generar_pdf(df, media, std)
    st.download_button("Descargar reporte PDF", data=pdf_buf, file_name="reporte_simulacion.pdf", mime="application/pdf")

    # Mostrar resumen estadístico
    st.subheader("📊 Resumen estadístico")
    st.dataframe(df[["ventas","costo_de_ventas","gastos","utilidad_neta"]].describe().applymap(lambda x: f"{x:,.2f}"))
