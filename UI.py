import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from matplotlib.backends.backend_pdf import PdfPages
import io
from datetime import datetime

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


# --- Funciones Base de Datos ---
def inicializar_db():
    conn = sqlite3.connect("simulaciones_restaurante.db")
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS simulaciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TIMESTAMP,
        tipo_analisis TEXT,
        num_iteraciones INTEGER,
        dias_min INTEGER,
        dias_max INTEGER,
        almuerzos_min INTEGER,
        almuerzos_max INTEGER,
        precio_min REAL,
        precio_max REAL,
        carne_min REAL,
        carne_max REAL,
        utilidad_media REAL,
        utilidad_std REAL,
        prob_perdida REAL,
        descripcion TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS resultados (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        simulacion_id INTEGER,
        dias REAL,
        almuerzos REAL,
        precio_almuerzo REAL,
        precio_carne REAL,
        ventas REAL,
        mano_de_obra REAL,
        materia_prima REAL,
        costos_indirectos REAL,
        costo_de_ventas REAL,
        utilidad_bruta_en_ventas REAL,
        rentabilidad_bruta REAL,
        personal REAL,
        comisiones REAL,
        arrendamiento REAL,
        depreciaciones REAL,
        impuesto REAL,
        gastos REAL,
        utilidad_bruta REAL,
        FOREIGN KEY (simulacion_id) REFERENCES simulaciones(id)
    )''')

    conn.commit()
    conn.close()


def guardar_simulacion(df, params, estadisticas, descripcion="", tipo_analisis="Monte Carlo"):
    conn = sqlite3.connect("simulaciones_restaurante.db")
    c = conn.cursor()

    c.execute('''INSERT INTO simulaciones 
        (fecha, tipo_analisis, num_iteraciones, dias_min, dias_max, almuerzos_min, almuerzos_max,
         precio_min, precio_max, carne_min, carne_max, utilidad_media, utilidad_std,
         prob_perdida, descripcion)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now(), tipo_analisis, params['N'], params['dias'][0], params['dias'][1],
               params['ventas'][0], params['ventas'][1], params['precio'][0], params['precio'][1],
               params['carne'][0], params['carne'][1], estadisticas['media'],
               estadisticas['std'], estadisticas['prob_perdida'], descripcion))

    simulacion_id = c.lastrowid
    df['simulacion_id'] = simulacion_id
    df.to_sql("resultados", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    return simulacion_id


def cargar_simulaciones():
    conn = sqlite3.connect("simulaciones_restaurante.db")
    try:
        df = pd.read_sql("SELECT * FROM simulaciones ORDER BY fecha DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df


def cargar_resultados(simulacion_id):
    conn = sqlite3.connect("simulaciones_restaurante.db")
    df = pd.read_sql(f"SELECT * FROM resultados WHERE simulacion_id = {simulacion_id}", conn)
    conn.close()
    return df


# --- Funciones de Cálculo ---
def calcular_estado(precio_almuerzo, almuerzos_diarios, dias, precio_carne,
                    servicio_por_almuerzo, impuesto_industria,
                    arrendamiento=850000, depreciacion=350000, factor_prestacional=0.52,
                    factor_costo_ventas=1.0, factor_gastos=1.0, usar_random=True):
    ventas = precio_almuerzo * almuerzos_diarios * dias

    mano_obra_base = sum([cant * sueldo for cant, sueldo in mano_obra.values()])
    mano_obra_totales = mano_obra_base * (1 + factor_prestacional) * factor_costo_ventas

    costo_mp = 0.0
    for nombre, (precio_kg, gramos) in insumos_base.items():
        if nombre == "Carne":
            precio_actual = precio_carne
        else:
            if usar_random:
                precio_actual = np.random.uniform(precio_kg * 0.9, precio_kg * 1.1)
            else:
                precio_actual = precio_kg
        costo_mp += (precio_actual * (gramos / 1000.0)) * almuerzos_diarios * dias

    costo_mp = costo_mp * factor_costo_ventas

    servicios_publicos = servicio_por_almuerzo * almuerzos_diarios * dias * factor_costo_ventas
    costo_de_ventas = mano_obra_totales + costo_mp + servicios_publicos
    utilidad_bruta_en_ventas = ventas - costo_de_ventas
    rentabilidad_bruta = (utilidad_bruta_en_ventas / ventas * 100) if ventas > 0 else 0

    gasto_en_personal = sum([cant * sueldo for cant, sueldo in personal.values()]) * factor_gastos
    comisiones_meseros = ventas * 0.01 * personal["Meseros"][0] * factor_gastos
    comisiones_admin = ventas * 0.075 * personal["Administrador"][0] * factor_gastos
    comisiones_totales = comisiones_admin + comisiones_meseros
    impuestos = ventas * impuesto_industria * factor_gastos
    arrendamiento_ajustado = arrendamiento * factor_gastos
    depreciacion_ajustada = depreciacion * factor_gastos
    gastos = gasto_en_personal + comisiones_totales + arrendamiento_ajustado + depreciacion_ajustada + impuestos
    utilidad_antes_de_impuesto = utilidad_bruta_en_ventas - gastos

    return {
        "ventas": ventas, "mano_de_obra": mano_obra_totales,
        "materia_prima": costo_mp, "costos_indirectos": servicios_publicos,
        "utilidad_bruta_en_ventas": utilidad_bruta_en_ventas, "costo_de_ventas": costo_de_ventas,
        "rentabilidad_bruta": rentabilidad_bruta,
        "personal": gasto_en_personal, "comisiones": comisiones_totales,
        "arrendamiento": arrendamiento_ajustado, "depreciaciones": depreciacion_ajustada,
        "impuesto": impuestos,
        "gastos": gastos, "utilidad_bruta": utilidad_antes_de_impuesto
    }


def monte_carlo(N, dias_range, ventas_range, precio_range, carne_range):
    rows = []
    for _ in range(N):
        dias = np.random.uniform(*dias_range)
        alm = np.random.uniform(*ventas_range)
        precio_almuerzo = np.random.uniform(*precio_range)
        precio_carne = np.random.uniform(*carne_range)
        res = calcular_estado(precio_almuerzo, alm, dias, precio_carne, 75, 0.05)
        res.update({
            "dias": dias, "almuerzos": alm, "precio_almuerzo": precio_almuerzo,
            "precio_carne": precio_carne
        })
        rows.append(res)
    return pd.DataFrame(rows)


# --- PUNTO 1.1: Análisis de Sensibilidad Utilidad vs Costos y Gastos ---
def analisis_sensibilidad_costos_gastos(base_params):
    """
    Análisis bidimensional de sensibilidad:
    - Eje X: Variación en TODOS los gastos de administración y ventas
    - Eje Y: Variación en TODOS los costos de ventas
    """
    variacion = np.linspace(0.8, 1.2, 20)
    resultados = []

    precio_almuerzo = np.mean(base_params['precio'])
    almuerzos = np.mean(base_params['ventas'])
    dias = np.mean(base_params['dias'])
    precio_carne = np.mean(base_params['carne'])

    for var_costo_ventas in variacion:
        for var_gastos in variacion:
            res = calcular_estado(
                precio_almuerzo, almuerzos, dias,
                precio_carne, 75, 0.005,
                factor_costo_ventas=var_costo_ventas,
                factor_gastos=var_gastos,
                usar_random=False  # Sin aleatoriedad para análisis consistente
            )
            resultados.append({
                'var_costo': var_costo_ventas,
                'var_gasto': var_gastos,
                'utilidad_antes_impuesto': res['utilidad_bruta'],
                'costo_de_ventas': res['costo_de_ventas'],
                'gastos_totales': res['gastos']
            })

    return pd.DataFrame(resultados)


# --- PUNTO 2.1: Análisis de Sensibilidad Variables Principales ---
def analisis_sensibilidad_variables(base_params):
    """
    Análisis univariado de sensibilidad:
    Varía cada variable individualmente manteniendo las demás constantes (ceteris paribus)
    """
    n_puntos = 20
    resultados = []

    precio_base = np.mean(base_params['precio'])
    almuerzos_base = np.mean(base_params['ventas'])
    dias_base = np.mean(base_params['dias'])
    carne_base = np.mean(base_params['carne'])

    # Calcular valor base para referencias
    res_base = calcular_estado(precio_base, almuerzos_base, dias_base, carne_base, 
                               75, 0.005, usar_random=False)
    utilidad_base = res_base['utilidad_bruta']
    ventas_base = res_base['ventas']

    for var in ['precio', 'almuerzos', 'dias', 'carne']:
        if var == 'precio':
            valores = np.linspace(base_params['precio'][0], base_params['precio'][1], n_puntos)
            for val in valores:
                res = calcular_estado(val, almuerzos_base, dias_base, carne_base, 
                                    75, 0.005, usar_random=False)
                resultados.append({
                    'variable': 'Precio Almuerzo',
                    'valor': val,
                    'ventas': res['ventas'],
                    'utilidad': res['utilidad_bruta'],
                    'var_pct': (val - precio_base) / precio_base * 100,
                    'utilidad_cambio_pct': (res['utilidad_bruta'] - utilidad_base) / abs(utilidad_base) * 100
                })
        elif var == 'almuerzos':
            valores = np.linspace(base_params['ventas'][0], base_params['ventas'][1], n_puntos)
            for val in valores:
                res = calcular_estado(precio_base, val, dias_base, carne_base, 
                                    75, 0.005, usar_random=False)
                resultados.append({
                    'variable': 'Almuerzos Diarios',
                    'valor': val,
                    'ventas': res['ventas'],
                    'utilidad': res['utilidad_bruta'],
                    'var_pct': (val - almuerzos_base) / almuerzos_base * 100,
                    'utilidad_cambio_pct': (res['utilidad_bruta'] - utilidad_base) / abs(utilidad_base) * 100
                })
        elif var == 'dias':
            valores = np.linspace(base_params['dias'][0], base_params['dias'][1], n_puntos)
            for val in valores:
                res = calcular_estado(precio_base, almuerzos_base, val, carne_base, 
                                    75, 0.005, usar_random=False)
                resultados.append({
                    'variable': 'Días Trabajados',
                    'valor': val,
                    'ventas': res['ventas'],
                    'utilidad': res['utilidad_bruta'],
                    'var_pct': (val - dias_base) / dias_base * 100,
                    'utilidad_cambio_pct': (res['utilidad_bruta'] - utilidad_base) / abs(utilidad_base) * 100
                })
        elif var == 'carne':
            valores = np.linspace(base_params['carne'][0], base_params['carne'][1], n_puntos)
            for val in valores:
                res = calcular_estado(precio_base, almuerzos_base, dias_base, val, 
                                    75, 0.005, usar_random=False)
                resultados.append({
                    'variable': 'Precio Carne',
                    'valor': val,
                    'ventas': res['ventas'],
                    'utilidad': res['utilidad_bruta'],
                    'var_pct': (val - carne_base) / carne_base * 100,
                    'utilidad_cambio_pct': (res['utilidad_bruta'] - utilidad_base) / abs(utilidad_base) * 100
                })

    return pd.DataFrame(resultados)


# --- PUNTO 4.1: Cálculo de Precio para Rentabilidad 55% ---
def calcular_precio_rentabilidad_objetivo(almuerzos_range, dias, objetivo_rentabilidad=0.55):
    resultados = []

    for almuerzos in np.linspace(almuerzos_range[0], almuerzos_range[1], 10):
        precio_min, precio_max = 3000, 6000
        mejor_precio = None

        for _ in range(50):
            precio_prueba = (precio_min + precio_max) / 2
            res = calcular_estado(precio_prueba, almuerzos, dias, 6000, 75, 0.005)
            rentabilidad_actual = res['rentabilidad_bruta'] / 100

            if abs(rentabilidad_actual - objetivo_rentabilidad) < 0.001:
                mejor_precio = precio_prueba
                break
            elif rentabilidad_actual < objetivo_rentabilidad:
                precio_min = precio_prueba
            else:
                precio_max = precio_prueba

        if mejor_precio:
            resultados.append({
                'almuerzos': almuerzos,
                'precio_optimo': mejor_precio,
                'rentabilidad': res['rentabilidad_bruta']
            })

    return pd.DataFrame(resultados)


# --- PUNTO 5.1: Análisis de Escenarios de Crisis ---
def analisis_escenarios_crisis():
    escenarios = {
        'Base': {
            'precio_carne': 6000, 'impuesto': 0.005, 'servicios': 75,
            'precio_almuerzo': 3750, 'almuerzos': 97.5
        },
        'Crisis Moderada': {
            'precio_carne': 7200, 'impuesto': 0.008, 'servicios': 95,
            'precio_almuerzo': 4000, 'almuerzos': 90
        },
        'Crisis Severa': {
            'precio_carne': 8400, 'impuesto': 0.012, 'servicios': 120,
            'precio_almuerzo': 4500, 'almuerzos': 80
        },
        'Crisis + Aumento Precio': {
            'precio_carne': 8400, 'impuesto': 0.012, 'servicios': 120,
            'precio_almuerzo': 5000, 'almuerzos': 70
        }
    }

    resultados = []
    for nombre, params in escenarios.items():
        res = calcular_estado(
            params['precio_almuerzo'], params['almuerzos'], 23,
            params['precio_carne'], params['servicios'], params['impuesto']
        )
        res['escenario'] = nombre
        resultados.append(res)

    return pd.DataFrame(resultados)


def generar_pdf(df, media, std, simulacion_info):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["utilidad_bruta"], bins=40, kde=True, ax=ax)
        ax.axvline(media, color="red", linestyle="--", label=f"Media ${media:,.0f}")
        ax.set_title(f"Distribución de Utilidades - {simulacion_info}")
        ax.set_xlabel("Utilidad Neta ($)")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df["utilidad_bruta"], ax=ax)
        ax.set_title("Boxplot de Utilidades")
        ax.set_xlabel("Utilidad Neta ($)")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        corr_vars = ['precio_almuerzo', 'almuerzos', 'dias', 'precio_carne', 'utilidad_bruta']
        corr_matrix = df[corr_vars].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title("Matriz de Correlación")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        estado_promedio = f"""
        ESTADO DE RESULTADOS PROMEDIO
        {'=' * 50}

        VENTAS:                      ${df['ventas'].mean():,.0f}

        COSTO DE VENTAS:
          Mano de Obra:              ${df['mano_de_obra'].mean():,.0f}
          Materia Prima:             ${df['materia_prima'].mean():,.0f}
          Costos Indirectos:         ${df['costos_indirectos'].mean():,.0f}
        TOTAL COSTO VENTAS:          ${df['costo_de_ventas'].mean():,.0f}

        UTILIDAD BRUTA EN VENTAS:    ${df['utilidad_bruta_en_ventas'].mean():,.0f}
        Rentabilidad Bruta:          {df['rentabilidad_bruta'].mean():.1f}%

        GASTOS:
          Personal:                  ${df['personal'].mean():,.0f}
          Comisiones:                ${df['comisiones'].mean():,.0f}
          Arrendamiento:             ${df['arrendamiento'].mean():,.0f}
          Depreciación:              ${df['depreciaciones'].mean():,.0f}
          Impuesto:                  ${df['impuesto'].mean():,.0f}
        TOTAL GASTOS:                ${df['gastos'].mean():,.0f}

        UTILIDAD ANTES IMPUESTOS:    ${df['utilidad_bruta'].mean():,.0f}
        """
        ax.text(0.1, 0.9, estado_promedio, transform=ax.transAxes,
                fontfamily='monospace', fontsize=9, verticalalignment='top')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    buf.seek(0)
    return buf


# --- UI Principal ---
inicializar_db()

st.title("🍽️ Simulación Financiera de Restaurante")
st.markdown("**Proyección de Estado de Resultados con Análisis de Sensibilidad**")

tabs = st.tabs([
    "🎲 Monte Carlo",
    "📊 Sensibilidad 1.1",
    "📈 Variables 2.1",
    "💰 Precio Óptimo 4.1",
    "⚠️ Crisis 5.1",
    "📚 Historial",
    "🔄 Comparación"
])

# --- TAB 1: Monte Carlo ---
with tabs[0]:
    st.header("Simulación Monte Carlo")
    st.markdown("**Punto b) y g) - Modelo con análisis de correlaciones**")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parámetros")
        N = st.number_input("Iteraciones", 100, 10000, 1000, step=100)
        dias_range = st.slider("Días trabajados", 22, 25, (23, 23))
        ventas_range = st.slider("Almuerzos diarios", 80, 120, (100, 120))
        precio_range = st.slider("Precio almuerzo ($)", 3000, 5000, (3500, 4000))
        carne_range = st.slider("Precio carne (kg)", 5000, 8000, (5500, 6500))
        descripcion = st.text_area("Descripción", "")

        if st.button("🚀 Ejecutar Simulación", type="primary"):
            with st.spinner("Ejecutando..."):
                df = monte_carlo(N, dias_range, ventas_range, precio_range, carne_range)
                media = df["utilidad_bruta"].mean()
                std = df["utilidad_bruta"].std()
                prob_perdida = (df["utilidad_bruta"] < 0).mean() * 100

                params = {
                    'N': N, 'dias': dias_range, 'ventas': ventas_range,
                    'precio': precio_range, 'carne': carne_range
                }
                estadisticas = {'media': media, 'std': std, 'prob_perdida': prob_perdida}
                sim_id = guardar_simulacion(df, params, estadisticas, descripcion, "Monte Carlo")
                st.session_state['ultimo_df'] = df
                st.session_state['ultimo_sim_id'] = sim_id
                st.session_state['ultimas_stats'] = estadisticas

    with col2:
        if 'ultimo_df' in st.session_state:
            df = st.session_state['ultimo_df']
            stats = st.session_state['ultimas_stats']

            st.success(f"✅ Simulación #{st.session_state['ultimo_sim_id']} completada")

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Utilidad Media", f"${stats['media']:,.0f}")
            col_m2.metric("Desv. Estándar", f"${stats['std']:,.0f}")
            col_m3.metric("Prob. Pérdida", f"{stats['prob_perdida']:.1f}%")

            # Gráficos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            sns.histplot(df["utilidad_bruta"], bins=40, kde=True, ax=ax1)
            ax1.axvline(stats['media'], color="red", linestyle="--", label="Media")
            ax1.axvline(0, color="orange", linestyle=":", label="Break-even")
            ax1.set_xlabel("Utilidad Neta ($)")
            ax1.legend()

            # Análisis de correlación (PUNTO g)
            corr_vars = ['precio_almuerzo', 'almuerzos', 'dias', 'utilidad_bruta']
            corr_matrix = df[corr_vars].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax2,
                        vmin=-1, vmax=1, center=0)
            ax2.set_title("Correlaciones (Punto g)")

            st.pyplot(fig)

            # Mostrar correlaciones fuertes
            st.subheader("📊 Análisis de Correlación (Punto g)")
            corr_con_utilidad = corr_matrix['utilidad_bruta'].drop('utilidad_bruta').sort_values(ascending=False)
            for var, corr in corr_con_utilidad.items():
                tipo = "Positiva" if corr > 0 else "Negativa"
                fuerza = "Muy fuerte" if abs(corr) > 0.7 else "Fuerte" if abs(corr) > 0.5 else "Moderada"
                st.write(f"**{var}**: Correlación {tipo} {fuerza} ({corr:.3f})")

            # Exportar
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 CSV", csv, f"sim_{st.session_state['ultimo_sim_id']}.csv")
            with col_e2:
                pdf_buf = generar_pdf(df, stats['media'], stats['std'],
                                      f"Simulación #{st.session_state['ultimo_sim_id']}")
                st.download_button("📥 PDF", pdf_buf,
                                   f"reporte_{st.session_state['ultimo_sim_id']}.pdf")

# --- TAB 2: Análisis Sensibilidad Costos/Gastos (1.1) ---
with tabs[1]:
    st.header("1.1 - Análisis de Sensibilidad: Utilidad vs Costos y Gastos")
    st.markdown("""
        **Objetivo**: Analizar cómo varía la utilidad antes de impuestos cuando cambian:

        **Eje Y - Costo de Ventas (TOTAL)**:
        - Mano de obra (cocinero + ayudantes + prestaciones)
        - Materia prima (arroz, papa, carne, otros insumos)
        - Costos indirectos (servicios públicos)

        **Eje X - Gastos de Administración y Ventas (TOTAL)**:
        - Personal administrativo (meseros + administrador)
        - Comisiones (meseros + administrador)
        - Arrendamiento
        - Depreciación
        - Impuesto de industria y comercio

        *El análisis varía ambas categorías completas simultáneamente del 80% al 120%*
        """)

    if st.button("Ejecutar Análisis 1.1", key="btn_analisis_11"):
        base_params = {
            'precio': (3500, 4000),
            'ventas': (100, 120),
            'dias': (23, 23),
            'carne': (5500, 6500)
        }

        with st.spinner("Analizando..."):
            df_sens = analisis_sensibilidad_costos_gastos(base_params)

            # Mostrar valores base para referencia (encontrar el más cercano a 1.0, 1.0)
            df_sens['dist_base'] = np.sqrt((df_sens['var_costo'] - 1.0) ** 2 + (df_sens['var_gasto'] - 1.0) ** 2)
            valor_base = df_sens.loc[df_sens['dist_base'].idxmin()]

            col1, col2, col3 = st.columns(3)
            col1.metric("Escenario Base - Costo Ventas", f"${valor_base['costo_de_ventas']:,.0f}")
            col2.metric("Escenario Base - Gastos", f"${valor_base['gastos_totales']:,.0f}")
            col3.metric("Escenario Base - Utilidad", f"${valor_base['utilidad_antes_impuesto']:,.0f}")

            # Crear heatmap
            pivot = df_sens.pivot(index='var_costo', columns='var_gasto',
                                  values='utilidad_antes_impuesto')

            # Redondear los índices para mejor legibilidad
            pivot.index = pivot.index.round(2)
            pivot.columns = pivot.columns.round(2)

            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax,
                        center=0, cbar_kws={'label': 'Utilidad Antes de Impuestos ($)'},
                        annot_kws={'fontsize': 8})

            ax.set_xlabel('Factor de Variación - GASTOS de Adm. y Ventas (1.0 = 100% base)', fontsize=12)
            ax.set_ylabel('Factor de Variación - COSTO DE VENTAS (1.0 = 100% base)', fontsize=12)
            ax.set_title(
                'Análisis de Sensibilidad: Utilidad vs Costos y Gastos\n(Variación conjunta de categorías completas)',
                fontsize=14, fontweight='bold')

            # Formatear etiquetas de los ejes con 2 decimales
            ax.set_xticklabels([f'{float(label.get_text()):.2f}' for label in ax.get_xticklabels()], rotation=45,
                               ha='right')
            ax.set_yticklabels([f'{float(label.get_text()):.2f}' for label in ax.get_yticklabels()], rotation=0)

            # Marcar el punto base
            base_idx_y = len(pivot) // 2
            base_idx_x = len(pivot.columns) // 2
            ax.add_patch(plt.Rectangle((base_idx_x, base_idx_y), 1, 1,
                                       fill=False, edgecolor='blue', lw=3))

            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("📊 Interpretación del Análisis")
            st.markdown("""
                **Lectura del Mapa de Calor:**

                - **Cuadro azul**: Escenario base (100% de costos y gastos)
                - **Eje Vertical (Costos de Ventas)**: Variación del 80% al 120%
                  - 0.8 = Reducción del 20% en TODOS los costos de ventas
                  - 1.2 = Aumento del 20% en TODOS los costos de ventas

                - **Eje Horizontal (Gastos)**: Variación del 80% al 120%
                  - 0.8 = Reducción del 20% en TODOS los gastos de adm. y ventas
                  - 1.2 = Aumento del 20% en TODOS los gastos de adm. y ventas

                **Hallazgos Clave:**

                - **Zona Verde**: Escenarios con utilidad positiva (rentables)
                - **Zona Roja**: Escenarios con pérdidas (utilidad negativa)
                - **Diagonal superior izquierda**: Mejor escenario (bajos costos, bajos gastos)
                - **Diagonal inferior derecha**: Peor escenario (altos costos, altos gastos)

                **Sensibilidad Relativa:**
                - Observe cómo los cambios VERTICALES (costos) afectan más que los HORIZONTALES (gastos)
                - Esto indica qué categoría tiene mayor impacto en la rentabilidad
                """)

            # Tabla de escenarios clave
            st.subheader("📋 Escenarios Clave")
            escenarios_clave = [
                ("Mejor Caso", 0.8, 0.8),
                ("Base", 1.0, 1.0),
                ("Peor Caso", 1.2, 1.2),
                ("Solo ↑ Costos +20%", 1.2, 1.0),
                ("Solo ↑ Gastos +20%", 1.0, 1.2),
                ("Solo ↓ Costos -20%", 0.8, 1.0),
                ("Solo ↓ Gastos -20%", 1.0, 0.8)
            ]

            tabla_escenarios = []
            for nombre, var_c, var_g in escenarios_clave:
                # Buscar el punto más cercano al escenario deseado
                df_sens['dist_temp'] = np.sqrt(
                    (df_sens['var_costo'] - var_c) ** 2 + (df_sens['var_gasto'] - var_g) ** 2)
                fila = df_sens.loc[df_sens['dist_temp'].idxmin()]

                tabla_escenarios.append({
                    'Escenario': nombre,
                    'Factor Costos': f"{fila['var_costo']:.2f}x",
                    'Factor Gastos': f"{fila['var_gasto']:.2f}x",
                    'Costo Ventas': f"${fila['costo_de_ventas']:,.0f}",
                    'Gastos': f"${fila['gastos_totales']:,.0f}",
                    'Utilidad': f"${fila['utilidad_antes_impuesto']:,.0f}",
                    'vs Base': f"{((fila['utilidad_antes_impuesto'] - valor_base['utilidad_antes_impuesto']) / abs(valor_base['utilidad_antes_impuesto']) * 100):+.1f}%"
                })

            df_tabla = pd.DataFrame(tabla_escenarios)
            st.dataframe(df_tabla, use_container_width=True, hide_index=True)

# --- TAB 3: Análisis Sensibilidad Variables (2.1) ---
with tabs[2]:
    st.header("2.1 - Análisis de Sensibilidad: Variables Principales")
    st.markdown("""
        **Objetivo**: Analizar el impacto INDIVIDUAL de cada variable en ventas y utilidad.
        
        **Método**: Análisis univariado (ceteris paribus) - cada variable se varía mientras las demás permanecen constantes.
        """)

    if st.button("Ejecutar Análisis 2.1", key="btn_analisis_21"):
        base_params = {
            'precio': (3500, 4000),
            'ventas': (100, 120),
            'dias': (22, 25),
            'carne': (5500, 6500)
        }

        with st.spinner("Analizando variables..."):
            df_vars = analisis_sensibilidad_variables(base_params)

            # Crear 3 gráficos: Utilidad absoluta, Utilidad % y Sensibilidad normalizada
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

            # Gráfico 1: Impacto en Utilidad (valores absolutos)
            for var in df_vars['variable'].unique():
                data = df_vars[df_vars['variable'] == var]
                ax1.plot(data['valor'], data['utilidad'], marker='o', label=var, linewidth=2)
            ax1.set_xlabel('Valor de Variable', fontsize=11)
            ax1.set_ylabel('Utilidad Antes de Impuestos ($)', fontsize=11)
            ax1.set_title('Impacto en Utilidad (Valores Absolutos)', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(0, color='red', linestyle='--', alpha=0.5, label='Break-even')
            
            # Formatear eje Y
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if abs(x) >= 1e6 else f'${x/1e3:.0f}K'))

            # Gráfico 2: Cambio porcentual en utilidad
            for var in df_vars['variable'].unique():
                data = df_vars[df_vars['variable'] == var]
                ax2.plot(data['var_pct'], data['utilidad_cambio_pct'], marker='o', label=var, linewidth=2)
            ax2.set_xlabel('Variación de la Variable (%)', fontsize=11)
            ax2.set_ylabel('Cambio en Utilidad (%)', fontsize=11)
            ax2.set_title('Sensibilidad Relativa', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)

            # Gráfico 3: Solo variables que afectan VENTAS
            variables_ventas = ['Precio Almuerzo', 'Almuerzos Diarios', 'Días Trabajados']
            for var in variables_ventas:
                if var in df_vars['variable'].values:
                    data = df_vars[df_vars['variable'] == var]
                    ax3.plot(data['valor'], data['ventas'], marker='o', label=var, linewidth=2)
            ax3.set_xlabel('Valor de Variable', fontsize=11)
            ax3.set_ylabel('Ventas ($)', fontsize=11)
            ax3.set_title('Impacto en Ventas\n(Precio Carne no afecta ventas)', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M' if abs(x) >= 1e6 else f'${x/1e3:.0f}K'))

            plt.tight_layout()
            st.pyplot(fig)

            # Tabla de análisis de sensibilidad
            st.subheader("📊 Tabla de Sensibilidad Comparativa")
            
            # Calcular sensibilidad (elasticidad) para cada variable
            sensibilidad_data = []
            for var in df_vars['variable'].unique():
                data = df_vars[df_vars['variable'] == var].sort_values('var_pct')
                if len(data) >= 2:
                    # Tomar extremos
                    minimo = data.iloc[0]
                    maximo = data.iloc[-1]
                    
                    cambio_var_pct = maximo['var_pct'] - minimo['var_pct']
                    cambio_util_pct = maximo['utilidad_cambio_pct'] - minimo['utilidad_cambio_pct']
                    
                    # Elasticidad = (% cambio utilidad) / (% cambio variable)
                    elasticidad = cambio_util_pct / cambio_var_pct if cambio_var_pct != 0 else 0
                    
                    sensibilidad_data.append({
                        'Variable': var,
                        'Rango': f"{minimo['valor']:.0f} - {maximo['valor']:.0f}",
                        'Variación (%)': f"{cambio_var_pct:+.1f}%",
                        'Impacto en Utilidad': f"{cambio_util_pct:+.1f}%",
                        'Elasticidad': f"{elasticidad:.2f}",
                        'Interpretación': 'Alta' if abs(elasticidad) > 2 else 'Media' if abs(elasticidad) > 1 else 'Baja'
                    })
            
            df_sensibilidad = pd.DataFrame(sensibilidad_data)
            df_sensibilidad = df_sensibilidad.sort_values('Elasticidad', key=lambda x: x.abs(), ascending=False)
            st.dataframe(df_sensibilidad, use_container_width=True, hide_index=True)
            
            st.info("""
                **Interpretación de Elasticidad:**
                - **> 2.0**: Muy sensible - Pequeños cambios causan grandes impactos
                - **1.0 - 2.0**: Sensibilidad media - Cambio proporcional
                - **< 1.0**: Poco sensible - Cambios grandes tienen impacto moderado
                
                **Elasticidad positiva** = Variable y utilidad se mueven en la misma dirección  
                **Elasticidad negativa** = Variable y utilidad se mueven en direcciones opuestas
                """)

            st.subheader("📝 Punto 3.1 - Diferencia Conceptual")
            st.info("""
                **Diferencias clave entre análisis 1.1 y 2.1**:

                **Análisis 1.1 (Bidimensional)**:
                - Examina la interacción CONJUNTA de costos y gastos
                - Muestra cómo dos factores interactúan simultáneamente
                - Útil para identificar combinaciones críticas de factores
                - Responde: "¿Qué pasa si ambos factores cambian al mismo tiempo?"

                **Análisis 2.1 (Univariado)**:
                - Examina el efecto INDIVIDUAL de cada variable (ceteris paribus)
                - Mantiene las demás variables constantes
                - Útil para identificar qué variable tiene mayor impacto individual
                - Responde: "¿Cuál es la sensibilidad a cada factor por separado?"

                **Aplicación Práctica**:
                - Use 1.1 para planificación de escenarios complejos
                - Use 2.1 para priorizar qué variables controlar primero
                """)

# --- TAB 4: Precio Óptimo (4.1) ---
with tabs[3]:
    st.header("4.1 - Cálculo de Precio para Rentabilidad del 55%")
    st.markdown("""
        **Escenario**: Ventas estabilizadas entre 100-120 almuerzos diarios, 23 días al mes.

        **Objetivo**: Determinar el precio del almuerzo necesario para alcanzar rentabilidad bruta del 55%.
        """)

    if st.button("Calcular Precio Óptimo"):
        with st.spinner("Calculando precios óptimos..."):
            df_precio = calcular_precio_rentabilidad_objetivo((100, 120), 23, 0.55)

            if len(df_precio) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df_precio['almuerzos'], df_precio['precio_optimo'],
                        marker='o', linewidth=2, markersize=8, color='green')
                ax.set_xlabel('Almuerzos Diarios')
                ax.set_ylabel('Precio Almuerzo ($)')
                ax.set_title('Precio Necesario para Rentabilidad Bruta del 55%')
                ax.grid(True, alpha=0.3)

                # Anotar puntos clave
                for idx in [0, len(df_precio) // 2, -1]:
                    row = df_precio.iloc[idx]
                    ax.annotate(f"${row['precio_optimo']:.0f}",
                                xy=(row['almuerzos'], row['precio_optimo']),
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

                st.pyplot(fig)

                st.subheader("Tabla de Precios Óptimos")
                df_display = df_precio.copy()
                df_display['precio_optimo'] = df_display['precio_optimo'].apply(lambda x: f"${x:,.0f}")
                df_display['rentabilidad'] = df_display['rentabilidad'].apply(lambda x: f"{x:.2f}%")
                df_display['almuerzos'] = df_display['almuerzos'].apply(lambda x: f"{x:.0f}")
                st.dataframe(df_display, use_container_width=True)

                st.success(f"""
                    **Conclusión**: 
                    - Con **100 almuerzos/día**: Precio requerido ≈ ${df_precio.iloc[0]['precio_optimo']:,.0f}
                    - Con **120 almuerzos/día**: Precio requerido ≈ ${df_precio.iloc[-1]['precio_optimo']:,.0f}
                    - A mayor volumen, se puede reducir el precio y mantener la rentabilidad objetivo
                    """)
            else:
                st.error("No se pudo calcular el precio óptimo con los parámetros actuales")

# --- TAB 5: Escenarios de Crisis (5.1) ---
with tabs[4]:
    st.header("5.1 - Análisis de Escenarios de Crisis")
    st.markdown("""
        **Situación**: Analizar el impacto de una crisis con:
        - Aumento en precio de insumos principales (carne)
        - Aumento en impuesto de industria y comercio
        - Aumento en servicios públicos
        - Respuesta con aumento de precios
        - Reacción del mercado con baja en ventas
        """)

    if st.button("Analizar Escenarios de Crisis"):
        with st.spinner("Analizando escenarios..."):
            df_crisis = analisis_escenarios_crisis()

            # Gráfico comparativo
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            # Ventas
            ax1.bar(df_crisis['escenario'], df_crisis['ventas'], color='steelblue')
            ax1.set_title('Ventas por Escenario')
            ax1.set_ylabel('Ventas ($)')
            ax1.tick_params(axis='x', rotation=15)
            for i, v in enumerate(df_crisis['ventas']):
                ax1.text(i, v, f'${v:,.0f}', ha='center', va='bottom')

            # Costos
            ax2.bar(df_crisis['escenario'], df_crisis['costo_de_ventas'],
                    color='coral', label='Costo Ventas')
            ax2.bar(df_crisis['escenario'], df_crisis['gastos'],
                    bottom=df_crisis['costo_de_ventas'], color='orange', label='Gastos')
            ax2.set_title('Estructura de Costos')
            ax2.set_ylabel('Monto ($)')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=15)

            # Rentabilidad
            ax3.bar(df_crisis['escenario'], df_crisis['rentabilidad_bruta'], color='green')
            ax3.axhline(55, color='red', linestyle='--', label='Objetivo 55%')
            ax3.set_title('Rentabilidad Bruta (%)')
            ax3.set_ylabel('Rentabilidad (%)')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=15)
            for i, v in enumerate(df_crisis['rentabilidad_bruta']):
                ax3.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

            # Utilidad Neta
            colors = ['green' if x > 0 else 'red' for x in df_crisis['utilidad_bruta']]
            ax4.bar(df_crisis['escenario'], df_crisis['utilidad_bruta'], color=colors)
            ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
            ax4.set_title('Utilidad Neta')
            ax4.set_ylabel('Utilidad ($)')
            ax4.tick_params(axis='x', rotation=15)
            for i, v in enumerate(df_crisis['utilidad_bruta']):
                ax4.text(i, v, f'${v:,.0f}', ha='center',
                         va='bottom' if v > 0 else 'top')

            plt.tight_layout()
            st.pyplot(fig)

            # Tabla comparativa
            st.subheader("Comparación Detallada de Escenarios")
            df_display = df_crisis[['escenario', 'ventas', 'costo_de_ventas',
                                    'rentabilidad_bruta', 'gastos', 'utilidad_bruta']].copy()
            for col in ['ventas', 'costo_de_ventas', 'gastos', 'utilidad_bruta']:
                df_display[col] = df_display[col].apply(lambda x: f"${x:,.0f}")
            df_display['rentabilidad_bruta'] = df_display['rentabilidad_bruta'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(df_display, use_container_width=True)

            # Informe ejecutivo
            st.subheader("📋 Informe Ejecutivo")
            st.markdown(f"""
                ### Análisis de Situación Empresarial

                **Escenario Base**:
                - Ventas: ${df_crisis.iloc[0]['ventas']:,.0f}
                - Utilidad: ${df_crisis.iloc[0]['utilidad_bruta']:,.0f}
                - Rentabilidad: {df_crisis.iloc[0]['rentabilidad_bruta']:.1f}%

                **Crisis Moderada** (↑20% carne, ↑60% impuesto, ↑27% servicios):
                - Ventas caen 7.7% (90 almuerzos vs 97.5)
                - Utilidad: ${df_crisis.iloc[1]['utilidad_bruta']:,.0f} 
                - Impacto: {((df_crisis.iloc[1]['utilidad_bruta'] - df_crisis.iloc[0]['utilidad_bruta']) / df_crisis.iloc[0]['utilidad_bruta'] * 100):.1f}%

                **Crisis Severa** (↑40% carne, ↑140% impuesto, ↑60% servicios):
                - Ventas caen 17.9% (80 almuerzos)
                - Utilidad: ${df_crisis.iloc[2]['utilidad_bruta']:,.0f}
                - Situación: {'CRÍTICA - Pérdidas' if df_crisis.iloc[2]['utilidad_bruta'] < 0 else 'Utilidad reducida'}

                **Respuesta con Aumento de Precio** (+33% precio, -28% volumen):
                - Precio sube a $5,000 pero volumen cae a 70 almuerzos
                - Utilidad: ${df_crisis.iloc[3]['utilidad_bruta']:,.0f}
                - Resultado: {'Mejor que crisis severa sin acción' if df_crisis.iloc[3]['utilidad_bruta'] > df_crisis.iloc[2]['utilidad_bruta'] else 'Peor estrategia'}

                ### Recomendaciones:
                1. **Optimizar costos**: Buscar proveedores alternativos para reducir dependencia del precio de la carne
                2. **Elasticidad precio-demanda**: El aumento de precio mejora la situación, sugiriendo que hay margen
                3. **Diversificación**: Considerar menú con opciones de menor costo
                4. **Negociación**: Reducir gastos fijos (arrendamiento) en época de crisis
                5. **Punto de equilibrio**: Monitorear constantemente para evitar operación en pérdida
                """)

# --- TAB 6: Historial ---
with tabs[5]:
    st.header("📚 Historial de Simulaciones")

    simulaciones = cargar_simulaciones()

    if len(simulaciones) == 0:
        st.info("No hay simulaciones guardadas. Ejecuta una simulación primero.")
    else:
        st.subheader("Simulaciones Guardadas")
        display_df = simulaciones[['id', 'fecha', 'tipo_analisis', 'num_iteraciones',
                                   'utilidad_media', 'utilidad_std',
                                   'prob_perdida', 'descripcion']].copy()
        display_df['fecha'] = pd.to_datetime(display_df['fecha']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['utilidad_media'] = display_df['utilidad_media'].apply(lambda x: f"${x:,.0f}")
        display_df['utilidad_std'] = display_df['utilidad_std'].apply(lambda x: f"${x:,.0f}")
        display_df['prob_perdida'] = display_df['prob_perdida'].apply(lambda x: f"{x:.1f}%")

        st.dataframe(display_df, use_container_width=True)

        # Selector de simulación
        st.subheader("Analizar Simulación Específica")
        sim_options = {f"#{row['id']} - {row['tipo_analisis']} - {row['fecha']}": row['id']
                       for _, row in simulaciones.iterrows()}

        selected = st.selectbox("Selecciona una simulación:", list(sim_options.keys()))

        if selected:
            sim_id = sim_options[selected]
            df_selected = cargar_resultados(sim_id)
            sim_info = simulaciones[simulaciones['id'] == sim_id].iloc[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Media", f"${sim_info['utilidad_media']:,.0f}")
            col2.metric("Desv. Std", f"${sim_info['utilidad_std']:,.0f}")
            col3.metric("Prob. Pérdida", f"{sim_info['prob_perdida']:.1f}%")

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df_selected["utilidad_bruta"], bins=40, kde=True, ax=ax)
                ax.axvline(sim_info['utilidad_media'], color="red", linestyle="--")
                ax.set_title("Distribución de Utilidades")
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                components = ['ventas', 'costo_de_ventas', 'gastos', 'utilidad_bruta']
                means = [df_selected[c].mean() for c in components]
                ax.barh(components, means, color=['green', 'red', 'orange', 'blue'])
                ax.set_xlabel('Monto ($)')
                ax.set_title('Componentes Promedio')
                st.pyplot(fig)

            st.subheader("Estadísticas Detalladas")
            st.dataframe(df_selected.describe().style.format("{:,.2f}"),
                         use_container_width=True)

# --- TAB 7: Comparación ---
with tabs[6]:
    st.header("🔄 Comparación entre Simulaciones")

    simulaciones = cargar_simulaciones()

    if len(simulaciones) < 2:
        st.info("Necesitas al menos 2 simulaciones para comparar.")
    else:
        sim_ids = st.multiselect(
            "Selecciona simulaciones para comparar (máximo 5):",
            options=simulaciones['id'].tolist(),
            format_func=lambda x: f"#{x} - {simulaciones[simulaciones['id'] == x]['tipo_analisis'].iloc[0]}",
            max_selections=5
        )

        if len(sim_ids) >= 2:
            st.subheader("Comparación de Métricas")
            comp_data = []
            for sim_id in sim_ids:
                info = simulaciones[simulaciones['id'] == sim_id].iloc[0]
                comp_data.append({
                    'ID': f"#{sim_id}",
                    'Tipo': info['tipo_analisis'],
                    'Fecha': str(info['fecha'])[:16],
                    'Iteraciones': info['num_iteraciones'],
                    'Media': f"${info['utilidad_media']:,.0f}",
                    'Desv. Std': f"${info['utilidad_std']:,.0f}",
                    'Prob. Pérdida': f"{info['prob_perdida']:.1f}%"
                })

            st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

            # Gráfico comparativo
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            for sim_id in sim_ids:
                df_sim = cargar_resultados(sim_id)
                ax1.hist(df_sim['utilidad_bruta'], bins=30, alpha=0.5, label=f"Sim #{sim_id}")

            ax1.set_xlabel("Utilidad Neta ($)")
            ax1.set_ylabel("Frecuencia")
            ax1.set_title("Distribuciones Comparadas")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Boxplot comparativo
            box_data = []
            labels = []
            for sim_id in sim_ids:
                df_sim = cargar_resultados(sim_id)
                box_data.append(df_sim['utilidad_bruta'])
                labels.append(f"#{sim_id}")

            ax2.boxplot(box_data, labels=labels)
            ax2.set_ylabel("Utilidad Neta ($)")
            ax2.set_title("Comparación de Rangos")
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig)

# Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📖 Guía de Uso")
st.sidebar.markdown("""
    **Tabs disponibles:**
    1. **Monte Carlo**: Simulación estocástica (Puntos b y g)
    2. **Sensibilidad 1.1**: Análisis Costos vs Gastos
    3. **Variables 2.1**: Análisis individual de variables
    4. **Precio Óptimo 4.1**: Cálculo para rentabilidad 55%
    5. **Crisis 5.1**: Escenarios de crisis empresarial
    6. **Historial**: Ver simulaciones guardadas
    7. **Comparación**: Comparar múltiples simulaciones

    **Punto 3.1** se explica en el Tab Variables 2.1
    """)