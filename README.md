# NimbusHR FAQ Bot — Chatbot de soporte con RAG

Proyecto Integrador — Módulo 2, AI Engineering (Henry).

Chatbot de preguntas frecuentes (FAQ) para NimbusHR, una plataforma de
gestión de recursos humanos ficticia. En vez de que el equipo de soporte
conteste manualmente las mismas ~200 preguntas repetitivas por día sobre
políticas y procedimientos, este sistema usa **RAG (Retrieval-Augmented
Generation)**: primero recupera los fragmentos de documentación
efectivamente relevantes a la pregunta, y recién después le pide al LLM
que genere una respuesta basada en ese contexto — en vez de responder "de
memoria" con lo que el modelo aprendió en su entrenamiento (que podría
estar desactualizado o directamente no conocer la documentación interna
de la empresa).

## Por qué RAG

El sistema implementa el flujo clásico de dos pasos: **recuperación**
(buscar los chunks más relevantes por similitud vectorial) y luego
**generación** (el LLM arma la respuesta usando esos chunks como
contexto). Esto resuelve dos problemas que un LLM "pelado" no puede
resolver por sí solo:

- **Conocimiento actualizable sin reentrenar**: si mañana cambia la
  política de vacaciones, alcanza con re-indexar el documento
  (`build_index.py`) — no hay que reentrenar ni hacer fine-tuning de
  ningún modelo.
- **Transparencia y auditabilidad**: cada respuesta viene acompañada de
  los `chunks_related` que se usaron para generarla, así que se puede
  verificar de dónde salió cada dato (algo que un LLM respondiendo de
  memoria no puede ofrecer).

## Estructura del repositorio

```
data/
  faq_document.txt     # Documento fuente (~2000 palabras) con las FAQs de NimbusHR
src/
  build_index.py        # Pipeline de indexación (offline, se corre una vez)
  query.py               # Pipeline de consulta (online, una corrida por pregunta)
  evaluator.py            # Agente evaluador (bonus): puntúa la calidad de la respuesta
  api_utils.py            # Reintentos con backoff ante rate-limit/errores de red
index/
  faq_index.json          # Chunks + embeddings generados por build_index.py
outputs/
  sample_queries.json     # 3+ ejemplos reales de consulta -> respuesta
tests/
  test_core.py            # Tests de chunking, similitud coseno y contrato JSON
```

## Setup

1. Cloná el repo y entrá a la carpeta:
   ```bash
   git clone <url-del-repo>
   cd pi2-rag-faq
   ```

2. Python 3.11 o superior. Creá un entorno virtual e instalá dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configurá tu API key:
   ```bash
   cp .env.example .env
   # editar .env y poner tu OPENAI_API_KEY real
   export OPENAI_API_KEY=tu-key-real   # PowerShell: $env:OPENAI_API_KEY = "tu-key-real"
   ```

## Uso

**Paso 1 — Indexar el documento (una sola vez, o cada vez que cambie `data/faq_document.txt`):**

```bash
python src/build_index.py
```

Esto carga el documento, lo divide en chunks, genera un embedding por
chunk con la API de OpenAI, y guarda todo en `index/faq_index.json`.

**Paso 2 — Consultar el sistema:**

```bash
python src/query.py "¿Cuántos días de vacaciones tengo por año?"
```

Salida esperada (ejemplo):

```json
{
  "user_question": "¿Cuántos días de vacaciones tengo por año?",
  "system_answer": "Tenés 15 días hábiles de vacaciones por año, acumulando 1.25 días por cada mes trabajado...",
  "chunks_related": [
    {"id": 1, "text": "Política de vacaciones. Todos los empleados...", "similarity": 0.6},
    {"id": 2, "text": "Si no responde en ese plazo...", "similarity": 0.58},
    {"id": 25, "text": "Preguntas frecuentes sobre feriados...", "similarity": 0.52}
  ]
}
```

Si corrés con `--evaluate`, además de ese JSON (sin tocar) se imprime la
evaluación como un bloque aparte:

```json
{
  "score": 10,
  "reason": "La respuesta es fiel al fragmento 1 y responde completamente la pregunta del usuario."
}
```

Opciones adicionales:
```bash
python src/query.py "pregunta" --top-k 5      # recuperar más chunks
python src/query.py "pregunta" --evaluate     # suma el agente evaluador (bonus), como bloque aparte
```

Nota importante: el JSON principal siempre tiene **exactamente 3 claves**
(`user_question`, `system_answer`, `chunks_related`), tal como pide la
consigna. La evaluación del agente bonus se imprime/guarda como un objeto
**separado**, nunca mezclado dentro de ese mismo JSON.

**Paso 3 — Generar los ejemplos de salida (`outputs/sample_queries.json`):**

```bash
python src/generate_samples.py
```

Esto corre 4 preguntas de ejemplo reales contra el sistema ya indexado y
genera dos archivos:
- `outputs/sample_queries.json` — array con las 4 respuestas, cada una con
  exactamente las 3 claves requeridas (evidencia de que el sistema
  funciona de punta a punta).
- `outputs/sample_evaluations.json` — (bonus) el score y la justificación
  del agente evaluador para cada una de esas mismas preguntas, en un
  archivo aparte.

## Decisiones técnicas

**Estrategia de chunking: por oraciones/párrafos.** Se eligió agrupar
oraciones consecutivas (sin cortar ninguna a la mitad ni mezclar
contenido de dos párrafos distintos) hasta acercarse a un target de ~150
tokens, dentro de un rango permitido de 50 a 500 tokens
(`MIN_CHUNK_TOKENS`, `TARGET_CHUNK_TOKENS`, `MAX_CHUNK_TOKENS` en
`build_index.py`). Se prefirió sobre tamaño fijo porque el documento fuente
está organizado en secciones temáticas claras (una política por párrafo),
y cortar por oraciones/párrafos preserva esa unidad de sentido en vez de
partir una idea a la mitad por cortar cada N caracteres. Un paso adicional
(`merge_short_chunks`) fusiona los chunks que quedaron demasiado cortos
(por ejemplo, un título suelto) con el chunk vecino, para que ninguno
quede fuera del rango 50-500 tokens.

**Método de búsqueda vectorial: k-NN exacto (fuerza bruta) con similitud
coseno.** Con ~26 chunks, calcular la similitud coseno contra todos los
vectores a la vez (vectorizado con numpy) es prácticamente instantáneo y
evita sumar una dependencia de vector store dedicado (FAISS, Pinecone,
etc.) que no aporta valor real a esta escala y complicaría la
reproducibilidad del proyecto. Para un corpus de miles de documentos, la
función `search_similar_chunks()` es el punto exacto donde se
reemplazaría esta búsqueda de fuerza bruta por un índice ANN.

**Modelo de embeddings: `text-embedding-3-small` (OpenAI).** Mismo
modelo tanto para indexar los chunks como para la pregunta del usuario —
es un requisito, ya que los vectores solo son comparables entre sí si
vienen del mismo modelo (dimensiones y espacio semántico deben coincidir).

**Manejo de errores de red/rate-limit.** `api_utils.py` implementa un
decorador de reintentos con backoff exponencial para las llamadas a la
API (embeddings, generación y evaluación), que captura específicamente
errores transitorios (`RateLimitError`, `APIConnectionError`,
`APITimeoutError`, `InternalServerError`) y reintenta hasta 3 veces antes
de fallar. Errores no transitorios (ej. API key inválida) se relanzan de
inmediato sin reintentar.

## Tests

```bash
pytest tests/test_core.py -v
```

No requieren `OPENAI_API_KEY`: cubren la lógica de chunking (incluyendo
que el documento produzca 20+ chunks dentro del rango 50-500 tokens), el
cálculo de similitud coseno, la búsqueda k-NN y la validación del
contrato de salida JSON — sin llamar a la API real.

## El agente evaluador (bonus)

`evaluator.py` recibe la pregunta, la respuesta generada y los chunks
usados, y le pide a un segundo llamado al LLM (temperatura 0, para ser lo
más determinístico posible) que evalúe la respuesta en dos dimensiones:
**fidelidad al contexto** (¿la respuesta usa información que realmente
está en los chunks, o alucina?) y **completitud** (¿responde toda la
pregunta?). Devuelve `{"score": 0-10, "reason": "..."}`. Si la llamada de
evaluación falla (JSON inválido, error de API), se devuelve un score de
0 con una razón que explica que fue un fallo técnico del evaluador — para
que un problema en la evaluación nunca tire abajo la respuesta principal
del pipeline de consulta.

## Limitaciones conocidas

- El sistema está construido para un único dominio (FAQs de RRHH de
  NimbusHR); generalizar a múltiples documentos o dominios requeriría
  sumar metadata de origen a cada chunk.
- La búsqueda k-NN de fuerza bruta escala linealmente con la cantidad de
  chunks — para un corpus de miles de documentos convendría migrar a un
  índice ANN (FAISS, HNSW) o un vector store dedicado.
- El chunking por oraciones puede generar chunks de tamaño variable
  (compensado parcialmente por `merge_short_chunks`), a diferencia de un
  esquema de tamaño fijo que da chunks más uniformes pero puede cortar
  ideas a la mitad.