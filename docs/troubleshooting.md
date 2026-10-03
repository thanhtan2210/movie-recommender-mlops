# Troubleshooting

Problems met while building and deploying the app, and what fixed them.

## `LanceError(Schema): No field named movieid`

- **Symptom**: raised by `table.search().where('"movieId" = 1')`.
- **Cause**: the SQL parser inside LanceDB lowercases column names, which no longer match the case-sensitive schema.
- **Fix**: look movies up in pandas instead of SQL. The catalogue is small enough to hold in memory.

## `only accept 2-D tensor shape, got: []`

- **Symptom**: LanceDB fails during a vector search.
- **Cause**: PyArrow inferred the vector column as a variable-length list when the table was created on Colab.
- **Fix**: declare the schema explicitly, `pa.list_(pa.float32(), 384)`, when creating the table, and pass query vectors as flat `float32` arrays.

## `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`

- **Cause**: `httpx` 0.28 renamed the `proxies` parameter to `proxy`, which breaks the Groq SDK.
- **Fix**: pin `httpx==0.27.2` in `requirements.txt`.

## Groq `RateLimitError`

- **Symptom**: the chat stops answering when the free quota is used up.
- **Fix**: `chatbot.py` catches the error and answers with a plain vector-search result (a list of matching movies) instead of an LLM-written reply, so the user still gets an answer.
