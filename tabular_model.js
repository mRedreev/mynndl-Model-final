// tabular_model.js — seeded initializers + regularized MLP (ELU), trained on log(price)

/**
 * Build the tabular model.
 * @param {object} schema - must contain: numCols, catCols, catMaps; optionally numInputDim.
 * @param {object} [seeds] - optional deterministic seeds:
 *   { initBase=2000, kDense1=301, kDense2=302, kDense3=303, kOut=304, kNum=101 }
 */
export function buildTabularModel(schema, seeds = {}) {
  // ---- seeds (deterministic initializers) ----
  const SEEDS = {
    initBase: 2000, // starting seed for Embedding matrices (will ++ per column)
    kDense1: 301,   // Dense trunk layer 1
    kDense2: 302,   // Dense trunk layer 2
    kDense3: 303,   // Dense trunk layer 3
    kOut:    304,   // Output layer
    kNum:    101,   // Numeric branch first Dense
    ...seeds
  };

  // ---- helpers: seeded initializers ----
  const KINIT = (s)=> tf.initializers.heNormal({ seed: s });
  const BINIT = tf.initializers.zeros;
  const EINIT = (s)=> tf.initializers.randomUniform({ minval: -0.05, maxval: 0.05, seed: s });

  const inputs = [];
  const parts  = [];

  // Use final numeric width that already includes target-encoding columns if present
  const numDim = (schema.numInputDim ?? schema.numCols.length);

  // ---- numeric branch ----
  const numInput = tf.input({ shape: [numDim], name: 'numeric' });
  inputs.push(numInput);
  let numBranch = tf.layers.dense({
    units: 64,
    activation: 'elu',
    kernelInitializer: KINIT(SEEDS.kNum),
    biasInitializer:   BINIT()
  }).apply(numInput);
  numBranch = tf.layers.batchNormalization().apply(numBranch);
  parts.push(numBranch);

  // ---- categorical branches (Embedding + Flatten + Dropout) ----
  let seedBase = SEEDS.initBase;
  for (const h of schema.catCols) {
    const size = schema.catMaps[h].size;
    const dim  = Math.min(24, Math.ceil(Math.sqrt(size)) + 1); // cap dims to avoid overfit
    const inp  = tf.input({ shape: [1], dtype: 'int32', name: `cat_${h}` });
    inputs.push(inp);

    const emb = tf.layers.embedding({
      inputDim: size,
      outputDim: dim,
      embeddingsInitializer: EINIT(seedBase++)  // <- seeded embeddings
    }).apply(inp);                               // [B, 1, dim]

    const flat = tf.layers.dropout({ rate: 0.2 })
                  .apply(tf.layers.flatten().apply(emb)); // -> [B, dim]
    parts.push(flat);
  }

  // ---- concat & trunk ----
  const concat = tf.layers.concatenate().apply(parts);

  let x = tf.layers.dense({
    units: 192,
    activation: 'elu',
    kernelInitializer: KINIT(SEEDS.kDense1),
    biasInitializer:   BINIT(),
    kernelRegularizer: tf.regularizers.l2({ l2: 4e-4 })
  }).apply(concat);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.dropout({ rate: 0.35 }).apply(x);

  x = tf.layers.dense({
    units: 96,
    activation: 'elu',
    kernelInitializer: KINIT(SEEDS.kDense2),
    biasInitializer:   BINIT(),
    kernelRegularizer: tf.regularizers.l2({ l2: 4e-4 })
  }).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.dropout({ rate: 0.35 }).apply(x);

  x = tf.layers.dense({
    units: 48,
    activation: 'elu',
    kernelInitializer: KINIT(SEEDS.kDense3),
    biasInitializer:   BINIT(),
    kernelRegularizer: tf.regularizers.l2({ l2: 4e-4 })
  }).apply(x);

  const out = tf.layers.dense({
    units: 1,
    activation: 'linear',
    kernelInitializer: KINIT(SEEDS.kOut),
    biasInitializer:   BINIT()
  }).apply(x);

  const model = tf.model({ inputs, outputs: out });

  // Lower LR for stability on small tabular datasets
  model.compile({
    optimizer: tf.train.adam(0.0006),
    loss: 'meanSquaredError',
    metrics: ['mae']
  });

  return model;
}

/**
 * Fit the model.
 * By default we keep shuffle=false so training is deterministic if your dataloader
 * already applied a single seeded shuffle to the train set.
 */
export async function fitModel(
  model,
  tensors,
  { epochs = 500, batchSize = 128, validationSplit = 0.20, shuffle = false, onEpoch } = {}
) {
  return await model.fit(tensors.Xtrain, tensors.ytrain, {
    epochs,
    batchSize,
    validationSplit,
    shuffle,
    callbacks: {
      onEpochEnd: async (ep, logs) => { if (onEpoch) await onEpoch(ep, logs); }
    }
  });
}

/**
 * Evaluate on (X, y), reporting MAE/RMSE/R² in **original price units**.
 * y is assumed to be log(price); we exponentiate predictions back.
 */
export async function evaluate(model, X, y, schema, returnPred = false) {
  const pred = model.predict(X);
  const yPredLog = await pred.data();
  const yTrueLog = await y.data();

  const yPred = Array.from(yPredLog, v => Math.expm1(v));
  const yTrue = Array.from(yTrueLog, v => Math.expm1(v));

  let ae = 0, se = 0, ssTot = 0;
  const meanY = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;

  for (let i = 0; i < yTrue.length; i++) {
    const e = yPred[i] - yTrue[i];
    ae += Math.abs(e);
    se += e * e;
    const d = yTrue[i] - meanY;
    ssTot += d * d;
  }

  const mae  = ae / yTrue.length;
  const rmse = Math.sqrt(se / yTrue.length);
  const r2   = 1 - (se / (ssTot + 1e-8));

  if (returnPred) return { mae, rmse, r2, yTrue, yPred };
  return { mae, rmse, r2 };
}
