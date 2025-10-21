// tabular_model.js — seeded initializers + regularized MLP (ELU), trained on log(price)

/**
 * Build the tabular model.
 * @param {object} schema - must contain: numCols, catCols, catMaps; optionally numInputDim.
 * @param {object} [seeds] - optional deterministic seeds:
 *   { initBase=2000, kDense1=301, kDense2=302, kDense3=303, kOut=304, kNum=101 }
 */
export function buildTabularModel(schema, seeds = {}) {
  const SEEDS = {
    initBase: 2000, kDense1: 301, kDense2: 302, kDense3: 303, kOut: 304, kNum: 101,
    ...seeds
  };

  const KINIT = (s)=> tf.initializers.heNormal({ seed: s });
  const BINIT = tf.initializers.zeros;
  const EINIT = (s)=> tf.initializers.randomUniform({ minval: -0.05, maxval: 0.05, seed: s });

  const inputs = [], parts = [];
  const numDim = (schema.numInputDim ?? schema.numCols.length);

  // numeric branch
  const numInput = tf.input({shape:[numDim], name:'numeric'});
  inputs.push(numInput);
  let numBranch = tf.layers.dense({
    units: 64, activation:'elu',
    kernelInitializer: KINIT(SEEDS.kNum),
    biasInitializer:   BINIT()
  }).apply(numInput);
  numBranch = tf.layers.batchNormalization().apply(numBranch);
  parts.push(numBranch);

  // categorical branches
  let seedBase = SEEDS.initBase;
  for (const h of schema.catCols) {
    const size = schema.catMaps[h].size;
    const dim  = Math.min(24, Math.ceil(Math.sqrt(size)) + 1);
    const inp  = tf.input({shape:[1], dtype:'int32', name:`cat_${h}`});
    inputs.push(inp);
    const emb = tf.layers.embedding({
      inputDim: size, outputDim: dim,
      embeddingsInitializer: EINIT(seedBase++)
    }).apply(inp); // [B,1,dim]
    const flat = tf.layers.dropout({rate:0.2}).apply(tf.layers.flatten().apply(emb));
    parts.push(flat);
  }

  const concat = tf.layers.concatenate().apply(parts);

  let x = tf.layers.dense({
    units:192, activation:'elu',
    kernelInitializer: KINIT(SEEDS.kDense1), biasInitializer: BINIT(),
    kernelRegularizer: tf.regularizers.l2({l2:4e-4})
  }).apply(concat);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.dropout({rate:0.35}).apply(x);

  x = tf.layers.dense({
    units:96, activation:'elu',
    kernelInitializer: KINIT(SEEDS.kDense2), biasInitializer: BINIT(),
    kernelRegularizer: tf.regularizers.l2({l2:4e-4})
  }).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.dropout({rate:0.35}).apply(x);

  x = tf.layers.dense({
    units:48, activation:'elu',
    kernelInitializer: KINIT(SEEDS.kDense3), biasInitializer: BINIT(),
    kernelRegularizer: tf.regularizers.l2({l2:4e-4})
  }).apply(x);

  const out = tf.layers.dense({
    units:1, activation:'linear',
    kernelInitializer: KINIT(SEEDS.kOut), biasInitializer: BINIT()
  }).apply(x);

  const model = tf.model({inputs, outputs: out});
  model.compile({ optimizer: tf.train.adam(0.0006), loss: 'meanSquaredError', metrics: ['mae'] });
  return model;
}

/**
 * Deterministic fit: set shuffle=false. We rely on a single seeded shuffle in the loader.
 */
export async function fitModel(
  model,
  tensors,
  { epochs = 500, batchSize = 128, validationSplit = 0.20, shuffle = false, onEpoch } = {}
) {
  return await model.fit(tensors.Xtrain, tensors.ytrain, {
    epochs, batchSize, validationSplit, shuffle,
    callbacks: { onEpochEnd: async (ep, logs)=> onEpoch && onEpoch(ep, logs) }
  });
}

export async function evaluate(model, X, y, schema, returnPred=false) {
  const pred = model.predict(X);
  const yPredLog = await pred.data();
  const yTrueLog = await y.data();
  const yPred = Array.from(yPredLog, v=> Math.expm1(v));
  const yTrue = Array.from(yTrueLog, v=> Math.expm1(v));
  let ae=0, se=0, ssTot=0;
  const meanY = yTrue.reduce((a,b)=>a+b,0)/yTrue.length;
  for (let i=0;i<yTrue.length;i++){ const e=yPred[i]-yTrue[i]; ae+=Math.abs(e); se+=e*e; const d=yTrue[i]-meanY; ssTot+=d*d; }
  const mae = ae/yTrue.length, rmse = Math.sqrt(se/yTrue.length), r2 = 1 - (se/(ssTot+1e-8));
  if (returnPred) return {mae, rmse, r2, yTrue, yPred};
  return {mae, rmse, r2};
}
