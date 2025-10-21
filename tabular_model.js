// tabular_model.js — Embeddings + Deep MLP with BN/Dropout, trained on log(price)
export function buildTabularModel(schema) {
  const inputs = [];
  const parts = [];

  const numInput = tf.input({shape: [schema.numCols.length], name: 'numeric'});
  inputs.push(numInput);
  let numBranch = tf.layers.dense({units: 64, activation: 'relu'}).apply(numInput);
  numBranch = tf.layers.batchNormalization().apply(numBranch);
  parts.push(numBranch);

  for (const h of schema.catCols) {
    const size = schema.catMaps[h].size;
    const dim = Math.min(50, Math.ceil(Math.sqrt(size))+1);
    const inp = tf.input({shape: [1], dtype:'int32', name: `cat_${h}`});
    inputs.push(inp);
    const emb = tf.layers.embedding({inputDim: size, outputDim: dim}).apply(inp); 
    const flat = tf.layers.flatten().apply(emb); 
    parts.push(flat);
  }

  const concat = tf.layers.concatenate().apply(parts);
  let x = tf.layers.dense({units: 256, activation:'relu', kernelRegularizer: tf.regularizers.l2({l2:1e-5})}).apply(concat);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.dropout({rate:0.15}).apply(x);
  x = tf.layers.dense({units: 128, activation:'relu'}).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.dropout({rate:0.15}).apply(x);
  x = tf.layers.dense({units: 64, activation:'relu'}).apply(x);
  const out = tf.layers.dense({units:1, activation:'linear'}).apply(x);

  const model = tf.model({inputs, outputs: out});
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError', metrics: ['mae'] });
  return model;
}

export async function fitModel(model, tensors, {epochs=200, batchSize=32, validationSplit=0.15, onEpoch}={}) {
  return await model.fit(tensors.Xtrain, tensors.ytrain, {
    epochs, batchSize, shuffle:true, validationSplit,
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
  for (let i=0;i<yTrue.length;i++) {
    const e = yPred[i] - yTrue[i];
    ae += Math.abs(e);
    se += e*e;
    const d = yTrue[i]-meanY;
    ssTot += d*d;
  }
  const mae = ae/yTrue.length;
  const rmse = Math.sqrt(se/yTrue.length);
  const r2 = 1 - (se/(ssTot+1e-8));
  if (returnPred) return {mae, rmse, r2, yTrue, yPred};
  return {mae, rmse, r2};
}
