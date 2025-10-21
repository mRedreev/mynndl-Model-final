// tabular_app.js — deterministic training + UI flow
import { parseCarsCSV, analyzeColumns, buildTabularTensors } from './tabular_loader.js';
import { buildTabularModel, fitModel, evaluate } from './tabular_model.js';

// ---- global deterministic seeds ----
export const SEEDS = {
  split: 42,        // stratified train/test by 'make'
  kfold: 1337,      // K-fold target encoding
  initBase: 2000,   // Embedding initializers base
  trainOrder: 777   // single shuffle of train set
};

const st = { model:null, schema:null, tensors:null, batch:128, epochs:500 };

// set deterministic backend before anything else
(async () => {
  await tf.ready();
  await tf.setBackend('webgl'); // 'cpu' is even more deterministic but slower
  if (tf.env().get('WEBGL_VERSION') != null) {
    tf.env().set('WEBGL_DETERMINISTIC', true);
  }
})();

const fileInput = document.getElementById('fileInput');
const fileName  = document.getElementById('fileName');
const statusEl  = document.getElementById('status');
const trainBtn  = document.getElementById('trainBtn');
const evalBtn   = document.getElementById('evalBtn');
const progressEl= document.getElementById('progress');
const epochText = document.getElementById('epochText');
const summary   = document.getElementById('summary');
const saveBtn   = document.getElementById('saveBtn');
const loadBtn   = document.getElementById('loadBtn');

fileInput.addEventListener('change', async (e)=> {
  const f = e.target.files?.[0];
  if (!f) return;
  fileName.textContent = f.name;
  statusEl.textContent = 'Parsing CSV…';
  try {
    const rows   = await parseCarsCSV(f);
    const schema = analyzeColumns(rows, 'price');
    // pass seeds down through schema so loader sees them
    schema.__seeds = { ...SEEDS };
    const tensors = buildTabularTensors(rows, schema, 0.8);
    st.schema = schema; st.tensors = tensors;
    statusEl.textContent = `Ready: ${rows.length} rows — numeric: ${schema.numCols.length}, categorical: ${schema.catCols.length}.`;
    trainBtn.disabled = false;
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Error: ' + err.message;
  }
});

trainBtn.addEventListener('click', async ()=> {
  trainBtn.disabled = true;
  statusEl.textContent = 'Building model…';
  st.model = buildTabularModel(st.schema, {
    initBase: SEEDS.initBase, kDense1:301, kDense2:302, kDense3:303, kOut:304, kNum:101
  });
  statusEl.textContent = 'Training…';
  progressEl.value = 0; epochText.textContent = '';
  st.__best = Infinity; st.__stall = 0;

  await fitModel(st.model, st.tensors, {
    epochs: st.epochs,
    batchSize: st.batch,
    validationSplit: 0.20,
    shuffle: false, // single deterministic shuffle happens in the loader
    onEpoch: async (ep, logs)=> {
      progressEl.value = (ep+1)/st.epochs;
      epochText.textContent = `Epoch ${ep+1}/${st.epochs} — loss ${logs.loss.toFixed(4)} — val_loss ${logs.val_loss?.toFixed(4) ?? '-'} — MAE ${logs.mae.toFixed(2)}`;

      // ReduceLROnPlateau (patience=8) + EarlyStopping (patience=20)
      if (logs.val_loss != null) {
        if (logs.val_loss + 1e-6 < st.__best) { st.__best = logs.val_loss; st.__stall = 0; }
        else { st.__stall++; }
        if (st.__stall === 8) {
          const opt = st.model.optimizer;
          opt.learningRate = (opt.learningRate ?? 0.0006) * 0.5;
        }
        if (st.__stall > 20) { st.model.stopTraining = true; } // not "this.stopTraining"
      }
    }
  });
  statusEl.textContent = 'Trained.';
  evalBtn.disabled = false;
  saveBtn.disabled = false;
});

evalBtn.addEventListener('click', async ()=> {
  const mTrain = await evaluate(st.model, st.tensors.Xtrain, st.tensors.ytrain, st.schema);
  const mTest  = await evaluate(st.model, st.tensors.Xtest,  st.tensors.ytest,  st.schema, true);
  appendMetrics('train', mTrain);
  appendMetrics('test',  mTest);
  summary.textContent = `Test MAE: ${mTest.mae.toFixed(2)}, RMSE: ${mTest.rmse.toFixed(2)}, R²: ${mTest.r2.toFixed(3)} (n=${st.tensors.ytest.shape[0]}).`;
  drawScatter(mTest.yTrue, mTest.yPred);
});

saveBtn.addEventListener('click', async ()=> {
  if (!st.model) return;
  await st.model.save('downloads://tfjs_cars_tabular_model');
});
loadBtn.addEventListener('click', async ()=> {
  try {
    const model = await tf.loadLayersModel('indexeddb://tfjs_cars_tabular_model');
    st.model = model;
    evalBtn.disabled = false; saveBtn.disabled = false;
    statusEl.textContent = 'Weights loaded from IndexedDB.';
  } catch(e) {
    statusEl.textContent = 'No saved weights found. Train and click “Save Weights”.';
  }
});

function appendMetrics(split, m) {
  const tb = document.querySelector('#metricsTbl tbody');
  const tr = document.createElement('tr');
  function td(x){ const e=document.createElement('td'); e.textContent=String(x); return e; }
  tr.appendChild(td(split));
  tr.appendChild(td(m.mae.toFixed(2)));
  tr.appendChild(td(m.rmse.toFixed(2)));
  tr.appendChild(td(m.r2.toFixed(3)));
  tb.appendChild(tr);
}

function drawScatter(yTrue, yPred) {
  const canvas = document.getElementById('scatter');
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth;
  const h = canvas.height = canvas.clientHeight;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = '#2a3872'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(40,h-30); ctx.lineTo(w-10,h-30); ctx.lineTo(w-10,20); ctx.stroke();
  const minY = Math.min(...yTrue, ...yPred);
  const maxY = Math.max(...yTrue, ...yPred);
  function xscale(v){ return 40 + ( (v - minY)/(maxY - minY + 1e-8) ) * (w-60); }
  function yscale(v){ return (h-30) - ( (v - minY)/(maxY - minY + 1e-8) ) * (h-60); }
  ctx.strokeStyle = '#6aa3ff'; ctx.beginPath();
  ctx.moveTo(xscale(minY), yscale(minY)); ctx.lineTo(xscale(maxY), yscale(maxY)); ctx.stroke();
  ctx.fillStyle = '#e7ecff';
  for (let i=0;i<yTrue.length;i++) {
    const x = xscale(yTrue[i]);
    const y = yscale(yPred[i]);
    ctx.beginPath(); ctx.arc(x,y,2,0,Math.PI*2); ctx.fill();
  }
  ctx.fillStyle = '#aab2d5'; ctx.font = '12px Inter, sans-serif';
  ctx.fillText('Actual', 44, h-10);
  ctx.save(); ctx.translate(w-6, 24); ctx.rotate(-Math.PI/2); ctx.fillText('Predicted', 0, 0); ctx.restore();
}
