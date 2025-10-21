// tabular_loader.js — build numeric z-scored features and categorical indices; target is log(price).
export async function parseCarsCSV(file) {
  const text = await file.text();
  const sep = text.includes(';') && !text.includes(',') ? ';' : ',';
  const lines = text.trim().split(/\r?\n/);
  const header = lines.shift().split(sep).map(s=>s.trim());
  const col = Object.fromEntries(header.map((h,i)=>[h,i]));
  if (!('price' in col)) throw new Error("Column 'price' not found");
  const rows = lines.map(line => {
    const cells = line.split(sep).map(s=>s.trim());
    const obj = {};
    header.forEach((h,i)=> obj[h] = cells[i] ?? '');
    return obj;
  });
  return rows;
}

function isNumeric(v) {
  if (v==null || v==='' || v==='?') return false;
  const x = Number(v);
  return Number.isFinite(x);
}

export function analyzeColumns(rows, targetName='price') {
  const headers = Object.keys(rows[0]);
  const numCols = [], catCols = [];
  for (const h of headers) {
    if (h===targetName) continue;
    let cnt=0, num=0;
    for (const r of rows) { cnt++; if (isNumeric(r[h])) num++; }
    if (num/cnt >= 0.8) numCols.push(h); else catCols.push(h);
  }
  const catMaps = {};
  for (const h of catCols) {
    const set = new Set();
    for (const r of rows) set.add((r[h] && r[h] !== '?') ? r[h] : '__NA__');
    const arr = Array.from(set.values()).sort();
    const map = {}; map['__UNK__'] = 0;
    arr.forEach((v, i)=> { map[v] = i+1; });
    catMaps[h] = { map, size: arr.length + 1 };
  }
  return { numCols, catCols, catMaps };
}

function zScore(arr) {
  const m = arr.reduce((a,b)=>a+b,0)/arr.length;
  const v = arr.reduce((a,b)=>a + (b-m)*(b-m),0)/arr.length;
  const s = Math.sqrt(v) || 1;
  return {mean: m, std: s};
}

export function buildTabularTensors(rows, schema, trainFrac=0.8) {
  const numeric = rows.map(r=> schema.numCols.map(h=> isNumeric(r[h]) ? Number(r[h]) : NaN ));
  const colMedians = schema.numCols.map((_,j)=> {
    const vals = numeric.map(row => row[j]).filter(x=> Number.isFinite(x)).sort((a,b)=>a-b);
    const k = Math.floor(vals.length/2);
    return vals.length? (vals.length%2? vals[k] : (vals[k-1]+vals[k])/2) : 0;
  });
  const Xnum = numeric.map(row => row.map((x,j)=> Number.isFinite(x)? x : colMedians[j]));

  const idx = [...Xnum.keys()];
  let seed=42; function rnd(){ seed=(seed*1664525+1013904223)%4294967296; return seed/4294967296; }
  idx.sort((a,b)=> rnd()-0.5);
  const split = Math.floor(idx.length*trainFrac);
  const trIdx = idx.slice(0,split), teIdx = idx.slice(split);

  const trainNum = trIdx.map(i=> Xnum[i]);
  const stats = schema.numCols.map((_,j)=> zScore(trainNum.map(r=> r[j])));
  const XnumScaled = Xnum.map(row => row.map((x,j)=> (x - stats[j].mean) / stats[j].std ));

  const Xcats = schema.catCols.map(h => rows.map(r=> {
    const v = (r[h] && r[h] !== '?') ? r[h] : '__NA__';
    return schema.catMaps[h].map[v] ?? 0;
  }));

  const y = rows.map(r=> isNumeric(r['price']) ? Math.log1p(Number(r['price'])) : NaN );
  const validIdx = y.map((v,i)=> Number.isFinite(v) ? i : -1).filter(i=> i>=0);

  function pick(indexes, arr){ return indexes.map(i=> arr[i]); }

  const XnumAll = pick(validIdx, XnumScaled);
  const yAll = pick(validIdx, y);
  const XcatsAll = Xcats.map(col => pick(validIdx, col));

  const tr = trIdx.filter(i=> validIdx.includes(i));
  const te = teIdx.filter(i=> validIdx.includes(i));

  const XnumTrain = pick(tr, XnumAll);
  const XnumTest  = pick(te, XnumAll);
  const yTrain = pick(tr, yAll);
  const yTest  = pick(te, yAll);
  const XcatsTrain = XcatsAll.map(col => pick(tr, col));
  const XcatsTest  = XcatsAll.map(col => pick(te, col));

  const XnumTrainT = tf.tensor2d(XnumTrain);
  const XnumTestT  = tf.tensor2d(XnumTest);
  const yTrainT = tf.tensor2d(yTrain, [yTrain.length, 1]);
  const yTestT  = tf.tensor2d(yTest,  [yTest.length, 1]);
  const XcatsTrainT = XcatsTrain.map(col => tf.tensor2d(col, [col.length, 1], 'int32'));
  const XcatsTestT  = XcatsTest.map(col => tf.tensor2d(col, [col.length, 1], 'int32'));

  return {
    Xtrain: [XnumTrainT, ...XcatsTrainT],
    Xtest:  [XnumTestT,  ...XcatsTestT],
    ytrain: yTrainT,
    ytest:  yTestT,
    numStats: stats
  };
}
