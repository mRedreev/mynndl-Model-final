// tabular_loader.js — stratified split, K-fold target encoding, robust scaling, deterministic order, log(price)
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

function isNumeric(v){ if (v==null || v==='' || v==='?') return false; const x=Number(v); return Number.isFinite(x); }

export function analyzeColumns(rows, targetName='price') {
  const headers = Object.keys(rows[0]);
  const numCols=[], catCols=[];
  for (const h of headers) {
    if (h===targetName) continue;
    let cnt=0, num=0;
    for (const r of rows) { cnt++; if (isNumeric(r[h])) num++; }
    (num/cnt >= 0.8 ? numCols : catCols).push(h);
  }
  const catMaps={};
  for (const h of catCols) {
    const set=new Set();
    for (const r of rows) set.add((r[h] && r[h] !== '?') ? r[h] : '__NA__');
    const arr=[...set].sort();
    const map={ '__UNK__':0 }; arr.forEach((v,i)=> map[v]=i+1);
    catMaps[h]={ map, size:arr.length+1 };
  }
  return { numCols, catCols, catMaps };
}

// ---------- helpers ----------
function robustStats(arr){
  const s=[...arr].sort((a,b)=>a-b);
  const mid=Math.floor(s.length/2);
  const median = s.length%2 ? s[mid] : (s[mid-1]+s[mid])/2;
  const absDev = s.map(x=>Math.abs(x-median)).sort((a,b)=>a-b);
  const mad = absDev.length%2 ? absDev[mid] : (absDev[mid-1]+absDev[mid])/2;
  const scale = (1.4826*mad) || 1;
  return {median, scale};
}
function quantile(a, q){ const s=[...a].sort((x,y)=>x-y); const p=(s.length-1)*q; const k=Math.floor(p); const d=p-k; return s[k]+d*((s[k+1]??s[k]) - s[k]); }
function clip(v, lo, hi){ return Math.min(hi, Math.max(lo, v)); }
function shuffleWithSeed(arr, seed=42){
  function rnd(){ seed=(seed*1664525+1013904223)>>>0; return seed/4294967296; }
  return arr.map(x=>[rnd(),x]).sort((a,b)=>a[0]-b[0]).map(p=>p[1]);
}
// stratify by "make"
function stratifiedSplit(rows, frac=0.8, seed=42, key='make'){
  const groups=new Map();
  rows.forEach((r,i)=> {
    const k = (r[key] && r[key] !== '?') ? r[key] : '__NA__';
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(i);
  });
  const train=[], test=[];
  for (const idxs of groups.values()){
    const s = shuffleWithSeed([...idxs], seed);
    const cut = Math.max(1, Math.floor(s.length*frac));
    train.push(...s.slice(0,cut));
    test.push(...s.slice(cut));
  }
  return {train, test};
}

// ---------- main ----------
export function buildTabularTensors(rows, schema, trainFrac=0.8) {
  const seeds = schema.__seeds ?? { split:42, kfold:1337, trainOrder:777 };

  // 1) stratified split
  const {train: trIdx, test: teIdx} = stratifiedSplit(rows, trainFrac, seeds.split, 'make');

  // 2) numeric matrix + median imputation
  const rawNum = rows.map(r=> schema.numCols.map(h=> isNumeric(r[h]) ? Number(r[h]) : NaN ));
  const medians = schema.numCols.map((_,j)=> {
    const vals = rawNum.map(row=>row[j]).filter(Number.isFinite).sort((a,b)=>a-b);
    const k = Math.floor(vals.length/2);
    return vals.length? (vals.length%2? vals[k] : (vals[k-1]+vals[k])/2) : 0;
  });
  const Xnum = rawNum.map(row=> row.map((x,j)=> Number.isFinite(x)? x : medians[j]));

  // 3) categorical indices (full length)
  const Xcats = schema.catCols.map(h => rows.map(r=> {
    const v = (r[h] && r[h] !== '?') ? r[h] : '__NA__';
    return schema.catMaps[h].map[v] ?? 0;
  }));

  // 4) price winsorization on train + log1p
  const pricesRaw = rows.map(r=> isNumeric(r['price']) ? Number(r['price']) : NaN);
  const trainPrices = trIdx.map(i=> pricesRaw[i]).filter(Number.isFinite);
  const pLo = quantile(trainPrices, 0.05), pHi = quantile(trainPrices, 0.95);
  const pricesClipped = pricesRaw.map(v=> Number.isFinite(v) ? clip(v, pLo, pHi) : NaN);
  const yLog = pricesClipped.map(v=> Number.isFinite(v)? Math.log1p(v) : NaN);

  // 5) K-fold target encoding for impactful categoricals (no leakage)
  const encCols = ['make','body-style','drive-wheels','engine-type','fuel-system','aspiration','num-of-doors'];
  const K=5;
  const folds = (()=>{ const s = shuffleWithSeed([...trIdx], seeds.kfold); const arr = Array.from({length:K}, ()=>[]); s.forEach((i,k)=> arr[k%K].push(i)); return arr; })();
  const foldOf = new Map(); folds.forEach((a,f)=>a.forEach(i=>foldOf.set(i,f)));
  const globalMean = trainPrices.reduce((a,b)=>a+b,0)/Math.max(1,trainPrices.length);
  function smoothed(sum,cnt,base,alpha=10){ return (sum + alpha*base)/(cnt+alpha); }

  const encMapsByFold = {};
  for (const c of encCols) {
    encMapsByFold[c]=Array.from({length:K}, ()=>({}));
    for (let f=0; f<K; f++) {
      const trF = trIdx.filter(i=> foldOf.get(i)!==f);
      const sums=new Map(), counts=new Map();
      for (const i of trF) {
        const key = (rows[i][c] && rows[i][c] !== '?') ? rows[i][c] : '__NA__';
        const yv = pricesRaw[i]; if (!Number.isFinite(yv)) continue;
        sums.set(key,(sums.get(key)||0)+yv); counts.set(key,(counts.get(key)||0)+1);
      }
      const m={ '__UNK__': globalMean };
      for (const [k,sum] of sums) m[k]=smoothed(sum, counts.get(k)||0, globalMean, 10);
      encMapsByFold[c][f]=m;
    }
  }
  // full-train maps for test
  const fullMaps = {};
  for (const c of encCols) {
    const sums=new Map(), counts=new Map();
    for (const i of trIdx) {
      const key=(rows[i][c] && rows[i][c] !== '?') ? rows[i][c] : '__NA__';
      const yv=pricesRaw[i]; if (!Number.isFinite(yv)) continue;
      sums.set(key,(sums.get(key)||0)+yv); counts.set(key,(counts.get(key)||0)+1);
    }
    const m={ '__UNK__': globalMean };
    for (const [k,sum] of sums) m[k]=smoothed(sum, counts.get(k)||0, globalMean, 10);
    fullMaps[c]=m;
  }

  // 6) append TE columns to numeric matrix
  const XnumPlus = Xnum.map(r=> r.slice());
  function encValue(i, col){
    const key=(rows[i][col] && rows[i][col] !== '?') ? rows[i][col] : '__NA__';
    if (foldOf.has(i)) { const f=foldOf.get(i); const m=encMapsByFold[col][f]; return m[key] ?? m['__UNK__']; }
    const m=fullMaps[col]; return m[key] ?? m['__UNK__'];
  }
  for (let i=0;i<rows.length;i++){
    for (const c of encCols) XnumPlus[i].push(encValue(i,c));
  }

  // expose final numeric width for model input
  schema.numInputDim = XnumPlus[0].length;

  // 7) robust-scale numerics (fit on training only)
  const trainNum = trIdx.map(i=> XnumPlus[i]);
  const stats = XnumPlus[0].map((_,j)=> robustStats(trainNum.map(r=> r[j])));
  const XnumScaled = XnumPlus.map(row => row.map((x,j)=> (x - stats[j].median) / stats[j].scale ));

  // 8) build final tensors — deterministic single shuffle for train
  const valid = yLog.map((v,i)=> Number.isFinite(v) ? i : -1).filter(i=> i>=0);
  const tr = trIdx.filter(i=> valid.includes(i));
  const te = teIdx.filter(i=> valid.includes(i));

  const trainOrder = shuffleWithSeed([...tr], seeds.trainOrder);

  // Train arrays in fixed order
  const XnumTrainArr = trainOrder.map(i=> XnumScaled[i]);
  const yTrainArr    = trainOrder.map(i=> yLog[i]);
  const XcatsTrainArrs = Xcats.map(col => trainOrder.map(i=> col[i]));

  // Test arrays (natural order)
  const XnumTestArr  = te.map(i=> XnumScaled[i]);
  const yTestArr     = te.map(i=> yLog[i]);
  const XcatsTestArrs= Xcats.map(col => te.map(i=> col[i]));

  // tensors
  const XnumTrainT = tf.tensor2d(XnumTrainArr);
  const XnumTestT  = tf.tensor2d(XnumTestArr);
  const yTrainT    = tf.tensor2d(yTrainArr, [yTrainArr.length,1]);
  const yTestT     = tf.tensor2d(yTestArr,  [yTestArr.length,1]);
  const XcatsTrainT= XcatsTrainArrs.map(col => tf.tensor2d(col, [col.length,1], 'int32'));
  const XcatsTestT = XcatsTestArrs.map(col => tf.tensor2d(col, [col.length,1], 'int32'));

  return { Xtrain:[XnumTrainT,...XcatsTrainT], Xtest:[XnumTestT,...XcatsTestT], ytrain:yTrainT, ytest:yTestT, numStats:stats };
}
