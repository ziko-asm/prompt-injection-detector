const NAMES = {"keyword":"Keyword rule","tfidf":"TF-IDF + LR","transformer":"Transformer","cascade":"Cascade",
  "ollama:llama3.2:3b":"Llama 3.2 3B","ollama:llama3.1:8b":"Llama 3.1 8B"};
const ORDER = ["keyword","tfidf","transformer","cascade","ollama:llama3.2:3b","ollama:llama3.1:8b"];
const SET_COLOR = {"in-domain":"var(--in)","cross-dataset":"var(--cross)","combined":"var(--comb)","zero-shot":"var(--zero)"};
const SET_LABEL = {"in-domain":"In-domain","cross-dataset":"Cross-dataset","combined":"Combined","zero-shot":"Zero-shot"};
const f3 = x => (x==null||isNaN(x)) ? "–" : x.toFixed(3);
const pct = x => Math.round(x*100) + "%";
const ms = x => x < 0.1 ? x.toFixed(2)+" ms" : x < 10 ? x.toFixed(1)+" ms" : x < 1000 ? Math.round(x)+" ms" : (x/1000).toFixed(1)+" s";
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs={}, parent) => { const e=document.createElementNS(NS,tag); for (const k in attrs) e.setAttribute(k,attrs[k]); if(parent) parent.appendChild(e); return e; };
const tip = document.getElementById("tip");
function showTip(evt, html){ tip.innerHTML=html; tip.style.opacity=1; const x=Math.min(evt.clientX+14, innerWidth-tip.offsetWidth-8); tip.style.left=x+"px"; tip.style.top=(evt.clientY+14)+"px"; }
function hideTip(){ tip.style.opacity=0; }

fetch("results/results.json").then(r=>r.json()).then(render).catch(e=>{
  document.getElementById("tiles").innerHTML='<p class="err">Could not load results/results.json ('+e+'). Run python -m src.evaluate first.</p>';
});

function render(R){
  const rows = R.results, meta = R.meta;
  const get = (d,tr,te) => rows.find(r=>r.detector===d && r.train===tr && r.test===te);
  const tests = meta.datasets.map(d=>d.dataset);
  const mean = a => a.reduce((s,x)=>s+x,0)/a.length;
  document.getElementById("gen").textContent = "Results generated " + meta.generated_at.replace("T"," ").replace("+00:00"," UTC");

  // ---------------- tiles
  const trained = ["keyword","tfidf","transformer"].filter(d=>rows.some(r=>r.detector===d));
  const inF1 = mean(rows.filter(r=>r.setting==="in-domain" && r.detector==="transformer").map(r=>r.f1));
  const crossF1 = mean(rows.filter(r=>r.setting==="cross-dataset" && r.detector==="transformer").map(r=>r.f1));
  const worstFPR = Math.max(...rows.filter(r=>r.setting==="cross-dataset").map(r=>r.false_positive_rate));
  const llm = rows.filter(r=>r.detector==="ollama:llama3.1:8b");
  const tr = rows.filter(r=>r.detector==="transformer");
  const speed = mean(llm.map(r=>r.ms_per_text)) / mean(tr.map(r=>r.ms_per_text));
  const tiles = [
    ["Transformer F1, same dataset", f3(inF1), "Mean over both datasets."],
    ["Same transformer, other dataset", f3(crossF1), "Trained on one source, tested on the other."],
    ["Worst false positive rate", pct(worstFPR), "Harmless texts blocked by a cross-dataset detector."],
    ["LLM judge vs transformer, time", Math.round(speed/100)*100 + "×", "How much slower Llama 3.1 8B is per text than the fine-tuned transformer."],
  ];
  document.getElementById("tiles").innerHTML = tiles.map(t=>`<div class="tile"><p class="k">${t[0]}</p><p class="v">${t[1]}</p><p class="n">${t[2]}</p></div>`).join("");

  // ---------------- findings
  const [A,B] = tests;
  const tCombA = get("transformer","all",A), tCombB = get("transformer","all",B);
  const tInA = get("transformer",A,A);
  const lA = get("ollama:llama3.1:8b","none",A), lB = get("ollama:llama3.1:8b","none",B);
  const tCrossB = get("transformer",A,B), tCrossA = get("transformer",B,A);
  const cas = rows.filter(r=>r.detector==="cascade");
  const casShareIn = cas.length ? mean(cas.filter(r=>r.setting!=="cross-dataset").map(r=>r.llm_share)) : null;
  const cAB = get("cascade",A,B), cBA = get("cascade",B,A);
  const cards = [
    ["Every trained detector overfits its source",
     `Trained on ${A} and tested on ${B}, the transformer flags ${pct(tCrossB.recall)} of injections but also ${pct(tCrossB.false_positive_rate)} of harmless texts. In the other direction it catches only ${pct(tCrossA.recall)} of injections. TF-IDF and the keyword rule fail the same way.`],
    ["Training on both sources closes most of the gap",
     `The combined transformer scores ${f3(tCombA.f1)} F1 on ${A} and ${f3(tCombB.f1)} on ${B}, with no false-positive blow-up. It pays a little on ${A} (${f3(tInA.f1)} when trained on ${A} alone), and it has still only seen two sources.`],
    ["Only the LLM transfers without training",
     `Llama 3.1 8B, never trained, reaches ${f3(lB.f1)} F1 on ${B} with a ${pct(lB.false_positive_rate)} false positive rate. On ${A}, with short and partly German texts, it finds only ${pct(lA.recall)} of injections. Llama 3.2 3B is far weaker.`],
    ["A confident wrong model cannot be rescued by a cascade",
     casShareIn==null ? "Run the cascade to fill this card." :
     `In-domain and combined, the cascade sends only ${pct(casShareIn)} of texts to the LLM and matches the transformer. Trained on ${A} and tested on ${B}, the transformer is confidently wrong: only ${pct(cAB.llm_share)} of texts reach the LLM and the false positives stay (${pct(cAB.false_positive_rate)}). In the other direction ${pct(cBA.llm_share)} reach the LLM and F1 rises only from ${f3(tCrossA.f1)} to ${f3(cBA.f1)}. Routing on confidence cannot spot data the model has never seen.`],
  ];
  document.getElementById("find").innerHTML = cards.map(c=>`<div class="tile"><h3>${c[0]}</h3><p>${c[1]}</p></div>`).join("");

  // ---------------- gap chart (dot plot, one facet per test set)
  const gap = document.getElementById("gapChart");
  const dets = ORDER.filter(d=>rows.some(r=>r.detector===d));
  tests.forEach(test=>{
    const box = document.createElement("div"); box.className="facet";
    box.innerHTML = `<h3>Tested on ${test}</h3>`; gap.appendChild(box);
    const W=520, rowH=38, top=10, left=118, right=16, H=top+dets.length*rowH+34;
    const svg = el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%",role:"img","aria-label":`F1 by detector, tested on ${test}`},box);
    const x = v => left + v*(W-left-right);
    [0,.25,.5,.75,1].forEach(t=>{
      el("line",{x1:x(t),x2:x(t),y1:top,y2:H-28,stroke:"var(--grid)","stroke-width":1},svg);
      const tx=el("text",{x:x(t),y:H-10,"text-anchor":"middle","font-size":12,fill:"var(--muted)"},svg); tx.textContent=t.toFixed(2).replace(/\.?0+$/,"")||"0";
    });
    const ax=el("text",{x:W-right,y:H-10,"text-anchor":"end","font-size":12,fill:"var(--muted)"},svg); ax.textContent="";
    dets.forEach((d,i)=>{
      const cy = top + i*rowH + rowH/2;
      const lab=el("text",{x:left-12,y:cy+4,"text-anchor":"end","font-size":13,fill:"var(--ink-2)"},svg); lab.textContent=NAMES[d]||d;
      const pts = rows.filter(r=>r.detector===d && r.test===test);
      if (pts.length>1){ const xs=pts.map(p=>x(p.f1)); el("line",{x1:Math.min(...xs),x2:Math.max(...xs),y1:cy,y2:cy,stroke:"var(--line)","stroke-width":2},svg); }
      pts.forEach(p=>{
        const zero = p.setting==="zero-shot";
        const c = el("circle",{cx:x(p.f1),cy,r:zero?5:6,fill:zero?"var(--surface)":SET_COLOR[p.setting],stroke:zero?"var(--zero)":"var(--surface)","stroke-width":zero?2.2:2},svg);
        const hit = el("circle",{cx:x(p.f1),cy,r:13,fill:"transparent",tabindex:0},svg);
        const html = `<b>${NAMES[d]}</b> · ${SET_LABEL[p.setting]}<br>train ${p.train} → test ${p.test}<br>F1 <b>${f3(p.f1)}</b> · recall ${f3(p.recall)}<br>precision ${f3(p.precision)} · FPR ${f3(p.false_positive_rate)}`;
        hit.addEventListener("mousemove",e=>showTip(e,html)); hit.addEventListener("mouseleave",hideTip);
        hit.addEventListener("focus",()=>{const b=c.getBoundingClientRect(); showTip({clientX:b.right,clientY:b.bottom},html);}); hit.addEventListener("blur",hideTip);
      });
    });
    const xl=el("text",{x:left,y:H-10,"font-size":12,fill:"var(--muted)","text-anchor":"start"},svg); xl.textContent="";
  });

  // ---------------- cost chart (log x)
  const pts = dets.map(d=>{
    const rs = rows.filter(r=>r.detector===d && (r.setting==="combined" || r.setting==="zero-shot"));
    return rs.length ? {d, f1:mean(rs.map(r=>r.f1)), ms:mean(rs.map(r=>r.ms_per_text))} : null;
  }).filter(Boolean);
  const cc = document.getElementById("costChart");
  const W=1000, H=380, L=56, Rr=30, T=16, Bm=46;
  const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...pts.map(p=>p.ms))))), hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...pts.map(p=>p.ms)))));
  const X = v => L + (Math.log10(v)-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo))*(W-L-Rr);
  const Y = v => T + (1-v)*(H-T-Bm);
  const svg = el("svg",{viewBox:`0 0 ${W} ${H}`,width:"100%",role:"img","aria-label":"Mean F1 against milliseconds per text"},cc);
  [0,.25,.5,.75,1].forEach(t=>{ el("line",{x1:L,x2:W-Rr,y1:Y(t),y2:Y(t),stroke:"var(--grid)"},svg); const tx=el("text",{x:L-10,y:Y(t)+4,"text-anchor":"end","font-size":12,fill:"var(--muted)"},svg); tx.textContent=t.toFixed(2); });
  for (let v=lo; v<=hi; v*=10){ el("line",{x1:X(v),x2:X(v),y1:T,y2:H-Bm,stroke:"var(--grid)"},svg); const tx=el("text",{x:X(v),y:H-Bm+20,"text-anchor":"middle","font-size":12,fill:"var(--muted)"},svg); tx.textContent=ms(v); }
  const yt=el("text",{x:L,y:T-2,"font-size":12,fill:"var(--muted)"},svg); yt.textContent="";
  const xt=el("text",{x:W-Rr,y:H-6,"text-anchor":"end","font-size":12,fill:"var(--muted)"},svg); xt.textContent="time per text (log scale) →";
  const ytl=el("text",{x:14,y:(H-Bm)/2,"font-size":12,fill:"var(--muted)",transform:`rotate(-90 14 ${(H-Bm)/2})`,"text-anchor":"middle"},svg); ytl.textContent="mean F1";
  pts.forEach(p=>{
    const llmPt = p.d.startsWith("ollama");
    el("circle",{cx:X(p.ms),cy:Y(p.f1),r:7,fill:llmPt?"var(--surface)":"var(--in)",stroke:llmPt?"var(--zero)":"var(--surface)","stroke-width":2.2},svg);
    const anchorEnd = X(p.ms) > W*0.8;
    const below = pts.some(q=>q!==p && q.ms>p.ms && Math.abs(X(q.ms)-X(p.ms))<140 && Math.abs(Y(q.f1)-Y(p.f1))<30);
    const lb=el("text",{x:X(p.ms)+(anchorEnd?-12:12),y:Y(p.f1)+(below?22:-10),"font-size":13,fill:"var(--ink)","text-anchor":anchorEnd?"end":"start"},svg); lb.textContent=NAMES[p.d];
    const hit=el("circle",{cx:X(p.ms),cy:Y(p.f1),r:16,fill:"transparent",tabindex:0},svg);
    const html=`<b>${NAMES[p.d]}</b> (${llmPt?"zero-shot":"combined training"})<br>mean F1 <b>${f3(p.f1)}</b><br>${ms(p.ms)} per text`;
    hit.addEventListener("mousemove",e=>showTip(e,html)); hit.addEventListener("mouseleave",hideTip);
  });

  // ---------------- cascade table
  if (cas.length){
    const head = "<thead><tr><th>Train</th><th>Test</th><th>Setting</th><th>Transformer F1</th><th>Cascade F1</th><th>Sent to LLM</th><th>Cascade time</th></tr></thead>";
    const body = cas.map(c=>{ const t=get("transformer",c.train,c.test);
      return `<tr><td>${c.train}</td><td>${c.test}</td><td><span class="chip"><i class="sw" style="background:${SET_COLOR[c.setting]}"></i>${SET_LABEL[c.setting]}</span></td><td>${f3(t&&t.f1)}</td><td>${f3(c.f1)}</td><td>${pct(c.llm_share)}</td><td>${ms(c.ms_per_text)}</td></tr>`;}).join("");
    document.getElementById("cascadeTbl").innerHTML = head + "<tbody>"+body+"</tbody>";
    document.getElementById("cascadeNote").textContent = "The cascade retrains its own transformer, so small differences from the Transformer column are training noise (one seed). LLM time is the latency measured when each answer was first computed.";
  }

  // ---------------- all results table with filter
  const settings = ["all","in-domain","cross-dataset","combined","zero-shot"];
  const fbox = document.getElementById("filters");
  let cur = "all";
  const draw = () => {
    const rs = rows.filter(r=>cur==="all"||r.setting===cur).sort((a,b)=>ORDER.indexOf(a.detector)-ORDER.indexOf(b.detector));
    document.getElementById("allTbl").innerHTML = "<thead><tr><th>Detector</th><th>Train</th><th>Test</th><th>Setting</th><th>Precision</th><th>Recall</th><th>F1</th><th>FPR</th><th>AUC</th><th>Time / text</th></tr></thead><tbody>" +
      rs.map(r=>`<tr><td>${NAMES[r.detector]||r.detector}</td><td>${r.train}</td><td>${r.test}</td><td><span class="chip">${r.setting==="zero-shot"?'<i class="sw ring"></i>':`<i class="sw" style="background:${SET_COLOR[r.setting]}"></i>`}${SET_LABEL[r.setting]}</span></td><td>${f3(r.precision)}</td><td>${f3(r.recall)}</td><td>${f3(r.f1)}</td><td>${f3(r.false_positive_rate)}</td><td>${f3(r.roc_auc)}</td><td>${ms(r.ms_per_text)}</td></tr>`).join("") + "</tbody>";
  };
  settings.forEach(s=>{ const b=document.createElement("button"); b.textContent = s==="all"?"All":SET_LABEL[s]; b.setAttribute("aria-pressed", s===cur);
    b.onclick=()=>{cur=s; [...fbox.children].forEach(c=>c.setAttribute("aria-pressed", c===b)); draw();}; fbox.appendChild(b); });
  draw();

  // ---------------- data list
  const SRC = {deepset:"https://huggingface.co/datasets/deepset/prompt-injections", jackhhao:"https://huggingface.co/datasets/jackhhao/jailbreak-classification"};
  document.getElementById("dataList").innerHTML = meta.datasets.map(d=>
    `<li><a href="${SRC[d.dataset]||"#"}">${d.dataset}</a>: ${d.rows.toLocaleString("en")} unique texts, ${d.test_rows} in test, ${pct(d.test_injection_share)} injections, median ${d.median_chars} characters.</li>`).join("") +
    "<li>Duplicates and empty texts removed; train and test never share a text.</li>";
}
