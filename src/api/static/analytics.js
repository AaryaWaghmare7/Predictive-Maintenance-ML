"use strict";

// Small SVG charts, local-only: no CDN, charting framework, raw-data persistence or fitting.
(() => {
  const el = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const id = (name) => document.getElementById(name);
  const colors = ["#b9b8bd", "#D10721", "#ff8877", "#e7b5c5"];
  const shortNames = ["Voltage", "Current", "Motor speed", "Temperature", "Vibration", "Ambient temp", "Humidity"];
  const sources = new Map();
  let reference = null;
  let uploadSequence = 0;
  let activeView = "distributions";
  const select = id("analytics-source");
  const filter = id("analytics-class");
  const host = id("analytics-charts");
  const tooltip = id("analytics-tooltip");
  const number = (value, digits=3) => value === null || value === undefined || !Number.isFinite(value) ? "N/A" : value.toFixed(digits);
  const tick = (value, span) => number(value, Math.min(8,Math.max(2,Math.ceil(-Math.log10(span))+1)));

  function svgNode(tag, attrs={}, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([name, value]) => node.setAttribute(name, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function tip(node, text, keyboard=true) {
    node.appendChild(svgNode("title", {}, text));
    node.setAttribute("aria-label", text);
    if (keyboard) node.setAttribute("tabindex", "0");
    ["pointerenter", "focus"].forEach((event) => node.addEventListener(event, () => { tooltip.textContent = text; }));
    return node;
  }
  function chart(title, width=560, height=280) {
    const card = el("section", undefined, "analytics-chart");
    card.appendChild(el("h3", title));
    const svg = svgNode("svg", {viewBox:`0 0 ${width} ${height}`, role:"img", "aria-label":title, xmlns:"http://www.w3.org/2000/svg"});
    card.appendChild(svg); host.appendChild(card);
    return {card, svg};
  }
  function text(svg, x, y, value, attrs={}) {
    svg.appendChild(svgNode("text", {x, y, fill:"#c8c4cb", "font-size":11, ...attrs}, value));
  }
  function axes(svg, width, height, xLabel, yLabel) {
    svg.appendChild(svgNode("path", {d:`M48 20V${height-40}H${width-15}`, fill:"none", stroke:"#69636e"}));
    text(svg, width/2, height-6, xLabel, {"text-anchor":"middle"});
    text(svg, 12, 14, yLabel);
  }
  function bars(title, labels, values, options={}) {
    const height = Math.max(190, labels.length*35+35), width=560;
    const {svg, card} = chart(title, width, height);
    const min = Math.min(0, ...values.filter(Number.isFinite)), max = Math.max(.001, ...values.filter(Number.isFinite));
    const x = (value) => 130+(value-min)/(max-min)*365;
    labels.forEach((label, index) => {
      text(svg, 123, index*35+29, label, {"text-anchor":"end"});
      const value = values[index];
      if (value === null || value === undefined) { text(svg, 140, index*35+29, "N/A"); return; }
      svg.appendChild(tip(svgNode("rect", {x:Math.min(x(0),x(value)), y:index*35+15, width:value===0 ? 0 : Math.max(1,Math.abs(x(value)-x(0))), height:19,
        fill:options.colors?.[index] || "#D10721", rx:3}), options.tips?.[index] || `${label}: ${number(value,5)}${options.suffix || ""}`));
      text(svg, Math.min(525, x(value)+6), index*35+29, number(value,options.decimals ?? 3));
    });
    return card;
  }
  function group(data) { return data.groups[filter.value] || data.groups.all; }
  function source() { return sources.get(select.value); }
  function refreshSources(preferred) {
    const previous = preferred || select.value;
    select.replaceChildren();
    sources.forEach((entry, key) => { const option=el("option",entry.name); option.value=key; select.appendChild(option); });
    select.value = sources.has(previous) ? previous : (sources.keys().next().value || "");
    select.disabled = !sources.size;
    render();
  }
  function histogram(data) {
    const current = group(data);
    data.features.forEach((feature, index) => {
      const stats = current.features[feature], {svg, card} = chart(`${shortNames[index]} distribution`);
      if(!stats.count) {
        text(svg,280,130,"No finite numeric values available",{"text-anchor":"middle"});
        card.appendChild(el("p",`${stats.excluded_nonfinite} missing, nonnumeric or nonfinite cells excluded from this chart; no measurements were imputed.`,"small"));
        return;
      }
      const actual=stats.actual_range_histogram;
      const counts=actual ? actual.counts : [stats.below_zero,...stats.histogram,stats.above_one];
      const max=Math.max(1,...counts), step=490/counts.length;
      axes(svg,560,280,"Value as uploaded / original normalized scale","Observations");
      counts.forEach((count, bin) => {
        const label=actual ? `[${number(actual.edges[bin])}, ${number(actual.edges[bin+1])}${bin===counts.length-1 ? "]" : ")"}` : bin===0 ? "Below 0" : bin===36 ? "Above 1" : `[${number((bin-1)/35)}, ${number(bin/35)}${bin===35 ? "]" : ")"}`;
        svg.appendChild(tip(svgNode("rect",{x:49+bin*step,y:239-count/max*200,width:Math.max(1,step-1),height:count/max*200,fill:bin===0 || bin===36 ? "#ff8877" : "#D10721"}),`${feature} · ${label}: ${count} observations`));
      });
      text(svg,48,255,actual ? number(stats.min) : "<0"); if(!actual) text(svg,68,255,"0"); text(svg,510,255,actual ? number(stats.max) : "1 / >1");
      [0,.5,1].forEach(fraction=>text(svg,43,243-fraction*200,String(Math.round(max*fraction)),{"text-anchor":"end"}));
      card.appendChild(el("p",`Finite: ${stats.count}; excluded missing/nonfinite: ${stats.excluded_nonfinite}. Median ${number(stats.median)}; min/max ${number(stats.min)} / ${number(stats.max)}. ${actual ? "Bins span actual uploaded values; no scaling or unit conversion. Prediction compatibility remains separate." : "Under/overflow bins are included, not hidden."}`,"small"));
    });
  }
  function boxes(data) {
    if (!data.class_counts || !data.groups["0"]) {
      host.appendChild(el("p","Class boxplots require valid confirmed labels or explicitly selected model predictions. Unlabeled observations are not assigned invented categories.","analytics-notice")); return;
    }
    data.features.forEach((feature,index) => {
      const {svg,card}=chart(`${shortNames[index]} by category`);
      const all=data.groups.all.features[feature];
      const low=all.min ?? 0, high=Math.max(low+.001,all.max ?? 1);
      const y=(value)=>230-(value-low)/(high-low)*185;
      axes(svg,560,280,"Fault category (not severity)","Value");
      [0,.25,.5,.75,1].forEach(fraction=>{const value=low+(high-low)*fraction;text(svg,43,y(value)+4,tick(value,high-low),{"text-anchor":"end"});});
      Object.entries(data.class_names).forEach(([code,name],position)=>{
        if (filter.value!=="all" && code!==filter.value) return;
        const stats=data.groups[code]?.features[feature];
        const x=105+position*120;
        text(svg,x,255,name.replace(" Fault",""),{"text-anchor":"middle"});
        if (!stats?.count) { text(svg,x,145,"No rows",{"text-anchor":"middle"}); return; }
        const box=svgNode("g");
        box.appendChild(svgNode("path",{d:`M${x} ${y(stats.min)}V${y(stats.max)}M${x-15} ${y(stats.min)}H${x+15}M${x-15} ${y(stats.max)}H${x+15}`,stroke:colors[position]}));
        box.appendChild(svgNode("rect",{x:x-28,y:y(stats.q3),width:56,height:Math.max(1,y(stats.q1)-y(stats.q3)),fill:colors[position],"fill-opacity":.55,stroke:colors[position]}));
        box.appendChild(svgNode("path",{d:`M${x-28} ${y(stats.median)}H${x+28}`,stroke:"#F5F5F5","stroke-width":2}));
        svg.appendChild(tip(box,`${name} · ${feature}: n=${stats.count}, min=${number(stats.min)}, Q1=${number(stats.q1)}, median=${number(stats.median)}, Q3=${number(stats.q3)}, max=${number(stats.max)}`));
      });
      card.appendChild(el("p","Boxes show Q1–Q3 and median; whiskers show the observed min/max. No outlier rows are removed.","small"));
    });
  }
  function heatmap(data) {
    const current=group(data), {svg,card}=chart("Sensor correlation · Pearson r",660,430);
    const size=45, left=140, top=65;
    shortNames.forEach((name,i)=>{
      text(svg,left-8,top+i*size+28,name,{"text-anchor":"end"});
      text(svg,left+i*size+12,top-10,String(i+1),{"text-anchor":"middle"});
      current.correlation[i].forEach((value,j)=>{
        const color=value===null ? "#27252b" : value>=0 ? `rgb(${70+Math.round(value*139)},${57-Math.round(value*35)},${65-Math.round(value*20)})` : `rgb(${65-Math.round(-value*25)},${65+Math.round(-value*40)},${80+Math.round(-value*95)})`;
        const node=svgNode("g");
        node.appendChild(svgNode("rect",{x:left+j*size,y:top+i*size,width:size-2,height:size-2,fill:color}));
        node.appendChild(svgNode("text",{x:left+j*size+size/2,y:top+i*size+28,fill:"#F5F5F5","text-anchor":"middle","font-size":12},value===null ? "—" : value.toFixed(2)));
        svg.appendChild(tip(node,`${data.features[i]} × ${data.features[j]}: r=${number(value,5)}; paired n=${current.pair_counts[i][j]}. ${value===null ? "Undefined: insufficient rows or constant values." : "Correlation is not causation."}`));
      });
    });
    card.appendChild(el("p",shortNames.map((name,i)=>`${i+1} ${name}`).join(" · ")+". Red: positive; blue: negative; −1 to +1. Null cells are undefined, not zero correlation; hover or focus for pair counts.","small"));
  }
  function scatter(data) {
    const first=Number(id("scatter-x").value), second=Number(id("scatter-y").value);
    if (first===second) { host.appendChild(el("p","Choose two different sensors for a feature-versus-feature scatterplot.","analytics-notice")); return; }
    const limits=["scatter-x-min","scatter-x-max","scatter-y-min","scatter-y-max"].map(name=>id(name).value.trim()==="" ? NaN : Number(id(name).value));
    const [xmin,xmax,ymin,ymax]=limits;
    if (!limits.every(Number.isFinite) || xmax<=xmin || ymax<=ymin) {
      host.appendChild(el("p","Scatter ranges must be finite, with minimum smaller than maximum.","analytics-notice")); return;
    }
    const {svg,card}=chart(`${shortNames[first]} × ${shortNames[second]}`,660,380);
    axes(svg,660,380,`${shortNames[first]}: ${number(xmin)} to ${number(xmax)}`,`${shortNames[second]}: ${number(ymin)} to ${number(ymax)}`);
    const x=value=>48+(value-xmin)/(xmax-xmin)*590;
    const y=value=>340-(value-ymin)/(ymax-ymin)*305;
    [0,.25,.5,.75,1].forEach(fraction=>{
      const a=xmin+(xmax-xmin)*fraction, b=ymin+(ymax-ymin)*fraction;
      text(svg,x(a),356,tick(a,xmax-xmin),{"text-anchor":"middle"});text(svg,43,y(b)+4,tick(b,ymax-ymin),{"text-anchor":"end"});
    });
    let visible=0;
    if (data.scatter?.kind==="sample") {
      data.scatter.points.forEach(point=>{
        if (filter.value!=="all" && point.class!==filter.value) return;
        const a=point.values[first],b=point.values[second];
        if (a===null || b===null || a<xmin || a>xmax || b<ymin || b>ymax) return;
        visible++;
        svg.appendChild(tip(svgNode("circle",{cx:x(a),cy:y(b),r:3,fill:colors[Number(point.class)] || "#8faabc","fill-opacity":.65}),`Observation ${point.row}: ${data.features[first]}=${a}; ${data.features[second]}=${b}; ${data.class_names[point.class] || "Unlabeled"}`,false));
      });
    } else if (data.scatter?.kind==="binned_reference" && first!==second) {
      const reverse=first>second, key=reverse ? `${second}:${first}` : `${first}:${second}`;
      Object.entries(data.scatter.pairs[key]).forEach(([code,bins])=>{
        if (filter.value!=="all" && code!==filter.value) return;
        bins.forEach(([i,j,count])=>{
          const a=((reverse ? j : i)+.5)/data.scatter.bins, b=((reverse ? i : j)+.5)/data.scatter.bins;
          if (a<xmin || a>xmax || b<ymin || b>ymax) return;
          visible++;
          svg.appendChild(tip(svgNode("circle",{cx:x(a),cy:y(b),r:Math.min(8,2+Math.sqrt(count)/2),fill:colors[Number(code)],"fill-opacity":.55}),`${data.class_names[code]}: ${count} original training observations in joint bin centered (${number(a)}, ${number(b)}); not an exact sensor row`,false));
        });
      });
    }
    card.appendChild(el("p",`${visible} visible ${data.scatter?.kind==="binned_reference" ? "occupied class/bin marks" : "sampled observations"}. ${data.scatter?.note || "No paired scatter data in this reference view; select training or an upload."} Edit range controls, then Apply plot ranges to zoom; Reset restores data bounds.`,"small"));
    const legend=el("p",undefined,"small");
    if (data.class_counts) Object.entries(data.class_names).forEach(([code,name])=>{
      const item=el("span",`● ${name}  `); item.style.color=colors[Number(code)]; legend.appendChild(item);
    });
    else legend.textContent="Blue marks: unlabeled observations (no inferred categories).";
    card.appendChild(legend);
  }
  function evidence(data, entry) {
    const association=data.associations || {};
    if (Object.keys(association).length) {
      bars(`Between-class association · eta squared · ${data.label_basis}`,shortNames,data.features.map(f=>association[f]?.eta_squared ?? null));
      host.appendChild(el("p","Eta squared describes class-mean separation. Association with predicted labels describes the classifier's own grouping, not independent fault evidence.","small"));
    } else host.appendChild(el("p","No feature–target association is calculated without usable labels and at least two classes.","small"));
    if (!reference) return;
    host.appendChild(el("h3","Original model evidence — never recomputed from your upload"));
    bars("Original training-only nonlinear association · MI (nats)",shortNames,data.features.map(f=>reference.historical_associations[f].mutual_information_nats));
    const importance=reference.historical_importance;
    const card=bars(`Historical validation permutation importance · ${importance.validation_rows} rows`,shortNames,data.features.map(f=>importance.permutation_macro_f1_drop[f].mean));
    card.appendChild(el("p","Mean macro-F1 decrease after permutation. Standard deviations are listed below; correlated predictors and dataset simplicity affect interpretation. This is not a causal explanation of an uploaded observation.","small"));
    card.appendChild(el("p",data.features.map((f,i)=>`${shortNames[i]}: ${number(importance.permutation_macro_f1_drop[f].mean,4)} ± ${number(importance.permutation_macro_f1_drop[f].std,4)}`).join(" · "),"small"));
    bars("Saved final Random Forest · impurity importance",shortNames,data.features.map(f=>reference.final_model_impurity_importance[f]));
  }
  function comparison(data) {
    if(source()?.origin==="reference") { host.appendChild(el("p","Select an uploaded source to compare with original training. Reference data is not an uploaded vehicle.","analytics-notice")); return; }
    const current=data.comparison;
    if (!current?.available) { host.appendChild(el("p",current?.reason || "Training comparison is shown for uploads only.","analytics-notice")); return; }
    host.appendChild(el("p",current.method+" "+current.limitation,"small"));
    const wrapper=el("div",undefined,"table-scroll"), table=el("table"), head=el("thead"), header=el("tr");
    ["Sensor","Training min–max","Upload min–max","Training / upload median","Outside training range","Histogram distance","Review"].forEach(name=>header.appendChild(el("th",name)));
    head.appendChild(header); table.appendChild(head); const body=el("tbody");
    data.features.forEach((feature,index)=>{
      const item=current.features[feature], row=el("tr");
      [shortNames[index],`${number(item.training_min)}–${number(item.training_max)}`,`${number(item.observed_min)}–${number(item.observed_max)}`,`${number(item.training_median)} / ${number(item.observed_median)}`,`${item.outside_training_range} / ${item.finite_rows}`,number(item.histogram_total_variation),item.shape_review ? "Shape differs — review" : item.finite_rows<30 ? "Small sample; no shape alert" : "No shape alert"].forEach(value=>row.appendChild(el("td",value)));
      body.appendChild(row);
    }); table.appendChild(body); wrapper.appendChild(table); host.appendChild(wrapper);
    host.appendChild(el("p","Table bounds are rounded for readability; the JSON download retains exact numeric values. This comparison covers the whole upload, not a class-filtered subset.","small"));
    const selected=Number(id("scatter-x").value), feature=data.features[selected];
    const train=reference?.datasets.training.groups.all.features[feature], upload=data.groups.all.features[feature];
    if (!train) return;
    const {svg,card}=chart(`${shortNames[selected]} · upload versus training proportions`);
    const a=[train.below_zero,...train.histogram,train.above_one].map(v=>train.count ? v/train.count : 0);
    const b=[upload.below_zero,...upload.histogram,upload.above_one].map(v=>upload.count ? v/upload.count : 0);
    const max=Math.max(.001,...a,...b), step=490/a.length;
    axes(svg,560,280,"Shared normalized bins; first/last are under/overflow","Proportion");
    a.forEach((value,index)=>{
      [value,b[index]].forEach((v,j)=>svg.appendChild(tip(svgNode("rect",{x:49+index*step+j*step/2,y:239-v/max*195,width:step/2-1,height:v/max*195,fill:j ? "#D10721" : "#b9b8bd"}),`${j ? "Upload" : "Original training"} bin ${index}: ${(v*100).toFixed(2)}%`)));
    }); card.appendChild(el("p","Silver: original training. Red: this upload. Select the X sensor above to compare another feature; histogram areas are proportions, not different sample sizes.","small"));
  }
  function render() {
    host.replaceChildren(); tooltip.textContent="Hover a mark, or focus a chart cell/bar, for exact values.";
    const entry=source();
    id("analytics-export-json").disabled=!entry; id("analytics-export-report").disabled=!entry; id("analytics-export-csv").disabled=!entry;
    if (!entry) { id("analytics-status").textContent="No analytics available. Load the original reference or inspect/upload a dataset."; return; }
    const data=entry.data;
    filter.disabled=!data.groups["0"] || ["evidence","comparison"].includes(activeView);
    if (filter.disabled) filter.value="all";
    if (!data.groups[filter.value]) filter.value="all";
    id("analytics-status").textContent=`${entry.name} · ${data.rows.toLocaleString()} observations · grouping: ${data.label_basis.replaceAll("_"," ")}. ${data.normalized ? "Normalized numeric contract; original physical-unit conversion unknown." : "Input scale unverified; values shown as uploaded, not physical-unit conversions."}`;
    const warnings=[];
    Object.entries(data.quality).forEach(([feature,q])=>{
      if(q.missing_or_invalid_for_charts) warnings.push(`${feature}: ${q.missing_or_invalid_for_charts} cells unavailable for charts.`);
      if(q.outside_normalized_range) warnings.push(`${feature}: ${q.outside_normalized_range} values outside normalized [0,1]; do not infer on raw physical values.`);
    });
    if (data.comparison?.available) {
      if (!data.comparison.normalization_confirmed) warnings.push("Training comparisons are numerical only: compatible normalization has NOT been confirmed.");
      Object.entries(data.comparison.features).forEach(([feature,item])=>{
        if(item.outside_training_range) warnings.push(`${feature}: ${item.outside_training_range} outside the observed training range (not a safety limit).`);
        if(item.shape_review) warnings.push(`${feature}: distribution shape differs under the documented review heuristic; not a calibrated OOD verdict.`);
      });
    }
    id("analytics-warnings").textContent=warnings.length ? warnings.join(" ") : "No counted missing/nonfinite/range/shape flags. This does not establish compatible units, safety or model validity.";
    if(data.class_counts) {
      const counts=Object.values(data.class_counts);
      bars(`Whole-source class distribution · ${data.label_basis}`,Object.values(data.class_names),counts,{colors,decimals:0,
        tips:Object.values(data.class_names).map((name,index)=>`${name}: ${counts[index]} ${counts[index]===1 ? "observation" : "observations"} (${data.rows ? (counts[index]/data.rows*100).toFixed(2) : "0.00"}%)`)});
    } else host.appendChild(el("p","Unlabeled data: no ground-truth or predicted class distribution is invented. Use confirmed labels or prediction mode for class-based charts.","analytics-notice"));
    if(activeView==="distributions") histogram(data);
    if(activeView==="classes") boxes(data);
    if(activeView==="relationships") { heatmap(data); scatter(data); }
    if(activeView==="evidence") evidence(data,entry);
    if(activeView==="comparison") comparison(data);
  }
  function resetZoom() {
    const data=source()?.data;
    if(!data) return;
    const features=[Number(id("scatter-x").value),Number(id("scatter-y").value)];
    ["x","y"].forEach((axis,index)=>{
      const stats=data.groups.all.features[data.features[features[index]]];
      id(`scatter-${axis}-min`).value=stats.min ?? 0;
      id(`scatter-${axis}-max`).value=Math.max((stats.min ?? 0)+.001,stats.max ?? 1);
    });
  }
  function download(content,type,name) {
    const url=URL.createObjectURL(new Blob([content],{type})), link=el("a"); link.href=url; link.download=name; link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  id("analytics-export-json").addEventListener("click",()=>{
    const entry=source(); if(entry) download(JSON.stringify({source:entry.name,scope:entry.origin,...entry.data},null,2),"application/json","evision_analytics.json");
  });
  id("analytics-export-csv").addEventListener("click",()=>{
    const entry=source(); if(!entry) return;
    const rows=[["Source","Scope","Label basis","Class filter","Feature","Finite observations","Excluded cells","Minimum","Q1","Median","Q3","Maximum","Mean","Standard deviation"]];
    entry.data.features.forEach(feature=>{const s=group(entry.data).features[feature];rows.push([entry.name,entry.origin,entry.data.label_basis,filter.value,feature,s.count,s.excluded_nonfinite,s.min,s.q1,s.median,s.q3,s.max,s.mean,s.std]);});
    download(rows.map(row=>row.map(v=>`"${String(v ?? "").replaceAll('"','""')}"`).join(",")).join("\r\n"),"text/csv","evision_sensor_summary.csv");
  });
  id("analytics-export-report").addEventListener("click",()=>{
    const entry=source(); if(!entry) return;
    const report=el("main");report.append(el("h1","EVision analytics report"),el("p",id("analytics-status").textContent),el("p",`Static export of view: ${activeView}; class filter: ${filter.selectedOptions[0].textContent}. Use EVision for interactive filtering. Print this report from your browser to save PDF.`),el("p",id("analytics-warnings").textContent),el("p",entry.data.caution));
    report.appendChild(host.cloneNode(true));
    const pre=el("pre",JSON.stringify({source:entry.name,scope:entry.origin,data:entry.data},null,2));
    const details=el("details"), summary=el("summary","Numeric data and sampling/method notes");details.append(summary,pre);report.appendChild(details);
    const html='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>EVision analytics report</title><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; img-src data:; base-uri \'none\'"><style>body{background:#08090B;color:#F5F5F5;font:15px system-ui;margin:32px}main{max-width:1100px;margin:auto}svg{width:100%;height:auto;max-height:480px}section{padding:18px;border:1px solid #474549;margin:18px 0}p{line-height:1.7}pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{padding:8px;text-align:left;border-bottom:1px solid #474549}@media print{body{background:white;color:black}svg text{fill:black}section{break-inside:avoid}details{display:none}}</style>'+report.outerHTML+'</html>';
    download(html,"text/html","evision_analytics_report.html");
  });
  select.addEventListener("change",()=>{filter.value="all";resetZoom();render();});
  filter.addEventListener("change",render);
  id("scatter-reset").addEventListener("click",()=>{resetZoom();render();});
  id("scatter-apply").addEventListener("click",render);
  ["scatter-x","scatter-y"].forEach(name=>{
    shortNames.forEach((feature,index)=>{const option=el("option",feature); option.value=String(index);id(name).appendChild(option);});
    id(name).addEventListener("change",()=>{resetZoom();render();});
  });id("scatter-y").value="4";
  ["scatter-x-min","scatter-x-max","scatter-y-min","scatter-y-max"].forEach(name=>id(name).addEventListener("change",render));
  document.querySelectorAll("[data-analytics-view]").forEach(button=>button.addEventListener("click",()=>{
    activeView=button.dataset.analyticsView;
    document.querySelectorAll("[data-analytics-view]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));
    id("scatter-controls").hidden=!["relationships","comparison"].includes(activeView);render();
    id("scatter-relationship-controls").hidden=activeView!=="relationships";
  }));
  window.EVisionAnalytics={
    setReference(value){reference=value; sources.set("reference-training",{name:"ORIGINAL NEV training · 7,700 rows",origin:"reference",data:value.datasets.training}); sources.set("reference-full",{name:"ORIGINAL NEV full EDA · 11,000 rows · descriptive",origin:"reference",data:value.datasets.full}); refreshSources();resetZoom();},
    referenceUnavailable(message){id("analytics-reference-note").textContent=message;},
    addUpload(origin,name,data){if(!data)return;const sequence=++uploadSequence;const key=`${origin}:${sequence}`;sources.set(key,{name:`UPLOAD · ${name} · file ${sequence}`,origin,data});refreshSources(key);resetZoom();render();},
    clearUploads(origin){for(const [key,value] of sources)if(value.origin===origin)sources.delete(key);refreshSources();resetZoom();},
  };
  render();
})();
