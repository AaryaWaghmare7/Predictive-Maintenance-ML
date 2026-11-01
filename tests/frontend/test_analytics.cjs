// Real dashboard module with a small DOM fixture; no browser/library dependency.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {readFileSync}=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const script=readFileSync(path.resolve(__dirname,'../../src/api/static/analytics.js'),'utf8');
const reference=JSON.parse(readFileSync(path.resolve(__dirname,'../../reports/metrics/experiment_2/dashboard_reference.json'),'utf8'));
const escape=text=>String(text).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');

class Node {
  constructor(tag='div') {this.tag=tag;this.children=[];this.attributes={};this.dataset={};this.style={};this.events={};this.ownText='';this._value='';this.disabled=false;}
  set textContent(value){this.ownText=String(value);this.children=[];}
  get textContent(){return this.ownText+this.children.map(c=>c.textContent).join(' ');}
  set value(value){this._value=String(value);}
  get value(){return this._value || (this.tag==='select' ? this.children[0]?.value || '' : '');}
  get selectedOptions(){return this.children.filter(c=>c.value===this.value);}
  setAttribute(name,value){this.attributes[name]=String(value);}
  appendChild(node){this.children.push(node);return node;}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(){this.ownText='';this.children=[];}
  addEventListener(name,fn){(this.events[name] ||= []).push(fn);}
  async trigger(name){for(const fn of this.events[name] || [])await fn({target:this});}
  click(){this.clicked=true;}
  cloneNode(deep){const result=new Node(this.tag);result.attributes={...this.attributes};result.ownText=this.ownText;result.className=this.className;if(deep)result.children=this.children.map(c=>c.cloneNode(true));return result;}
  get outerHTML(){return `<${this.tag} ${Object.entries(this.attributes).map(([k,v])=>`${k}="${escape(v)}"`).join(' ')}>${escape(this.ownText)}${this.children.map(c=>c.outerHTML).join('')}</${this.tag}>`;}
}
function setup() {
  const ids=new Map(), element=id=>{if(!ids.has(id))ids.set(id,new Node(['analytics-source','analytics-class','scatter-x','scatter-y'].includes(id)?'select':'div'));return ids.get(id);};
  element('analytics-class').value='all';
  const opt=new Node('option');opt.value='all';opt.textContent='All categories';element('analytics-class').appendChild(opt);
  ['scatter-x-min','scatter-y-min'].forEach(id=>element(id).value=0);
  ['scatter-x-max','scatter-y-max'].forEach(id=>element(id).value=1);
  const views=['distributions','classes','relationships','evidence','comparison'].map(view=>{const b=new Node('button');b.dataset.analyticsView=view;return b;});
  const window={},downloads=[];
  vm.runInNewContext(script,{document:{getElementById:element,createElement:tag=>new Node(tag),createElementNS:(_ns,tag)=>new Node(tag),querySelectorAll:()=>views},window,Blob,URL:{createObjectURL:blob=>{downloads.push(blob);return 'blob:test';},revokeObjectURL(){}},setTimeout:fn=>fn()});
  return {element,views,api:window.EVisionAnalytics,downloads};
}
function all(node,tag){return [...(node.tag===tag?[node]:[]),...node.children.flatMap(child=>all(child,tag))];}
function upload(rows=4){const data=structuredClone(reference.datasets.training);data.rows=rows;data.label_basis='model_predictions';data.normalized=true;return data;}

test('initial state has no fake upload counts and exports are disabled',()=>{
  const ui=setup();assert.match(ui.element('analytics-status').textContent,/No analytics available/);assert.equal(ui.element('analytics-export-json').disabled,true);
});
test('original training reference draws all seven histograms and preserves provenance',()=>{
  const ui=setup();ui.api.setReference(reference);
  assert.match(ui.element('analytics-status').textContent,/ORIGINAL NEV training/);
  assert.equal(all(ui.element('analytics-charts'),'svg').length,8);
  const bars=all(ui.element('analytics-charts'),'rect');
  assert.match(bars[0].attributes['aria-label'],/Normal: 3500 observations \(45.45%\)/);
  bars[0].trigger('focus');assert.match(ui.element('analytics-tooltip').textContent,/3500/);
});
test('source switching uses full EDA rather than training class-dependent charts',async()=>{
  const ui=setup();ui.api.setReference(reference);ui.element('analytics-source').value='reference-full';await ui.element('analytics-source').trigger('change');
  assert.match(ui.element('analytics-status').textContent,/11,000/);assert.equal(ui.element('analytics-class').disabled,true);
  await ui.views[1].trigger('click');assert.match(ui.element('analytics-charts').textContent,/Class boxplots require/);
});
test('all reference chart views render genuine data and zoom validation is useful',async()=>{
  const ui=setup();ui.api.setReference(reference);
  await ui.views[1].trigger('click');assert.equal(all(ui.element('analytics-charts'),'svg').length,8);
  await ui.views[2].trigger('click');assert.equal(all(ui.element('analytics-charts'),'svg').length,3);
  assert.match(ui.element('analytics-charts').textContent,/joint-bin counts/);
  ui.element('scatter-x-min').value=2;ui.element('scatter-x-max').value=1;await ui.element('scatter-apply').trigger('click');
  assert.match(ui.element('analytics-charts').textContent,/minimum smaller than maximum/);
  await ui.element('scatter-reset').trigger('click');assert.doesNotMatch(ui.element('analytics-charts').textContent,/minimum smaller than maximum/);
  await ui.views[3].trigger('click');assert.match(ui.element('analytics-charts').textContent,/1540 rows/);assert.match(ui.element('analytics-charts').textContent,/not a causal explanation/);
});
test('new uploads replace the active chart source and same-name files stay separate',()=>{
  const ui=setup();ui.api.setReference(reference);
  ui.api.addUpload('dataset','same.csv',upload(4));assert.match(ui.element('analytics-status').textContent,/4 observations/);
  ui.api.addUpload('dataset','same.csv',upload(40));assert.match(ui.element('analytics-status').textContent,/40 observations/);
  assert.equal(ui.element('analytics-source').children.length,4);
  ui.api.clearUploads('dataset');assert.equal(ui.element('analytics-source').children.length,2);assert.match(ui.element('analytics-status').textContent,/ORIGINAL/);
});
test('unlabeled values never become invented class boxes or associations',async()=>{
  const ui=setup(),data=upload();data.class_counts=null;data.label_basis='unlabeled';data.groups={all:data.groups.all};data.associations={};
  ui.api.addUpload('dataset','unlabeled.csv',data);assert.match(ui.element('analytics-charts').textContent,/Unlabeled data/);
  await ui.views[1].trigger('click');assert.match(ui.element('analytics-charts').textContent,/Class boxplots require/);
});
test('JSON and CSV downloads contain actual selected summaries, not fake accuracy',async()=>{
  const ui=setup();ui.api.setReference(reference);ui.api.addUpload('csv','actual.csv',upload(12));
  await ui.element('analytics-export-json').trigger('click');const exported=JSON.parse(await ui.downloads[0].text());
  assert.equal(exported.rows,12);assert.match(exported.source,/UPLOAD.*actual.csv/);assert.equal(exported.accuracy,undefined);
  await ui.element('analytics-export-csv').trigger('click');const csv=await ui.downloads[1].text();assert.match(csv,/Feature/);assert.match(csv,/Voltage \(V\)/);
});
test('standalone reports escape uploaded filenames and contain no executable scripts',async()=>{
  const ui=setup();ui.api.setReference(reference);ui.api.addUpload('dataset','<img src=x onerror=bad>.csv',upload());
  await ui.element('analytics-export-report').trigger('click');const report=await ui.downloads[0].text();
  assert.ok(report.startsWith('<!doctype html>'));assert.match(report,/&lt;img src=x onerror=bad&gt;/);assert.doesNotMatch(report,/<img src=x/);assert.doesNotMatch(report,/<script/);assert.match(report,/Content-Security-Policy/);
});
