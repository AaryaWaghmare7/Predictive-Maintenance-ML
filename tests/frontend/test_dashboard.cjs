// Lightweight control-flow tests using Node's built-in test runner. No npm
// packages, framework, production model fitting or browser mocking libraries.
// Run: node --test tests/frontend/test_dashboard.cjs
const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = readFileSync(path.resolve(__dirname, '../../src/api/static/app.js'), 'utf8');
const features = ['Voltage (V)', 'Current (A)', 'Motor Speed (RPM)', 'Temperature (°C)', 'Vibration (g)', 'Ambient Temp (°C)', 'Humidity (%)'];
const sampleRows = [
  [.939, .200, .544, .240, .301, .582, .397],
  [.805, .597, .073, .180, .921, .128, .125],
  [.586, .275, .355, .398, .452, .728, .326],
  [.180, .453, .393, .782, .227, .287, .515],
];

class Element {
  constructor() {
    this.textContent = ''; this.value = ''; this.hidden = false; this.disabled = false;
    this.dataset = {}; this.style = {setProperty(name,value){this[name]=value;}};
    this.children = []; this.files = [];
    this.listeners = {}; this.attributes = {}; this.classes = new Set();
    this.classList = {
      add: (name) => this.classes.add(name),
      remove: (name) => this.classes.delete(name),
      toggle: (name, active) => active ? this.classes.add(name) : this.classes.delete(name),
    };
  }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  removeEventListener(type, handler) { this.listeners[type]=(this.listeners[type] || []).filter(item=>item!==handler); }
  async trigger(type) {
    await Promise.all((this.listeners[type] || []).map(handler => handler({target:this, preventDefault(){}})));
  }
  reportValidity() { return true; }
  querySelectorAll() { return this.inputs || []; }
  setAttribute(name, value) { this.attributes[name] = value; }
  appendChild(child) { this.children.push(child); }
  append(...children) { this.children.push(...children); }
  replaceChildren() { this.children = []; }
  click() { this.clicked = true; }
  getBoundingClientRect() { return this.bounds || {top:0, height:400}; }
}
const prediction = {predicted_class:0, predicted_fault:'Normal', model_confidence: .991,
  status:'normal', model_version:'trusted-test-artifact'};
const summary = {rows_analyzed:4, mean_model_confidence:.9975, dominant_abnormal_faults:[],
  classes:{Normal:{count:4, percentage:100}}, predictions:[prediction],
  predictions_returned:1, predictions_truncated:false,
  ignored_columns:['Fault Label'], normalized_headers:{' Voltage (V) ':'Voltage (V)'}};
const response = (data, ok=true, status=200) => ({ok, status, json:async()=>data});

async function setup(handler, browserWindow) {
  const ids = new Map();
  const element = (id) => { if (!ids.has(id)) ids.set(id,new Element()); return ids.get(id); };
  const inputs = features.map(() => new Element());
  element('fault-form').inputs = inputs;
  element('fault-form').elements = {namedItem:name=>inputs[features.indexOf(name)]};
  const buttons = sampleRows.map((_,index) => { const item = new Element(); item.dataset.sample=String(index); return item; });
  const calls = [];
  const downloads = [];
  class UploadForm { constructor() { this.entries=[]; } append(...entry) { this.entries.push(entry); } }
  vm.runInNewContext(source, {
    document:{documentElement:element('html'), getElementById:element, querySelectorAll:selector=>selector==='[data-sample]' ? buttons : [], createElement:()=>new Element()},
    ...(browserWindow ? {window:browserWindow} : {}),
    FormData:UploadForm,
    Blob,
    URL:{createObjectURL:blob=>{downloads.push(blob); return 'blob:test';}, revokeObjectURL:()=>{}},
    setTimeout:handler=>handler(),
    fetch:async(url, options) => {
      calls.push({url,options});
      if (url==='/health') return response({fault_model_status:'ready', model_version:'trusted-test-artifact'});
      return handler ? handler(url,options) : response(url==='/predict-fault' ? prediction : summary);
    },
  });
  await new Promise(setImmediate); // Settle the real script's initial health request.
  return {element, inputs, buttons, calls, downloads};
}

test('direct file preview redirects to localhost; HTTP stays on its own host and port', () => {
  const html=readFileSync(path.resolve(__dirname,'../../src/api/static/index.html'),'utf8');
  const guard=html.match(/<script id="local-preview-guard">([\s\S]*?)<\/script>/)[1];
  for (const protocol of ['file:','http:','https:']) {
    const destinations=[];
    vm.runInNewContext(guard,{location:{protocol,replace:url=>destinations.push(url)}});
    assert.deepEqual(destinations,protocol==='file:' ? ['http://127.0.0.1:8012/'] : []);
  }
});

function motionBrowser({reduced=false, compact=false, observe=false}={}) {
  const events={}; const frames=new Map(); const observers=[]; let nextFrame=0;
  const preference=matches=>({matches, addEventListener(_name,callback){this.changed=callback;},
    removeEventListener(){this.changed=null;}});
  const media=preference(reduced); const mobile=preference(compact);
  const browserWindow={matchMedia:query=>query.includes('reduced-motion') ? media : mobile,
    requestAnimationFrame:callback=>{frames.set(++nextFrame,callback); return nextFrame;},
    cancelAnimationFrame:id=>frames.delete(id),
    addEventListener:(name,callback,options)=>{events[name]=callback; if(name==='scroll') assert.equal(options.passive,true);},
    removeEventListener:name=>{delete events[name];}};
  if(observe) browserWindow.IntersectionObserver=class {
    constructor(callback){this.callback=callback; this.targets=[]; observers.push(this);}
    observe(target){this.targets.push(target);}
    unobserve(target){this.targets=this.targets.filter(item=>item!==target);}
    disconnect(){this.disconnected=true;}
  };
  return {window:browserWindow,events,frames,media,mobile,observers,
    flush(){const pending=[...frames.values()];frames.clear();pending.forEach(callback=>callback());}};
}

test('hero scroll work is frame-bounded, reversible and stops with reduced motion', async () => {
  const browser=motionBrowser();
  const {element}=await setup(undefined,browser.window);
  element('home').bounds={top:-200,height:400};
  browser.events.scroll(); browser.events.scroll();
  assert.equal(browser.frames.size,1);
  browser.flush();
  assert.equal(element('home').style['--hero-progress'],'0.5');
  element('home').bounds.top=-100; browser.events.scroll(); browser.flush();
  assert.equal(element('home').style['--hero-progress'],'0.25');
  browser.media.matches=true; browser.media.changed();
  assert.equal(element('home').style['--hero-progress'],'0');
  assert.equal(browser.events.scroll,undefined);
  assert.equal(element('html').dataset.motion,'reduced');
});

test('manual reduced-motion control works and compact screens have no scroll listener', async () => {
  const browser=motionBrowser();
  const {element}=await setup(undefined,browser.window);
  await element('motion-toggle').trigger('click');
  assert.equal(element('html').dataset.motion,'reduced');
  assert.equal(element('motion-toggle').attributes['aria-pressed'],'true');
  assert.equal(browser.events.scroll,undefined);
  await element('motion-toggle').trigger('click');
  assert.equal(element('html').dataset.motion,'full');
  assert.ok(browser.events.scroll);
  browser.mobile.matches=true; browser.mobile.changed();
  assert.equal(browser.events.scroll,undefined);
  assert.equal(element('home').style['--hero-progress'],'0');
});

test('hero visibility gates scroll work; pagehide cleans listeners, observers and pending frames', async () => {
  const browser=motionBrowser({observe:true});
  const {element}=await setup(undefined,browser.window);
  assert.equal(browser.events.scroll,undefined);
  browser.observers[1].callback([{isIntersecting:true}]);
  browser.events.scroll();
  assert.equal(browser.frames.size,1);
  browser.events.pagehide();
  assert.equal(browser.frames.size,0);
  assert.equal(browser.events.scroll,undefined);
  assert.equal(browser.media.changed,null);
  assert.ok(browser.observers.every(observer=>observer.disconnected));
  browser.events.pageshow({persisted:true});
  assert.equal(browser.observers.length,4);
  assert.ok(browser.media.changed);
});

test('four samples only fill canonical fields and clear stale predictions', async () => {
  const {element,inputs,buttons,calls} = await setup();
  for (let index=0; index<4; index++) {
    await buttons[index].trigger('click');
    assert.deepEqual(inputs.map(input=>Number(input.value)),sampleRows[index]);
    assert.equal(element('fault-result').hidden,true);
    assert.equal(element('fault-empty').hidden,false);
  }
  assert.deepEqual(calls.map(call=>call.url),['/health']);
});

test('manual pending state blocks duplicates and recovers after success', async () => {
  let release;
  const pending = new Promise(resolve=>{release=resolve;});
  const {element,inputs,buttons,calls} = await setup(()=>pending);
  await buttons[0].trigger('click');
  const first = element('fault-form').trigger('submit');
  assert.equal(element('analyze-button').disabled,true);
  assert.equal(element('fault-form').attributes['aria-busy'],'true');
  assert.ok(inputs.every(input=>input.disabled));
  assert.ok(buttons.every(button=>button.disabled));
  await element('fault-form').trigger('submit');
  assert.equal(calls.filter(call=>call.url==='/predict-fault').length,1);
  release(response(prediction)); await first;
  assert.equal(element('analyze-button').disabled,false);
  assert.equal(element('model-confidence').textContent,'99.1%');
  assert.equal(element('fault-result').hidden,false);
  assert.ok(inputs.every(input=>!input.disabled));
  assert.ok(buttons.every(button=>!button.disabled));
});

test('manual backend error is shown and a later submission can succeed', async () => {
  let fail = true;
  const {element,buttons} = await setup(()=> fail ? response({detail:'Model unavailable'},false,503) : response(prediction));
  await buttons[0].trigger('click');
  await element('fault-form').trigger('submit');
  assert.equal(element('fault-error').textContent,'Model unavailable');
  assert.equal(element('fault-result').hidden,true);
  assert.equal(element('analyze-button').disabled,false);
  fail=false; await element('fault-form').trigger('submit');
  assert.equal(element('fault-error').textContent,'');
  assert.equal(element('fault-result').hidden,false);
});

test('CSV uses file FormData without manually setting multipart headers and blocks duplicates', async () => {
  let release;
  const pending=new Promise(resolve=>{release=resolve;});
  const {element,calls} = await setup(()=>pending);
  const file={name:'telemetry.csv',size:200}; element('telemetry-file').files=[file];
  await element('telemetry-file').trigger('change');
  const first = element('csv-form').trigger('submit');
  assert.equal(element('telemetry-file').disabled,true);
  assert.equal(element('upload-button').disabled,true);
  await element('csv-form').trigger('submit');
  const uploads = calls.filter(call=>call.url==='/analyze-csv');
  assert.equal(uploads.length,1);
  assert.equal(uploads[0].options.method,'POST');
  assert.equal(uploads[0].options.headers,undefined);
  assert.equal(uploads[0].options.body.entries[0][0],'file');
  assert.equal(uploads[0].options.body.entries[0][1],file);
  release(response(summary)); await first;
  assert.equal(element('csv-result').hidden,false);
  assert.equal(element('telemetry-file').disabled,false);
  assert.equal(element('upload-button').disabled,false);
  assert.equal(element('csv-confidence').textContent,'99.8%');
  assert.match(element('csv-parsing-note').textContent,/Fault Label/);
  assert.match(element('csv-parsing-note').textContent,/whitespace/);
});

test('CSV validation error clears stale summary and permits corrected upload', async () => {
  let fail=true;
  const {element} = await setup(()=>fail ? response({detail:'CSV is missing required sensor columns: Voltage (V)'},false,422) : response(summary));
  element('telemetry-file').files=[{name:'missing.csv',size:200}];
  await element('csv-form').trigger('submit');
  assert.match(element('csv-error').textContent,/Voltage \(V\)/);
  assert.equal(element('csv-result').hidden,true);
  assert.equal(element('upload-button').disabled,false);
  fail=false; await element('csv-form').trigger('submit');
  assert.equal(element('csv-error').textContent,'');
  assert.equal(element('csv-result').hidden,false);
});

test('missing, empty and oversized CSV are not sent for inference', async () => {
  const {element,calls} = await setup();
  for (const files of [[], [{size:0}], [{size:5*1024*1024+1}]]) {
    element('telemetry-file').files=files;
    await element('csv-form').trigger('submit');
    assert.ok(element('csv-error').textContent);
    assert.equal(element('telemetry-file').disabled,false);
  }
  assert.equal(calls.filter(call=>call.url==='/analyze-csv').length,0);
});

test('network and non-JSON errors are readable, not raw HTML or stale results', async () => {
  for (const handler of [()=>{throw new TypeError('failed fetch');}, ()=>({ok:false,status:500,json:async()=>{throw new SyntaxError();}})]) {
    const {element,buttons} = await setup(handler);
    await buttons[0].trigger('click');
    await element('fault-form').trigger('submit');
    assert.match(element('fault-error').textContent,/localhost|unreadable/);
    assert.equal(element('fault-result').hidden,true);
    assert.equal(element('analyze-button').disabled,false);
  }
});

test('chart filtering preserves whole-file counts and original row numbers without reinference', async () => {
  const batch = {...summary, rows_analyzed:3000, predictions_returned:3, predictions_truncated:true,
    predictions:[prediction, {...prediction, predicted_fault:'Motor Fault'}, {...prediction, predicted_fault:'Battery Fault'}],
    classes:{Normal:{count:2000,percentage:200/3}, 'Motor Fault':{count:500,percentage:50/3},
      'Inverter Fault':{count:0,percentage:0}, 'Battery Fault':{count:500,percentage:50/3}}};
  const {element,calls}=await setup(()=>response(batch));
  element('telemetry-file').files=[{name:'batch.csv',size:200}];
  await element('csv-form').trigger('submit');
  const chart=element('class-distribution').children;
  assert.equal(chart.length,4);
  assert.equal(chart[1].attributes['aria-pressed'],'false');
  await chart[1].trigger('click');
  assert.equal(chart[1].attributes['aria-pressed'],'true');
  assert.equal(element('row-details').open,true);
  assert.equal(element('row-predictions').children.length,1);
  assert.equal(element('row-predictions').children[0].children[0].textContent,'2');
  assert.equal(element('csv-rows').textContent,3000);
  assert.match(element('row-limit-note').textContent,/first 1,000 only/);
  await element('clear-csv-filter').trigger('click');
  assert.equal(element('row-predictions').children.length,3);
  assert.equal(element('clear-csv-filter').disabled,true);
  assert.equal(calls.filter(call=>call.url==='/analyze-csv').length,1);
});

test('Explore Demo reuses the genuine Motor sample without calling inference', async () => {
  const {element,inputs,calls} = await setup();
  await element('explore-demo').trigger('click');
  assert.deepEqual(inputs.map(input=>Number(input.value)),sampleRows[1]);
  assert.match(element('sample-status').textContent,/Motor Fault sample loaded/);
  assert.deepEqual(calls.map(call=>call.url),['/health']);
});

test('donut uses exact counts and changing files clears stale results and downloads', async () => {
  const batch = {...summary, classes:{Normal:{count:3,percentage:75},'Motor Fault':{count:1,percentage:25}}};
  const {element} = await setup(()=>response(batch));
  element('telemetry-file').files=[{name:'batch.csv',size:200}];
  await element('csv-form').trigger('submit');
  assert.equal(element('donut-total').textContent,4);
  assert.match(element('fault-donut').style.background,/conic-gradient\(#a8a6ab 0% 75%, #D10721 75% 100%\)/);
  assert.match(element('fault-donut').attributes['aria-label'],/Motor Fault: 1/);
  assert.equal(element('distribution-empty').hidden,true);
  assert.equal(element('download-results').disabled,false);
  await element('telemetry-file').trigger('change');
  assert.equal(element('donut-total').textContent,'—');
  assert.equal(element('fault-donut').style.background,'');
  assert.equal(element('distribution-empty').hidden,false);
  assert.equal(element('download-results').disabled,true);
  assert.equal(element('class-distribution').children.length,0);
});

test('download contains only actual returned predictions, in original order, regardless of filter', async () => {
  const batch = {...summary, rows_analyzed:2000, predictions_truncated:true,
    predictions_returned:2, predictions:[prediction,{...prediction,predicted_class:1,predicted_fault:'Motor Fault'}]};
  const {element,downloads,calls}=await setup(()=>response(batch));
  await element('download-results').trigger('click');
  assert.equal(downloads.length,0);
  element('telemetry-file').files=[{name:'batch.csv',size:200}];
  await element('csv-form').trigger('submit');
  await element('download-results').trigger('click');
  const csv=await downloads[0].text();
  assert.equal(csv.split('\r\n').length,3);
  assert.match(csv,/"1","0","Normal","0.991"/);
  assert.match(csv,/"2","1","Motor Fault","0.991"/);
  assert.equal(element('download-results').textContent,'Download 2 returned predictions');
  assert.equal(calls.filter(call=>call.url==='/analyze-csv').length,1);
});
