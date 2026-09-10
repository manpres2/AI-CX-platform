const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
for (const path of ['static/admin.html', 'techsupport-voice-bot/static_tech/admin.html', 'bot-template/static_bot/admin.html']) {
  const html = fs.readFileSync(path, 'utf8');
  for (const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)) new vm.Script(match[1], {filename:path});
  const local = html.match(/<select id="tts-local-engine"[\s\S]*?<\/select>/)[0];
  const cloud = html.match(/<select id="tts-cloud-engine"[\s\S]*?<\/select>/)[0];
  assert(local.includes('value="chatterbox"'));
  assert(!local.includes('value="qwen3"'));
  assert(!cloud.includes('value="chatterbox"'));
  const nodes = {};
  const node = id => nodes[id] ||= {value:'', style:{}, parentNode:{insertBefore(){}, style:{}}};
  let mode = 'local';
  const released = [];
  const ctx = {document:{
    getElementById:node,
    querySelector:() => ({value:mode})
  }, releaseTtsSelection: engine => released.push(engine)};
  vm.createContext(ctx);
  vm.runInContext(html.match(/function onTtsModeChange\([^)]*\) \{[\s\S]*?\n\}/)[0],ctx);
  for (const engine of ['chatterbox','kokoro']) {
    node('tts-local-engine').value=engine;
    ctx.onTtsModeChange(true);
    assert.equal(node('tts-chatterbox-fields').style.display,engine==='chatterbox'?'':'none');
    assert.equal(node('tts-local-fields').style.display,engine==='kokoro'?'':'none');
    assert.equal(node('tts-cloud-fields').style.display,'none');
  }
  mode='cloud';
  ctx.onTtsModeChange(true);
  assert.equal(node('tts-chatterbox-fields').style.display,'none');
  assert.equal(node('tts-cloud-fields').style.display,'');
  assert.deepEqual(released,['chatterbox','kokoro','cloud']);
  console.log(path + ': scripts and TTS controls passed');
}
