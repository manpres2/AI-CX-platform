'use strict';
const $ = id => document.getElementById(id);
const API = '/admin/api/outbound';
let campaigns = [], current = null, preview = null, permissions = [], dirty = false, ready = false;
function message(text, error = false) { $('message').textContent = text; $('message').className = error ? 'error' : ''; }
async function api(path, options = {}) {
  const response = await fetch(API + path, options);
  if (!response.ok) {
    let body; try { body = await response.json(); } catch { body = {}; }
    throw new Error(typeof body.detail === 'string' ? body.detail : 'Unable to complete request. Check the fields and your permissions.');
  }
  return response.json();
}
function can(key) { return permissions.includes('outbound.' + key); }
function enforce() {
  $('edit-fields').disabled = !ready || !can('manage');
  $('new').disabled = !ready;
  document.querySelectorAll('[data-manage]').forEach(el => el.hidden = !can('manage'));
  $('export').hidden = !can('export');
  $('export').disabled = !current;
  $('file').disabled = !current;
}
function table(target, headers, rows) {
  target.replaceChildren();
  const table = document.createElement('table'), head = document.createElement('tr');
  headers.forEach(text => { const th = document.createElement('th'); th.textContent = text; head.append(th); });
  const thead = document.createElement('thead'); thead.append(head); table.append(thead);
  const tbody = document.createElement('tbody');
  rows.forEach(row => { const tr = document.createElement('tr'); row.forEach(text => { const td = document.createElement('td'); td.textContent = text; tr.append(td); }); tbody.append(tr); });
  table.append(tbody); target.append(table);
}
function question(data = {}) {
  const box = document.createElement('div'); box.className = 'question';
  box.innerHTML = '<div class="two"><label>Answer field<input class="q-field" placeholder="interested" required pattern="[a-zA-Z][a-zA-Z0-9_]*" maxlength="80"></label><label>Answer type<select class="q-type" aria-label="Answer type"><option value="text">Text</option><option value="yes_no">Yes / No</option><option value="number">Number</option><option value="datetime">Date & time</option></select></label></div><label>Question to ask<input class="q-question" placeholder="Would you like to renew?" required maxlength="500"></label><button type="button" class="secondary">Remove question</button>';
  box.querySelector('.q-field').value = data.field || '';
  box.querySelector('.q-type').value = data.type || 'text';
  box.querySelector('.q-question').value = data.question || '';
  box.querySelector('button').onclick = () => { box.remove(); dirty = true; };
  $('questions').append(box);
}
function agentLink() {
  const slug = $('agent').value;
  $('agent-settings').hidden = !slug;
  $('agent-settings').href = '/proxy/' + encodeURIComponent(slug) + '/admin';
  $('agent-settings').target = '_blank'; $('agent-settings').rel = 'noopener';
}
async function refresh() {
  campaigns = (await api('/campaigns')).campaigns;
  $('campaign-count').textContent = campaigns.length;
  $('lead-count').textContent = campaigns.reduce((sum, c) => sum + c.lead_count, 0);
  $('campaign-list').replaceChildren();
  if (!campaigns.length) { const p = document.createElement('p'); p.className = 'empty'; p.textContent = 'Your first campaign starts here. Choose an agent and save a draft.'; $('campaign-list').append(p); }
  campaigns.forEach(c => {
    const button = document.createElement('button'); button.className = 'campaign' + (c.id === current ? ' selected' : '');
    button.textContent = c.name;
    const meta = document.createElement('span'); meta.textContent = `${c.lead_count} leads · Draft`; button.append(meta);
    button.onclick = () => { if (!dirty || confirm('Discard unsaved campaign changes?')) open(c); };
    $('campaign-list').append(button);
  });
}
async function loadLeads() {
  if (!current) { $('lead-table').textContent = 'No leads yet. Save your campaign, then upload a lead list.'; return; }
  const rows = (await api('/campaigns/' + current + '/leads')).leads;
  if (!rows.length) { $('lead-table').textContent = 'No leads imported yet.'; return; }
  table($('lead-table'), ['Customer', 'Phone', 'Purpose', 'Status'], rows.map(r => [r.name || '—', r.phone, r.purpose, 'Not called']));
}
async function open(c = null) {
  current = c?.id || null; dirty = false; preview = null;
  $('campaign-form').reset(); $('questions').replaceChildren(); $('mapping').hidden = true; $('file').value = '';
  $('import-result').textContent = ''; $('editor-title').textContent = c?.name || 'Create a campaign';
  for (const key of ['name', 'agent', 'purpose', 'opening', 'instructions']) if (c) $(key).value = c[key] || '';
  $('sheet-url').value = c?.sheet_url || '';
  (c?.questions || []).forEach(question);
  agentLink(); enforce();
  try { await refresh(); await loadLeads(); } catch (e) { message(e.message, true); }
}
$('new').onclick = () => { if (!dirty || confirm('Discard unsaved campaign changes?')) open(); };
$('campaign-form').oninput = () => { dirty = true; };
$('agent').onchange = agentLink;
$('add-question').onclick = () => { question(); dirty = true; };
$('campaign-form').onsubmit = async event => {
  event.preventDefault(); $('save').disabled = true;
  try {
    const data = {};
    for (const key of ['name', 'agent', 'purpose', 'opening', 'instructions']) data[key] = $(key).value.trim();
    data.sheet_url = $('sheet-url').value.trim();
    data.questions = [...document.querySelectorAll('.question')].map(el => ({field: el.querySelector('.q-field').value.trim(), type: el.querySelector('.q-type').value, question: el.querySelector('.q-question').value.trim()}));
    const id = current || (globalThis.crypto?.randomUUID?.() || 'draft-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2));
    await api('/campaigns/' + id, {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
    current = id; dirty = false; $('editor-title').textContent = data.name; enforce(); await refresh(); message('Campaign draft saved. No calls have been scheduled.');
  } catch (e) { message(e.message, true); } finally { $('save').disabled = false; }
};
$('file').onchange = async () => {
  preview = null; $('mapping').hidden = true;
  const file = $('file').files[0]; if (!file) return;
  if (file.size > 5 * 1024 * 1024) { message('Maximum upload size is 5 MB.', true); return; }
  const form = new FormData(); form.append('file', file);
  try {
    preview = await api('/preview', {method: 'POST', body: form});
    for (const field of ['phone', 'name', 'purpose']) {
      const select = $('map-' + field); select.replaceChildren(new Option(field === 'phone' ? 'Select phone column' : 'Not mapped', ''));
      preview.headers.forEach(h => select.add(new Option(h, h))); select.value = preview.suggested[field] || '';
    }
    table($('preview'), preview.headers, preview.rows.slice(0, 5)); $('mapping').hidden = false;
    $('import-result').textContent = `${preview.rows.length} rows found. Preview shows the first five. Confirm the column mapping before importing.`;
  } catch (e) { message(e.message, true); }
};
$('import').onclick = async () => {
  if (!preview || !current) return;
  if (dirty) { message('Save your draft changes before importing leads.', true); return; }
  if (!$('country-code').checkValidity()) { $('country-code').reportValidity(); return; }
  $('import').disabled = true;
  try {
    const result = await api('/campaigns/' + current + '/leads', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({headers: preview.headers, rows: preview.rows, phone: $('map-phone').value, name: $('map-name').value, purpose: $('map-purpose').value, country_code: $('country-code').value.trim()})});
    $('import-result').textContent = `${result.imported} imported · ${result.duplicates} duplicates skipped · ${result.rejected.length} rejected\n` + result.rejected.slice(0, 30).map(r => `Row ${r.row}: ${r.reason}`).join('\n') + (result.rejected.length > 30 ? '\nOnly the first 30 errors shown. Correct invalid phone numbers and missing purposes, then re-upload; existing numbers are skipped.' : '');
    await refresh(); await loadLeads(); message('Lead import complete. No calls have been scheduled.');
  } catch (e) { message(e.message, true); } finally { $('import').disabled = false; }
};
$('export').onclick = async () => {
  try {
    const response = await fetch(API + '/campaigns/' + current + '/export');
    if (!response.ok) throw new Error('Download denied or unavailable. Check your export permission.');
    const url = URL.createObjectURL(await response.blob()), a = document.createElement('a');
    a.href = url; a.download = 'outbound-results.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) { message(e.message, true); }
};
window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });
(async () => {
  try {
    permissions = (await api('/session')).permissions;
    enforce();
    if (!can('view')) throw new Error('Outbound view permission is required to open campaigns.');
    const data = await api('/agents');
    $('agent').add(new Option('Select a customer care bot', ''));
    data.agents.forEach(a => $('agent').add(new Option(a.label, a.slug)));
    await refresh(); await open(campaigns[0] || null);
    ready = true; enforce();
  } catch (e) { message(e.message, true); }
})();
