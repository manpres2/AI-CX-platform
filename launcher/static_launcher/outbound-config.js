'use strict';
let callingCatalog = {numbers: [], default_number_id: ''};
let providerRevision = 0, providerDirty = false, voicePane = 'voicemodel';
const callDefaults = {caller_number_id:'', timezone:'Asia/Kolkata', days:[0,1,2,3,4], start_time:'09:00', end_time:'18:00', concurrent_calls:1, ring_timeout_seconds:30, max_call_minutes:10, max_attempts:1, retry_delay_minutes:60, retry_busy:true, retry_no_answer:true, voicemail:'hang_up', voicemail_message:'', record_calls:false, allow_interruptions:true, disclosure:'', opt_out_phrase:'Please do not call me again', transfer_number:''};
const callNumeric = ['concurrent_calls','ring_timeout_seconds','max_call_minutes','max_attempts','retry_delay_minutes'];
const callBoolean = ['retry_busy','retry_no_answer','record_calls','allow_interruptions'];
function newConfigId() { return globalThis.crypto?.randomUUID?.() || 'number-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2); }
function fillCallerNumbers(selected) {
  const select = $('caller-number'); select.replaceChildren(new Option('Not selected — configure later', ''));
  callingCatalog.numbers.forEach(n => {
    const option = new Option(`${n.label} · ${n.phone}${n.enabled ? '' : ' (disabled)'}`, n.id);
    option.disabled = !n.enabled; select.add(option);
  });
  if (selected && !callingCatalog.numbers.some(n => n.id === selected)) {
    const unavailable = new Option('Previously selected number is unavailable', selected); unavailable.disabled = true; select.add(unavailable);
  }
  select.value = selected || '';
  const number = callingCatalog.numbers.find(n => n.id === select.value);
  $('caller-number-hint').textContent = number && !number.enabled ? 'This number is disabled. Choose an enabled number before calling can be activated.' : 'This number will be used for this campaign. Numbers are manually configured and not yet verified by Twilio.';
}
async function loadCallingCatalog() {
  const selected = $('caller-number').value;
  callingCatalog = await api('/calling/catalog');
  fillCallerNumbers(selected);
  $('provider-summary').textContent = callingCatalog.numbers.length ? `Twilio · ${callingCatalog.numbers.length} numbers (not connected)` : 'Not connected';
}
function populateCallOptions(saved, isNew) {
  const options = {...callDefaults, ...(saved || {})};
  fillCallerNumbers(isNew ? callingCatalog.default_number_id : options.caller_number_id);
  for (const key of Object.keys(callDefaults)) {
    if (key === 'caller_number_id' || key === 'days') continue;
    const el = $('call-' + key);
    if (callBoolean.includes(key)) el.checked = options[key]; else el.value = options[key];
  }
  document.querySelectorAll('.days input').forEach(el => el.checked = options.days.includes(Number(el.value)));
}
function readCallOptions() {
  const options = {caller_number_id:$('caller-number').value, days:[...document.querySelectorAll('.days input:checked')].map(el => Number(el.value))};
  for (const key of Object.keys(callDefaults)) {
    if (key === 'caller_number_id' || key === 'days') continue;
    const el = $('call-' + key);
    options[key] = callBoolean.includes(key) ? el.checked : callNumeric.includes(key) ? Number(el.value) : el.value.trim();
  }
  if (!options.days.length) throw new Error('Select at least one calling day.');
  if (options.start_time >= options.end_time) throw new Error('Calling window end must be later than its start on the same day.');
  if (options.voicemail === 'leave_message' && !options.voicemail_message) throw new Error('Enter the message to leave on voicemail.');
  return options;
}
function numberRows() {
  return [...document.querySelectorAll('.number-entry')].map(el => ({id:el.dataset.id, label:el.querySelector('.number-label').value.trim(), phone:el.querySelector('.number-phone').value.trim(), enabled:el.querySelector('.number-enabled').checked}));
}
function refreshDefaultNumbers(selected = $('default-number').value) {
  const select = $('default-number'); select.replaceChildren(new Option('No default — choose per campaign', ''));
  numberRows().filter(n => n.enabled).forEach(n => select.add(new Option(`${n.label || 'Unnamed number'} · ${n.phone || 'Enter phone number'}`, n.id)));
  select.value = selected;
}
function addNumber(data = {}) {
  const el = document.createElement('div'); el.className = 'number-entry'; el.dataset.id = data.id || newConfigId();
  el.innerHTML = '<div class="two"><label>Number label<input class="number-label" required maxlength="80" placeholder="Sales line"></label><label>Phone number (international)<input class="number-phone" required pattern="\\+[1-9][0-9]{7,14}" placeholder="+12025550101"></label></div><div class="section-heading"><label class="check-label"><input class="number-enabled" type="checkbox" checked>Enabled for campaign selection</label><button type="button" class="secondary">Remove</button></div><p class="hint">Provider verification pending</p>';
  el.querySelector('.number-label').value = data.label || '';
  el.querySelector('.number-phone').value = data.phone || '';
  el.querySelector('.number-enabled').checked = data.enabled !== false;
  el.querySelector('button').onclick = () => { el.remove(); providerDirty = true; refreshDefaultNumbers(); };
  el.oninput = () => { providerDirty = true; refreshDefaultNumbers(); };
  $('number-list').append(el); refreshDefaultNumbers();
}
function callingMessage(text, error = false) { $('calling-message').textContent = text; $('calling-message').className = error ? 'error' : ''; }
function fillProvider(data) {
  providerRevision = data.revision;
  $('provider').value = data.provider;
  $('provider-label').value = data.account_label;
  $('provider-sid').value = data.account_sid;
  $('provider-url').value = data.public_url;
  $('provider-token').value = ''; $('clear-token').checked = false;
  $('token-status').textContent = data.auth_token_set ? 'Auth token saved securely. Leave blank to keep it, or enter a replacement.' : 'No auth token saved. You can add it later.';
  $('number-list').replaceChildren(); data.numbers.forEach(addNumber);
  refreshDefaultNumbers(data.default_number_id); providerDirty = false;
}
async function showCallingSettings() {
  $('calling-dialog').showModal(); $('provider-fields').disabled = true; callingMessage('Loading calling settings…');
  try { fillProvider(await api('/calling/settings')); $('provider-fields').disabled = false; callingMessage(''); }
  catch (e) { callingMessage(e.message, true); }
}
function closeCalling(event) {
  if (event) event.preventDefault();
  if (providerDirty && !confirm('Discard unsaved calling settings?')) return;
  $('provider-token').value = ''; providerDirty = false; $('calling-dialog').close();
}
function applyVoicePane() {
  const frame = $('voice-frame');
  try {
    if (typeof frame.contentWindow.switchPane === 'function') {
      frame.contentWindow.switchPane(voicePane);
      $('voice-status').textContent = 'Connected to the selected bot’s existing configuration panel.';
    } else {
      $('voice-status').textContent = 'The bot panel is unavailable or still loading. Start the bot in AI Bots and ensure you have both Voice AI and selected-bot access.';
    }
  } catch { $('voice-status').textContent = 'Unable to open the selected bot settings. Check bot availability and permissions.'; }
}
function initializeCallingUI() {
  $('calling-settings-button').onclick = showCallingSettings;
  $('close-calling').onclick = closeCalling;
  $('calling-dialog').addEventListener('cancel', closeCalling);
  $('add-number').onclick = () => { addNumber(); providerDirty = true; };
  $('calling-form').oninput = () => { providerDirty = true; };
  $('caller-number').onchange = () => fillCallerNumbers($('caller-number').value);
  $('calling-form').onsubmit = async event => {
    event.preventDefault(); $('save-calling').disabled = true;
    try {
      const data = {revision:providerRevision, provider:$('provider').value, account_label:$('provider-label').value.trim(), account_sid:$('provider-sid').value.trim(), public_url:$('provider-url').value.trim(), numbers:numberRows(), default_number_id:$('default-number').value, clear_auth_token:$('clear-token').checked};
      const token = $('provider-token').value.trim(); if (token) data.auth_token = token;
      fillProvider(await api('/calling/settings', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)}));
      await loadCallingCatalog(); callingMessage('Calling settings saved. Provider verification and dialing are not connected yet.');
    } catch (e) { callingMessage(e.message, true); } finally { $('save-calling').disabled = false; }
  };
  $('voice-settings-button').onclick = () => {
    const slug = $('agent').value;
    if (!slug) { message('Select a customer care agent first.', true); return; }
    voicePane = 'voicemodel'; $('voice-title').textContent = 'Voice AI · ' + $('agent').selectedOptions[0].textContent;
    $('voice-status').textContent = 'Loading the selected bot…'; $('voice-dialog').showModal();
    $('voice-frame').src = '/outbound/voice/' + encodeURIComponent(slug) + '/admin';
  };
  $('voice-frame').onload = applyVoicePane;
  document.querySelectorAll('#voice-navigation button').forEach(button => button.onclick = () => { voicePane = button.dataset.pane; applyVoicePane(); });
  $('close-voice').onclick = () => $('voice-dialog').close();
  $('voice-dialog').addEventListener('close', () => { $('voice-frame').src = 'about:blank'; });
}
