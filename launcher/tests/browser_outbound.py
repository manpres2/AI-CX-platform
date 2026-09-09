"""Optional browser smoke test: pip install -r launcher/tests/requirements.txt.
Uses installed Microsoft Edge, temporary databases, and synthetic customer data.
Run: python launcher/tests/browser_outbound.py
"""
import threading
import json
import time
from pathlib import Path
import uvicorn
from playwright.sync_api import sync_playwright, expect
from test_outbound import OutboundTests

OutboundTests.setUpClass()
server = uvicorn.Server(uvicorn.Config(OutboundTests.app, host='127.0.0.1', port=18084, log_level='error'))
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
try:
    for _ in range(100):
        if server.started:
            break
        time.sleep(.05)
    if not server.started:
        raise RuntimeError('Preview server failed to start')
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        context = browser.new_context(http_credentials={'username': 'owner', 'password': 'test-password'}, viewport={'width': 1440, 'height': 1080})
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto('http://127.0.0.1:18084/outbound')
        output = Path(__file__).resolve().parents[2] / 'docs' / 'outbound'
        output.mkdir(parents=True, exist_ok=True)
        page.get_by_role('button', name='Calling settings', exact=True).click()
        page.get_by_label('Account label', exact=True).fill('Sample customer operations account')
        page.get_by_label('Twilio Account SID', exact=True).fill('AC' + 'a' * 32)
        page.get_by_label('Public HTTPS base URL', exact=True).fill('https://calling.example.com')
        for label, number in [('Sales line', '+12025550101'), ('Support line', '+12025550102')]:
            page.get_by_role('button', name='+ Add phone number', exact=True).click()
            row = page.locator('.number-entry').last
            row.get_by_label('Number label', exact=True).fill(label)
            row.get_by_label('Phone number (international)', exact=True).fill(number)
        sales_id = page.locator('.number-entry').nth(0).get_attribute('data-id')
        support_id = page.locator('.number-entry').nth(1).get_attribute('data-id')
        page.get_by_label('Default number for new campaigns', exact=True).select_option(sales_id)
        page.get_by_role('button', name='Save calling settings', exact=True).click()
        expect(page.locator('#calling-message')).to_contain_text('Calling settings saved')
        page.locator('#calling-dialog').evaluate('(dialog) => dialog.scrollTop = 0')
        page.screenshot(path=str(output / 'calling-settings.png'), full_page=False)
        page.locator('#close-calling').click()
        page.get_by_role('button', name='+ New campaign', exact=True).click()
        expect(page.locator('#caller-number')).to_have_value(sales_id)
        page.get_by_label('Call from number', exact=True).select_option(support_id)
        page.locator('#call-max_attempts').fill('3')
        page.get_by_label('Campaign name', exact=True).fill('September renewal follow-up')
        page.get_by_label('Customer care agent', exact=True).select_option('tech')
        # Serve the actual template with synthetic API responses to verify its UI
        # without starting GPU models or editing any real bot configuration.
        template = (Path(__file__).resolve().parents[2] / 'bot-template/static_bot/admin.html').read_text(encoding='utf-8')
        template = template.replace('<head>', "<head><script>window.__BASE_PATH__='/outbound/voice/tech';</script>", 1)
        def voice_fixture(route):
            if route.request.url.split('?')[0].endswith('/admin'):
                route.fulfill(status=200, content_type='text/html', body=template)
            else:
                route.fulfill(status=200, content_type='application/json', body=json.dumps({
                    'tts_mode':'local', 'llm_mode':'local', 'stt_mode':'local', 'tts_cloud':{}, 'llm_cloud':{},
                    'files':[], 'models':[{'name':'small', 'size_mb':500, 'downloaded':True}], 'configured':'small', 'loaded':'small', 'company_name':'Sample voice agent', 'available_models':['sample-local-model'],
                    'ollama_model':'sample-local-model', 'available_voices':{'en':{'female':['af_heart']}}, 'kokoro_voice':'af_heart'}))
        page.route('**/outbound/voice/tech/**', voice_fixture)
        page.get_by_role('button', name='Configure Voice AI', exact=True).click()
        frame = page.frame_locator('#voice-frame')
        expect(frame.locator('#pane-voicemodel')).to_be_visible()
        expect(frame.locator('#voice-select')).to_be_visible()
        expect(frame.locator('#stt-model-select')).to_be_visible()
        page.screenshot(path=str(output / 'voice-ai.png'), full_page=False)
        for pane in ['prompt', 'guardrails', 'bgaudio', 'kb', 'search', 'branding']:
            page.locator('#voice-navigation [data-pane="' + pane + '"]').click()
            expect(frame.locator('#pane-' + pane)).to_be_visible()
        page.locator('#close-voice').click()
        page.get_by_label('Default call purpose', exact=True).fill('Explain renewal options and collect the customer’s decision.')
        page.get_by_label('Opening message', exact=True).fill('Hello {{customer_name}}, I’m an AI assistant calling about {{purpose}}. Is now a good time?')
        page.get_by_label('Additional conversation instructions', exact=True).fill('Answer from the approved knowledge base. Ask for a callback time if the customer is busy.')
        page.get_by_role('button', name='+ Add question', exact=True).click()
        page.get_by_label('Answer field', exact=True).fill('interested')
        page.get_by_label('Answer type', exact=True).select_option('yes_no')
        page.get_by_label('Question to ask', exact=True).fill('Would you like to renew your plan?')
        page.get_by_role('button', name='Save draft', exact=True).click()
        expect(page.locator('#message')).to_contain_text('Campaign draft saved')
        page.locator('#file').set_input_files({'name': 'sample.csv', 'mimeType': 'text/csv', 'buffer': b'Name,Phone,Purpose\nSample Customer A,+12025550101,Renew annual plan\nSample Customer B,+12025550102,Discuss service options\n'})
        expect(page.locator('#mapping')).to_be_visible()
        expect(page.locator('#map-phone')).to_have_value('Phone')
        page.get_by_role('button', name='Import mapped leads', exact=True).click()
        expect(page.locator('#import-result')).to_contain_text('2 imported')
        expect(page.locator('#lead-table')).to_contain_text('Sample Customer A')
        with page.expect_download() as download:
            page.get_by_role('button', name='Download CSV', exact=True).click()
        assert download.value.suggested_filename == 'outbound-results.csv'
        page.reload()
        expect(page.locator('#name')).to_have_value('September renewal follow-up')
        expect(page.locator('#caller-number')).to_have_value(support_id)
        expect(page.locator('#call-max_attempts')).to_have_value('3')
        expect(page.locator('#lead-table')).to_contain_text('Sample Customer A')
        expect(page.get_by_role('button', name='Start calling — not connected')).to_be_disabled()
        output = Path(__file__).resolve().parents[2] / 'docs' / 'outbound'
        output.mkdir(parents=True, exist_ok=True)
        page.evaluate('window.scrollTo(0, 0)')
        page.screenshot(path=str(output / 'desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'Mobile page overflows horizontally'
        page.screenshot(path=str(output / 'mobile.png'), full_page=True)
        assert not errors, errors
        context.close()
        viewer = browser.new_context(http_credentials={'username': 'viewer', 'password': 'test-password'})
        page = viewer.new_page(); page.goto('http://127.0.0.1:18084/outbound')
        expect(page.locator('#name')).to_have_value('September renewal follow-up')
        expect(page.locator('#name')).to_be_disabled()
        expect(page.locator('#new')).to_be_hidden()
        expect(page.locator('#export')).to_be_hidden()
        expect(page.locator('#calling-settings-button')).to_be_hidden()
        expect(page.locator('#voice-settings-button')).to_be_hidden()
        browser.close()
        print('Browser checks passed: provider settings, two caller numbers, default/selection persistence, actual template Voice AI panels, save, import, download, reload, disabled dialing, viewer controls, mobile layout; no JS errors.')
finally:
    server.should_exit = True
    thread.join(timeout=5)
    OutboundTests.tearDownClass()
