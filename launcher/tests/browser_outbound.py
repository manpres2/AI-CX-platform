"""Optional browser smoke test: pip install -r launcher/tests/requirements.txt.
Uses installed Microsoft Edge, temporary databases, and synthetic customer data.
Run: python launcher/tests/browser_outbound.py
"""
import threading
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
        page.get_by_label('Campaign name', exact=True).fill('September renewal follow-up')
        page.get_by_label('Customer care agent', exact=True).select_option('tech')
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
        browser.close()
        print('Browser checks passed: save, import, download, reload, disabled dialing, viewer controls, mobile layout; no JS errors.')
finally:
    server.should_exit = True
    thread.join(timeout=5)
    OutboundTests.tearDownClass()
