"""Run: python -m unittest discover -s launcher/tests -v (uses temporary databases)."""
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook
import auth
from outbound import create_router, parse_upload, normalize_phone


class OutboundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        auth.configure(cls.root / 'users.db')
        auth.ensure_bootstrap_user('owner', 'test-password')
        cls.users = {}
        for name, keys in [('viewer', ['outbound.view']), ('manager', ['outbound.view', 'outbound.manage']),
                           ('exporter', ['outbound.view', 'outbound.export']), ('blocked', [])]:
            cls.users[name] = auth.create_user(name, 'test-password', False, keys)
        cls.app = FastAPI()
        cls.db = cls.root / 'campaigns.db'
        cls.app.include_router(create_router(cls.db, Path(__file__).resolve().parents[1] / 'static_launcher',
                                            lambda: [{'slug': 'tech', 'label': 'Tech Support'}]))
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.temp.cleanup()

    def request(self, method, path, user='owner', **kwargs):
        return self.client.request(method, '/admin/api/outbound' + path, auth=(user, 'test-password'), **kwargs)

    def draft(self, cid='sample', **updates):
        config = {'name': 'Renewals', 'agent': 'tech', 'purpose': 'Renew membership',
                  'questions': [{'field': 'interested', 'question': 'Would you renew?', 'type': 'yes_no'}]}
        config.update(updates)
        return self.request('PUT', '/campaigns/' + cid, json=config)

    def test_anonymous_denied(self):
        self.assertEqual(self.client.get('/admin/api/outbound/campaigns').status_code, 401)
        self.assertEqual(self.client.get('/outbound').status_code, 401)

    def test_viewer_cannot_write_or_export(self):
        self.assertEqual(self.request('GET', '/campaigns', 'viewer').status_code, 200)
        self.assertEqual(self.request('PUT', '/campaigns/no', 'viewer', json={'name': 'x', 'agent': 'tech'}).status_code, 403)
        self.assertEqual(self.request('POST', '/preview', 'viewer', files={'file': ('x.csv', b'Phone\n+919876543210')}).status_code, 403)
        self.assertEqual(self.request('GET', '/campaigns/sample/export', 'viewer').status_code, 403)

    def test_blocked_user_cannot_read(self):
        self.assertEqual(self.request('GET', '/campaigns', 'blocked').status_code, 403)
        self.assertEqual(self.client.get('/outbound/assets/outbound.js', auth=('blocked', 'test-password')).status_code, 403)

    def test_manage_and_export_are_separate(self):
        self.assertEqual(self.request('PUT', '/campaigns/managed', 'manager', json={'name': 'x', 'agent': 'tech'}).status_code, 200)
        self.assertEqual(self.request('GET', '/campaigns/managed/export', 'manager').status_code, 403)
        self.assertEqual(self.request('GET', '/campaigns/managed/export', 'exporter').status_code, 200)
        self.assertEqual(self.request('PUT', '/campaigns/managed', 'exporter', json={'name': 'x', 'agent': 'tech'}).status_code, 403)

    def test_save_persistence_and_validation(self):
        self.assertEqual(self.draft('persist').status_code, 200)
        items = self.request('GET', '/campaigns').json()['campaigns']
        self.assertTrue(any(c['id'] == 'persist' and c['status'] == 'Draft' for c in items))
        self.assertEqual(self.draft('bad', agent='missing').status_code, 400)
        self.assertEqual(self.draft('bad', name=' ').status_code, 400)
        self.assertEqual(self.draft('bad', questions=[{'field': 'x', 'question': 'x'}, {'field': 'X', 'question': 'y'}]).status_code, 400)

    def test_csv_mapping_and_ambiguity(self):
        data = parse_upload(b'Customer Name,Phone Number,Purpose\r\n"Doe, Jane",+919876543210,Renewal', 'leads.csv')
        self.assertEqual(data['suggested']['phone'], 'Phone Number')
        self.assertEqual(data['rows'][0][0], 'Doe, Jane')
        self.assertEqual(parse_upload(b'Phone,Mobile\n1,2', 'x.csv')['suggested']['phone'], '')

    def test_excel_upload(self):
        book = Workbook(); sheet = book.active
        sheet.append(['Name', 'Phone']); sheet.append(['Jane', '+919876543210'])
        data = io.BytesIO(); book.save(data); book.close()
        response = self.request('POST', '/preview', 'manager', files={'file': ('leads.xlsx', data.getvalue())})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['rows'][0][1], '+919876543210')

    def test_bad_uploads(self):
        for contents in [b'Phone,Phone\n1,2', b'Phone\n1,2', b'', b'Phone\n']:
            self.assertEqual(self.request('POST', '/preview', files={'file': ('x.csv', contents)}).status_code, 400)
        self.assertEqual(self.request('POST', '/preview', files={'file': ('x.csv', b'x' * (5*1024*1024+1))}).status_code, 413)

    def test_phone_normalization(self):
        self.assertEqual(normalize_phone('+91 (98765) 43210', ''), '+919876543210')
        self.assertEqual(normalize_phone('00919876543210', ''), '+919876543210')
        self.assertEqual(normalize_phone('9876543210', '+91'), '+919876543210')
        self.assertEqual(normalize_phone('9876543210', ''), '')
        self.assertEqual(normalize_phone('call me', '+91'), '')

    def test_import_dedupe_rejections_and_safe_export(self):
        self.draft('imports')
        payload = {'headers': ['Name', 'Phone', 'Purpose'], 'phone': 'Phone', 'name': 'Name', 'purpose': 'Purpose',
                   'rows': [['=HYPERLINK("bad")', '+919876543210', ''], ['Jane', '+91 98765 43210', 'Renew'], ['Bad', '123', 'Renew']]}
        first = self.request('POST', '/campaigns/imports/leads', json=payload).json()
        self.assertEqual(first['imported'], 1); self.assertEqual(first['duplicates'], 1)
        self.assertEqual(len(first['rejected']), 1)
        second = self.request('POST', '/campaigns/imports/leads', json=payload).json()
        self.assertEqual(second['imported'], 0)
        leads = self.request('GET', '/campaigns/imports/leads').json()['leads']
        self.assertEqual(leads[0]['purpose'], 'Renew membership')
        text = self.request('GET', '/campaigns/imports/export', 'exporter').text.lstrip('\ufeff')
        rows = list(csv.reader(io.StringIO(text)))
        self.assertTrue(rows[1][1].startswith("'=")); self.assertEqual(rows[1][4], 'Not called')
        self.assertEqual(rows[1][-1], '')

    def test_missing_purpose_and_invalid_mapping(self):
        self.draft('no-purpose', purpose='')
        data = {'headers': ['Phone'], 'rows': [['+919876543210']], 'phone': 'Phone'}
        result = self.request('POST', '/campaigns/no-purpose/leads', json=data).json()
        self.assertEqual(len(result['rejected']), 1)
        data['name'] = 'missing'
        self.assertEqual(self.request('POST', '/campaigns/no-purpose/leads', json=data).status_code, 400)

    def test_permission_revocation_immediate(self):
        uid = auth.create_user('temporary', 'test-password', False, ['outbound.view'])
        self.assertEqual(self.request('GET', '/campaigns', 'temporary').status_code, 200)
        auth.set_permissions(uid, [])
        self.assertEqual(self.request('GET', '/campaigns', 'temporary').status_code, 403)

    def test_no_dialing_endpoint(self):
        self.assertEqual(self.request('POST', '/campaigns/sample/start').status_code, 404)

    def test_page_and_assets(self):
        self.assertEqual(self.client.get('/outbound', auth=('viewer', 'test-password')).status_code, 200)
        self.assertEqual(self.client.get('/outbound/assets/outbound.js', auth=('viewer', 'test-password')).status_code, 200)
        self.assertEqual(self.client.get('/outbound/assets/no.txt', auth=('viewer', 'test-password')).status_code, 404)


if __name__ == '__main__':
    unittest.main()
