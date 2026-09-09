"""Calling configuration persistence, validation, and permission tests."""
import json
import sqlite3
import unittest
from contextlib import closing
from unittest.mock import patch, AsyncMock, MagicMock
import importlib
import httpx
from fastapi.testclient import TestClient
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials
import test_outbound as fixtures
import auth
from outbound import require_voice_access


class CallingSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.OutboundTests.setUpClass()
        cls.client = fixtures.OutboundTests.client
        cls.db = fixtures.OutboundTests.db
        auth.create_user('configurer', 'test-password', False, ['outbound.view', 'outbound.configure'])
        auth.create_user('voice-only', 'test-password', False, ['outbound.view', 'outbound.voice'])
        auth.create_user('voice-tech', 'test-password', False, ['outbound.view', 'outbound.voice', 'tech'])

    @classmethod
    def tearDownClass(cls):
        fixtures.OutboundTests.tearDownClass()

    def setUp(self):
        # Initialize schema, then isolate each test's configuration and drafts.
        self.request('GET', '/calling/settings')
        with closing(sqlite3.connect(self.db)) as con:
            for table in ('campaigns', 'leads', 'calling_settings', 'audit'):
                con.execute('DELETE FROM ' + table)
            con.commit()

    request = fixtures.OutboundTests.request
    draft = fixtures.OutboundTests.draft

    def settings(self, **changes):
        current = self.request('GET', '/calling/settings').json()
        data = {k: v for k, v in current.items() if k not in ('auth_token_set', 'connected')}
        data.update(changes)
        return self.request('PUT', '/calling/settings', 'configurer', json=data)

    def numbers(self):
        return [{'id':'sales', 'label':'Sales', 'phone':'+12025550101', 'enabled':True},
                {'id':'support', 'label':'Support', 'phone':'+12025550102', 'enabled':True}]

    def test_configuration_permission_is_separate(self):
        for user in ('viewer', 'manager', 'exporter', 'blocked'):
            self.assertEqual(self.request('GET', '/calling/settings', user).status_code, 403)
            self.assertEqual(self.request('PUT', '/calling/settings', user, json={}).status_code, 403)
        self.assertEqual(self.request('GET', '/calling/settings', 'configurer').status_code, 200)
        self.assertEqual(self.request('GET', '/calling/catalog', 'viewer').status_code, 200)
        catalog = self.request('GET', '/calling/catalog', 'viewer').json()
        self.assertNotIn('account_sid', catalog)
        self.assertNotIn('auth_token_set', catalog)

    def test_multiple_numbers_and_independent_campaign_selection(self):
        self.assertEqual(self.settings(numbers=self.numbers(), default_number_id='sales').status_code, 200)
        self.assertEqual(self.draft('one', calling={'caller_number_id':'sales'}).status_code, 200)
        self.assertEqual(self.draft('two', calling={'caller_number_id':'support'}).status_code, 200)
        self.assertEqual(self.settings(default_number_id='support').status_code, 200)
        campaigns = {c['id']: c for c in self.request('GET', '/campaigns').json()['campaigns']}
        self.assertEqual(campaigns['one']['calling']['caller_number_id'], 'sales')
        self.assertEqual(campaigns['two']['calling']['caller_number_id'], 'support')
        # Older clients omitting calling options must not erase them.
        self.assertEqual(self.draft('one').status_code, 200)
        campaigns = {c['id']: c for c in self.request('GET', '/campaigns').json()['campaigns']}
        self.assertEqual(campaigns['one']['calling']['caller_number_id'], 'sales')

    def test_number_identity_and_disable_rules(self):
        self.settings(numbers=self.numbers())
        self.draft('one', calling={'caller_number_id':'sales'})
        self.assertEqual(self.settings(numbers=[self.numbers()[1]]).status_code, 409)
        changed = self.numbers(); changed[0]['phone'] = '+12025550103'
        self.assertEqual(self.settings(numbers=changed).status_code, 409)
        disabled = self.numbers(); disabled[0]['enabled'] = False
        self.assertEqual(self.settings(numbers=disabled).status_code, 200)
        self.assertEqual(self.draft('new', calling={'caller_number_id':'sales'}).status_code, 400)
        self.assertEqual(self.draft('one', calling={'caller_number_id':'sales'}).status_code, 200)
        self.assertEqual(self.draft('new', calling={'caller_number_id':'missing'}).status_code, 400)

    def test_settings_validation(self):
        nums = self.numbers(); nums[1]['phone'] = nums[0]['phone']
        self.assertEqual(self.settings(numbers=nums).status_code, 422)
        self.assertEqual(self.settings(default_number_id='missing').status_code, 422)
        self.assertEqual(self.settings(public_url='http://example.com').status_code, 422)
        self.assertEqual(self.settings(public_url='https://user:secret@example.com').status_code, 422)
        self.assertEqual(self.settings(account_sid='not-a-sid').status_code, 422)
        self.assertEqual(self.settings(provider='unsupported').status_code, 422)

    def test_stale_settings_are_rejected(self):
        self.settings(account_label='First update')
        response = self.request('PUT', '/calling/settings', json={'revision':0, 'account_label':'Stale overwrite'})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.request('GET', '/calling/settings').json()['account_label'], 'First update')

    def test_credentials_protected_redacted_preserved_and_cleared(self):
        token = 'synthetic-token-for-unit-tests-only'
        response = self.settings(auth_token=token, account_sid='AC' + 'a' * 32)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['auth_token_set'])
        self.assertNotIn(token, response.text)
        self.assertNotIn('auth_token', response.json())
        with closing(sqlite3.connect(self.db)) as con:
            row = con.execute('SELECT config,protected_token FROM calling_settings').fetchone()
            self.assertNotIn(token, row[0]); self.assertNotIn(token, row[1]); self.assertTrue(row[1])
            self.assertNotIn(token, str(con.execute('SELECT * FROM audit').fetchall()))
        self.settings(account_label='Renamed')
        self.assertTrue(self.request('GET', '/calling/settings').json()['auth_token_set'])
        self.assertEqual(self.settings(account_sid='AC' + 'b' * 32).status_code, 400)
        self.assertEqual(self.settings(clear_auth_token=True).status_code, 200)
        self.assertFalse(self.request('GET', '/calling/settings').json()['auth_token_set'])

    def test_behavior_validation_and_roundtrip(self):
        for options in ({'timezone':'Mars/Olympus'}, {'days':[]}, {'days':[7]}, {'days':[1,1]},
                        {'start_time':'18:00', 'end_time':'09:00'}, {'start_time':'25:00'},
                        {'concurrent_calls':0}, {'max_attempts':6}, {'voicemail':'leave_message'},
                        {'transfer_number':'123'}):
            self.assertEqual(self.draft('invalid', calling=options).status_code, 422, options)
        valid = {'timezone':'Europe/London', 'max_attempts':3, 'record_calls':True,
                 'voicemail':'leave_message', 'voicemail_message':'Please call us back', 'days':[1,3]}
        self.assertEqual(self.draft('valid', calling=valid).status_code, 200)
        data = self.request('GET', '/campaigns').json()['campaigns'][0]['calling']
        for key, value in valid.items():
            self.assertEqual(data[key], value)

    def test_voice_proxy_forwards_settings_and_preserves_rbac(self):
        # Avoid touching production authentication/audit databases during import.
        with patch.object(auth, 'configure'), patch.object(auth, 'ensure_bootstrap_user'), patch('sqlite3.connect'):
            launcher = importlib.import_module('main_launcher')
        upstream = AsyncMock()
        upstream.request.return_value = httpx.Response(200, text='<html><head></head><body>Voice settings</body></html>',
                                                       headers={'Content-Type':'text/html'})
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=upstream)
        context.__aexit__ = AsyncMock(return_value=False)
        with TestClient(launcher.app) as client, patch.object(launcher.httpx, 'AsyncClient', return_value=context), patch.object(launcher, 'log_access'):
            response = client.get('/outbound/voice/tech/admin', auth=('voice-tech','test-password'))
            self.assertEqual(response.status_code, 200)
            self.assertIn("window.__BASE_PATH__='/outbound/voice/tech'", response.text)
            self.assertEqual(upstream.request.call_args.args[:2], ('GET', 'http://localhost:8001/admin'))
            upstream.request.return_value = httpx.Response(200, json={'status':'saved'})
            response = client.post('/outbound/voice/tech/admin/api/prompt', auth=('voice-tech','test-password'), json={'greeting':'Hello'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(upstream.request.call_args.args[:2], ('POST', 'http://localhost:8001/admin/api/prompt'))
            self.assertIn('authorization', upstream.request.call_args.kwargs['headers'])
            self.assertEqual(json.loads(upstream.request.call_args.kwargs['content']), {'greeting':'Hello'})
            before = upstream.request.call_count
            self.assertEqual(client.get('/outbound/voice/tech/admin', auth=('voice-only','test-password')).status_code, 403)
            self.assertEqual(client.get('/outbound/voice/portal/admin', auth=('voice-tech','test-password')).status_code, 404)
            self.assertEqual(client.get('/proxy/tech/admin', auth=('voice-tech','test-password')).status_code, 403)
            self.assertEqual(upstream.request.call_count, before)
            upstream.request.side_effect = httpx.ConnectError('offline')
            self.assertEqual(client.get('/outbound/voice/tech/admin', auth=('voice-tech','test-password')).status_code, 503)

    def test_voice_requires_both_outbound_and_bot_access(self):
        credentials = lambda name: HTTPBasicCredentials(username=name, password='test-password')
        for name in ('viewer', 'manager', 'voice-only', 'configurer'):
            with self.assertRaises(HTTPException) as error:
                require_voice_access('tech', credentials(name), {'tech'})
            self.assertEqual(error.exception.status_code, 403)
        self.assertEqual(require_voice_access('tech', credentials('voice-tech'), {'tech'}), 'voice-tech')
        self.assertEqual(require_voice_access('tech', credentials('owner'), {'tech'}), 'owner')
        with self.assertRaises(HTTPException) as error:
            require_voice_access('bank', credentials('voice-tech'), {'tech'})
        self.assertEqual(error.exception.status_code, 404)


if __name__ == '__main__':
    unittest.main()
