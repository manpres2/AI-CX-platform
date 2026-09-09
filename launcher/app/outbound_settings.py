"""Saved calling configuration; saving never contacts a provider or places calls."""
import base64
import ctypes
import os
import re
from datetime import time
from typing import Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class CallerNumber(BaseModel):
    id: str = Field(pattern=r'^[a-zA-Z0-9-]{1,80}$')
    label: str = Field(min_length=1, max_length=80)
    phone: str = Field(pattern=r'^\+[1-9]\d{7,14}$')
    enabled: bool = True

    @field_validator('label')
    @classmethod
    def clean_label(cls, value):
        if not value.strip():
            raise ValueError('Enter a number label')
        return value.strip()


class CallingSettings(BaseModel):
    revision: int = Field(default=0, ge=0)
    provider: Literal['twilio'] = 'twilio'
    account_label: str = Field(default='', max_length=100)
    account_sid: str = Field(default='', pattern=r'^(AC[a-fA-F0-9]{32})?$')
    public_url: str = Field(default='', max_length=500)
    auth_token: SecretStr | None = None
    clear_auth_token: bool = False
    numbers: list[CallerNumber] = Field(default_factory=list, max_length=100)
    default_number_id: str = ''

    @field_validator('public_url')
    @classmethod
    def validate_url(cls, value):
        value = value.strip().rstrip('/')
        if value:
            parsed = urlsplit(value)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError('Use a public HTTPS base URL without credentials, query, or fragment')
        return value

    @model_validator(mode='after')
    def validate_settings(self):
        if len({n.id for n in self.numbers}) != len(self.numbers):
            raise ValueError('Number IDs must be unique')
        if len({n.phone for n in self.numbers}) != len(self.numbers):
            raise ValueError('Each caller number can only be added once')
        if self.default_number_id and not any(n.id == self.default_number_id and n.enabled for n in self.numbers):
            raise ValueError('The default must be an enabled caller number')
        token = self.auth_token.get_secret_value() if self.auth_token else ''
        if token and (len(token) > 512 or len(token) < 16):
            raise ValueError('The auth token must contain 16–512 characters')
        if token and self.clear_auth_token:
            raise ValueError('Choose either replace or remove the auth token')
        return self


class CallOptions(BaseModel):
    caller_number_id: str = Field(default='', max_length=80)
    timezone: str = Field(default='Asia/Kolkata', max_length=80)
    days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4], min_length=1, max_length=7)
    start_time: str = '09:00'
    end_time: str = '18:00'
    concurrent_calls: int = Field(default=1, ge=1, le=20)
    ring_timeout_seconds: int = Field(default=30, ge=10, le=120)
    max_call_minutes: int = Field(default=10, ge=1, le=60)
    max_attempts: int = Field(default=1, ge=1, le=5)
    retry_delay_minutes: int = Field(default=60, ge=5, le=10080)
    retry_busy: bool = True
    retry_no_answer: bool = True
    voicemail: Literal['hang_up', 'leave_message'] = 'hang_up'
    voicemail_message: str = Field(default='', max_length=1500)
    record_calls: bool = False
    allow_interruptions: bool = True
    disclosure: str = Field(default='', max_length=1000)
    opt_out_phrase: str = Field(default='Please do not call me again', max_length=300)
    transfer_number: str = Field(default='', pattern=r'^(\+[1-9]\d{7,14})?$')

    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError('Use an IANA timezone such as Asia/Kolkata or Europe/London')
        return value

    @field_validator('start_time', 'end_time')
    @classmethod
    def validate_time(cls, value):
        if not re.fullmatch(r'\d{2}:\d{2}', value):
            raise ValueError('Use HH:MM time')
        time.fromisoformat(value)
        return value

    @model_validator(mode='after')
    def validate_options(self):
        if any(day < 0 or day > 6 for day in self.days) or len(set(self.days)) != len(self.days):
            raise ValueError('Choose unique calling days, Monday (0) through Sunday (6)')
        if self.start_time >= self.end_time:
            raise ValueError('The calling window must end after it starts on the same day')
        if self.voicemail == 'leave_message' and not self.voicemail_message.strip():
            raise ValueError('Enter a voicemail message')
        return self


def protect_token(token: str) -> str:
    """Windows DPAPI binds the stored secret to the current OS account and machine."""
    if os.name != 'nt':
        raise RuntimeError('Secure credential storage currently requires a Windows host')
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]

    raw = token.encode('utf-8')
    buffer = ctypes.create_string_buffer(raw)
    incoming = Blob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    outgoing = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    crypt.CryptProtectData.argtypes = [ctypes.POINTER(Blob), wintypes.LPCWSTR, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    crypt.CryptProtectData.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    if not crypt.CryptProtectData(ctypes.byref(incoming), 'AI CX outbound', None, None, None, 1, ctypes.byref(outgoing)):
        raise RuntimeError('Unable to protect calling credential on this host')
    try:
        return base64.b64encode(ctypes.string_at(outgoing.data, outgoing.size)).decode('ascii')
    finally:
        kernel.LocalFree(outgoing.data)
