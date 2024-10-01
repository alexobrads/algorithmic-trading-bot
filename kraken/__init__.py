import requests
import time
import hashlib
import hmac
import base64
import urllib.parse

class KrakenRestAPI:
    def __init__(self, api_key=None, api_secret=None):
        self.base_url = 'https://api.kraken.com'
        self.api_key = api_key
        self.api_secret = api_secret

    def _make_public_request(self, endpoint, params=None):
        url = f"{self.base_url}/0/public/{endpoint}"
        response = requests.get(url, params=params)
        return response.json()

    def _make_private_request(self, endpoint, params=None):
        if params is None:
            params = {}

        url_path = f"/0/private/{endpoint}"
        url = f"{self.base_url}{url_path}"
        nonce = str(int(1000 * time.time()))  # Kraken requires a nonce
        params['nonce'] = nonce

        headers = self._generate_auth_headers(url_path, params)
        response = requests.post(url, data=params, headers=headers)
        return response.json()

    def _generate_auth_headers(self, url_path, data):
        # Create the message signature
        post_data = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + post_data).encode('utf-8')
        message = url_path.encode('utf-8') + hashlib.sha256(encoded).digest()

        # Decode the secret key and create the signature
        secret = base64.b64decode(self.api_secret)
        signature = hmac.new(secret, message, hashlib.sha512)
        signature_digest = base64.b64encode(signature.digest())

        headers = {
            'API-Key': self.api_key,
            'API-Sign': signature_digest.decode('utf-8')
        }
        return headers

    def get_ticker(self, pair):
        """Get the ticker information for a currency pair"""
        return self._make_public_request('Ticker', {'pair': pair})

    def get_ohlc(self, pair, interval=1, since=None):
        """
        Fetches OHLC (Open, High, Low, Close) data for a given trading pair.

        :param pair: The trading pair (e.g., 'BTCUSD').
        :param interval: Time frame interval in minutes. Options are: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600.
        :param since: Return committed OHLC data since given timestamp (optional).
        :return: JSON response from Kraken API with OHLC data.
        """
        params = {
            'pair': pair,
            'interval': interval
        }
        if since:
            params['since'] = since

        return self._make_public_request('OHLC', params)

