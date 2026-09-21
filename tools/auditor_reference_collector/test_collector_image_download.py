import unittest
from unittest.mock import patch
from collector_image_download import validate_image_url

class ImageURLTests(unittest.TestCase):
    def test_private_network_rejected(self):
        with patch('socket.getaddrinfo',return_value=[(2,1,6,'',('127.0.0.1',443))]):
            with self.assertRaises(ValueError):validate_image_url('https://example.com/a.jpg')
    def test_public_https_accepted(self):
        with patch('socket.getaddrinfo',return_value=[(2,1,6,'',('8.8.8.8',443))]):
            self.assertEqual(validate_image_url('https://example.com/a.jpg'),'https://example.com/a.jpg')
    def test_credentials_and_other_protocols_rejected(self):
        for url in ['file:///tmp/image','http://example.com/a.jpg','https://user:pass@example.com/a.jpg','https://example.com:8443/a.jpg']:
            with self.assertRaises(ValueError):validate_image_url(url)
if __name__=='__main__':unittest.main()
