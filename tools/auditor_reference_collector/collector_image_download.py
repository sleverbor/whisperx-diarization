"""Download operator-selected public HTTPS images into the private library."""
import ipaddress
import socket
from urllib.parse import urlparse
from urllib.request import Request,build_opener,HTTPRedirectHandler


def validate_image_url(url):
    p=urlparse(url)
    if p.scheme!='https' or not p.hostname or p.username or p.password or p.port not in (None,443):
        raise ValueError('Use a direct public HTTPS image URL')
    addresses=socket.getaddrinfo(p.hostname,443,type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(x[4][0]).is_global for x in addresses):
        raise ValueError('Private or local network image addresses are not allowed')
    return url


class CheckedRedirect(HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        validate_image_url(newurl)
        return super().redirect_request(request,fp,code,msg,headers,newurl)


def download_image(url):
    validate_image_url(url)
    request=Request(url,headers={'User-Agent':'AuditorReferenceCollector/1.0','Accept':'image/*'})
    with build_opener(CheckedRedirect()).open(request,timeout=25) as response:
        if not response.headers.get_content_type().startswith('image/'):
            raise ValueError('URL returned a web page, not an image. Copy the image address, not the Google result-page address.')
        data=response.read(20*1024*1024+1)
        if len(data)>20*1024*1024:raise ValueError('Image exceeds 20 MB')
        return data
