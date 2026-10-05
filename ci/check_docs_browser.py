"""Inspect built Sphinx pages in real browsers, including search and responsive themes."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--chromium',type=Path)
    parser.add_argument('--firefox',type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    root=args.root.resolve()
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(root)))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    results=[]
    try:
        with sync_playwright() as playwright:
            for name,executable in (('chromium',args.chromium),('firefox',args.firefox)):
                engine=getattr(playwright,name)
                browser=engine.launch(executable_path=str(executable) if executable else None)
                context=browser.new_context(viewport={'width':1440,'height':1000})
                page=context.new_page()
                errors=[]
                page.on('pageerror',lambda error: errors.append(str(error)))
                for path in sorted(root.rglob('*.html')):
                    response=page.goto(base+'/'+str(path.relative_to(root)),wait_until='load')
                    assert response.ok, path
                    assert page.locator('body').inner_text().strip(), path
                for path in ('index.html','content/openspliceai_variant.html','content/migration.html','content/function_manual.html'):
                    for width in (1440,390):
                        page.set_viewport_size({'width':width,'height':1000})
                        for theme in ('light','dark'):
                            page.goto(base+'/'+path,wait_until='load')
                            page.evaluate('(theme)=>{document.body.dataset.theme=theme}',theme)
                            overflow=page.evaluate('Math.max(document.body.scrollWidth,document.documentElement.scrollWidth)>innerWidth+1')
                            assert not overflow, f'{name} {path} {width} {theme}: horizontal overflow'
                            images=page.locator('.sidebar img').all()
                            for image in images:
                                if image.is_visible():
                                    assert image.evaluate('(image)=>image.complete&&image.naturalWidth>0')
                            page.screenshot(path=str(args.output_dir/f'{name}-{Path(path).stem}-{width}-{theme}.png'))
                            results.append({'browser':name,'page':path,'width':width,'theme':theme,'overflow':False})
                page.set_viewport_size({'width':1440,'height':1000})
                page.goto(base+'/search.html?q=focal',wait_until='load')
                page.wait_for_function('document.querySelectorAll("#search-results li").length>0',timeout=10000)
                assert not errors, errors
                context.close()
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
    (args.output_dir/'summary.json').write_text(json.dumps({'pages':len(list(root.rglob('*.html'))),
        'browsers':['chromium','firefox'],'search_passed':True,'checks':results},indent=2)+'\n')
    print(f'{len(results)} responsive/theme checks, all pages and search passed in two browsers')


if __name__=='__main__':
    main()
