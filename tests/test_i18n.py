"""English UI coverage and display-only localisation invariants."""
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "webapp/static"
EN = STATIC / "en"


@pytest.fixture(scope="module", autouse=True)
def built_english_bundle():
    """webapp/static/en/ is generated and git-ignored, so rebuild it for a clean checkout."""
    spec = importlib.util.spec_from_file_location("build_english", ROOT / "scripts/20_build_english.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build()
VI = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯ\u1ea0-\u1ef9]")


@pytest.mark.parametrize("name", ["manager.html", "index.html", "manager.js", "app.js"])
def test_english_ui_has_no_untranslated_vietnamese(name):
    source = (EN / name).read_text(encoding="utf-8")
    assert not VI.search(source), name
    if name.endswith(".html"):
        assert '<html lang="en">' in source
        assert '/static/i18n.js' in source


@pytest.mark.skipif(not shutil.which("node"), reason="Node is needed for browser-runtime checks")
def test_localisation_preserves_forecasts_identifiers_and_raw_data():
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const context = {URL, Intl, window: {}, location: {href:'http://localhost/?lang=en'},
 document: {documentElement:{lang:'en'},addEventListener(){}},
 localStorage: {getItem(){return null},setItem(){}}};
vm.createContext(context);
for (const file of ['en/dictionary.js','i18n.js']) vm.runInContext(fs.readFileSync('webapp/static/'+file,'utf8'),context);
const i18n=context.window.I18N;
assert.equal(i18n.text('Đang lên'),'Emerging');
assert.equal(i18n.text('Ổn định'),'Stable');
assert.equal(i18n.text('Đang giảm'),'Declining');
assert.equal(i18n.date(new Date('2019-10-14T00:00:00Z')),'14 Oct 2019');
assert.equal(i18n.text('maximumFractionDigits'),'maximumFractionDigits');
assert.equal(i18n.text('long cardigan'),'long cardigan');
assert.equal(i18n.text('maxi'),'maxi dress');
assert.equal(i18n.query('Long-sleeved top'),'long sleeve');
assert.equal(i18n.text('Không có sản phẩm này trong dữ liệu.'),'This product was not found in the dataset.');
const source = {key:'áo dài tay', term:'long sleeve', category_vi:'áo dài tay',
 name:'Blouse sáng màu/pastel', p:[.1,.2,.7], code:123,
 data:[['áo sơ mi',2]], columns:['màu'], filename:'ảnh-gốc.jpg'};
const result=i18n.data(source);
assert.equal(result.category_vi,'long-sleeved top');
assert.equal(result.name,'Light-coloured and pastel blouses');
assert.equal(result.key,source.key); assert.equal(result.term,source.term);
assert.equal(result.filename,source.filename);
assert.equal(JSON.stringify(result.p),JSON.stringify(source.p));
assert.equal(JSON.stringify(result.data),JSON.stringify(source.data));
assert.equal(JSON.stringify(result.columns),JSON.stringify(source.columns));
assert.equal(i18n.data({name:'Phong cách của tôi',created:'2026-01-01'}).name,'Phong cách của tôi');
const vi=/[ăâđêôơưĂÂĐÊÔƠƯ\u1ea0-\u1ef9]/u;
for (const enc of ['resnet50','fashionclip']) {
 if (!fs.existsSync('outputs/web/'+enc+'/meta.json')) continue;
 const meta=JSON.parse(fs.readFileSync('outputs/web/'+enc+'/meta.json','utf8'));
 const en=i18n.data(meta);
 for (const style of en.styles) assert.equal(vi.test(style.name),false,style.name);
 assert.equal(JSON.stringify(en.forecasts),JSON.stringify(meta.forecasts));
}
console.log('English labels, dates, style names, and data integrity verified.');
"""
    result = subprocess.run(["node", "-e", script], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not shutil.which("node"), reason="Node is needed for syntax checks")
@pytest.mark.parametrize("name", ["en/manager.js", "en/app.js", "i18n.js", "en/dictionary.js"])
def test_english_javascript_syntax(name):
    result = subprocess.run(["node", "--check", str(STATIC / name)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
