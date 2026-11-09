"""Vector PDF composition for the figures that contain the author's Illustrator schematics (Figs. 1 and 2).

PyMuPDF (``pymupdf`` / ``fitz``) is used when installed.  Otherwise pikepdf or pypdf (placement),
pypdfium2 (PNG preview) and pdfplumber (inventory) give the same result.  In all cases
page 1 of the schematic PDF is placed as vector content, uniformly scaled and centred in the target
rectangle (aspect ratio preserved); nothing is rasterised.  Rectangles are (x, y, w, h) in millimetres from
the lower-left corner of the page.
"""
from __future__ import annotations

MM_PT = 72.0 / 25.4


def _fitz():
    for name in ('pymupdf', 'fitz'):
        try:
            module = __import__(name)
        except ImportError:
            continue
        if hasattr(module, 'open'):
            return module
    return None


def backend():
    if _fitz() is not None:
        return 'pymupdf'
    try:
        import pikepdf, pypdfium2, pdfplumber  # noqa: F401
        return 'pikepdf'
    except ImportError:
        pass
    try:
        import pypdf, pypdfium2, pdfplumber  # noqa: F401
        return 'pypdf'
    except ImportError:
        return None


def require_backend():
    b = backend()
    if b is None:
        raise RuntimeError('PDF composition needs PyMuPDF (pip install pymupdf) '
                           'or (pikepdf or pypdf) + pypdfium2 + pdfplumber.')
    return b


def page_info(pdf):
    """(width_pt, height_pt, page_count) of page 1."""
    fz = _fitz()
    if fz is not None:
        with fz.open(str(pdf)) as doc:
            return float(doc[0].rect.width), float(doc[0].rect.height), doc.page_count
    try:
        import pikepdf
        with pikepdf.open(str(pdf)) as doc:
            x0, y0, x1, y1 = (float(v) for v in doc.pages[0].mediabox)
            return x1 - x0, y1 - y0, len(doc.pages)
    except ImportError:
        import pypdf
        reader = pypdf.PdfReader(str(pdf))
        mb = reader.pages[0].mediabox
        return float(mb.width), float(mb.height), len(reader.pages)


def page_text(pdf):
    fz = _fitz()
    if fz is not None:
        with fz.open(str(pdf)) as doc:
            return doc[0].get_text('text')
    import pdfplumber
    with pdfplumber.open(str(pdf)) as doc:
        return doc.pages[0].extract_text() or ''


def overlay(base_pdf, art_pdf, rect_mm, out_pdf, clip_pt=None):
    x, y, w, h = (float(v) for v in rect_mm)
    fz = _fitz()
    if fz is not None:
        with fz.open(str(base_pdf)) as target, fz.open(str(art_pdf)) as art:
            ph = target[0].rect.height
            rect = fz.Rect(x * MM_PT, ph - (y + h) * MM_PT, (x + w) * MM_PT, ph - y * MM_PT)
            kw = {'keep_proportion': True, 'overlay': True}
            if clip_pt is not None:
                kw['clip'] = fz.Rect(*clip_pt)
            target[0].show_pdf_page(rect, art, 0, **kw)
            target.save(str(out_pdf), garbage=4, deflate=True)
        return 'pymupdf'
    if clip_pt is not None:
        raise RuntimeError('Clipped placement (legacy Fig2a only) requires PyMuPDF.')
    try:
        import pikepdf
        with pikepdf.open(str(base_pdf)) as target, pikepdf.open(str(art_pdf)) as art:
            page = target.pages[0]
            rect = pikepdf.Rectangle(x * MM_PT, y * MM_PT, (x + w) * MM_PT, (y + h) * MM_PT)
            try:
                page.add_overlay(art.pages[0], rect)
            except Exception:
                page.add_overlay(target.copy_foreign(art.pages[0].as_form_xobject()), rect)
            target.save(str(out_pdf))
        return 'pikepdf'
    except ImportError:
        import pypdf
        from pypdf import Transformation
        target = pypdf.PdfReader(str(base_pdf))
        art = pypdf.PdfReader(str(art_pdf))
        writer = pypdf.PdfWriter()
        page = target.pages[0]
        art_page = art.pages[0]
        x0, y0, rw, rh = x * MM_PT, y * MM_PT, w * MM_PT, h * MM_PT
        aw, ah = float(art_page.mediabox.width), float(art_page.mediabox.height)
        s = min(rw / aw, rh / ah)
        tx = x0 + (rw - aw * s) / 2.0 - float(art_page.mediabox.left) * s
        ty = y0 + (rh - ah * s) / 2.0 - float(art_page.mediabox.bottom) * s
        page.merge_transformed_page(art_page, Transformation().scale(s, s).translate(tx, ty))
        writer.add_page(page)
        with open(out_pdf, 'wb') as f:
            writer.write(f)
        return 'pypdf'


def to_png(pdf, png, dpi=600):
    fz = _fitz()
    if fz is not None:
        with fz.open(str(pdf)) as doc:
            pix = doc[0].get_pixmap(matrix=fz.Matrix(dpi / 72., dpi / 72.), alpha=False)
            try:
                pix.set_dpi(int(dpi), int(dpi))
            except Exception:
                pass
            pix.save(str(png))
        return
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    try:
        img = doc[0].render(scale=dpi / 72.).to_pil().convert('RGB')
    finally:
        doc.close()
    img.save(str(png), dpi=(dpi, dpi))


def inventory(pdf):
    fz = _fitz()
    if fz is not None:
        with fz.open(str(pdf)) as doc:
            page = doc[0]
            return {'pages': doc.page_count,
                    'pdf_mm': [round(page.rect.width / MM_PT, 2), round(page.rect.height / MM_PT, 2)],
                    'vector_drawings': len(page.get_drawings()), 'embedded_rasters': len(page.get_images())}
    import pdfplumber
    with pdfplumber.open(str(pdf)) as doc:
        page = doc.pages[0]
        return {'pages': len(doc.pages),
                'pdf_mm': [round(float(page.width) / MM_PT, 2), round(float(page.height) / MM_PT, 2)],
                'vector_drawings': len(page.lines) + len(page.rects) + len(page.curves),
                'embedded_rasters': len(page.images)}


def text_outside(pdf, tol=1.0):
    """Text spans lying outside the page (should be empty)."""
    out = []
    fz = _fitz()
    if fz is not None:
        with fz.open(str(pdf)) as doc:
            page = doc[0]
            W, H = page.rect.width, page.rect.height
            for b in page.get_text('dict')['blocks']:
                for line in b.get('lines', []):
                    for sp in line.get('spans', []):
                        x0, y0, x1, y1 = sp['bbox']
                        if sp['text'].strip() and (x0 < -tol or y0 < -tol or x1 > W + tol or y1 > H + tol):
                            out.append(sp['text'])
        return out
    import pdfplumber
    with pdfplumber.open(str(pdf)) as doc:
        page = doc.pages[0]
        W, H = float(page.width), float(page.height)
        for ch in page.chars:
            if ch['x0'] < -tol or ch['top'] < -tol or ch['x1'] > W + tol or ch['bottom'] > H + tol:
                out.append(ch['text'])
    return out
