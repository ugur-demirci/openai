#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
translate.py (v0.4.1-argos-fix)
- PDF→PDF: stable (span tabanlı) + ocr (PaddleOCR) + ocr_doctr (PyTorch/docTR)
- Çeviri motoru seçimi:
    * ARGOS: tamamen offline, yerel paketlerden (argostranslate >=1.9 API uyumlu)
    * Ollama: http://127.0.0.1:11434 (model adı ile)
- NotoSans fontfile gömme (varsa); yoksa Helvetica
- FastAPI:
    /translate (async, background thread)
    /status/{task_id}, /logs/{task_id}, /download/{task_id}
    /metrics (CPU/GPU), /models (ARGOS + Ollama tags), root UI
- Tunables (env): OCR_SCALE, BATCH_SIZE, OLLAMA_TIMEOUT
"""
from __future__ import annotations

import argparse
import os
# --- force ArgosTranslate to use CUDA (CTranslate2) ---
try:
    # only set if not explicitly provided
    if os.environ.get("ARGOS_DEVICE_TYPE") not in ("cuda", "auto"):
        os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
except Exception:
    pass

import sys
import time
import uuid
import json
import traceback
import threading
import re
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import uvicorn

# ---- Opsiyonel metrikler ----
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore

# ---- PaddleOCR (GPU/CPU) ----
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:  # pragma: no cover
    PaddleOCR = None  # type: ignore

try:
    import paddle  # type: ignore
except Exception:  # pragma: no cover
    paddle = None  # type: ignore

# ---- docTR (PyTorch) OCR ----
try:
    import torch  # type: ignore
    from doctr.io import DocumentFile  # type: ignore
    from doctr.models import ocr_predictor  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    DocumentFile = None  # type: ignore
    ocr_predictor = None  # type: ignore

# ---- FastText language detection ----
try:
    import fasttext  # type: ignore
    _FT = fasttext.load_model(os.path.expanduser("~/apps/ai/models/lid.176.ftz"))
except Exception:  # pragma: no cover
    _FT = None

# ---- Argos Translate (modern API) ----
try:
    from argostranslate import translate as _argo  # type: ignore
except Exception:  # pragma: no cover
    _argo = None  # type: ignore

# ------------------------------------------------------------
# Konfig
# ------------------------------------------------------------
NOTO_FONT_DIR = os.path.expanduser('~/apps/ai/fonts/Noto')
FONT_MAP = {
    'regular': os.path.join(NOTO_FONT_DIR, 'NotoSans-Regular.ttf'),
    'bold': os.path.join(NOTO_FONT_DIR, 'NotoSans-Bold.ttf'),
    'italic': os.path.join(NOTO_FONT_DIR, 'NotoSans-Italic.ttf'),
    'bolditalic': os.path.join(NOTO_FONT_DIR, 'NotoSans-BoldItalic.ttf'),
}
TMP_DIR = os.path.expanduser('~/apps/ai/tmp')
LOG_DIR = os.path.expanduser('~/apps/ai/logs')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Performans ayarları (env)
OCR_SCALE = float(os.environ.get('OCR_SCALE', '2.0'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '12'))
OLLAMA_TIMEOUT = float(os.environ.get('OLLAMA_TIMEOUT', '180'))

# ------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------

def _choose_fontfile(span_fontname: str) -> Tuple[str, bool]:
    """Span font adına göre bold/italic heuristics -> NotoSans eşlemesi.
    True dönerse fontfile kullan, False dönerse yerleşik Helvetica adı kullan.
    """
    fname = span_fontname or ''
    is_bold = ('Bold' in fname) or ('SemiBold' in fname) or ('Medium' in fname and 'Italic' not in fname)
    is_italic = ('Italic' in fname) or ('Oblique' in fname)
    key = 'bolditalic' if (is_bold and is_italic) else ('bold' if is_bold else ('italic' if is_italic else 'regular'))
    ff = FONT_MAP.get(key)
    if ff and os.path.exists(ff):
        return ff, True
    reg = FONT_MAP.get('regular')
    if reg and os.path.exists(reg):
        return reg, True
    return 'helv', False  # Helvetica yerleşik ad


def _rgb_from_int(i: int) -> Tuple[float, float, float]:
    r, g, b = ((i >> 16) & 255), ((i >> 8) & 255), (i & 255)
    return (r/255.0, g/255.0, b/255.0)


def detect_lang(t: str) -> Optional[str]:
    if not _FT or not t.strip():
        return None
    try:
        lbl = _FT.predict(t)[0][0]
        return lbl.replace("__label__", "")
    except Exception:
        return None

# ------------------------------------------------------------
# ÇEVİRİ MOTORLARI
# ------------------------------------------------------------

def _ollama_translate(texts: List[str], model: str, src: str, tgt: str) -> List[str]:
    """Batch çeviri: timeout ve retry ile; eko/çevirisiz yanıta STRICT retry uygular."""
    if not texts:
        return []

    def _make_prompt(t: str, strict: bool = False) -> str:
        auto = (src or "").strip().lower() == "auto"
        if auto:
            base = f"Translate the following text into {tgt}.\nReturn only the translation without commentary."
        else:
            base = (
                "You are a professional translator. Preserve meaning, tone, and style.\n"
                "Return only the translation without commentary.\n"
                f"Source language: {src}\nTarget language: {tgt}\nText:"
            )
        if strict:
            # Eko / çevirmeme durumunda daha net talimat
            if auto:
                base = f"Translate strictly into {tgt}. Return ONLY the translation text. Do NOT repeat the source. Do NOT explain."
            else:
                base = (
                    "You are a STRICT translator. Output ONLY the target-language translation.\n"
                    "Do NOT echo the source. Do NOT add notes or brackets.\n"
                    f"Source language: {src}\nTarget language: {tgt}\nText:"
                )
        return base + f"\n<<<\n{t}\n>>>"

    try:
        to = httpx.Timeout(connect=5.0, read=OLLAMA_TIMEOUT, write=OLLAMA_TIMEOUT, pool=OLLAMA_TIMEOUT)
    except Exception:
        to = OLLAMA_TIMEOUT

    bs = max(1, min(int(os.environ.get("BATCH_SIZE", str(BATCH_SIZE))), 16))
    out: List[str] = []

    def _one_call(client: httpx.Client, payload: dict, retries: int = 1) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                r = client.post("http://127.0.0.1:11434/api/generate", json=payload)
                r.raise_for_status()
                return (r.json().get("response") or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise last_err  # type: ignore

    def _norm(x: str) -> str:
        return re.sub(r"\s+", " ", (x or "")).strip()

    thread_task = getattr(threading.current_thread(), "task_id", None)

    with httpx.Client(timeout=to) as c:
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            for t in chunk:
                payload = {"model": model, "prompt": _make_prompt(t, strict=False), "stream": False, "options": {"temperature": 0.2}}
                try:
                    resp = _one_call(c, payload, retries=1)
                    ns, nr = _norm(t), _norm(resp)
                    need_retry = (not nr) or (nr.lower() == ns.lower()) or (nr in ns) or (ns in nr)
                    if need_retry:
                        # STRICT retry
                        payload["prompt"] = _make_prompt(t, strict=True)
                        try:
                            resp2 = _one_call(c, payload, retries=1)
                        except Exception as e2:
                            if thread_task:
                                try: task_log(thread_task, f"Ollama strict-retry error: {e2}")
                                except Exception: pass
                            resp2 = ""
                        nr2 = _norm(resp2)
                        if nr2 and (nr2.lower() != ns.lower()) and (nr2 not in ns):
                            out.append(resp2)
                            continue
                        else:
                            if thread_task:
                                try: task_log(thread_task, "Ollama: untranslated/echo → keeping source")
                                except Exception: pass
                            out.append(t)
                    else:
                        out.append(resp)
                except Exception as e:
                    if thread_task:
                        try: task_log(thread_task, f"Ollama error: {e}")
                        except Exception: pass
                    out.append(t)
    return out

# ------------------------------------------------------------
# STABLE modu (span tabanlı)
# ------------------------------------------------------------

def _extract_spans(page: fitz.Page) -> List[dict]:
    data = page.get_text("rawdict")
    spans: List[dict] = []
    for b in data.get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                txt = s.get("text", "")
                if txt.strip():
                    spans.append({
                        "bbox": s.get("bbox"),
                        "text": txt,
                        "size": float(s.get("size", 10)),
                        "color": _rgb_from_int(int(s.get("color", 0))),
                        "font": s.get("font", "NotoSans"),
                    })
    return spans


def _stable_translate_page(page: fitz.Page, model: str, src: str, tgt: str) -> None:

    spans = _extract_spans(page)

    if not spans:

        spans = _extract_spans_textpage(page)

        if not spans:

            task_log(getattr(threading.current_thread(), "task_id", "NA"), "stable: low span density -> fallback to docTR")

            _ocr_doctr_translate_page(page, model, src, tgt)

            return

    sample = " ".join(s["text"] for s in spans[:8])[:2000]

    translations = _translate_batch([s["text"] for s in spans], model, src, tgt, sample_text=sample)

    for s in spans:

        page.add_redact_annot(fitz.Rect(s["bbox"]), fill=None)

    for s, t in zip(spans, translations):

        rect = fitz.Rect(s["bbox"])

        page.insert_textbox(rect, t, fontsize=float(s.get("size", 10)), fontname=s.get("font", "NotoSans"), color=s.get("color", [0, 0, 0]), align=0)

# OCR modları
# ------------------------------------------------------------

_OCR_INSTANCE = None

def _src_to_paddle_lang(src: str) -> str:
    m = {
        'tr': 'tr', 'en': 'en', 'de': 'de', 'fr': 'fr', 'es': 'es', 'ru': 'ru', 'nl': 'latin', 'uk': 'cyrillic'
    }
    s = (src or '').lower()
    if s == 'auto':
        return 'en'
    return m.get(s, 'en')

def _get_ocr(lang: str = 'en'):
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        if PaddleOCR is None:
            raise RuntimeError("PaddleOCR yüklü değil. 'pip install paddleocr paddlepaddle-gpu' gerekli.")
        try:
            if paddle:
                # CUDA varsa GPU, yoksa CPU
                paddle.device.set_device('gpu' if paddle.device.is_compiled_with_cuda() else 'cpu')
        except Exception as e:
            print('Paddle device init warning:', e)
        _OCR_INSTANCE = PaddleOCR(
            use_angle_cls=False,
            use_gpu=(paddle.device.is_compiled_with_cuda() if paddle else False),
            lang=lang
        )
    return _OCR_INSTANCE


def _page_to_png(page: fitz.Page, scale: Optional[float] = None) -> Tuple[str, float]:
    import tempfile
    s = scale if (isinstance(scale, (int, float)) and (scale or 0) > 0) else 2.0
    mat = fitz.Matrix(s, s)
    pix = page.get_pixmap(matrix=mat)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pix.save(tmp.name)
    tmp_png = tmp.name
    tmp.close()
    return tmp_png, s


def _ocr_translate_page(page: fitz.Page, model: str, src: str, tgt: str) -> None:
    ocr_lang = _src_to_paddle_lang(src)
    ocr = _get_ocr(lang=ocr_lang)
    png_path, scale = _page_to_png(page, scale=OCR_SCALE)
    try:
        result = ocr.ocr(png_path, cls=False)
        boxes: List[Tuple[float, float, float, float]] = []
        texts: List[str] = []
        count_lines = 0
        for line in (result[0] or []):
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = line[0]
            rx0 = min(x0, x1, x2, x3) / scale
            ry0 = min(y0, y1, y2, y3) / scale
            rx1 = max(x0, x1, x2, x3) / scale
            ry1 = max(y0, y1, y2, y3) / scale
            boxes.append((rx0, ry0, rx1, ry1))
            texts.append(line[1][0])
            count_lines += 1
        if not texts:
            task_log(getattr(threading.current_thread(), 'task_id', 'NA'), "PaddleOCR: no text detected")
            return
        task_log(getattr(threading.current_thread(), 'task_id', 'NA'), f"PaddleOCR: lines={count_lines}, to_translate={len(texts)}")
        sample = " ".join(texts[:40])[:2000]
        translations = _translate_batch(texts, model, src, tgt, sample_text=sample)
        for bb, txt in zip(boxes, translations):
            rect = fitz.Rect(bb)
            fontsize = max(10, min(28, rect.height))
            fontfile, is_file = _choose_fontfile("NotoSans")
            if is_file:
                page.insert_textbox(rect, txt, fontfile=fontfile, fontsize=fontsize, color=(0.0, 0.0, 0.0))
            else:
                page.insert_textbox(rect, txt, fontname=fontfile, fontsize=fontsize, color=(0.0, 0.0, 0.0))
    finally:
        try:
            os.remove(png_path)
        except Exception:
            pass

# ---- docTR (PyTorch) OCR ----
_DOCTR_PRED = None

def _get_doctr():
    global _DOCTR_PRED
    if _DOCTR_PRED is None:
        if ocr_predictor is None:
            raise RuntimeError("docTR kurulu değil veya import başarısız. 'pip install python-doctr[torch]'")
        device = 'cuda' if (torch and hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
        pred = ocr_predictor(pretrained=True)
        if pred is None:
            raise RuntimeError("docTR ocr_predictor() None döndü.")
        _DOCTR_PRED = pred.to(device)
    return _DOCTR_PRED


def _ocr_doctr_translate_page(page: fitz.Page, model: str, src: str, tgt: str) -> None:
    if DocumentFile is None:
        raise RuntimeError("docTR DocumentFile kullanılamıyor (import başarısız).")
    png_path, _ = _page_to_png(page, scale=max(OCR_SCALE, 3.0))
    try:
        pred = _get_doctr()
        doc = DocumentFile.from_images([png_path])
        res = pred(doc)
        export = res.export()  # normalized [0,1]
        if not export.get('pages'):
            task_log(getattr(threading.current_thread(), 'task_id', 'NA'), "docTR: no pages in export")
            return
        page_w, page_h = page.rect.width, page.rect.height
        texts: List[str] = []
        boxes: List[Tuple[float, float, float, float]] = []
        count_lines = 0
        for blk in export['pages'][0].get('blocks', []):
            for line in blk.get('lines', []):
                line_text = " ".join([w.get('value', '') for w in line.get('words', [])]).strip()
                if not line_text:
                    continue
                (x0, y0), (x1, y1) = line.get('geometry', ((0, 0), (0, 0)))
                rx0, ry0 = x0 * page_w, y0 * page_h
                rx1, ry1 = x1 * page_w, y1 * page_h
                texts.append(line_text)
                boxes.append((rx0, ry0, rx1, ry1))
                count_lines += 1
        if not texts:
            task_log(getattr(threading.current_thread(), 'task_id', 'NA'), "docTR: no text detected")
            return
        task_log(getattr(threading.current_thread(), 'task_id', 'NA'), f"docTR: lines={count_lines}, to_translate={len(texts)}")
        sample = " ".join(texts[:40])[:2000]
        translations = _translate_batch(texts, model, src, tgt, sample_text=sample)
        for bb, txt in zip(boxes, translations):
            rect = fitz.Rect(bb)
            fontsize = max(10, min(28, rect.height))
            fontfile, is_file = _choose_fontfile("NotoSans")
            if is_file:
                page.insert_textbox(rect, txt, fontfile=fontfile, fontsize=fontsize, color=(0.0, 0.0, 0.0))
            else:
                page.insert_textbox(rect, txt, fontname=fontfile, fontsize=fontsize, color=(0.0, 0.0, 0.0))
    finally:
        try:
            os.remove(png_path)
        except Exception:
            pass

# ------------------------------------------------------------
# Ana PDF dönüştürücü
# ------------------------------------------------------------

def translate_pdf(in_path: str, out_path: str, src: str, tgt: str, *, model: str = "ARGOS", pdf_mode: str = "stable", rebuild_output: bool = False, task_id: Optional[str] = None) -> None:
    doc = fitz.open(in_path)
    # Rebuild tüm belge: tamamen yeniden çizim
    if rebuild_output:
        doc.close()
        translate_pdf_rebuild(in_path, out_path, src, tgt, model=model, task_id=task_id)
        return
    pages_total = len(doc)
    start_time = time.time()
    if task_id:
        task_update(task_id, pages_total=pages_total, status="running")
        task_log(task_id, f"Started translation with {pages_total} pages")

    for page_index, page in enumerate(doc, start=1):
        if task_id:
            task_update(task_id, page=page_index, progress=round((page_index - 1) / pages_total * 100, 1))
            task_log(task_id, f"Processing page {page_index}/{pages_total}")

        if pdf_mode == "stable":
            _stable_translate_page(page, model, src, tgt)
        elif pdf_mode == "ocr":
            _ocr_translate_page(page, model, src, tgt)
        elif pdf_mode == "ocr_doctr":
            _ocr_doctr_translate_page(page, model, src, tgt)
        elif pdf_mode == "ocr_rebuild":
            doc.close()
            translate_pdf_rebuild(in_path, out_path, src, tgt, model=model, task_id=task_id)
            return
        else:
            raise ValueError("pdf_mode 'stable' veya 'ocr' veya 'ocr_doctr' veya 'ocr_rebuild' olmalı.")

        if task_id:
            elapsed = max(0.001, time.time() - start_time)
            done = page_index
            eta = (elapsed / done) * max(0, pages_total - done)
            task_update(task_id, progress=round(done / pages_total * 100, 1), eta_sec=int(eta))
            task_log(task_id, f"Page {page_index}/{pages_total} done, ETA ~{int(eta)}s")

    doc.save(out_path, garbage=4, deflate=True, clean=True)

    if task_id:
        task_update(task_id, progress=100.0)
        task_log(task_id, f"Saved output -> {out_path}")
    doc.close()

# ------------------------------------------------------------
# Task / Progress Registry (thread-safe)
# ------------------------------------------------------------

TASKS: dict[str, dict] = {}
_TASK_LOCK = threading.Lock()


def _now_iso() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _gpu_snapshot():
    try:
        if pynvml:
            n = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append({
                    "index": i,
                    "name": str(pynvml.nvmlDeviceGetName(h)),
                    "gpu_util": int(getattr(util, 'gpu', 0)),
                    "mem_util": int(100 * mem.used / mem.total) if getattr(mem, 'total', 0) else 0,
                })
            return gpus
    except Exception:
        pass
    return None


def _cpu_snapshot():
    try:
        if psutil:
            return float(psutil.cpu_percent(interval=0.0))
    except Exception:
        pass
    return None


def _task_write_snap(task_id: str) -> None:
    with _TASK_LOCK:
        d = TASKS.get(task_id)
        if not d:
            return
        try:
            with open(d["snap_path"], "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def task_init(task_id: str, meta: dict) -> dict:
    d = {
        "task_id": task_id,
        "status": "queued",
        "progress": 0.0,
        "page": 0,
        "pages_total": 0,
        "mode": meta.get("pdf_mode"),
        "src": meta.get("src"),
        "tgt": meta.get("tgt"),
        "model": meta.get("model"),
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "eta_sec": None,
        "cpu_pct": None,
        "gpu": None,
        "output_path": None,
        "error": None,
        "log_path": str(os.path.join(LOG_DIR, f"{task_id}.log")),
        "snap_path": str(os.path.join(LOG_DIR, f"{task_id}.json")),
    }
    with _TASK_LOCK:
        TASKS[task_id] = d
    try:
        with open(d["log_path"], "a", encoding="utf-8") as f:
            f.write(f"[{_now_iso()}] TASK INIT: task_id={task_id} mode={d['mode']} src={d['src']} tgt={d['tgt']} model={d['model']}\n")
    except Exception:
        pass
    _task_write_snap(task_id)
    return d


def task_log(task_id: str, msg: str) -> None:
    line = f"[{_now_iso()}] {msg}\n"
    try:
        with open(os.path.join(LOG_DIR, f"{task_id}.log"), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def task_update(task_id: str, **kw) -> None:
    with _TASK_LOCK:
        d = TASKS.get(task_id)
        if not d:
            return
        d.update(kw)
        d["updated_at"] = _now_iso()
        d["cpu_pct"] = _cpu_snapshot()
        d["gpu"] = _gpu_snapshot()
    _task_write_snap(task_id)


def task_error(task_id: str, err: Exception) -> None:
    task_update(task_id, status="error", error=str(err))
    task_log(task_id, f"ERROR: {err}")


def task_done(task_id: str, output_path: str) -> None:
    task_update(task_id, status="done", progress=100.0, output_path=output_path)
    task_log(task_id, f"DONE: {output_path}")

# ------------------------------------------------------------
# FastAPI App & Endpoints
# ------------------------------------------------------------

app = FastAPI(title="offlineAI translate-api", version="v0.4.1-argos-fix")


@app.post("/translate")
async def api_translate(
    file: UploadFile = File(...),
    src: str = Form(...),
    tgt: str = Form(...),
    model: str = Form("ARGOS"),
    pdf_mode: str = Form("stable"),
    rebuild_output: bool = Form(False),
):
    task_id = uuid.uuid4().hex
    meta = {"pdf_mode": pdf_mode, "src": src, "tgt": tgt, "model": model, "rebuild_output": rebuild_output}
    task_init(task_id, meta)
    task_log(task_id, "received request")

    ts = time.strftime("%Y%m%d-%H%M%S")
    in_path = os.path.join(TMP_DIR, f"upload_{ts}_{task_id}.pdf")
    out_path = os.path.join(TMP_DIR, f"translated_{ts}_{task_id}.pdf")
    try:
        with open(in_path, "wb") as f:
            f.write(await file.read())
        task_update(task_id, input_path=in_path, original_name=getattr(file, 'filename', None))
        task_log(task_id, "translation started")

        t = threading.Thread(target=_run_translate, args=(in_path, out_path, src, tgt, model, pdf_mode, task_id, rebuild_output), daemon=True)
        t.start()

        return JSONResponse({"ok": True, "task_id": task_id})
    except Exception as e:
        try:
            task_log(task_id, "TRACEBACK:\n" + traceback.format_exc())
        except Exception:
            pass
        task_error(task_id, e)
        return JSONResponse({"ok": False, "task_id": task_id, "error": str(e)}, status_code=500)


@app.get("/metrics")
async def metrics():
    info: dict = {"cpu": None, "gpu": None}
    if psutil:
        info["cpu"] = {"percent": psutil.cpu_percent(interval=0.1)}
        try:
            info["ram"] = {"percent": psutil.virtual_memory().percent}
        except Exception:
            info["ram"] = None
    if pynvml:
        try:
            n = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append({
                    "index": i,
                    "name": str(pynvml.nvmlDeviceGetName(h)),
                    "gpu_util": int(getattr(util, 'gpu', 0)),
                    "mem_util": int(100 * mem.used / mem.total) if getattr(mem, 'total', 0) else 0,
                })
            info["gpu"] = gpus
        except Exception:
            info["gpu"] = None
    return info


@app.get("/status/{task_id}")
def get_status(task_id: str):
    d = None
    with _TASK_LOCK:
        d = TASKS.get(task_id)
    if not d and os.path.exists(os.path.join(LOG_DIR, f"{task_id}.json")):
        try:
            with open(os.path.join(LOG_DIR, f"{task_id}.json"), "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            d = None
    if not d:
        return JSONResponse({"ok": False, "error": "task_id not found"}, status_code=404)
    return JSONResponse({"ok": True, "status": d})


@app.get("/logs/{task_id}")
def get_logs(task_id: str, tail: int = 200):
    d = None
    with _TASK_LOCK:
        d = TASKS.get(task_id)
    log_path = d["log_path"] if d else os.path.join(LOG_DIR, f"{task_id}.log")
    if not os.path.exists(log_path):
        return JSONResponse({"ok": True, "log": ""})
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        tail = max(1, min(int(tail), 5000))
        return JSONResponse({"ok": True, "log": "".join(lines[-tail:])})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/")
def ui() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>offlineAI — PDF Çeviri</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 12px}
  h1{font-size:20px;margin:0 0 12px}
  fieldset{border:1px solid #ddd;border-radius:10px;padding:12px;margin-bottom:12px}
  label{display:block;margin:6px 0 4px}
  input[type=file],select,input[type=text]{width:100%;padding:8px;border:1px solid #ccc;border-radius:8px;box-sizing:border-box}
  button{padding:10px 16px;border:0;border-radius:10px;background:#111;color:#fff;cursor:pointer}
  button:disabled{opacity:.6;cursor:not-allowed}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .bar{height:12px;background:#eee;border-radius:6px;overflow:hidden;position:relative}
  .bar>span{display:block;height:100%;background:#0a84ff;width:0%}
  .bar.small{height:16px}
  .bar em{position:absolute;left:50%;top:0;height:100%;display:flex;align-items:center;transform:translateX(-50%);font-size:12px;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.6)}
  .bars3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:10px}
  pre{background:#0b0b0b;color:#d7ffd9;padding:10px;border-radius:8px;max-height:240px;overflow:auto}
  .muted{color:#666;font-size:12px}
</style>
</head>
<body>
  <h1>offlineAI — PDF Çeviri (PDF→PDF)</h1>

  <form id="f" onsubmit="return false">
    <fieldset>
      <legend>Dosya &amp; Ayarlar</legend>
      <label>PDF Yükle</label>
      <input id="file" name="file" type="file" accept="application/pdf" required>

      <div class="row">
        <div>
          <label>Kaynak Dil (src)</label>
          <select id="src" name="src">
<option value="auto" selected>Otomatik (algıla)</option>
<option value="tr">Türkçe</option>
<option value="en">İngilizce</option>
<option value="fr">Fransızca</option>
<option value="de">Almanca</option>
<option value="es">İspanyolca</option>
<option value="ru">Rusça</option>
<option value="nl">Felemenkçe</option>
<option value="uk">Ukraynaca</option>
</select>
        </div>
        <div>
          <label>Hedef Dil (tgt)</label>
          <select id="tgt" name="tgt">
<option value="tr">Türkçe</option>
<option value="en" selected>İngilizce</option>
<option value="fr">Fransızca</option>
<option value="de">Almanca</option>
<option value="es">İspanyolca</option>
<option value="ru">Rusça</option>
<option value="nl">Felemenkçe</option>
<option value="uk">Ukraynaca</option>
</select>
        </div>
      </div>

      <div class="row">
        <div>
          <label>PDF Modu</label>
          <select id="pdf_mode" name="pdf_mode">
            <option value="stable">stable (span)</option>
            <option value="ocr">ocr (Paddle)</option>
            <option value="ocr_doctr" selected>ocr_doctr (PyTorch)</option>
            <option value="ocr_rebuild">ocr_rebuild (Paddle+ReportLab rebuild)</option>
</select>
        </div>
        <div>
          <label>Model</label>
          <select id="model" name="model"><option value="ARGOS">ARGOS</option></select>
        </div>
      </div>

      <div style="margin-top:12px;display:flex;gap:8px">
        <button id="btn" onclick="start()">Translate</button>
        <button type="button" onclick="clearLog()">Log Temizle</button>
        <button type="button" onclick="loadModels()">Ollama Yenile</button>
      </div>
    </fieldset>
  </form>

  <fieldset>
    <legend>Durum</legend>
    <div class="bar"><span id="prog"></span></div>
    <div class="bars3">
      <div class="bar small"><span id="bar_cpu"></span><em id="txt_cpu">CPU 0%</em></div>
      <div class="bar small"><span id="bar_gpu"></span><em id="txt_gpu">GPU 0%</em></div>
      <div class="bar small"><span id="bar_ram"></span><em id="txt_ram">RAM 0%</em></div>
    </div>
    <div style="margin-top:8px">
      <a id="dl" href="#" download style="display:none">İndir (çıktı PDF)</a>
    </div>
  </fieldset>

  <fieldset>
    <legend>Log</legend>
    <pre id="log"></pre>
  </fieldset>

<script>
let task_id=null, poll=null, mpoll=null;
function setBar(id, pct){
  pct = Math.max(0, Math.min(100, Number(pct)||0));
  const bar = document.getElementById(id);
  if(bar){ bar.style.width = pct+'%'; }
  const txt = document.getElementById(id.replace('bar','txt'));
  if(txt){
    const label = txt.id.includes('cpu')?'CPU':(txt.id.includes('gpu')?'GPU':'RAM');
    txt.textContent = label+' '+pct+'%';
  }
}
async function pollMetrics(){
  try{
    const r = await fetch('/metrics');
    const j = await r.json();
    const cpu = (j.cpu && typeof j.cpu.percent==='number') ? Math.round(j.cpu.percent) : 0;
    const ram = (j.ram && typeof j.ram.percent==='number') ? Math.round(j.ram.percent) : 0;
    let gpu = 0;
    if(Array.isArray(j.gpu) && j.gpu.length && typeof j.gpu[0].gpu_util==='number'){
      gpu = Math.round(j.gpu[0].gpu_util);
    }
    setBar('bar_cpu', cpu);
    setBar('bar_gpu', gpu);
    setBar('bar_ram', ram);
  }catch(e){ /* sessiz geç */ }
}
async function loadModels(){
  try{
    const r = await fetch('/models');
    const j = await r.json();
    const sel = document.getElementById('model');
    if(!j.ok && !Array.isArray(j.models)) throw new Error(j.error||'models failed');
    const cur = sel.value;
    sel.innerHTML = '';
    // ARGOS her zaman en üstte
    const opt0=document.createElement('option');
    opt0.value='ARGOS'; opt0.textContent='ARGOS'; sel.appendChild(opt0);
    (j.models||[]).forEach(name=>{
      if(name && String(name).toUpperCase()!=='ARGOS'){
        const opt=document.createElement('option');
        opt.value=name; opt.textContent=name; sel.appendChild(opt);
      }
    });
    const found = Array.from(sel.options).some(o=>o.value===cur);
    if(found) sel.value = cur;
  }catch(e){ console.warn('models:', e.message); }
}
function clearLog(){ document.getElementById('log').textContent=''; }
async function start(){
  const btn = document.getElementById('btn');
  const prog = document.getElementById('prog');
  const dl = document.getElementById('dl');
  const file = document.getElementById('file').files[0];
  if(!file){ alert('PDF seçin'); return; }
  btn.disabled = true; dl.style.display = 'none'; dl.href='#'; task_id=null;
  prog.style.width='0%';  clearLog();
  const form = new FormData();
  form.append('file', file);
  form.append('src', document.getElementById('src').value);
  form.append('tgt', document.getElementById('tgt').value);
  form.append('model', document.getElementById('model').value || 'ARGOS');
  form.append('pdf_mode', document.getElementById('pdf_mode').value);
  try{
    const r = await fetch('/translate', {method:'POST', body:form});
    const j = await r.json();
    if(!j.ok){ throw new Error(j.error||'Unknown error'); }
    task_id = j.task_id;
    poll = setInterval(pollStatus, 1000);
    pollStatus();
  }catch(e){
    alert(e.message);
    btn.disabled=false;
  }
}
async function pollStatus(){
  if(!task_id) return;
  try{
    const r = await fetch('/status/'+task_id);
    const j = await r.json();
    if(!j.ok){ throw new Error('status failed'); }
    const st = j.status;
    document.getElementById('prog').style.width = (st.progress||0)+'%';
    const r2 = await fetch('/logs/'+task_id+'?tail=200');
    const j2 = await r2.json();
    if(j2.ok){ document.getElementById('log').textContent = j2.log; }
    if(st.status==='done'){
      clearInterval(poll); poll = null;
      const dl = document.getElementById('dl');
      dl.href = '/download/'+task_id;
      dl.style.display = 'inline-block';
      document.getElementById('btn').disabled=false;
    }
    if(st.status==='error'){
      clearInterval(poll); poll = null;
      alert('Hata: '+(st.error||'bilinmiyor'));
      document.getElementById('btn').disabled=false;
    }
  }catch(e){ console.error(e); }
}
window.addEventListener('load', ()=>{ loadModels(); pollMetrics(); mpoll = setInterval(pollMetrics, 1000); });
</script>
</body>
</html>
        """
    )


@app.get("/download/{task_id}")
def download(task_id: str):
    with _TASK_LOCK:
        d = TASKS.get(task_id)
    if not d:
        snap = os.path.join(LOG_DIR, f"{task_id}.json")
        if os.path.exists(snap):
            try:
                with open(snap, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                d = None
    if not d:
        return JSONResponse({"ok": False, "error": "task_id not found"}, status_code=404)

    out_path = d.get("output_path")
    if not out_path or not os.path.exists(out_path):
        return JSONResponse({"ok": False, "error": "file not found"}, status_code=404)

    abs_tmp = os.path.abspath(TMP_DIR)
    abs_out = os.path.abspath(out_path)
    if not abs_out.startswith(abs_tmp + os.sep):
        return JSONResponse({"ok": False, "error": "invalid path"}, status_code=403)

    orig_name = d.get("original_name") or os.path.basename(out_path)
    base = os.path.splitext(os.path.basename(orig_name))[0]

    tgt = (d.get("tgt") or "xx").upper()

    src_cfg = (d.get("src") or "xx").lower()
    src = src_cfg
    if src_cfg == "auto":
        src = "xx"
        try:
            ip = d.get("input_path")
            if ip and os.path.exists(ip):
                with fitz.open(ip) as doc:
                    if len(doc) > 0:
                        txt = (doc[0].get_text("") or "").strip()
                        if txt:
                            det = detect_lang(txt[:4000])
                            if det:
                                src = det
        except Exception:
            pass
    src = (src or "xx").upper()

    safe = re.sub(r"[^0-9A-Za-z_.-]+", " ", base).strip()
    dl_name = f"{safe} [{src}-{tgt}].pdf" if safe else f"translated_{task_id} [{src}-{tgt}].pdf"

    return FileResponse(os.path.abspath(out_path), media_type="application/pdf", filename=dl_name)


@app.get("/models")
def list_models():
    """ARGOS'u sabit döndür + Ollama tag'lerini ekle. 'Ollama Yenile' bu uca bakıyor."""
    models: List[str] = []
    try:
        models.append("ARGOS")  # her zaman en üstte
        with httpx.Client(timeout=5.0) as c:
            r = c.get("http://127.0.0.1:11434/api/tags")
            r.raise_for_status()
            data = r.json()
            names = [m.get("name") for m in data.get("models", []) if m.get("name")]
            for n in names:
                if n and str(n).upper() != "ARGOS":
                    models.append(n)
        return JSONResponse({"ok": True, "models": models})
    except Exception as e:
        # Ollama kapalıysa bile ARGOS'u göstereceğiz
        return JSONResponse({"ok": True, "models": models, "warning": str(e)})


def _run_translate(in_path: str, out_path: str, src: str, tgt: str, model: str, pdf_mode: str, task_id: str, rebuild_output: bool = False) -> None:
    try:
        threading.current_thread().task_id = task_id  # type: ignore[attr-defined]
        translate_pdf(in_path, out_path, src, tgt, model=model, pdf_mode=pdf_mode, rebuild_output=rebuild_output, task_id=task_id)
        task_done(task_id, out_path)
    except Exception as e:
        try:
            task_log(task_id, "THREAD TRACEBACK:\n" + traceback.format_exc())
        except Exception:
            pass
        task_error(task_id, e)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=False)
    ap.add_argument("--out", dest="out_path", required=False)
    ap.add_argument("--src", default="fr")
    ap.add_argument("--tgt", default="en")
    ap.add_argument("--model", default="ARGOS")
    ap.add_argument("--pdf_mode", choices=["stable", "ocr", "ocr_doctr", "ocr_rebuild"], default="stable")
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()

    if args.serve:
        uvicorn.run("translate:app", host="0.0.0.0", port=8080)
        return

    if not args.in_path or not args.out_path:
        print("CLI kullanım: --in input.pdf --out out.pdf --src xx --tgt yy [--model ARGOS|ollama-model-adi] [--pdf_mode stable|ocr|ocr_doctr]")
        sys.exit(2)

    translate_pdf(args.in_path, args.out_path, args.src, args.tgt, model=args.model, pdf_mode=args.pdf_mode)
    print(f"OK -> {args.out_path}")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------
# OCR_REBUILD: PaddleOCR + ReportLab ile sayfayı baştan çiz
# ------------------------------------------------------------
def translate_pdf_rebuild(in_path: str, out_path: str, src: str, tgt: str, *, model: str = "ARGOS", task_id: Optional[str] = None) -> None:
    from reportlab.pdfgen import canvas as _rl_canvas
    from reportlab.lib.pagesizes import portrait
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    src_doc = fitz.open(in_path)
    new_doc = fitz.open()
    pages_total = len(src_doc)
    start_time = time.time()
    if task_id:
        task_update(task_id, pages_total=pages_total, status="running")
        task_log(task_id, f"OCR_REBUILD start: {pages_total} pages")

    # Font seçimi (Noto varsa onu kullan)
    noto_regular = FONT_MAP.get('regular')
    font_name = 'Helvetica'
    if noto_regular and os.path.exists(noto_regular):
        try:
            pdfmetrics.registerFont(TTFont('NotoSans', noto_regular))
            font_name = 'NotoSans'
        except Exception:
            font_name = 'Helvetica'

    # OCR örneği
    ocr_lang = _src_to_paddle_lang(src)
    ocr = _get_ocr(lang=ocr_lang)

    for page_index, page in enumerate(src_doc, start=1):
        if task_id:
            task_update(task_id, page=page_index, progress=round((page_index - 1) / pages_total * 100, 1))
            task_log(task_id, f"OCR_REBUILD page {page_index}/{pages_total}")

        pw, ph = page.rect.width, page.rect.height

        # OCR için raster
        png_path, scale = _page_to_png(page, scale=OCR_SCALE)
        try:
            result = ocr.ocr(png_path, cls=False)
            boxes, texts = [], []
            for line in (result[0] or []):
                (x0,y0),(x1,y1),(x2,y2),(x3,y3) = line[0]
                rx0 = min(x0,x1,x2,x3) / scale
                ry0 = min(y0,y1,y2,y3) / scale
                rx1 = max(x0,x1,x2,x3) / scale
                ry1 = max(y0,y1,y2,y3) / scale
                boxes.append((rx0, ry0, rx1, ry1))
                texts.append(line[1][0])

            # Metin yoksa boş sayfa
            if not texts:
                tmp_pdf = _render_empty_pdf(pw, ph)
                with fitz.open(tmp_pdf) as rd:
                    new_doc.insert_pdf(rd)
                try: os.remove(tmp_pdf)
                except Exception: pass
                continue

            # Çeviri
            sample = " ".join(texts[:40])[:2000]
            translations = _translate_batch(texts, model, src, tgt, sample_text=sample)

            # ReportLab ile tek sayfa PDF üret ve bbox'lara yaz
            tmp_path = os.path.join(TMP_DIR, f"reb_{uuid.uuid4().hex}.pdf")
            c = _rl_canvas.Canvas(tmp_path, pagesize=portrait((pw, ph)))
            def yflip(y): return ph - y  # RL alt-sol, fitz üst-sol
            # --- helpers for wrapping & font measuring ---
            from reportlab.pdfbase import pdfmetrics
            def _wrap_text_to_width(text, font_name, font_size, max_width):
                lines = []
                for para in (text or "").splitlines():
                    words = para.split()
                    if not words:
                        lines.append("")
                        continue
                    cur = words[0]
                    for w in words[1:]:
                        test = cur + " " + w
                        if pdfmetrics.stringWidth(test, font_name, font_size) <= max_width:
                            cur = test
                        else:
                            lines.append(cur)
                            cur = w
                    lines.append(cur)
                return lines

            def _shrink_to_fit(lines, font_name, font_size, max_width, max_height, leading_ratio=1.15, min_size=6):
                size = font_size
                while size > min_size:
                    too_wide = any(pdfmetrics.stringWidth(ln, font_name, size) > max_width for ln in lines if ln)
                    leading = size * leading_ratio
                    needed_h = leading * max(1, len(lines))
                    if (not too_wide) and (needed_h <= max_height):
                        return size, leading
                    size *= 0.92
                return max(min_size, size), max(min_size, size) * leading_ratio


            # draw original page image as background
            try:
                c.drawImage(png_path, 0, 0, width=pw, height=ph, preserveAspectRatio=False, mask=None)
            except Exception:
                pass

            for (x0, y0, x1, y1), txt in zip(boxes, translations):
                h = (y1 - y0)
                fontsize = max(8, min(28, h))
                try:
                    c.setFont(font_name, fontsize)
                except Exception:
                    c.setFont("Helvetica", fontsize)
                # wrapping + shrink-to-fit inside bbox
                box_w = max(1.0, (x1 - x0))
                box_h = max(1.0, (y1 - y0))
                lines = _wrap_text_to_width(txt, font_name, fontsize, box_w)
                fontsize, leading = _shrink_to_fit(lines, font_name, fontsize, box_w, box_h)
                try:
                    c.setFont(font_name, fontsize)
                except Exception:
                    c.setFont("Helvetica", fontsize)
                cur_y = (ph - y0) - (0.2 * fontsize)  # start near top of bbox
                for ln in lines:
                    c.drawString(x0, cur_y, ln)
                    cur_y -= leading

            c.showPage(); c.save()

            with fitz.open(tmp_path) as rd:
                new_doc.insert_pdf(rd)
            try: os.remove(tmp_path)
            except Exception: pass

        finally:
            try: os.remove(png_path)
            except Exception: pass

        if task_id:
            elapsed = max(0.001, time.time() - start_time)
            done = page_index
            eta = (elapsed / done) * max(0, pages_total - done)
            task_update(task_id, progress=round(done / pages_total * 100, 1), eta_sec=int(eta))

    new_doc.save(out_path, garbage=4, deflate=True, clean=True)
    new_doc.close()
    src_doc.close()
    if task_id:
        task_update(task_id, progress=100.0)
        task_log(task_id, f"OCR_REBUILD saved -> {out_path}")

def _render_empty_pdf(pw: float, ph: float) -> str:
    from reportlab.pdfgen import canvas as _rl_canvas
    from reportlab.lib.pagesizes import portrait
    tmp_path = os.path.join(TMP_DIR, f"empty_{uuid.uuid4().hex}.pdf")
    c = _rl_canvas.Canvas(tmp_path, pagesize=portrait((pw, ph)))
    c.showPage(); c.save()
    return tmp_path


# ------------------------------------------------------------
# Fallback batch translator for OCR_REBUILD
# ------------------------------------------------------------

def _force_argos_cuda():
    import os, importlib, sys
    # Çeviri başlamadan evvel env'i garantiye al
    if os.environ.get("ARGOS_DEVICE_TYPE") not in ("cuda", "auto"):
        os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
    # CT2 daha gürültülü olsun ki log'da görelim
    os.environ.setdefault("CT2_VERBOSE", "1")

    # Tanılama log'u
    try:
        import ctranslate2 as ct
        ct_file = getattr(ct, "__file__", "n/a")
        has_cuda_attr = hasattr(ct, "has_cuda")
        has_cuda_val = None
        if has_cuda_attr:
            try:
                has_cuda_val = ct.has_cuda()
            except Exception:
                has_cuda_val = "error"
        print(f"[ARGOSDBG] ARGOS_DEVICE_TYPE={os.environ.get('ARGOS_DEVICE_TYPE')} CT2_FILE={ct_file} HAS_CUDA_ATTR={has_cuda_attr} HAS_CUDA={has_cuda_val}", flush=True)
    except Exception as e:
        print(f"[ARGOSDBG] import ctranslate2 failed: {e}", flush=True)
    return True



# --- ARGOS (CT2/CUDA) helper: translator injection + reuse ---
_ARGOS_TR_CACHE = {}  # key: (src_code, tgt_code) -> CachedTranslation

def _argos_get_translation_cuda(src: str, tgt: str):
    """
    Argos'un Translation nesnesini döndürürken, alttaki PackageTranslation.translator
    alanına CTranslate2.Translator(device="cuda", compute_type="float16") enjekte eder.
    Sonraki çağrılar aynı objeyi yeniden kullanır.
    """
    try:
        import argostranslate.translate as at
    except Exception:
        return None

    # ortamı garantiye al
    try:
        _force_argos_cuda()
    except Exception:
        pass

    # normalize
    src = (src or "").lower()
    tgt = (tgt or "").lower()

    # cache?
    k = (src, tgt)
    if k in _ARGOS_TR_CACHE:
        return _ARGOS_TR_CACHE[k]

    langs = at.get_installed_languages()
    s = next((l for l in langs if l.code == src), None)
    t = next((l for l in langs if l.code == tgt), None)
    if not s or not t:
        return None

    tr = s.get_translation(t)  # CachedTranslation
    underlying = getattr(tr, "underlying", None)  # PackageTranslation
    if underlying is None:
        _ARGOS_TR_CACHE[k] = tr
        return tr

    if getattr(underlying, "translator", None) is not None:
        _ARGOS_TR_CACHE[k] = tr
        return tr

    # Paketin model dizinini bul (Argos 1.9.x)
    pkg = getattr(underlying, "pkg", None)
    model_dir = None
    for cand in ("package_path", "install_path", "path"):
        if pkg is not None and hasattr(pkg, cand):
            try:
                pth = getattr(pkg, cand)
                if pth:
                    model_dir = str(pth)
                    break
            except Exception:
                pass

    # CTranslate2 translator oluştur ve enjekte et
    try:
        import ctranslate2 as ct
        if model_dir:
            translator = ct.Translator(model_dir, device="cuda", compute_type="float16")
            setattr(underlying, "translator", translator)
    except Exception:
        # sessiz düş
        pass

    _ARGOS_TR_CACHE[k] = tr
    return tr


def _translate_batch(texts, model, src, tgt, sample_text=None):
    out = []

    # --- ARGOS (GPU via CTranslate2) early-return path ---
    if (model or "").strip().upper() == "ARGOS":
        out = []
        tr_obj = _argos_get_translation_cuda(src, tgt)
        if tr_obj is None:
            try:
                import argostranslate.translate as at
                for _t in texts:
                    try:
                        out.append(at.translate(_t, src, tgt))
                    except Exception:
                        out.append(_t)
                return out
            except Exception:
                return [t for t in texts]

        # mini-batch: ardışık çağrılar aynı CT2 translator'ı paylaşır
        import os
        bs = max(1, min(int(os.environ.get("BATCH_SIZE", "12")), 64))
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            for _t in chunk:
                try:
                    out.append(tr_obj.translate(_t))
                except Exception:
                    out.append(_t)
        return out
    try:
        if (model or "").strip().upper() == "ARGOS":
            try:
                import argostranslate.package, argostranslate.translate
            except Exception:
                return [t for t in texts]
            for idx, t in enumerate(texts):
                try:
                    out.append(argostranslate.translate.translate(t, src, tgt))
                except Exception:
                    out.append(t)
            return out
        else:
            import requests, json
            url = "http://127.0.0.1:11434/api/generate"
            headers = {"Content-Type":"application/json"}
            prompt_tpl = (
                "You are a professional translator. Preserve meaning, tone, and style.\\n"
                "Return only the translation without commentary.\\n"
                "Source language: {src}\\n"
                "Target language: {tgt}\\n"
                "Text:\\n<<<\\n{txt}\\n>>>"
            )
            for t in texts:
                payload = {
                    "model": model,
                    "prompt": prompt_tpl.format(src=src, tgt=tgt, txt=t),
                    "stream": False,
                    "options": {"temperature": 0.2}
                }
                try:
                    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
                    if r.ok:
                        js = r.json()
                        out.append((js.get("response") or "").strip() or t)
                    else:
                        out.append(t)
                except Exception:
                    out.append(t)
            return out
    except Exception:
        return [t for t in texts]# -------- stable mode fallback span extractor (PyMuPDF textpage) --------
def _extract_spans_textpage(page: fitz.Page) -> list[dict]:
    spans = []
    try:
        tp = page.get_textpage()
        # extractBLOCKS: (x0, y0, x1, y1, "text", block_no, block_type, ...)
        for blk in tp.extractBLOCKS() or []:
            if len(blk) < 5: 
                continue
            x0, y0, x1, y1, txt = blk[0], blk[1], blk[2], blk[3], blk[4]
            if not txt:
                continue
            for line in txt.splitlines():
                t = line.strip()
                if not t:
                    continue
                spans.append({
                    "text": t,
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "size": 10.0,
                    "color": [0, 0, 0],
                    "font": "Helvetica",
                })
    except Exception:
        pass
    return spans
