import io
import json
import os
import queue
import random
import shutil
import threading
import uuid
import zipfile
from pathlib import Path

import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, send_file, session, url_for)
from PIL import Image

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'widroit-cow-dev-key-change-in-prod')

WORKSPACE = Path(__file__).parent / 'workspace'
WORKSPACE.mkdir(exist_ok=True)

PAGE_SIZE = 48  # images per gallery page

_task_queues: dict[str, queue.Queue] = {}
_task_lock = threading.Lock()


def _new_queue(task_id: str) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    with _task_lock:
        _task_queues[task_id] = q
    return q


def _get_queue(task_id: str) -> 'queue.Queue | None':
    with _task_lock:
        return _task_queues.get(task_id)


def get_ws(sid: str) -> Path:
    p = WORKSPACE / sid
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_state(sid: str) -> dict:
    p = get_ws(sid) / 'state.json'
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(sid: str, state: dict) -> None:
    (get_ws(sid) / 'state.json').write_text(json.dumps(state, indent=2))


def _counts(review: dict) -> dict:
    c = {'accepted': 0, 'rejected': 0, 'pending': 0}
    for v in review.values():
        if v in c:
            c[v] += 1
    return c


# ─── Step 1: Upload ──────────────────────────────────────────────────────────

@app.route('/')
def step1():
    return render_template('step1.html')


@app.route('/step1/upload', methods=['POST'])
def step1_upload():
    sid = str(uuid.uuid4())[:8]
    session['sid'] = sid
    ws = get_ws(sid)

    dataset_dir = ws / 'dataset'
    dataset_dir.mkdir(exist_ok=True)

    f = request.files.get('dataset')
    if not f:
        return 'No file uploaded.', 400
    if not f.filename.lower().endswith('.zip'):
        return 'Please upload a ZIP file containing images.', 400

    zip_path = ws / 'upload.zip'
    f.save(str(zip_path))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if Path(member).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    name = Path(member).name
                    if name:
                        (dataset_dir / name).write_bytes(zf.read(member))
    finally:
        zip_path.unlink(missing_ok=True)

    seed_pct = max(1.0, min(50.0, float(request.form.get('seed_pct', 10))))

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = sorted(p.name for p in dataset_dir.iterdir() if p.suffix.lower() in img_exts)
    if not all_images:
        return 'No images found in ZIP.', 400

    random.seed(42)
    n_seed = max(1, int(len(all_images) * seed_pct / 100))
    seed_files = random.sample(all_images, n_seed)
    remaining = [f for f in all_images if f not in set(seed_files)]

    seed_dir = ws / 'seed'
    seed_dir.mkdir(exist_ok=True)
    for name in seed_files:
        shutil.copy2(dataset_dir / name, seed_dir / name)

    state = {
        'step': 2,
        'total_images': len(all_images),
        'seed_pct': seed_pct,
        'seed_files': seed_files,
        'remaining_files': remaining,
        'sam2_done': False,
        'review': {f: 'pending' for f in seed_files},
        'accepted_files': [],
        'training_done': False,
        'training_metrics': {},
        'inference_done': False,
        'prediction_review': {},
    }
    save_state(sid, state)
    return redirect(url_for('step2'))


# ─── Step 2: SAM2 Annotation + Review ────────────────────────────────────────

@app.route('/step2')
def step2():
    sid = session.get('sid')
    if not sid:
        return redirect(url_for('step1'))
    state = load_state(sid)
    page = max(1, int(request.args.get('page', 1)))
    files = state.get('seed_files', [])
    total_pages = max(1, (len(files) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages)
    page_files = files[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    return render_template('step2.html', state=state, sid=sid,
                           page=page, total_pages=total_pages, page_files=page_files,
                           counts=_counts(state.get('review', {})))


@app.route('/step2/start_sam2', methods=['POST'])
def step2_start_sam2():
    from tasks.sam2_runner import run_sam2
    sid = session.get('sid')
    if not sid:
        return jsonify(error='no session'), 400
    state = load_state(sid)
    task_id = str(uuid.uuid4())[:8]
    q = _new_queue(task_id)
    threading.Thread(target=run_sam2,
                     args=(get_ws(sid), state['seed_files'], q),
                     daemon=True).start()
    return jsonify(task_id=task_id)


@app.route('/step2/mark_done', methods=['POST'])
def step2_mark_done():
    sid = session.get('sid')
    if not sid:
        return jsonify(error='no session'), 400
    state = load_state(sid)
    state['sam2_done'] = True
    save_state(sid, state)
    return jsonify(ok=True)


@app.route('/step2/review', methods=['POST'])
def step2_review():
    sid = session.get('sid')
    if not sid:
        return jsonify(error='no session'), 400
    data = request.get_json()
    action = data.get('action')
    fname = data.get('filename')
    state = load_state(sid)

    if action == 'no_additional':
        for k in state['review']:
            if state['review'][k] == 'pending':
                state['review'][k] = 'rejected'
    elif action in ('accept', 'reject') and fname:
        state['review'][fname] = action + 'ed'

    save_state(sid, state)
    return jsonify(ok=True, counts=_counts(state['review']))


@app.route('/step2/save', methods=['POST'])
def step2_save():
    sid = session.get('sid')
    if not sid:
        return redirect(url_for('step1'))
    state = load_state(sid)
    ws = get_ws(sid)

    acc_imgs = ws / 'accepted_images'
    acc_masks = ws / 'accepted_masks'
    acc_imgs.mkdir(exist_ok=True)
    acc_masks.mkdir(exist_ok=True)

    accepted = []
    for fname, decision in state['review'].items():
        if decision == 'accepted':
            mask_name = Path(fname).stem + '.png'
            src_mask = ws / 'sam_masks' / mask_name
            if src_mask.exists():
                shutil.copy2(src_mask, acc_masks / mask_name)
            src_img = ws / 'seed' / fname
            if src_img.exists():
                shutil.copy2(src_img, acc_imgs / fname)
            accepted.append(fname)

    state['accepted_files'] = accepted
    state['step'] = 3
    save_state(sid, state)
    return redirect(url_for('step3'))


# ─── Step 3: Train PicoCowUNet ────────────────────────────────────────────────

@app.route('/step3')
def step3():
    sid = session.get('sid')
    if not sid:
        return redirect(url_for('step1'))
    return render_template('step3.html', state=load_state(sid), sid=sid)


@app.route('/step3/start_training', methods=['POST'])
def step3_start_training():
    from tasks.trainer import run_training
    sid = session.get('sid')
    if not sid:
        return jsonify(error='no session'), 400
    state = load_state(sid)
    data = request.get_json() or {}
    epochs = max(1, min(200, int(data.get('epochs', 25))))
    task_id = str(uuid.uuid4())[:8]
    q = _new_queue(task_id)
    threading.Thread(target=run_training,
                     args=(get_ws(sid), state['accepted_files'], q, epochs),
                     daemon=True).start()
    return jsonify(task_id=task_id)


@app.route('/step3/save_metrics', methods=['POST'])
def step3_save_metrics():
    sid = session.get('sid')
    if not sid:
        return jsonify(error='no session'), 400
    data = request.get_json()
    state = load_state(sid)
    state['training_done'] = True
    state['training_metrics'] = data.get('metrics', {})
    state['step'] = 4
    save_state(sid, state)
    return jsonify(ok=True)


# ─── Step 4: Predict + Review ─────────────────────────────────────────────────

@app.route('/step4')
def step4():
    sid = session.get('sid')
    if not sid:
        return redirect(url_for('step1'))
    state = load_state(sid)
    page = max(1, int(request.args.get('page', 1)))
    files = state.get('remaining_files', [])
    total_pages = max(1, (len(files) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages)
    page_files = files[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    return render_template('step4.html', state=state, sid=sid,
                           page=page, total_pages=total_pages, page_files=page_files,
                           counts=_counts(state.get('prediction_review', {})))


@app.route('/step4/start_predict', methods=['POST'])
def step4_start_predict():
    from tasks.predictor import run_prediction
    sid = session.get('sid')
    if not sid:
        return jsonify(error='no session'), 400
    state = load_state(sid)
    ws = get_ws(sid)
    model_path = ws / 'model' / 'pico.pth'
    if not model_path.exists():
        return jsonify(error='Trained model not found. Complete Step 3 first.'), 400
    task_id = str(uuid.uuid4())[:8]
    q = _new_queue(task_id)
    threading.Thread(target=run_prediction,
                     args=(ws, state['remaining_files'], str(model_path), q),
                     daemon=True).start()
    return jsonify(task_id=task_id)


@app.route('/step4/mark_done', methods=['POST'])
def step4_mark_done():
    sid = session.get('sid')
    if not sid:
        return jsonify(error='no session'), 400
    state = load_state(sid)
    state['inference_done'] = True
    state['prediction_review'] = {f: 'pending' for f in state['remaining_files']}
    save_state(sid, state)
    return jsonify(ok=True)


@app.route('/step4/review', methods=['POST'])
def step4_review():
    sid = session.get('sid')
    if not sid:
        return jsonify(error='no session'), 400
    data = request.get_json()
    action = data.get('action')
    fname = data.get('filename')
    state = load_state(sid)
    if fname and action in ('accept', 'reject'):
        state['prediction_review'][fname] = action + 'ed'
    save_state(sid, state)
    return jsonify(ok=True, counts=_counts(state['prediction_review']))


@app.route('/step4/save', methods=['POST'])
def step4_save():
    sid = session.get('sid')
    if not sid:
        return redirect(url_for('step1'))
    state = load_state(sid)
    ws = get_ws(sid)
    # Export accepted predictions to workspace/final/
    final_dir = ws / 'final'
    final_dir.mkdir(exist_ok=True)
    for fname, decision in state['prediction_review'].items():
        if decision == 'accepted':
            mask_name = Path(fname).stem + '.png'
            src = ws / 'predictions' / mask_name
            if src.exists():
                shutil.copy2(src, final_dir / mask_name)
    state['finished'] = True
    save_state(sid, state)
    return redirect(url_for('step4') + '?saved=1')


# ─── SSE Progress Stream ──────────────────────────────────────────────────────

@app.route('/progress/<task_id>')
def progress_stream(task_id: str):
    q = _get_queue(task_id)
    if q is None:
        return 'Task not found', 404

    def generate():
        while True:
            try:
                event = q.get(timeout=60)
                yield f'data: {json.dumps(event)}\n\n'
                if event.get('done') or event.get('error'):
                    break
            except queue.Empty:
                yield ': heartbeat\n\n'

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ─── Image Serving ────────────────────────────────────────────────────────────

@app.route('/img/<sid>/<folder>/<filename>')
def serve_image(sid: str, folder: str, filename: str):
    if folder not in {'seed', 'dataset', 'accepted_images'}:
        return 'Forbidden', 403
    path = get_ws(sid) / folder / filename
    return send_file(str(path)) if path.exists() else ('Not found', 404)


@app.route('/overlay/<sid>/seed/<filename>')
def overlay_seed(sid: str, filename: str):
    ws = get_ws(sid)
    return _overlay(ws / 'seed' / filename,
                    ws / 'sam_masks' / (Path(filename).stem + '.png'))


@app.route('/overlay/<sid>/pred/<filename>')
def overlay_pred(sid: str, filename: str):
    ws = get_ws(sid)
    return _overlay(ws / 'dataset' / filename,
                    ws / 'predictions' / (Path(filename).stem + '.png'))


def _overlay(img_path: Path, mask_path: Path) -> Response:
    img = Image.open(img_path).convert('RGB')
    if mask_path.exists():
        mask = Image.open(mask_path).convert('L').resize(img.size, Image.NEAREST)
        arr = np.array(img, dtype=np.float32)
        obj = np.array(mask) < 128  # object pixels (0=object convention)
        arr[obj, 0] = arr[obj, 0] * 0.3
        arr[obj, 1] = arr[obj, 1] * 0.3 + 170
        arr[obj, 2] = arr[obj, 2] * 0.3 + 50
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=82)
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)
