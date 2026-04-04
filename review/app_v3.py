#!/usr/bin/env python3
"""PhyMetric QA Review server for question_v3.json"""

from flask import Flask, jsonify, request, send_from_directory, Response
import json, pathlib

BASE    = pathlib.Path(__file__).parent
DATA    = BASE.parent / 'data' / 'question_v3.json'
RECORDS = BASE / 'records_v3.json'

app = Flask(__name__)


def load_data():
    return json.loads(DATA.read_text(encoding='utf-8'))


def load_records():
    return json.loads(RECORDS.read_text(encoding='utf-8')) if RECORDS.exists() else {}


def save_records(records):
    RECORDS.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')


# ── routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(str(BASE), 'index_v3.html')


@app.route('/api/data')
def api_data():
    return jsonify(load_data())


@app.route('/api/records', methods=['GET'])
def api_get_records():
    return jsonify(load_records())


@app.route('/api/records', methods=['POST'])
def api_post_record():
    body = request.json
    key, record = body['key'], body['record']
    records = load_records()
    records[key] = record
    save_records(records)
    return jsonify({'ok': True})


@app.route('/api/export')
def api_export():
    """Return question_v3.json with reviewer edits applied."""
    data = load_data()
    records = load_records()

    for key, rec in records.items():
        if '::' not in key:
            continue
        cat, sid = key.split('::', 1)
        cats = data.get('categories', {})
        if cat not in cats or sid not in cats[cat]['scenes']:
            continue

        edits = rec.get('edits', {})
        scene = cats[cat]['scenes'][sid]
        for field in ('scene_name', 'scene_caption', 'generation_prompt'):
            if field in edits:
                scene[field] = edits[field]
        if 'causal_chain' in edits:
            scene['causal_chain'] = edits['causal_chain']
        if 'questions' in edits:
            scene['questions'] = edits['questions']

    content = json.dumps(data, indent=2, ensure_ascii=False)
    return Response(content, mimetype='application/json',
                    headers={'Content-Disposition':
                             'attachment; filename=question_v3_reviewed.json'})


if __name__ == '__main__':
    print('Open http://localhost:5001')
    app.run(host='0.0.0.0', port=5001, debug=False)
