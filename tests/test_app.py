from app import app
from flask import json

def test_normal():
    response = app.test_client().post(
        '/twitter',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'hashtag': 'jokowi 3 periode'}),
    )

    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200

def test_bahasa_asing():
    response = app.test_client().post(
        '/twitter',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'hashtag': 'electoral vote'}),
    )

    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200

def test_number():
    response = app.test_client().post(
        '/twitter',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'hashtag': 123}),
    )

    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200

def test_number_str():
    response = app.test_client().post(
        '/twitter',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'hashtag': '123'}),
    )

    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200

def test_charachter():
    response = app.test_client().post(
        '/twitter',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'hashtag': ':"('}),
    )

    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200

def test_emoticon():
    response = app.test_client().post(
        '/twitter',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'hashtag': '😀' }),
    )

    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200

def test_empty_string():
    response = app.test_client().post(
        '/twitter',
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'hashtag': '' }),
    )

    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200


def test_empty():
    
    response = app.test_client().post(
        '/twitter',
        headers={'Content-Type': 'application/json'},
        data=json.dumps(),
    )

    data = json.loads(response.get_data(as_text=True))

    assert response.status_code == 200
