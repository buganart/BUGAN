#!/usr/bin/env python3
from datetime import datetime
import json
import sys

from typeform import Client, Typeform


def mk_choice(label, href):
    return {
        "label": label,
        "attachment": {
            "type": "image",
            "href": href,
            "properties": {"description": "Heart"},
        },
    }


def mk_tree_question(choices):
    return {
        "title": "Which tree resembles the book the most?",
        "properties": {
            "randomize": False,
            "allow_multiple_selection": False,
            "allow_other_choice": False,
            "supersized": False,
            "show_labels": True,
            "choices": choices,
        },
        "validations": {"required": False},
        "type": "picture_choice",
    }


def book_question():
    return {
        "title": "Name a book you like?",
        "properties": {},
        "validations": {"required": False},
        "type": "short_text",
    }


def mk_survey(fields):
    return {
        "type": "form",
        "title": f"Books to Trees (from API) {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "theme": {"href": "https://api.typeform.com/themes/qHWOQ7"},
        "settings": {
            "language": "en",
            "progress_bar": "proportion",
            "meta": {"allow_indexing": False},
            "is_public": True,
            "is_trial": False,
            "show_progress_bar": True,
            "show_typeform_branding": True,
            "are_uploads_public": False,
        },
        "fields": fields,
    }


token = open("typeform_token.txt").read().strip()
t = Typeform(token)
client = Client(token)

images = client.request("get", "/images")
images_by_name = {item["file_name"]: item for item in images}

choices = []
for i in range(100):
    fname = f"{i:05}.jpg"
    href = images_by_name[fname]["src"]
    choice = mk_choice(fname, href)
    choices.append(choice)


tree_question = mk_tree_question(choices)
fields = [book_question(), tree_question]
survey = mk_survey(fields)

print(t.forms.create(survey))
