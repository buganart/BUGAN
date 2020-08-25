#!/usr/bin/env python3
import json
import sys
from datetime import datetime
from pathlib import Path

from typeform import Client, Typeform

LONG_INTRO = open("about.md").read().strip()
CONTINUE_TITLE = "Press ENTER to do another book!"
BOOK_QUESTION = "Type in the title of the book."
AUTHOR_QUESTION = "Type in the author of the book."


def mk_choice(*, label, href, fname):
    return {
        "label": label,
        "attachment": {
            "type": "image",
            "href": href,
            "properties": {"description": fname},
        },
    }


def mk_tree_question(choices):
    return {
        "title": f"Click on a tree that you think should be on its cover.",
        "properties": {
            "randomize": True,
            "allow_multiple_selection": False,
            "allow_other_choice": False,
            "supersized": False,
            "show_labels": False,
            "choices": choices,
        },
        "validations": {"required": False},
        "type": "picture_choice",
    }


def mk_text_question(title):
    return {
        "title": title,
        "properties": {},
        "validations": {"required": False},
        "type": "short_text",
    }


def mk_survey(fields):
    return {
        "type": "form",
        "title": f"Books to Trees (from API) {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "theme": {"href": "https://api.typeform.com/themes/Jml7Xb"},
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


def mk_fields(title, fields):
    return {
        "title": title,
        "properties": {
            "button_text": "Continue",
            "show_button": True,
            "fields": fields,
        },
        "type": "group",
    }


token = open("typeform_token.txt").read().strip()
t = Typeform(token)
client = Client(token)

images = client.request("get", "/images")
images_by_name = {item["file_name"]: item for item in images}

choices = []

paths = sorted(list(Path(sys.argv[1]).glob("*.*")))

for index, path in enumerate(paths, start=1):
    fname = path.name
    href = images_by_name[fname]["src"]
    choice = mk_choice(label=fname, href=href, fname=fname)
    choices.append(choice)


fields = [
    mk_fields(
        LONG_INTRO,
        [
            mk_text_question(BOOK_QUESTION),
            mk_text_question(AUTHOR_QUESTION),
            mk_tree_question(choices),
        ],
    )
]

num_questions = 2
for question_number in range(1, num_questions):
    question_fields = mk_fields(
        CONTINUE_TITLE,
        [
            mk_text_question(BOOK_QUESTION),
            mk_text_question(AUTHOR_QUESTION),
            mk_tree_question(choices),
        ],
    )
    fields.append(question_fields)

survey = mk_survey(fields)

print(t.forms.create(survey))
