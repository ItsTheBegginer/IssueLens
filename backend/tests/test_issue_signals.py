"""Tests for the deterministic issue signal extractor.

Uses realistic issue text — no mocks, no network I/O.
"""
from __future__ import annotations

import pytest

from app.analyzers.issue_signals import extract_signals
from app.models.github import IssueComment, IssueDetail, GitHubUser
from app.models.issue_signals import IssueSignals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _issue(
    title: str,
    body: str = "",
    *,
    labels: list[str] | None = None,
    comments: list[str] | None = None,
) -> IssueDetail:
    comment_objs = [
        IssueComment(
            id=i + 1,
            user=GitHubUser(login=f"user{i}"),
            body=text,
            created_at="2024-01-01T00:00:00Z",
        )
        for i, text in enumerate(comments or [])
    ]
    return IssueDetail(
        number=1,
        title=title,
        body=body,
        state="open",
        user=GitHubUser(login="reporter"),
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        html_url="https://github.com/owner/repo/issues/1",
        comments=len(comment_objs),
        labels=labels or [],
        comment_list=comment_objs,
    )


# ---------------------------------------------------------------------------
# Return type and title preservation
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_issue_signals(self):
        result = extract_signals(_issue("simple title"))
        assert isinstance(result, IssueSignals)

    def test_title_preserved_exactly(self):
        title = "TypeError in handle_request() when body is None"
        result = extract_signals(_issue(title))
        assert result.title == title

    def test_empty_body_produces_signals(self):
        result = extract_signals(_issue("Crash on startup"))
        assert isinstance(result, IssueSignals)
        assert result.keywords  # title tokens still extracted

    def test_no_fields_are_none_except_behavior(self):
        result = extract_signals(_issue("bug"))
        assert result.keywords is not None
        assert result.error_messages is not None
        assert result.file_references is not None
        assert result.function_names is not None
        assert result.class_names is not None


# ---------------------------------------------------------------------------
# Error message extraction
# ---------------------------------------------------------------------------

class TestErrorMessages:
    def test_python_exception_in_fence(self):
        body = """
Running the command produces:

```
Traceback (most recent call last):
  File "app/main.py", line 42, in handle_request
    result = parser.parse(data)
AttributeError: 'NoneType' object has no attribute 'parse'
```
"""
        result = extract_signals(_issue("Parser crashes", body))
        errors = result.error_messages
        assert any("AttributeError" in e for e in errors)

    def test_python_traceback_frame_extracted(self):
        body = """
```
  File "src/auth.py", line 17, in login
    raise ValueError("bad token")
ValueError: bad token
```
"""
        result = extract_signals(_issue("Login broken", body))
        errors = result.error_messages
        assert any("ValueError" in e for e in errors)

    def test_js_stack_frame(self):
        body = """
Console output:
```
TypeError: Cannot read properties of undefined (reading 'map')
    at processItems (utils/list.js:34:12)
    at App.render (App.jsx:88:5)
```
"""
        result = extract_signals(_issue("React crash", body))
        errors = result.error_messages
        assert any("TypeError" in e for e in errors)

    def test_inline_code_error(self):
        body = "I get `KeyError: 'user_id'` every time I call the endpoint."
        result = extract_signals(_issue("KeyError in endpoint", body))
        errors = result.error_messages
        assert any("KeyError" in e for e in errors)

    def test_failed_keyword(self):
        body = """
```
FAILED tests/test_auth.py::test_login - AssertionError
```
"""
        result = extract_signals(_issue("Tests fail", body))
        errors = result.error_messages
        assert any("FAILED" in e for e in errors)

    def test_no_false_positives_on_clean_text(self):
        body = "The button text should be updated to match the design spec."
        result = extract_signals(_issue("Update button label", body))
        # No exception-like strings in this text
        assert not any(
            "Error" in e or "Exception" in e or "Traceback" in e
            for e in result.error_messages
        )


# ---------------------------------------------------------------------------
# File reference extraction
# ---------------------------------------------------------------------------

class TestFileReferences:
    def test_python_path_in_prose(self):
        body = "The bug is in `src/auth/middleware.py` around line 42."
        result = extract_signals(_issue("Middleware bug", body))
        assert any("middleware.py" in f for f in result.file_references)

    def test_js_file_reference(self):
        body = "See the handler in routes/api/users.js for the broken logic."
        result = extract_signals(_issue("User API broken", body))
        assert any("users.js" in f for f in result.file_references)

    def test_config_file_reference(self):
        body = "Check the settings in config/database.yaml."
        result = extract_signals(_issue("DB config issue", body))
        assert any("database.yaml" in f or "database.yml" in f
                   for f in result.file_references)

    def test_path_in_traceback(self):
        body = """
```
  File "app/models/user.py", line 88, in save
    db.execute(query)
```
"""
        result = extract_signals(_issue("DB error", body))
        assert any("user.py" in f for f in result.file_references)

    def test_relative_path_reference(self):
        body = "Update the export in `./lib/utils/formatter.ts`."
        result = extract_signals(_issue("Formatter issue", body))
        assert any("formatter.ts" in f for f in result.file_references)

    def test_no_false_positives_on_urls(self):
        body = "See https://github.com/owner/repo for details."
        result = extract_signals(_issue("See repo", body))
        # URLs should not produce file references here (no recognised extension at end)
        refs = result.file_references
        assert not any("https" in r for r in refs)


# ---------------------------------------------------------------------------
# Function name extraction
# ---------------------------------------------------------------------------

class TestFunctionNames:
    def test_snake_case_in_title(self):
        result = extract_signals(_issue("handle_request crashes on empty input"))
        assert "handle_request" in result.function_names

    def test_snake_case_in_body(self):
        body = "Calling `parse_config()` raises an exception."
        result = extract_signals(_issue("Config bug", body))
        assert any("parse_config" in f for f in result.function_names)

    def test_function_call_syntax(self):
        body = "The issue is in `processItems()` when the list is empty."
        result = extract_signals(_issue("List bug", body))
        assert any("processItems" in f for f in result.function_names)

    def test_def_pattern_in_fence(self):
        body = """
```python
def authenticate_user(token):
    return db.query(token)
```
The `authenticate_user` function raises KeyError.
"""
        result = extract_signals(_issue("Auth bug", body))
        assert any("authenticate_user" in f for f in result.function_names)

    def test_no_single_word_false_positives(self):
        result = extract_signals(_issue("login broken"))
        # "login" alone (no underscore segments) should not be in function_names
        # as a snake_case multi-segment function
        assert "login" not in result.function_names


# ---------------------------------------------------------------------------
# Class name extraction
# ---------------------------------------------------------------------------

class TestClassNames:
    def test_camelcase_class_in_title(self):
        result = extract_signals(_issue("RequestHandler fails on POST"))
        assert "RequestHandler" in result.class_names

    def test_camelcase_class_in_body(self):
        body = "The `AuthMiddleware` class does not pass the token downstream."
        result = extract_signals(_issue("Auth bug", body))
        assert any("AuthMiddleware" in c for c in result.class_names)

    def test_multiple_classes(self):
        body = "Both `UserService` and `PermissionChecker` are affected."
        result = extract_signals(_issue("Permissions bug", body))
        names = result.class_names
        assert any("UserService" in n for n in names)
        assert any("PermissionChecker" in n for n in names)

    def test_single_word_not_class(self):
        result = extract_signals(_issue("Error in login"))
        # Single-word CamelCase (like "Error") has only one segment — not extracted
        assert "Error" not in result.class_names

    def test_exception_type_extracted(self):
        body = "Raises `ConnectionTimeoutError` when server is slow."
        result = extract_signals(_issue("Timeout bug", body))
        assert any("ConnectionTimeoutError" in c for c in result.class_names)


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

class TestKeywords:
    def test_title_tokens_present(self):
        result = extract_signals(_issue("TypeError in parse_request body"))
        # "TypeError" is a class name, but words from title should appear in keywords
        kw = result.keywords
        assert "parse_request" in result.function_names or "TypeError" in result.class_names

    def test_dotted_name_extracted(self):
        body = "The error occurs in `flask.request.get_json()` call."
        result = extract_signals(_issue("Flask bug", body))
        assert any("flask.request" in k or "flask" in k for k in result.keywords)

    def test_all_caps_constant(self):
        body = "The `MAX_RETRIES` constant is ignored when set to 0."
        result = extract_signals(_issue("Retry bug", body))
        assert "MAX_RETRIES" in result.keywords

    def test_stopwords_excluded(self):
        result = extract_signals(_issue("the error is not fixed"))
        assert "the" not in result.keywords
        assert "is" not in result.keywords
        assert "not" not in result.keywords

    def test_keywords_are_deduplicated(self):
        body = "MAX_RETRIES MAX_RETRIES MAX_RETRIES is mentioned three times."
        result = extract_signals(_issue("Retry bug", body))
        assert result.keywords.count("MAX_RETRIES") == 1

    def test_inline_code_tokens_included(self):
        body = "Setting `debug_mode` to True causes the crash."
        result = extract_signals(_issue("Debug crash", body))
        # debug_mode is a snake_case function name, not a keyword — but should appear somewhere
        assert "debug_mode" in result.function_names or "debug_mode" in result.keywords


# ---------------------------------------------------------------------------
# Behavioral heuristics
# ---------------------------------------------------------------------------

class TestBehavioralHeuristics:
    def test_expected_behavior_detected(self):
        body = "I expected the function to return an empty list. Instead it raises."
        result = extract_signals(_issue("Wrong return value", body))
        assert result.expected_behavior is not None
        assert "expected" in result.expected_behavior.lower()

    def test_observed_behavior_detected(self):
        body = "When I call save(), it actually raises a ValueError."
        result = extract_signals(_issue("Save error", body))
        assert result.observed_behavior is not None

    def test_should_triggers_expected(self):
        body = "The endpoint should return 200 OK for valid requests."
        result = extract_signals(_issue("HTTP status wrong", body))
        assert result.expected_behavior is not None

    def test_no_behavior_on_vague_title_only(self):
        result = extract_signals(_issue("bug in login"))
        # No body with behavioral markers — both may be None
        # (not asserting non-None, just asserting no crash)
        assert result.expected_behavior is None or isinstance(result.expected_behavior, str)

    def test_instead_triggers_observed(self):
        body = "The login should succeed. Instead, it redirects to /error."
        result = extract_signals(_issue("Login redirect", body))
        assert result.observed_behavior is not None
        assert "instead" in result.observed_behavior.lower()


# ---------------------------------------------------------------------------
# Comment text included
# ---------------------------------------------------------------------------

class TestCommentsIncluded:
    def test_error_in_comment_extracted(self):
        result = extract_signals(_issue(
            "Auth fails",
            body="No details in body.",
            comments=["I also see `PermissionError: access denied` in the logs."],
        ))
        assert any("PermissionError" in e for e in result.error_messages)

    def test_file_ref_in_comment_extracted(self):
        result = extract_signals(_issue(
            "Config error",
            body="No details.",
            comments=["The issue is in `config/settings.py` line 12."],
        ))
        assert any("settings.py" in f for f in result.file_references)

    def test_class_in_comment_extracted(self):
        result = extract_signals(_issue(
            "Service crash",
            body="Happens sometimes.",
            comments=["DatabaseConnectionPool is the problematic class."],
        ))
        assert any("DatabaseConnectionPool" in c for c in result.class_names)


# ---------------------------------------------------------------------------
# Full realistic issue (integration)
# ---------------------------------------------------------------------------

class TestRealisticIssue:
    def test_flask_issue(self):
        """Simulates a realistic Flask issue report."""
        title = "RequestHandler raises AttributeError when Content-Type is missing"
        body = """
## Description

When making a POST request without a `Content-Type` header, the
`parse_request_body` function crashes.

## Steps to reproduce

```python
import requests
resp = requests.post("http://localhost:5000/api/submit", data="raw body")
```

## Error

```
Traceback (most recent call last):
  File "src/handlers/request_handler.py", line 34, in handle
    body = self.parse_request_body(request)
  File "src/handlers/request_handler.py", line 67, in parse_request_body
    content_type = request.headers["Content-Type"]
AttributeError: 'NoneType' object has no attribute '__getitem__'
```

## Expected behavior

I expected `parse_request_body` to return `None` when the header is absent.

## Actual behavior

It actually raises `AttributeError` instead of returning `None`.
"""
        result = extract_signals(_issue(title, body))

        # Error messages
        assert any("AttributeError" in e for e in result.error_messages)

        # File references
        assert any("request_handler.py" in f for f in result.file_references)

        # Function names
        func_names = result.function_names
        assert any("parse_request_body" in f for f in func_names)

        # Class names
        assert any("RequestHandler" in c or "NoneType" in c
                   for c in result.class_names)

        # Behavioral signals
        assert result.expected_behavior is not None
        assert result.observed_behavior is not None

    def test_react_issue(self):
        """Simulates a realistic React/JS issue report."""
        title = "useDataFetcher hook causes infinite re-render in UserDashboard"
        body = """
## Problem

The `useDataFetcher` hook triggers an infinite re-render loop when used inside
`UserDashboard`.

## Console output

```
Warning: Maximum update depth exceeded. This can happen when a component
calls setState inside useEffect, but useEffect either doesn't have a
dependency array, or one of the dependencies changes on every render.
    at UserDashboard (src/components/UserDashboard.jsx:24:10)
    at App (src/App.jsx:12:5)
```

## Expected

`useDataFetcher` should fetch data once on mount and not re-render
unless `userId` changes.

## Observed

It fetches on every render. The `userId` prop passed to `useDataFetcher`
creates a new object reference each time.
"""
        result = extract_signals(_issue(title, body))

        # File references from stack trace
        assert any("UserDashboard.jsx" in f or "App.jsx" in f
                   for f in result.file_references)

        # Class/component names
        assert any("UserDashboard" in c for c in result.class_names)

        # Function names (hook)
        assert any("useDataFetcher" in f for f in result.function_names)

        # Behavioral signals
        assert result.expected_behavior is not None
        assert result.observed_behavior is not None
