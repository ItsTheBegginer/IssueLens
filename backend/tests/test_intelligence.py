"""Tests for the Python AST analyzer and relationship builder."""
from __future__ import annotations

import textwrap

import pytest

from app.analyzers.python_ast import analyze_file, is_test_file
from app.analyzers.relationships import build_relationships
from app.models.intelligence import RelationshipType, SymbolType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dedent(s: str) -> str:
    return textwrap.dedent(s).strip()


# ---------------------------------------------------------------------------
# is_test_file
# ---------------------------------------------------------------------------

class TestIsTestFile:
    def test_test_prefix(self):
        assert is_test_file("tests/test_auth.py") is True

    def test_test_suffix(self):
        assert is_test_file("src/auth_test.py") is True

    def test_test_directory(self):
        assert is_test_file("src/tests/utils.py") is True

    def test_regular_file(self):
        assert is_test_file("src/auth/oauth.py") is False

    def test_test_word_in_name(self):
        assert is_test_file("src/test_runner.py") is True


# ---------------------------------------------------------------------------
# analyze_file — functions
# ---------------------------------------------------------------------------

class TestFunctionExtraction:
    def test_simple_function(self):
        src = dedent("""
            def greet(name):
                return f"Hello, {name}"
        """)
        fs = analyze_file("module.py", src)
        assert len(fs.symbols) == 1
        sym = fs.symbols[0]
        assert sym.name == "greet"
        assert sym.type == SymbolType.function
        assert sym.start_line == 1
        assert sym.parent is None

    def test_async_function(self):
        src = dedent("""
            async def fetch():
                pass
        """)
        fs = analyze_file("module.py", src)
        assert fs.symbols[0].type == SymbolType.async_function

    def test_multiple_functions(self):
        src = dedent("""
            def alpha():
                pass

            def beta():
                pass
        """)
        fs = analyze_file("module.py", src)
        names = [s.name for s in fs.symbols]
        assert "alpha" in names
        assert "beta" in names

    def test_function_line_numbers(self):
        src = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        fs = analyze_file("module.py", src)
        foo = next(s for s in fs.symbols if s.name == "foo")
        bar = next(s for s in fs.symbols if s.name == "bar")
        assert foo.start_line == 1
        assert bar.start_line == 4


# ---------------------------------------------------------------------------
# analyze_file — classes and methods
# ---------------------------------------------------------------------------

class TestClassExtraction:
    def test_class_detected(self):
        src = dedent("""
            class MyClass:
                pass
        """)
        fs = analyze_file("module.py", src)
        cls = next(s for s in fs.symbols if s.name == "MyClass")
        assert cls.type == SymbolType.class_

    def test_method_has_parent(self):
        src = dedent("""
            class Auth:
                def login(self):
                    pass
                async def logout(self):
                    pass
        """)
        fs = analyze_file("module.py", src)
        methods = [s for s in fs.symbols if s.parent == "Auth"]
        names = {s.name for s in methods}
        assert "login" in names
        assert "logout" in names

    def test_method_types(self):
        src = dedent("""
            class Auth:
                def login(self):
                    pass
                async def logout(self):
                    pass
        """)
        fs = analyze_file("module.py", src)
        login = next(s for s in fs.symbols if s.name == "login")
        logout = next(s for s in fs.symbols if s.name == "logout")
        assert login.type == SymbolType.method
        assert logout.type == SymbolType.async_method

    def test_class_end_line_after_methods(self):
        src = dedent("""
            class Big:
                def alpha(self):
                    pass

                def beta(self):
                    pass
        """)
        fs = analyze_file("module.py", src)
        cls = next(s for s in fs.symbols if s.name == "Big")
        assert cls.end_line > cls.start_line


# ---------------------------------------------------------------------------
# analyze_file — imports
# ---------------------------------------------------------------------------

class TestImportExtraction:
    def test_bare_import(self):
        src = "import os\n"
        fs = analyze_file("module.py", src)
        assert any(imp.module == "os" for imp in fs.imports)

    def test_from_import(self):
        src = "from pathlib import Path\n"
        fs = analyze_file("module.py", src)
        imp = next(i for i in fs.imports if i.module == "pathlib")
        assert any(n.name == "Path" for n in imp.names)

    def test_relative_import(self):
        src = "from .utils import helper\n"
        fs = analyze_file("app/module.py", src)
        imp = next(i for i in fs.imports if "utils" in i.module)
        assert imp.is_relative is True

    def test_aliased_import(self):
        src = "import numpy as np\n"
        fs = analyze_file("module.py", src)
        assert any(imp.module == "numpy" for imp in fs.imports)

    def test_import_line_number(self):
        src = "\nimport os\n"
        fs = analyze_file("module.py", src)
        imp = next(i for i in fs.imports if i.module == "os")
        assert imp.line == 2


# ---------------------------------------------------------------------------
# analyze_file — test detection
# ---------------------------------------------------------------------------

class TestTestDetection:
    def test_test_file_flagged(self):
        fs = analyze_file("tests/test_auth.py", "def test_login(): pass\n")
        assert fs.is_test is True

    def test_non_test_file_not_flagged(self):
        fs = analyze_file("app/auth.py", "def login(): pass\n")
        assert fs.is_test is False


# ---------------------------------------------------------------------------
# analyze_file — malformed Python
# ---------------------------------------------------------------------------

class TestMalformedFile:
    def test_syntax_error_does_not_crash(self):
        src = "def broken(\n    # unclosed\n"
        fs = analyze_file("broken.py", src)
        assert fs.parse_error is not None
        assert "SyntaxError" in fs.parse_error

    def test_empty_file(self):
        fs = analyze_file("empty.py", "")
        assert fs.parse_error is None
        assert fs.symbols == []
        assert fs.imports == []

    def test_comments_only(self):
        fs = analyze_file("comments.py", "# just a comment\n")
        assert fs.parse_error is None

    def test_malformed_file_does_not_affect_others(self):
        """Ensure a broken file doesn't stop analysis of surrounding files."""
        broken = analyze_file("bad.py", "def (: pass")
        good = analyze_file("good.py", "def foo(): pass\n")
        assert broken.parse_error is not None
        assert good.parse_error is None
        assert good.symbols[0].name == "foo"


# ---------------------------------------------------------------------------
# analyze_file — filtering (no directory traversal here; handled upstream)
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_dunder_init_analyzed(self):
        src = "from .module import Foo\n"
        fs = analyze_file("app/__init__.py", src)
        assert fs.parse_error is None

    def test_nested_class(self):
        src = dedent("""
            class Outer:
                class Inner:
                    def inner_method(self):
                        pass
        """)
        fs = analyze_file("module.py", src)
        names = {s.name for s in fs.symbols}
        assert "Outer" in names
        assert "Inner" in names


# ---------------------------------------------------------------------------
# Relationship builder
# ---------------------------------------------------------------------------

class TestRelationshipBuilder:
    def _fs(self, path: str, src: str):
        return analyze_file(path, src)

    def test_file_imports_module(self):
        fs = self._fs("app/auth.py", "import os\n")
        rels = build_relationships([fs])
        assert any(
            r.type == RelationshipType.file_imports_module and r.target == "os"
            for r in rels
        )

    def test_class_contains_method(self):
        src = dedent("""
            class Auth:
                def login(self):
                    pass
        """)
        fs = self._fs("app/auth.py", src)
        rels = build_relationships([fs])
        assert any(r.type == RelationshipType.class_contains_method for r in rels)

    def test_function_belongs_to_module(self):
        fs = self._fs("app/utils.py", "def helper(): pass\n")
        rels = build_relationships([fs])
        assert any(r.type == RelationshipType.function_belongs_to_module for r in rels)

    def test_test_references_module(self):
        src = "import flask\ndef test_app(): pass\n"
        fs = self._fs("tests/test_app.py", src)
        rels = build_relationships([fs])
        assert any(r.type == RelationshipType.test_references_module and r.target == "flask" for r in rels)

    def test_no_relationships_for_empty_file(self):
        fs = self._fs("empty.py", "")
        rels = build_relationships([fs])
        assert rels == []
