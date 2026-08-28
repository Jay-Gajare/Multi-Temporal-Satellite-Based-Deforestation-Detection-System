"""Theme configuration — loads custom CSS."""
from pathlib import Path

_css_path = Path(__file__).parent / "main.css"
CUSTOM_CSS = _css_path.read_text(encoding="utf-8")
