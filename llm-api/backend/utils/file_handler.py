"""
File handling utilities for uploads
"""
import base64
import os
import posixpath
import re
import shutil
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from fastapi import UploadFile

import config


# Upload filename sanitising (save_uploaded_files). The name comes straight
# from the multipart request, so it must never be trusted as a path.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")
# Characters Windows refuses in a filename (':' would also open an NTFS
# alternate data stream). Only replaced on Windows; Linux keeps them.
_WINDOWS_BAD_CHARS_RE = re.compile(r'[<>:"|?*]')
_WINDOWS_RESERVED_STEMS = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
# Stay well under the usual 255-byte NAME_MAX, leaving room for a "_N" suffix.
_MAX_NAME_BYTES = 200


def _truncate_name(name: str) -> str:
    """Shorten *name* to _MAX_NAME_BYTES (UTF-8), keeping a short extension."""
    if len(name.encode("utf-8")) <= _MAX_NAME_BYTES:
        return name
    stem, ext = os.path.splitext(name)
    if len(ext.encode("utf-8")) > 32:
        ext = ""
    budget = _MAX_NAME_BYTES - len(ext.encode("utf-8"))
    stem = stem.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
    return stem + ext


def sanitize_upload_filename(raw: Optional[str]) -> Optional[str]:
    """Reduce a client-supplied upload filename to a safe bare file name.

    Both '/' and '\\' are treated as separators and only the last component
    is kept, NULs/control characters are removed, and the result is capped in
    length. Returns None when nothing usable is left ('', '.', '..').
    """
    if raw is None:
        return None
    name = str(raw).replace("\\", "/")
    # posixpath (not os.path): identical on every OS once '\\' is '/'. On
    # Windows os.path would also split "C:name" at the drive; here ':' is
    # replaced below instead.
    name = posixpath.basename(name)
    name = _CONTROL_CHARS_RE.sub("", name).strip()
    if os.name == "nt":
        name = _WINDOWS_BAD_CHARS_RE.sub("_", name)
        # Windows silently drops trailing dots/spaces ("..." would name the
        # directory itself).
        name = name.rstrip(". ")
        if name.split(".", 1)[0].strip().lower() in _WINDOWS_RESERVED_STEMS:
            name = "_" + name
    if name in ("", ".", ".."):
        return None
    name = _truncate_name(name)
    if name in ("", ".", ".."):
        return None
    return name


def _dedupe_name(name: str, used: Set[str]) -> str:
    """Return *name*, or name_2.ext / name_3.ext ... if already in *used*.

    Compared case-insensitively (Windows/macOS filesystems are), and the
    chosen name is added to *used*.
    """
    candidate = name
    if candidate.casefold() in used:
        stem, ext = os.path.splitext(name)
        n = 2
        while True:
            # name is <= _MAX_NAME_BYTES, so the suffix still fits NAME_MAX.
            candidate = f"{stem}_{n}{ext}"
            if candidate.casefold() not in used:
                break
            n += 1
    used.add(candidate.casefold())
    return candidate


def is_image_file(file_path: str) -> bool:
    """Check if a file is a supported image format."""
    return Path(file_path).suffix.lower() in config.IMAGE_SUPPORTED_FORMATS


def encode_image_base64(file_path: str) -> Optional[Dict[str, Any]]:
    """Read an image file and return a base64-encoded data URL dict.

    Returns ``{"url": "data:image/<subtype>;base64,..."}`` or *None* on failure.
    Resizes the image if either dimension exceeds ``IMAGE_MAX_DIMENSION``.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    }
    mime = mime_map.get(suffix, "image/png")

    try:
        max_dim = getattr(config, "IMAGE_MAX_DIMENSION", 4096)
        max_bytes = getattr(config, "IMAGE_MAX_SIZE_MB", 20) * 1024 * 1024

        # Check file size
        if path.stat().st_size > max_bytes:
            print(f"[IMAGE] {path.name} exceeds {config.IMAGE_MAX_SIZE_MB}MB limit, skipping")
            return None

        # Try to resize with Pillow if available and image is large
        try:
            from PIL import Image
            import io
            img = Image.open(path)
            w, h = img.size
            if w > max_dim or h > max_dim:
                ratio = min(max_dim / w, max_dim / h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                print(f"[IMAGE] Resized {path.name}: {w}x{h} -> {new_size[0]}x{new_size[1]}")
                buf = io.BytesIO()
                fmt = "PNG" if suffix == ".png" else "JPEG" if suffix in (".jpg", ".jpeg") else "WEBP" if suffix == ".webp" else "PNG"
                img.save(buf, format=fmt)
                raw = buf.getvalue()
            else:
                raw = path.read_bytes()
        except ImportError:
            # Pillow not installed — send raw bytes
            raw = path.read_bytes()

        b64 = base64.b64encode(raw).decode("ascii")
        return {"url": f"data:{mime};base64,{b64}"}
    except Exception as e:
        print(f"[IMAGE] Failed to encode {path.name}: {e}")
        return None


def save_uploaded_files(
    files: List[UploadFile],
    username: str,
    session_id: str
) -> List[str]:
    """
    Save uploaded files to both persistent and scratch directories

    Args:
        files: List of uploaded files
        username: Username for persistent storage
        session_id: Session ID for scratch storage

    Returns:
        List of file paths in scratch directory. File names are sanitised
        (sanitize_upload_filename) and de-duplicated within this call
        (report.pdf, report_2.pdf, ...), so they can differ from the names
        the client sent; files whose name is unusable are skipped.
    """
    scratch_paths = []
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024

    # Create directories
    user_upload_dir = config.UPLOAD_DIR / username
    session_scratch_dir = config.SCRATCH_DIR / session_id

    user_upload_dir.mkdir(parents=True, exist_ok=True)
    session_scratch_dir.mkdir(parents=True, exist_ok=True)
    upload_root = user_upload_dir.resolve()
    scratch_root = session_scratch_dir.resolve()

    used_names: Set[str] = set()
    for i, file in enumerate(files):
        if file.filename:
            safe_name = sanitize_upload_filename(file.filename)
            if not safe_name:
                print(f"[FILE_HANDLER] Skipping upload with unusable filename {file.filename!r}")
                continue
            safe_name = _dedupe_name(safe_name, used_names)
            if safe_name != file.filename:
                print(f"[FILE_HANDLER] Upload {file.filename!r} saved as {safe_name!r}")
            try:
                user_file_path = user_upload_dir / safe_name
                scratch_file_path = session_scratch_dir / safe_name
                # Belt and braces: the sanitised name must stay a direct child.
                if (user_file_path.resolve().parent != upload_root
                        or scratch_file_path.resolve().parent != scratch_root):
                    raise ValueError(f"Upload filename {file.filename!r} escapes the upload directory")

                with open(user_file_path, 'wb') as f:
                    shutil.copyfileobj(file.file, f)
                user_size = user_file_path.stat().st_size
                if user_size > max_bytes:
                    user_file_path.unlink()
                    raise ValueError(
                        f"File '{safe_name}' exceeds the {config.MAX_FILE_SIZE_MB}MB limit"
                    )

                shutil.copy2(user_file_path, scratch_file_path)

                scratch_paths.append(str(scratch_file_path))
            except Exception as e:
                print(f"[FILE_HANDLER] ERROR saving {file.filename!r}: {e}")
                import traceback
                traceback.print_exc()

    return scratch_paths


def extract_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract rich metadata from files based on type

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file-type-specific metadata
    """
    path = Path(file_path)
    metadata = {}

    try:
        file_type = path.suffix.lstrip('.').lower()

        # JSON files
        if file_type == 'json':
            metadata.update(_extract_json_metadata(path))

        # CSV files
        elif file_type == 'csv':
            metadata.update(_extract_csv_metadata(path))

        # Excel files
        elif file_type in ['xlsx', 'xls']:
            metadata.update(_extract_excel_metadata(path))

        # Text/Code files
        elif file_type in ['txt', 'md', 'py', 'js', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'ts', 'jsx', 'tsx', 'html', 'css', 'xml']:
            metadata.update(_extract_text_metadata(path))

    except Exception as e:
        metadata['metadata_error'] = str(e)

    return metadata


def _extract_json_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from JSON files"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = {}

        # Determine structure type
        if isinstance(data, dict):
            metadata['structure'] = 'object'
            metadata['keys'] = list(data.keys())
            metadata['key_count'] = len(data.keys())

            # Sample values for first few keys
            sample = {}
            for key in list(data.keys())[:5]:
                value = data[key]
                if isinstance(value, (dict, list)):
                    sample[key] = f"<{type(value).__name__} with {len(value)} items>"
                else:
                    sample[key] = value
            metadata['sample'] = sample

        elif isinstance(data, list):
            metadata['structure'] = 'array'
            metadata['length'] = len(data)
            if len(data) > 0:
                metadata['first_item_type'] = type(data[0]).__name__
                # Show first 2 items as sample
                metadata['sample'] = data[:2]
        else:
            metadata['structure'] = 'primitive'
            metadata['value_type'] = type(data).__name__

        return metadata

    except json.JSONDecodeError as e:
        return {'parse_error': f'Invalid JSON: {str(e)}'}
    except Exception as e:
        return {'error': str(e)}


def _extract_csv_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from CSV files — reads only headers + sample rows, not the whole file."""
    try:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            # Detect delimiter from first 1 KB
            sample = f.read(1024)
            f.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter

            reader = csv.reader(f, delimiter=delimiter)
            # Read only the first 4 rows (header + 3 data rows); count rest cheaply
            head_rows = []
            for _ in range(4):
                try:
                    head_rows.append(next(reader))
                except StopIteration:
                    break

            # Count remaining rows without loading them
            row_count = len(head_rows) + sum(1 for _ in reader)

        metadata: Dict[str, Any] = {
            'rows': row_count,
            'delimiter': delimiter,
        }
        if head_rows:
            metadata['columns'] = len(head_rows[0])
            metadata['headers'] = head_rows[0]
            if len(head_rows) > 1:
                metadata['sample_rows'] = head_rows[1:]

        return metadata

    except Exception as e:
        return {'error': str(e)}


def _extract_excel_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from Excel files"""
    try:
        import pandas as pd

        # Read all sheets
        excel_file = pd.ExcelFile(path)
        sheet_names = excel_file.sheet_names

        metadata = {
            'sheet_count': len(sheet_names),
            'sheet_names': sheet_names,
            'sheets': {}
        }

        # Get info for each sheet
        for sheet_name in sheet_names[:5]:  # Limit to first 5 sheets
            df = pd.read_excel(path, sheet_name=sheet_name)

            sheet_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'sample_rows': df.head(3).to_dict('records')  # First 3 rows
            }

            metadata['sheets'][sheet_name] = sheet_info

        return metadata

    except ImportError:
        return {'error': 'pandas not available for Excel parsing'}
    except Exception as e:
        return {'error': str(e)}


def _extract_text_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from text/code files"""
    try:
        line_count = 0
        char_count = 0
        preview_lines: List[str] = []
        scan_lines: List[str] = []

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line_count += 1
                char_count += len(line)
                if line_count <= 10:
                    preview_lines.append(line)
                if line_count <= 100:
                    scan_lines.append(line)

        metadata = {
            'lines': line_count,
            'chars': char_count,
            'preview': ''.join(preview_lines),
        }

        # For code files, try to detect structure
        file_type = path.suffix.lstrip('.').lower()
        if file_type in ['py', 'js', 'java', 'cpp', 'c', 'go', 'rs', 'ts']:
            # Simple detection of functions/classes
            imports = []
            definitions = []

            for line in scan_lines:
                line_stripped = line.strip()

                # Python
                if file_type == 'py':
                    if line_stripped.startswith('import ') or line_stripped.startswith('from '):
                        imports.append(line_stripped)
                    elif line_stripped.startswith('def ') or line_stripped.startswith('class '):
                        definitions.append(line_stripped.split('(')[0].split(':')[0])

                # JavaScript/TypeScript
                elif file_type in ['js', 'ts', 'jsx', 'tsx']:
                    if 'import ' in line_stripped:
                        imports.append(line_stripped)
                    if 'function ' in line_stripped or 'class ' in line_stripped:
                        definitions.append(line_stripped.split('{')[0].strip())

            if imports:
                metadata['imports'] = imports[:10]  # First 10 imports
            if definitions:
                metadata['definitions'] = definitions[:15]  # First 15 definitions

        return metadata

    except Exception as e:
        return {'error': str(e)}
