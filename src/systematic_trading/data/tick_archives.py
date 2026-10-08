"""Read-only access to vendor ZIPs; extracted CSVs are temporary raw inputs."""

import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


def extract_csv_from_zip(
    zip_path: Path,
    destination: Path,
    csv_name: str,
    *,
    output_name: str | None = None,
) -> Path:
    """Safely extract one unambiguous CSV and verify its CRC through EOF."""
    target_name = csv_name if output_name is None else output_name
    if Path(target_name).name != target_name or target_name in ("", ".", ".."):
        raise ValueError(f"Expected a plain CSV output filename: {target_name!r}")
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = []
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename.replace("\\", "/"))
            if (
                member.is_dir()
                or member_path.suffix.lower() != ".csv"
                or "__MACOSX" in member_path.parts
                or member_path.name.startswith("._")
            ):
                continue
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise RuntimeError(f"Unsafe tick CSV member in {zip_path}: {member.filename}")
            csv_members.append((member_path.name, member))
        if len(csv_members) != 1 or csv_members[0][0] != csv_name:
            raise RuntimeError(
                f"Expected exactly one tick CSV {csv_name!r} in {zip_path}; "
                f"found {[member.filename for _, member in csv_members]}."
            )
        target = destination / target_name
        created = False
        try:
            with target.open("xb") as dst:
                created = True
                with archive.open(csv_members[0][1]) as src:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
    return target
