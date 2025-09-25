import importlib
import sys
from typing import Callable, Optional, Tuple


class CheckResult:
    def __init__(self, package_name: str, ok: bool, detected_version: Optional[str], message: str):
        self.package_name = package_name
        self.ok = ok
        self.detected_version = detected_version
        self.message = message


def parse_version(version_str: str) -> Tuple[int, ...]:
    numeric = []
    part = ''
    for ch in version_str:
        if ch.isdigit():
            part += ch
        else:
            if part:
                numeric.append(int(part))
                part = ''
    if part:
        numeric.append(int(part))
    return tuple(numeric)


def compare_exact(required: str) -> Callable[[str], bool]:
    required_tuple = parse_version(required)

    def _cmp(found: str) -> bool:
        return parse_version(found)[: len(required_tuple)] == required_tuple

    return _cmp


def compare_min(required_min: str) -> Callable[[str], bool]:
    required_tuple = parse_version(required_min)

    def _cmp(found: str) -> bool:
        return parse_version(found) >= required_tuple

    return _cmp


def try_import(module_name: str):
    return importlib.import_module(module_name)


def get_version_attr(mod, attr_candidates=("__version__", "version", "__VERSION__")) -> Optional[str]:
    for attr in attr_candidates:
        if hasattr(mod, attr):
            v = getattr(mod, attr)
            return str(v)
    return None


def check_package(
    display_name: str,
    module_name: str,
    version_check: Optional[Callable[[str], bool]] = None,
    required_version_human: Optional[str] = None,
) -> CheckResult:
    try:
        mod = try_import(module_name)
    except Exception as e:
        return CheckResult(display_name, False, None, f"not installed ({e.__class__.__name__}: {e})")

    detected_version = get_version_attr(mod)
    if version_check is None or detected_version is None:
        return CheckResult(display_name, True, detected_version, "ok")

    if version_check(detected_version):
        return CheckResult(display_name, True, detected_version, "ok")
    else:
        rv = required_version_human or "<unspecified>"
        return CheckResult(
            display_name,
            False,
            detected_version,
            f"version mismatch (have {detected_version}, require {rv})",
        )


def check_torch_cuda(mod_torch) -> Optional[CheckResult]:
    try:
        version = getattr(mod_torch, "__version__", "?")
        has_cuda = bool(getattr(mod_torch, "cuda").is_available())
        if "+cu" in version and not has_cuda:
            return CheckResult("torch-cuda", False, version, "CUDA build detected but CUDA is not available")
        return CheckResult("torch-cuda", True, version, "ok" if has_cuda else "cuda not available (ok if CPU-only)")
    except Exception as e:
        return CheckResult("torch-cuda", False, None, f"error querying CUDA: {e}")


def main() -> int:
    checks: list[CheckResult] = []

    # Essential packages per requirements.txt
    checks.append(
        check_package(
            display_name="torch",
            module_name="torch",
            version_check=compare_exact("1.13.1"),
            required_version_human="1.13.1 (+cu117 if using CUDA 11.7)",
        )
    )

    # Torch CUDA environment sanity
    try:
        torch_mod = importlib.import_module("torch")
        cuda_check = check_torch_cuda(torch_mod)
        if cuda_check:
            checks.append(cuda_check)
    except Exception:
        pass  # torch already failed, skip CUDA check

    checks.append(
        check_package(
            display_name="torch-geometric",
            module_name="torch_geometric",
            version_check=compare_exact("2.2.0"),
            required_version_human="2.2.0",
        )
    )

    # PyG binary extensions: import is a good smoke test
    for name, module in (
        ("torch-scatter", "torch_scatter"),
        ("torch-sparse", "torch_sparse"),
        ("torch-cluster", "torch_cluster"),
        ("torch-spline-conv", "torch_spline_conv"),
    ):
        checks.append(check_package(display_name=name, module_name=module))

    checks.append(
        check_package(
            display_name="numpy",
            module_name="numpy",
            version_check=compare_exact("1.25.0"),
            required_version_human="1.25.0",
        )
    )
    checks.append(
        check_package(
            display_name="torchmetrics",
            module_name="torchmetrics",
            version_check=compare_exact("1.2.1"),
            required_version_human="1.2.1",
        )
    )
    checks.append(
        check_package(
            display_name="scipy",
            module_name="scipy",
            version_check=compare_min("1.10.0"),
            required_version_human=">=1.10.0",
        )
    )
    checks.append(
        check_package(
            display_name="tqdm",
            module_name="tqdm",
            version_check=compare_min("4.66.0"),
            required_version_human=">=4.66.0",
        )
    )
    checks.append(
        check_package(
            display_name="PyYAML",
            module_name="yaml",
            version_check=compare_min("6.0.0"),
            required_version_human=">=6.0",
        )
    )

    # RDKit version strings are like "2023.09.4" while requirements may say "2023.9.4"
    def rdkit_cmp(found: str) -> bool:
        # Compare first three numeric components
        return parse_version(found)[:3] == parse_version("2023.9.4")[:3]

    checks.append(
        check_package(
            display_name="rdkit-pypi",
            module_name="rdkit",
            version_check=rdkit_cmp,
            required_version_human="2023.9.4",
        )
    )

    max_name_len = max(len(c.package_name) for c in checks)
    failures = 0
    print("\nEssential package check:")
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        version_str = f" ({c.detected_version})" if c.detected_version else ""
        print(f"- {c.package_name.ljust(max_name_len)} : {status}{version_str} - {c.message}")
        if not c.ok:
            failures += 1

    if failures:
        print(f"\n{failures} issue(s) detected. Please fix the failures above.")
        return 1
    else:
        print("\nAll essential packages are installed and satisfy version requirements.")
        return 0


if __name__ == "__main__":
    sys.exit(main())


