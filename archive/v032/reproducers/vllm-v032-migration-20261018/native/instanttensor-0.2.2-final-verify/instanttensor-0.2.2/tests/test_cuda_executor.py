import pathlib
import shutil
import subprocess
import sysconfig
import tempfile
import unittest


class CUDAExecutorTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ required")
    def test_state_machine(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "test_cuda_executor"
            includes = [
                root / "csrc",
                root / "csrc/third_party/atomic_queue/include",
                root / "csrc/third_party/pybind11/include",
                root / "csrc/third_party/libaio/src",
                pathlib.Path(sysconfig.get_path("include")),
            ]
            library = pathlib.Path(sysconfig.get_config_var("LIBDIR"))
            python_library = library / sysconfig.get_config_var("LDLIBRARY")
            if not python_library.exists():
                python_library = library / f"libpython{sysconfig.get_config_var('LDVERSION')}.so"
            build = subprocess.run([
                "g++", "-std=c++17", "-pthread", "-ffunction-sections",
                "-fdata-sections", "-Wl,--gc-sections",
                *(f"-I{path}" for path in includes),
                str(root / "tests/cpp/test_cuda_executor.cpp"),
                str(python_library),
                f"-Wl,-rpath,{library}", "-ldl", "-o", str(executable),
            ], capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            subprocess.run([str(executable)], check=True, timeout=30)
