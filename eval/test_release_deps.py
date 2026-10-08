"""Exercise the release workflow's apt recovery without changing system packages."""
import os
from pathlib import Path
import subprocess
import unittest

import yaml


class ReleaseDependenciesTest(unittest.TestCase):
    def test_cached_install_recovery(self):
        workflow = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / '.github/workflows/release.yml').read_text()
        )
        script = next(step['run'] for step in workflow['jobs']['build']['steps']
                      if step.get('name') == 'Install and verify system dependencies')
        # Substitute only external commands; execute the real workflow shell logic.
        mocks = '''
        attempts=0
        mkdir() { :; }
        sudo() {
          if [[ "$1" == tee ]]; then cat >/dev/null; return; fi
          for arg in "$@"; do
            case "$arg" in
              update) echo update; return "$FAIL_UPDATE" ;;
              clean) echo clean; return 0 ;;
              install)
                attempts=$((attempts + 1))
                echo install
                if (( attempts <= FAIL_INSTALLS )); then return 100; fi
                return 0 ;;
            esac
          done
          return 99
        }
        pkg-config() { echo verified; }
        '''
        for failures, update_error, expected, success in [
            (0, 0, ['update', 'install', 'verified'], True),
            (1, 0, ['update', 'install', 'clean', 'update', 'install', 'verified'], True),
            (2, 0, ['update', 'install', 'clean', 'update', 'install'], False),
            (0, 100, ['update'], False),
        ]:
            with self.subTest(failures=failures, update_error=update_error):
                result = subprocess.run(
                    ['bash', '-euo', 'pipefail', '-c', mocks + script],
                    env={**os.environ, 'FAIL_INSTALLS': str(failures),
                         'FAIL_UPDATE': str(update_error)},
                    text=True, capture_output=True,
                )
                calls = [line for line in result.stdout.splitlines()
                         if line in {'update', 'install', 'clean', 'verified'}]
                self.assertEqual(calls, expected, result.stderr)
                self.assertEqual(result.returncode == 0, success, result.stderr)


if __name__ == '__main__':
    unittest.main()
