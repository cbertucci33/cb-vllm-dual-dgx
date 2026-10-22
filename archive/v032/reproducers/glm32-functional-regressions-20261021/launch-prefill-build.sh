#!/usr/bin/env bash
set +e
bash /opt/project/evidence/glm32-functional-regressions-20261021/run-prefill-repair-build.sh > /opt/project/evidence/glm32-functional-regressions-20261021/prefill-repair-build.log 2>&1
status=$?
printf '%s\n' "$status" > /opt/project/evidence/glm32-functional-regressions-20261021/prefill-repair-build.exit
exit "$status"
