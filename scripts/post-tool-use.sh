#!/usr/bin/env bash
# Pass the JSON hook payload through stdin; fail open so the agent's
# session continues if the binary is missing or the handler errors.

"${CLAUDE_PLUGIN_ROOT}/bin/claudix" hook PostToolUse <&0 || true
exit 0
