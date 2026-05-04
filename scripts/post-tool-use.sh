#!/usr/bin/env bash

"${CLAUDE_PLUGIN_ROOT}/bin/claudix" hook PostToolUse || exit 0
