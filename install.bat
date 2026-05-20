@echo off
setlocal

where claude >nul 2>&1
if errorlevel 1 (
    echo claudix: claude CLI not found. Open Claude Code and run:
    echo   /plugin marketplace add uwuclxdy/claudix
    echo   /plugin install claudix@claudix
    echo Then restart Claude Code; the binary downloads on first session.
    exit /b 0
)

claude plugin uninstall claudix@claudix >nul 2>&1
claude plugin marketplace rm claudix >nul 2>&1
claude plugin marketplace add uwuclxdy/claudix
claude plugin install claudix@claudix

REM Prime the binary cache from the local checkout so the first session
REM does not stall on download. Uses the documented plugin-data path,
REM avoiding any dependency on node or jq.
set "SCRIPT_DIR=%~dp0"
set "PLUGIN_DATA=%USERPROFILE%\.claude\plugins\data\claudix-claudix"
if not exist "%PLUGIN_DATA%" mkdir "%PLUGIN_DATA%"
set "CLAUDE_PLUGIN_ROOT=%SCRIPT_DIR%"
set "CLAUDE_PLUGIN_DATA=%PLUGIN_DATA%"
bash "%SCRIPT_DIR%scripts\ensure-binary.sh" --install >nul
if errorlevel 1 (
    echo claudix: binary install will retry on first session
)

echo.
echo claudix installed. Restart Claude Code to activate.
