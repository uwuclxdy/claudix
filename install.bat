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

REM refresh the marketplace source (add if missing, update if present), then
REM install or update without uninstalling an existing install.
claude plugin marketplace add uwuclxdy/claudix >nul 2>&1
if errorlevel 1 claude plugin marketplace update claudix >nul 2>&1

claude plugin list 2>&1 | findstr /c:"claudix@claudix" >nul
if errorlevel 1 (
    claude plugin install claudix@claudix
) else (
    claude plugin update claudix@claudix >nul 2>&1
)

REM The plugin's MCP server and hooks run through node (Claude Code is a node
REM app, so node is expected on PATH). Detect it so a box without node gets a
REM clear error instead of a silently-broken plugin.
where node >nul 2>&1
if errorlevel 1 (
    echo claudix: node not found on PATH.
    echo claudix needs node for its MCP server and hooks; Claude Code normally provides it.
    echo Skipping binary priming; it will retry on first session if node is available then.
    goto :done
)

REM Prime the binary cache from the local checkout so the first session
REM does not stall on download.
set "SCRIPT_DIR=%~dp0"
set "PLUGIN_DATA=%USERPROFILE%\.claude\plugins\data\claudix-claudix"
if not exist "%PLUGIN_DATA%" mkdir "%PLUGIN_DATA%"
set "CLAUDE_PLUGIN_ROOT=%SCRIPT_DIR%"
set "CLAUDE_PLUGIN_DATA=%PLUGIN_DATA%"
node "%SCRIPT_DIR%bin\claudix-bootstrap.js" --install >nul
if errorlevel 1 (
    echo claudix: binary install will retry on first session
)

:done

echo.
echo claudix installed. Restart Claude Code to activate.
