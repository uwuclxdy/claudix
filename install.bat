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

REM The plugin's MCP server and all hooks run through bash, so bash is required
REM at runtime. Detect it now so a bashless Windows box gets a clear error
REM instead of silently installing a plugin that cannot start.
where bash >nul 2>&1
if errorlevel 1 (
    echo claudix: bash not found on PATH.
    echo claudix needs bash (Git Bash, MSYS2, or WSL) for its MCP server and hooks.
    echo Install Git for Windows or MSYS2, add bash to PATH, then rerun this script.
    echo Skipping binary priming; the plugin cannot run until bash is available.
    goto :done
)

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

:done

echo.
echo claudix installed. Restart Claude Code to activate.
