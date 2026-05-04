@echo off
setlocal

where cargo >nul 2>&1
if errorlevel 1 (
    echo claudix: cargo is required. Install Rust from https://rustup.rs/ 1>&2
    exit /b 1
)

if "%CLAUDIX_INSTALL_REPO%"=="" (
    set REPO=https://github.com/uwuclxdy/claudix
) else (
    set REPO=%CLAUDIX_INSTALL_REPO%
)

if exist "%REPO%\Cargo.toml" (
    cargo install --path "%REPO%"
) else (
    cargo install --git %REPO%
)
if errorlevel 1 exit /b 1

where claude >nul 2>&1
if errorlevel 1 (
    echo.
    echo claudix binary installed. Open Claude Code and run:
    echo   /plugin marketplace add uwuclxdy/claudix
    echo   /plugin install claudix@claudix
    echo Then restart Claude Code.
) else (
	claude plugin uninstall claudix@claudix >nul 2>&1
	claude plugin marketplace rm claudix >nul 2>&1
    claude plugin marketplace add uwuclxdy/claudix
    claude plugin install claudix@claudix
    for /f "usebackq delims=" %%P in (`claude plugin list --json ^| node -e "let input=''; process.stdin.on('data', chunk => input += chunk); process.stdin.on('end', () => { const plugins = JSON.parse(input); const plugin = plugins.find(item => item.id === 'claudix@claudix'); if (plugin) process.stdout.write(plugin.installPath); });"`) do set CLAUDE_PLUGIN_ROOT=%%P
    claudix install
    if errorlevel 1 exit /b 1
    echo.
    echo claudix installed. Restart Claude Code to activate.
)
