@echo off
setlocal

where cargo >nul 2>&1
if errorlevel 1 (
    echo claudix: cargo is required. Install Rust from https://rustup.rs/ 1>&2
    exit /b 1
)

set REPO=https://github.com/uwuclxdy/claudix

cargo install --git %REPO%
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
	claude plugin marketplace rm uwuclxdy/claudix >nul 2>&1
    claude plugin marketplace add uwuclxdy/claudix
    claude plugin install claudix@claudix
    echo.
    echo claudix installed. Restart Claude Code to activate.
)
