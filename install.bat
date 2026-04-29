@echo off
setlocal

set BUNDLED=0
:parse_args
if "%~1"=="" goto done_args
if "%~1"=="--bundled" set BUNDLED=1
shift
goto parse_args
:done_args

where cargo >nul 2>&1
if errorlevel 1 (
    echo claudix: cargo is required. Install Rust from https://rustup.rs/ 1>&2
    exit /b 1
)

set REPO=https://github.com/uwuclxdy/claudix

if "%BUNDLED%"=="1" (
    cargo install --git %REPO%
) else (
    cargo install --git %REPO% --no-default-features
)
if errorlevel 1 exit /b 1

where claude >nul 2>&1
if errorlevel 1 (
    echo.
    echo claudix binary installed. Open Claude Code and run:
    echo   claude plugin marketplace add uwuclxdy/claudix
    echo   claude plugin install claudix@uwuclxdy
    echo Then restart Claude Code.
) else (
    claude plugin marketplace add uwuclxdy/claudix
    claude plugin install claudix@uwuclxdy
    echo.
    echo claudix installed. Restart Claude Code to activate.
)
