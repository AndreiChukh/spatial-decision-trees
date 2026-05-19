@echo off
setlocal enabledelayedexpansion
REM Lightweight offline launcher for Windows. Supports:
REM   gradlew.bat run
REM   gradlew.bat run --args="..."
REM   gradlew.bat clean
REM
REM Important: source list is written with RELATIVE paths. This avoids javac
REM MalformedInputException when the project folder contains Cyrillic symbols,
REM for example "Рабочий стол".

set CMD=%1
if "%CMD%"=="" set CMD=help
shift /1

if "%CMD%"=="clean" (
  if exist build\classes-gradlew rmdir /s /q build\classes-gradlew
  if exist build\sources-gradlew.txt del /q build\sources-gradlew.txt
  echo Cleaned offline build output.
  exit /b 0
)

if "%CMD%"=="help" goto :help
if "%CMD%"=="--help" goto :help
if "%CMD%"=="-h" goto :help
if "%CMD%"=="tasks" goto :help
if not "%CMD%"=="run" (
  echo Unsupported command: %CMD%
  echo Use: gradlew.bat run --args="..."
  exit /b 2
)

if not exist build mkdir build
if not exist build\classes-gradlew mkdir build\classes-gradlew

REM Create a javac argument file without absolute paths.
REM Absolute paths may contain non-UTF-8 console characters on Windows.
if exist build\sources-gradlew.txt del /q build\sources-gradlew.txt
for /f "delims=" %%F in ('dir /s /b src\main\java\*.java') do (
  set "P=%%F"
  set "P=!P:%CD%\=!"
  echo !P!>> build\sources-gradlew.txt
)

javac -encoding UTF-8 -d build\classes-gradlew @build\sources-gradlew.txt
if errorlevel 1 exit /b 1

set ARGS_STRING=
:parse
if "%~1"=="" goto :afterparse
set A=%~1
if "%A:~0,7%"=="--args=" set ARGS_STRING=%A:~7%
shift /1
goto :parse

:afterparse
set SCENE=LC09_L2SP_167041_20230317_20230320_02_T1
set DEFAULT=data\%SCENE%_SR_B2.TIF data\%SCENE%_SR_B3.TIF data\%SCENE%_SR_B4.TIF data\%SCENE%_SR_B5.TIF data\%SCENE%_SR_B6.TIF data\%SCENE%_SR_B7.TIF data\%SCENE%_QA_PIXEL.TIF

if "%ARGS_STRING%"=="" (
  set RUNARGS=%DEFAULT%
) else (
  echo %ARGS_STRING% | findstr /I ".TIF" >nul
  if errorlevel 1 (
    set RUNARGS=%DEFAULT% %ARGS_STRING%
  ) else (
    set RUNARGS=%ARGS_STRING%
  )
)

for %%F in (%DEFAULT%) do (
  if not exist "%%F" (
    echo Missing default raster: %%F
    echo Put Landsat .TIF files into data or pass all paths through --args.
    exit /b 3
  )
)

if "%JAVA_OPTS%"=="" set JAVA_OPTS=-Xmx4g
java %JAVA_OPTS% -Djava.awt.headless=true -cp build\classes-gradlew sdt.Main %RUNARGS%
exit /b %ERRORLEVEL%

:help
echo Available commands:
echo   gradlew.bat run
echo   gradlew.bat run --args="..."
echo   gradlew.bat clean
echo.
echo Default data folder: .\data\ with full Landsat file names.
exit /b 0
