@echo off
title 智枢星启动器
setlocal
cd /d "%~dp0"

set "PORT=7860"
set "URL=http://127.0.0.1:%PORT%/"
set /a tries=0

rem 端口已在监听则不重复起服务,直接开浏览器
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
if %errorlevel%==0 goto open

echo [智枢星] 正在启动 Web 控制台(端口 %PORT%,服务器窗口已最小化)...
start "zhishuxing server" /min cmd /c "zhishuxing serve --host 127.0.0.1 --port %PORT% & echo. & echo 服务器已退出(正常停止或报错均停在此处)。 & pause"

:waitloop
rem 用 ping 做延时,不依赖控制台 stdin(timeout 在重定向下会报错退出)
%SystemRoot%\System32\ping.exe -n 2 127.0.0.1 >nul
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>nul
if %errorlevel%==0 goto open
set /a tries+=1
if %tries% lss 20 goto waitloop
echo [智枢星] 等待超时:请查看最小化的服务器窗口中的报错信息。
pause
exit /b 1

:open
if %tries%==0 (
    echo [智枢星] 服务已在运行,直接打开浏览器:%URL%
) else (
    echo [智枢星] 已启动,打开浏览器:%URL%
)
start "" "%URL%"
%SystemRoot%\System32\ping.exe -n 3 127.0.0.1 >nul
endlocal
