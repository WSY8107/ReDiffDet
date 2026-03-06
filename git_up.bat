@echo off
setlocal enabledelayedexpansion

echo === [1/3] 正在拉取远程最新更改... ===
git pull origin main
if %errorlevel% neq 0 (
    echo [ERROR] 拉取失败，请检查网络或是否存在冲突！
    pause
    exit /b %errorlevel%
)

echo.
echo === [2/3] 正在添加更改并提交... ===
git add .

:: 获取当前时间作为默认提交信息
set msg=Update at %date% %time%
set /p user_msg="请输入提交描述 (直接回车将使用默认描述): "
if not "!user_msg!"=="" set msg=!user_msg!

git commit -m "!msg!"

echo.
echo === [3/3] 正在推送到远程仓库... ===
git push origin main
if %errorlevel% neq 0 (
    echo [ERROR] 推送失败！
    pause
    exit /b %errorlevel%
)

echo.
echo === [OK] 全部同步完成！ ===
pause