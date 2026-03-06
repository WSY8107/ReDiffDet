@echo off
chcp 65001 >nul
echo ===== Git 同步脚本 =====

:: 步骤 1：拉取远程最新更改
echo [1/3] 正在拉取远程最新代码...
git pull origin main

if %errorlevel% neq 0 (
    echo [错误] 拉取失败，请检查网络或解决冲突
    pause
    exit /b 1
)

echo [成功] 代码已是最新

:: 步骤 2：检查是否有本地更改
echo [2/3] 检查本地更改...
git status --short

:: 询问是否继续提交
set /p choice="是否有更改需要提交？(y/n): "

if /i "%choice%"=="y" (
    :: 步骤 3：提交并推送
    echo [3/3] 正在提交并推送...
    
    git add .
    if %errorlevel% neq 0 (
        echo [错误] git add 失败
        pause
        exit /b 1
    )
    
    set /p msg="请输入提交信息: "
    git commit -m "%msg%"
    if %errorlevel% neq 0 (
        echo [错误] git commit 失败，可能没有要提交的更改
        pause
        exit /b 1
    )
    
    git push origin main
    if %errorlevel% neq 0 (
        echo [错误] git push 失败
        pause
        exit /b 1
    )
    
    echo [成功] 推送完成！
) else (
    echo 跳过提交步骤
)

echo ===== 同步完成 =====
pause

