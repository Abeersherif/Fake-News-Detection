@echo off
echo ========================================
echo GitHub Push Script
echo ========================================
echo.
echo This script will push your code to GitHub
echo Repository: https://github.com/Abeersherif/CODE-4-MODELS
echo.
echo You will be prompted for authentication.
echo Use your GitHub username and Personal Access Token as password.
echo.
pause

cd /d "c:\Users\Dell PC\Desktop\CODE 4 MODELS"

echo.
echo Checking Git status...
git status

echo.
echo Pushing to GitHub...
git push -u origin main

echo.
echo ========================================
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS! Code pushed to GitHub
    echo Visit: https://github.com/Abeersherif/CODE-4-MODELS
) else (
    echo FAILED! There was an error pushing to GitHub
    echo.
    echo Troubleshooting:
    echo 1. Make sure you're using your Personal Access Token as password
    echo 2. Check that the token has 'repo' permissions
    echo 3. Try regenerating the token at: https://github.com/settings/tokens
)
echo ========================================
echo.
pause
