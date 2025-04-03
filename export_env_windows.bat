@echo off
for /f "usebackq tokens=1,2 delims==" %%i in (.env) do (
    set %%i=%%j
)
