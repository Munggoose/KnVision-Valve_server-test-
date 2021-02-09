@echo on
set root=C:\anaconda3
call %root%\Scripts\activate.bat %root%

call activate dl

:START

call python server.py

@GOTO START