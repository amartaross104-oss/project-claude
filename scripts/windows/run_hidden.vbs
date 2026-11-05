' Runs a .cmd file with NO visible window (wscript has no console).
' Usage: wscript.exe run_hidden.vbs "C:\full\path\run_scalp.cmd"
Set sh = CreateObject("WScript.Shell")
sh.Run "cmd /c """ & WScript.Arguments(0) & """", 0, True
