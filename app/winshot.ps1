param([string]$OutDir = "$env:TEMP\o2bshots")
$ErrorActionPreference = "Stop"
Add-Type @"
using System; using System.Runtime.InteropServices;
public class W {
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int hh, bool r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L; public int T; public int R; public int B; }
  [DllImport("user32.dll")] public static extern uint GetDpiForWindow(IntPtr h);
  [DllImport("shcore.dll")] public static extern int SetProcessDpiAwareness(int v);
}
"@
[W]::SetProcessDpiAwareness(2) | Out-Null
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Get-Process osu2bs -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -m 500
Start-Process "$env:USERPROFILE\Desktop\osu2bs.exe"

$p = $null
for ($i = 0; $i -lt 30 -and -not $p; $i++) {
  Start-Sleep -m 500
  $p = Get-Process osu2bs -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
}
if (-not $p) { throw "no window" }
$h = $p.MainWindowHandle
[W]::SetForegroundWindow($h) | Out-Null
Start-Sleep -Seconds 2

$scale = [W]::GetDpiForWindow($h) / 96.0
$sizes = @(@(720,560,"min"), @(980,720,"default"), @(1500,950,"large"))
foreach ($s in $sizes) {
  $pw = [int]($s[0] * $scale); $ph = [int]($s[1] * $scale)
  [W]::MoveWindow($h, 40, 40, $pw, $ph, $true) | Out-Null
  Start-Sleep -m 900
  $r = New-Object W+RECT
  [W]::GetWindowRect($h, [ref]$r) | Out-Null
  $wd = $r.R - $r.L; $ht = $r.B - $r.T
  $bmp = New-Object System.Drawing.Bitmap($wd, $ht)
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.CopyFromScreen($r.L, $r.T, 0, 0, $bmp.Size)
  $bmp.Save("$OutDir\app-$($s[2]).png", [System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose(); $bmp.Dispose()
}
Get-Process osu2bs -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Output "captured to $OutDir"
