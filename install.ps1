# Install the latest indexio release on Windows.
#
#   irm https://raw.githubusercontent.com/IOServicesLabs/indexio/main/install.ps1 | iex
#
# Options (environment):
#   INDEXIO_VERSION      a tag such as v0.1.0 (default: the latest release)
#   INDEXIO_INSTALL_DIR  where the binary goes (default: %LOCALAPPDATA%\Programs\indexio)
#
# The script downloads the Windows archive from the GitHub release, checks
# its SHA-256 against the published checksum, puts indexio.exe in the
# install directory and adds that directory to the user PATH.
$ErrorActionPreference = 'Stop'

$repo = 'IOServicesLabs/indexio'
$dir = if ($env:INDEXIO_INSTALL_DIR) { $env:INDEXIO_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Programs\indexio' }
$arch = if ([Environment]::Is64BitOperatingSystem) { 'x86_64' } else { throw 'indexio needs 64-bit Windows' }
if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { $arch = 'aarch64' }

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$tag = $env:INDEXIO_VERSION
if (-not $tag) {
    $tag = (Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" -Headers @{ 'User-Agent' = 'indexio-install' }).tag_name
}
$version = $tag.TrimStart('v')
$name = "indexio-$version-windows-$arch"
$base = "https://github.com/$repo/releases/download/$tag"

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("indexio-" + [IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "downloading $name.zip"
    Invoke-WebRequest "$base/$name.zip" -OutFile (Join-Path $tmp "$name.zip") -UseBasicParsing
    $want = ((Invoke-WebRequest "$base/$name.zip.sha256" -UseBasicParsing).Content -split '\s+')[0].ToLower()
    $have = (Get-FileHash (Join-Path $tmp "$name.zip") -Algorithm SHA256).Hash.ToLower()
    if ($want -ne $have) { throw 'checksum mismatch' }

    Expand-Archive (Join-Path $tmp "$name.zip") -DestinationPath $tmp -Force
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Copy-Item (Join-Path $tmp "$name\indexio.exe") (Join-Path $dir 'indexio.exe') -Force
    Write-Host "installed $dir\indexio.exe ($tag)"

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($userPath -split ';') -notcontains $dir) {
        [Environment]::SetEnvironmentVariable('Path', "$dir;$userPath", 'User')
        $env:Path = "$dir;$env:Path"
        Write-Host "added $dir to your user PATH (open a new terminal to use it)"
    }
    Write-Host 'next:  indexio add ~/code ; indexio setup claude ; indexio hook install'
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
