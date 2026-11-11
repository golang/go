// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && race && !cgo && windows && amd64

package race

//go:cgo_import_dynamic WaitOnAddress WaitOnAddress "api-ms-win-core-synch-l1-2-0.dll"
//go:cgo_import_dynamic WakeByAddressAll WakeByAddressAll "api-ms-win-core-synch-l1-2-0.dll"
//go:cgo_import_dynamic WakeByAddressSingle WakeByAddressSingle "api-ms-win-core-synch-l1-2-0.dll"
//go:cgo_import_dynamic _errno _errno "msvcrt.dll"
//go:cgo_import_dynamic exit exit "msvcrt.dll"
//go:cgo_import_dynamic memset memset "msvcrt.dll"
//go:cgo_import_dynamic K32GetProcessMemoryInfo K32GetProcessMemoryInfo "kernel32.dll"
//go:cgo_import_dynamic CloseHandle CloseHandle "kernel32.dll"
//go:cgo_import_dynamic CreateDirectoryA CreateDirectoryA "kernel32.dll"
//go:cgo_import_dynamic CreateFileA CreateFileA "kernel32.dll"
//go:cgo_import_dynamic GetCurrentProcess GetCurrentProcess "kernel32.dll"
//go:cgo_import_dynamic GetCurrentThreadId GetCurrentThreadId "kernel32.dll"
//go:cgo_import_dynamic GetEnvironmentVariableA GetEnvironmentVariableA "kernel32.dll"
//go:cgo_import_dynamic GetFileAttributesA GetFileAttributesA "kernel32.dll"
//go:cgo_import_dynamic GetLastError GetLastError "kernel32.dll"
//go:cgo_import_dynamic GetModuleFileNameW GetModuleFileNameW "kernel32.dll"
//go:cgo_import_dynamic GetNativeSystemInfo GetNativeSystemInfo "kernel32.dll"
//go:cgo_import_dynamic GetProcessId GetProcessId "kernel32.dll"
//go:cgo_import_dynamic GetStdHandle GetStdHandle "kernel32.dll"
//go:cgo_import_dynamic GetSystemInfo GetSystemInfo "kernel32.dll"
//go:cgo_import_dynamic IsDebuggerPresent IsDebuggerPresent "kernel32.dll"
//go:cgo_import_dynamic OutputDebugStringA OutputDebugStringA "kernel32.dll"
//go:cgo_import_dynamic QueryPerformanceCounter QueryPerformanceCounter "kernel32.dll"
//go:cgo_import_dynamic QueryPerformanceFrequency QueryPerformanceFrequency "kernel32.dll"
//go:cgo_import_dynamic ReadFile ReadFile "kernel32.dll"
//go:cgo_import_dynamic Sleep Sleep "kernel32.dll"
//go:cgo_import_dynamic TerminateProcess TerminateProcess "kernel32.dll"
//go:cgo_import_dynamic VirtualAlloc VirtualAlloc "kernel32.dll"
//go:cgo_import_dynamic VirtualFree VirtualFree "kernel32.dll"
//go:cgo_import_dynamic VirtualProtect VirtualProtect "kernel32.dll"
//go:cgo_import_dynamic VirtualQuery VirtualQuery "kernel32.dll"
//go:cgo_import_dynamic WideCharToMultiByte WideCharToMultiByte "kernel32.dll"
//go:cgo_import_dynamic WriteFile WriteFile "kernel32.dll"
