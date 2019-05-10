// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

// TODO(brainman): should not need those
const (
	_NSIG = 65
)

//go:cgo_import_dynamic runtime._AddVectoredExceptionHandler AddVectoredExceptionHandler%2 "kernel32.dll"
//go:cgo_import_dynamic runtime._CloseHandle CloseHandle%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._CreateEventA CreateEventA%4 "kernel32.dll"
//go:cgo_import_dynamic runtime._CreateIoCompletionPort CreateIoCompletionPort%4 "kernel32.dll"
//go:cgo_import_dynamic runtime._CreateThread CreateThread%6 "kernel32.dll"
//go:cgo_import_dynamic runtime._CreateWaitableTimerA CreateWaitableTimerA%3 "kernel32.dll"
//go:cgo_import_dynamic runtime._DuplicateHandle DuplicateHandle%7 "kernel32.dll"
//go:cgo_import_dynamic runtime._ExitProcess ExitProcess%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._FreeEnvironmentStringsW FreeEnvironmentStringsW%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._GetConsoleMode GetConsoleMode%2 "kernel32.dll"
//go:cgo_import_dynamic runtime._GetEnvironmentStringsW GetEnvironmentStringsW%0 "kernel32.dll"
//go:cgo_import_dynamic runtime._GetProcAddress GetProcAddress%2 "kernel32.dll"
//go:cgo_import_dynamic runtime._GetProcessAffinityMask GetProcessAffinityMask%3 "kernel32.dll"
//go:cgo_import_dynamic runtime._GetQueuedCompletionStatus GetQueuedCompletionStatus%5 "kernel32.dll"
//go:cgo_import_dynamic runtime._GetStdHandle GetStdHandle%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._GetSystemDirectoryA GetSystemDirectoryA%2 "kernel32.dll"
//go:cgo_import_dynamic runtime._GetSystemInfo GetSystemInfo%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._GetThreadContext GetThreadContext%2 "kernel32.dll"
//go:cgo_import_dynamic runtime._LoadLibraryW LoadLibraryW%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._LoadLibraryA LoadLibraryA%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._ResumeThread ResumeThread%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._SetConsoleCtrlHandler SetConsoleCtrlHandler%2 "kernel32.dll"
//go:cgo_import_dynamic runtime._SetErrorMode SetErrorMode%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._SetEvent SetEvent%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._SetProcessPriorityBoost SetProcessPriorityBoost%2 "kernel32.dll"
//go:cgo_import_dynamic runtime._SetThreadPriority SetThreadPriority%2 "kernel32.dll"
//go:cgo_import_dynamic runtime._SetUnhandledExceptionFilter SetUnhandledExceptionFilter%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._SetWaitableTimer SetWaitableTimer%6 "kernel32.dll"
//go:cgo_import_dynamic runtime._SuspendThread SuspendThread%1 "kernel32.dll"
//go:cgo_import_dynamic runtime._SwitchToThread SwitchToThread%0 "kernel32.dll"
//go:cgo_import_dynamic runtime._TlsAlloc TlsAlloc%0 "kernel32.dll"
//go:cgo_import_dynamic runtime._VirtualAlloc VirtualAlloc%4 "kernel32.dll"
//go:cgo_import_dynamic runtime._VirtualFree VirtualFree%3 "kernel32.dll"
//go:cgo_import_dynamic runtime._VirtualQuery VirtualQuery%3 "kernel32.dll"
//go:cgo_import_dynamic runtime._WaitForSingleObject WaitForSingleObject%2 "kernel32.dll"
//go:cgo_import_dynamic runtime._WriteConsoleW WriteConsoleW%5 "kernel32.dll"
//go:cgo_import_dynamic runtime._WriteFile WriteFile%5 "kernel32.dll"

type stdFunction unsafe.Pointer

var (
	// Following syscalls are available on every Windows PC.
	// All these variables are set by the Windows executable
	// loader before the Go program starts.
	_AddVectoredExceptionHandler,
	_CloseHandle,
	_CreateEventA,
	_CreateIoCompletionPort,
	_CreateThread,
	_CreateWaitableTimerA,
	_DuplicateHandle,
	_ExitProcess,
	_FreeEnvironmentStringsW,
	_GetConsoleMode,
	_GetEnvironmentStringsW,
	_GetProcAddress,
	_GetProcessAffinityMask,
	_GetQueuedCompletionStatus,
	_GetStdHandle,
	_GetSystemDirectoryA,
	_GetSystemInfo,
	_GetSystemTimeAsFileTime,
	_GetThreadContext,
	_LoadLibraryW,
	_LoadLibraryA,
	_QueryPerformanceCounter,
	_QueryPerformanceFrequency,
	_ResumeThread,
	_SetConsoleCtrlHandler,
	_SetErrorMode,
	_SetEvent,
	_SetProcessPriorityBoost,
	_SetThreadPriority,
	_SetUnhandledExceptionFilter,
	_SetWaitableTimer,
	_SuspendThread,
	_SwitchToThread,
	_TlsAlloc,
	_VirtualAlloc,
	_VirtualFree,
	_VirtualQuery,
	_WaitForSingleObject,
	_WriteConsoleW,
	_WriteFile,
	_ stdFunction

	// Following syscalls are only available on some Windows PCs.
	// We will load syscalls, if available, before using them.
	_AddDllDirectory,
	_AddVectoredContinueHandler,
	_GetQueuedCompletionStatusEx,
	_LoadLibraryExA,
	_LoadLibraryExW,
	_ stdFunction

	// Use RtlGenRandom to generate cryptographically random data.
	// This approach has been recommended by Microsoft (see issue
	// 15589 for details).
	// The RtlGenRandom is not listed in advapi32.dll, instead
	// RtlGenRandom function can be found by searching for SystemFunction036.
	// Also some versions of Mingw cannot link to SystemFunction036
	// when building executable as Cgo. So load SystemFunction036
	// manually during runtime startup.
	_RtlGenRandom stdFunction

	// Load ntdll.dll manually during startup, otherwise Mingw
	// links wrong printf function to cgo executable (see issue
	// 12030 for details).
	_NtWaitForSingleObject stdFunction

	// These are from non-kernel32.dll, so we prefer to LoadLibraryEx them.
	_timeBeginPeriod,
	_timeEndPeriod,
	_WSAGetOverlappedResult,
	_ stdFunction
)

// Function to be called by windows CreateThread
// to start new os thread.
func tstart_stdcall(newm *m)

// Called by OS using stdcall ABI.
func ctrlhandler()

type mOS struct {
	waitsema uintptr // semaphore for parking on locks
}

//go:linkname os_sigpipe os.sigpipe
func os_sigpipe() {
	throw("too many writes on closed pipe")
}

// Stubs so tests can link correctly. These should never be called.
func open(name *byte, mode, perm int32) int32 {
	throw("unimplemented")
	return -1
}
func closefd(fd int32) int32 {
	throw("unimplemented")
	return -1
}
func read(fd int32, p unsafe.Pointer, n int32) int32 {
	throw("unimplemented")
	return -1
}

type sigset struct{}

// Call a Windows function with stdcall conventions,
// and switch to os stack during the call.
func asmstdcall(fn unsafe.Pointer)

var asmstdcallAddr unsafe.Pointer

func windowsFindfunc(lib uintptr, name []byte) stdFunction {
	if name[len(name)-1] != 0 {
		throw("usage")
	}
	f := stdcall2(_GetProcAddress, lib, uintptr(unsafe.Pointer(&name[0])))
	return stdFunction(unsafe.Pointer(f))
}

var sysDirectory [521]byte
var sysDirectoryLen uintptr

func windowsLoadSystemLib(name []byte) uintptr {
	if useLoadLibraryEx {
		return stdcall3(_LoadLibraryExA, uintptr(unsafe.Pointer(&name[0])), 0, _LOAD_LIBRARY_SEARCH_SYSTEM32)
	} else {
		if sysDirectoryLen == 0 {
			l := stdcall2(_GetSystemDirectoryA, uintptr(unsafe.Pointer(&sysDirectory[0])), uintptr(len(sysDirectory)-1))
			if l == 0 || l > uintptr(len(sysDirectory)-1) {
				throw("Unable to determine system directory")
			}
			sysDirectory[l] = '\\'
			sysDirectoryLen = l + 1
		}
		absName := append(sysDirectory[:sysDirectoryLen], name...)
		return stdcall1(_LoadLibraryA, uintptr(unsafe.Pointer(&absName[0])))
	}
}

func loadOptionalSyscalls() {
	var kernel32dll = []byte("kernel32.dll\000")
	k32 := stdcall1(_LoadLibraryA, uintptr(unsafe.Pointer(&kernel32dll[0])))
	if k32 == 0 {
		throw("kernel32.dll not found")
	}
	_AddDllDirectory = windowsFindfunc(k32, []byte("AddDllDirectory\000"))
	_AddVectoredContinueHandler = windowsFindfunc(k32, []byte("AddVectoredContinueHandler\000"))
	_GetQueuedCompletionStatusEx = windowsFindfunc(k32, []byte("GetQueuedCompletionStatusEx\000"))
	_LoadLibraryExA = windowsFindfunc(k32, []byte("LoadLibraryExA\000"))
	_LoadLibraryExW = windowsFindfunc(k32, []byte("LoadLibraryExW\000"))
	useLoadLibraryEx = (_LoadLibraryExW != nil && _LoadLibraryExA != nil && _AddDllDirectory != nil)

	var advapi32dll = []byte("advapi32.dll\000")
	a32 := windowsLoadSystemLib(advapi32dll)
	if a32 == 0 {
		throw("advapi32.dll not found")
	}
	_RtlGenRandom = windowsFindfunc(a32, []byte("SystemFunction036\000"))

	var ntdll = []byte("ntdll.dll\000")
	n32 := windowsLoadSystemLib(ntdll)
	if n32 == 0 {
		throw("ntdll.dll not found")
	}
	_NtWaitForSingleObject = windowsFindfunc(n32, []byte("NtWaitForSingleObject\000"))

	if GOARCH == "arm" {
		_QueryPerformanceCounter = windowsFindfunc(k32, []byte("QueryPerformanceCounter\000"))
		if _QueryPerformanceCounter == nil {
			throw("could not find QPC syscalls")
		}
	}

	var winmmdll = []byte("winmm.dll\000")
	m32 := windowsLoadSystemLib(winmmdll)
	if m32 == 0 {
		throw("winmm.dll not found")
	}
	_timeBeginPeriod = windowsFindfunc(m32, []byte("timeBeginPeriod\000"))
	_timeEndPeriod = windowsFindfunc(m32, []byte("timeEndPeriod\000"))
	if _timeBeginPeriod == nil || _timeEndPeriod == nil {
		throw("timeBegin/EndPeriod not found")
	}

	var ws232dll = []byte("ws2_32.dll\000")
	ws232 := windowsLoadSystemLib(ws232dll)
	if ws232 == 0 {
		throw("ws2_32.dll not found")
	}
	_WSAGetOverlappedResult = windowsFindfunc(ws232, []byte("WSAGetOverlappedResult\000"))
	if _WSAGetOverlappedResult == nil {
		throw("WSAGetOverlappedResult not found")
	}

	if windowsFindfunc(n32, []byte("wine_get_version\000")) != nil {
		// running on Wine
		initWine(k32)
	}
}

//go:nosplit
func getLoadLibrary() uintptr {
	return uintptr(unsafe.Pointer(_LoadLibraryW))
}

//go:nosplit
func getLoadLibraryEx() uintptr {
	return uintptr(unsafe.Pointer(_LoadLibraryExW))
}

//go:nosplit
func getGetProcAddress() uintptr {
	return uintptr(unsafe.Pointer(_GetProcAddress))
}

func getproccount() int32 {
	var mask, sysmask uintptr
	ret := stdcall3(_GetProcessAffinityMask, currentProcess, uintptr(unsafe.Pointer(&mask)), uintptr(unsafe.Pointer(&sysmask)))
	if ret != 0 {
		n := 0
		maskbits := int(unsafe.Sizeof(mask) * 8)
		for i := 0; i < maskbits; i++ {
			if mask&(1<<uint(i)) != 0 {
				n++
			}
		}
		if n != 0 {
			return int32(n)
		}
	}
	// use GetSystemInfo if GetProcessAffinityMask fails
	var info systeminfo
	stdcall1(_GetSystemInfo, uintptr(unsafe.Pointer(&info)))
	return int32(info.dwnumberofprocessors)
}

func getPageSize() uintptr {
	var info systeminfo
	stdcall1(_GetSystemInfo, uintptr(unsafe.Pointer(&info)))
	return uintptr(info.dwpagesize)
}

const (
	currentProcess = ^uintptr(0) // -1 = current process
	currentThread  = ^uintptr(1) // -2 = current thread
)

// in sys_windows_386.s and sys_windows_amd64.s:
func externalthreadhandler()
func getlasterror() uint32
func setlasterror(err uint32)

// When loading DLLs, we prefer to use LoadLibraryEx with
// LOAD_LIBRARY_SEARCH_* flags, if available. LoadLibraryEx is not
// available on old Windows, though, and the LOAD_LIBRARY_SEARCH_*
// flags are not available on some versions of Windows without a
// security patch.
//
// https://msdn.microsoft.com/en-us/library/ms684179(v=vs.85).aspx says:
// "Windows 7, Windows Server 2008 R2, Windows Vista, and Windows
// Server 2008: The LOAD_LIBRARY_SEARCH_* flags are available on
// systems that have KB2533623 installed. To determine whether the
// flags are available, use GetProcAddress to get the address of the
// AddDllDirectory, RemoveDllDirectory, or SetDefaultDllDirectories
// function. If GetProcAddress succeeds, the LOAD_LIBRARY_SEARCH_*
// flags can be used with LoadLibraryEx."
var useLoadLibraryEx bool

var timeBeginPeriodRetValue uint32

// osRelaxMinNS indicates that sysmon shouldn't osRelax if the next
// timer is less than 60 ms from now. Since osRelaxing may reduce
// timer resolution to 15.6 ms, this keeps timer error under roughly 1
// part in 4.
const osRelaxMinNS = 60 * 1e6

// osRelax is called by the scheduler when transitioning to and from
// all Ps being idle.
//
// On Windows, it adjusts the system-wide timer resolution. Go needs a
// high resolution timer while running and there's little extra cost
// if we're already using the CPU, but if all Ps are idle there's no
// need to consume extra power to drive the high-res timer.
func osRelax(relax bool) uint32 {
	if relax {
		return uint32(stdcall1(_timeEndPeriod, 1))
	} else {
		return uint32(stdcall1(_timeBeginPeriod, 1))
	}
}

func osinit() {
	asmstdcallAddr = unsafe.Pointer(funcPC(asmstdcall))
	usleep2Addr = unsafe.Pointer(funcPC(usleep2))
	switchtothreadAddr = unsafe.Pointer(funcPC(switchtothread))

	setBadSignalMsg()

	loadOptionalSyscalls()

	disableWER()

	initExceptionHandler()

	stdcall2(_SetConsoleCtrlHandler, funcPC(ctrlhandler), 1)

	timeBeginPeriodRetValue = osRelax(false)

	ncpu = getproccount()

	physPageSize = getPageSize()

	// Windows dynamic priority boosting assumes that a process has different types
	// of dedicated threads -- GUI, IO, computational, etc. Go processes use
	// equivalent threads that all do a mix of GUI, IO, computations, etc.
	// In such context dynamic priority boosting does nothing but harm, so we turn it off.
	stdcall2(_SetProcessPriorityBoost, currentProcess, 1)
}

func nanotime() int64

// useQPCTime controls whether time.now and nanotime use QueryPerformanceCounter.
// This is only set to 1 when running under Wine.
var useQPCTime uint8

var qpcStartCounter int64
var qpcMultiplier int64

//go:nosplit
func nanotimeQPC() int64 {
	var counter int64 = 0
	stdcall1(_QueryPerformanceCounter, uintptr(unsafe.Pointer(&counter)))

	// returns number of nanoseconds
	return (counter - qpcStartCounter) * qpcMultiplier
}

//go:nosplit
func nowQPC() (sec int64, nsec int32, mono int64) {
	var ft int64
	stdcall1(_GetSystemTimeAsFileTime, uintptr(unsafe.Pointer(&ft)))

	t := (ft - 116444736000000000) * 100

	sec = t / 1000000000
	nsec = int32(t - sec*1000000000)

	mono = nanotimeQPC()
	return
}

func initWine(k32 uintptr) {
	_GetSystemTimeAsFileTime = windowsFindfunc(k32, []byte("GetSystemTimeAsFileTime\000"))
	if _GetSystemTimeAsFileTime == nil {
		throw("could not find GetSystemTimeAsFileTime() syscall")
	}

	_QueryPerformanceCounter = windowsFindfunc(k32, []byte("QueryPerformanceCounter\000"))
	_QueryPerformanceFrequency = windowsFindfunc(k32, []byte("QueryPerformanceFrequency\000"))
	if _QueryPerformanceCounter == nil || _QueryPerformanceFrequency == nil {
		throw("could not find QPC syscalls")
	}

	// We can not simply fallback to GetSystemTimeAsFileTime() syscall, since its time is not monotonic,
	// instead we use QueryPerformanceCounter family of syscalls to implement monotonic timer
	// https://msdn.microsoft.com/en-us/library/windows/desktop/dn553408(v=vs.85).aspx

	var tmp int64
	stdcall1(_QueryPerformanceFrequency, uintptr(unsafe.Pointer(&tmp)))
	if tmp == 0 {
		throw("QueryPerformanceFrequency syscall returned zero, running on unsupported hardware")
	}

	// This should not overflow, it is a number of ticks of the performance counter per second,
	// its resolution is at most 10 per usecond (on Wine, even smaller on real hardware), so it will be at most 10 millions here,
	// panic if overflows.
	if tmp > (1<<31 - 1) {
		throw("QueryPerformanceFrequency overflow 32 bit divider, check nosplit discussion to proceed")
	}
	qpcFrequency := int32(tmp)
	stdcall1(_QueryPerformanceCounter, uintptr(unsafe.Pointer(&qpcStartCounter)))

	// Since we are supposed to run this time calls only on Wine, it does not lose precision,
	// since Wine's timer is kind of emulated at 10 Mhz, so it will be a nice round multiplier of 100
	// but for general purpose system (like 3.3 Mhz timer on i7) it will not be very precise.
	// We have to do it this way (or similar), since multiplying QPC counter by 100 millions overflows
	// int64 and resulted time will always be invalid.
	qpcMultiplier = int64(timediv(1000000000, qpcFrequency, nil))

	useQPCTime = 1
}

//go:nosplit
func getRandomData(r []byte) {
	n := 0
	if stdcall2(_RtlGenRandom, uintptr(unsafe.Pointer(&r[0])), uintptr(len(r)))&0xff != 0 {
		n = len(r)
	}
	extendRandom(r, n)
}

func goenvs() {
	// strings is a pointer to environment variable pairs in the form:
	//     "envA=valA\x00envB=valB\x00\x00" (in UTF-16)
	// Two consecutive zero bytes end the list.
	strings := unsafe.Pointer(stdcall0(_GetEnvironmentStringsW))
	p := (*[1 << 24]uint16)(strings)[:]

	n := 0
	for from, i := 0, 0; true; i++ {
		if p[i] == 0 {
			// empty string marks the end
			if i == from {
				break
			}
			from = i + 1
			n++
		}
	}
	envs = make([]string, n)

	for i := range envs {
		envs[i] = gostringw(&p[0])
		for p[0] != 0 {
			p = p[1:]
		}
		p = p[1:] // skip nil byte
	}

	stdcall1(_FreeEnvironmentStringsW, uintptr(strings))
}

// exiting is set to non-zero when the process is exiting.
var exiting uint32

//go:nosplit
func exit(code int32) {
	atomic.Store(&exiting, 1)
	stdcall1(_ExitProcess, uintptr(code))
}

//go:nosplit
func write(fd uintptr, buf unsafe.Pointer, n int32) int32 {
	const (
		_STD_OUTPUT_HANDLE = ^uintptr(10) // -11
		_STD_ERROR_HANDLE  = ^uintptr(11) // -12
	)
	var handle uintptr
	switch fd {
	case 1:
		handle = stdcall1(_GetStdHandle, _STD_OUTPUT_HANDLE)
	case 2:
		handle = stdcall1(_GetStdHandle, _STD_ERROR_HANDLE)
	default:
		// assume fd is real windows handle.
		handle = fd
	}
	isASCII := true
	b := (*[1 << 30]byte)(buf)[:n]
	for _, x := range b {
		if x >= 0x80 {
			isASCII = false
			break
		}
	}

	if !isASCII {
		var m uint32
		isConsole := stdcall2(_GetConsoleMode, handle, uintptr(unsafe.Pointer(&m))) != 0
		// If this is a console output, various non-unicode code pages can be in use.
		// Use the dedicated WriteConsole call to ensure unicode is printed correctly.
		if isConsole {
			return int32(writeConsole(handle, buf, n))
		}
	}
	var written uint32
	stdcall5(_WriteFile, handle, uintptr(buf), uintptr(n), uintptr(unsafe.Pointer(&written)), 0)
	return int32(written)
}

var (
	utf16ConsoleBack     [1000]uint16
	utf16ConsoleBackLock mutex
)

// writeConsole writes bufLen bytes from buf to the console File.
// It returns the number of bytes written.
func writeConsole(handle uintptr, buf unsafe.Pointer, bufLen int32) int {
	const surr2 = (surrogateMin + surrogateMax + 1) / 2

	// Do not use defer for unlock. May cause issues when printing a panic.
	lock(&utf16ConsoleBackLock)

	b := (*[1 << 30]byte)(buf)[:bufLen]
	s := *(*string)(unsafe.Pointer(&b))

	utf16tmp := utf16ConsoleBack[:]

	total := len(s)
	w := 0
	for _, r := range s {
		if w >= len(utf16tmp)-2 {
			writeConsoleUTF16(handle, utf16tmp[:w])
			w = 0
		}
		if r < 0x10000 {
			utf16tmp[w] = uint16(r)
			w++
		} else {
			r -= 0x10000
			utf16tmp[w] = surrogateMin + uint16(r>>10)&0x3ff
			utf16tmp[w+1] = surr2 + uint16(r)&0x3ff
			w += 2
		}
	}
	writeConsoleUTF16(handle, utf16tmp[:w])
	unlock(&utf16ConsoleBackLock)
	return total
}

// writeConsoleUTF16 is the dedicated windows calls that correctly prints
// to the console regardless of the current code page. Input is utf-16 code points.
// The handle must be a console handle.
func writeConsoleUTF16(handle uintptr, b []uint16) {
	l := uint32(len(b))
	if l == 0 {
		return
	}
	var written uint32
	stdcall5(_WriteConsoleW,
		handle,
		uintptr(unsafe.Pointer(&b[0])),
		uintptr(l),
		uintptr(unsafe.Pointer(&written)),
		0,
	)
	return
}

//go:nosplit
func semasleep(ns int64) int32 {
	const (
		_WAIT_ABANDONED = 0x00000080
		_WAIT_OBJECT_0  = 0x00000000
		_WAIT_TIMEOUT   = 0x00000102
		_WAIT_FAILED    = 0xFFFFFFFF
	)

	// store ms in ns to save stack space
	if ns < 0 {
		ns = _INFINITE
	} else {
		ns = int64(timediv(ns, 1000000, nil))
		if ns == 0 {
			ns = 1
		}
	}

	result := stdcall2(_WaitForSingleObject, getg().m.waitsema, uintptr(ns))
	switch result {
	case _WAIT_OBJECT_0: //signaled
		return 0

	case _WAIT_TIMEOUT:
		return -1

	case _WAIT_ABANDONED:
		systemstack(func() {
			throw("runtime.semasleep wait_abandoned")
		})

	case _WAIT_FAILED:
		systemstack(func() {
			print("runtime: waitforsingleobject wait_failed; errno=", getlasterror(), "\n")
			throw("runtime.semasleep wait_failed")
		})

	default:
		systemstack(func() {
			print("runtime: waitforsingleobject unexpected; result=", result, "\n")
			throw("runtime.semasleep unexpected")
		})
	}

	return -1 // unreachable
}

//go:nosplit
func semawakeup(mp *m) {
	if stdcall1(_SetEvent, mp.waitsema) == 0 {
		systemstack(func() {
			print("runtime: setevent failed; errno=", getlasterror(), "\n")
			throw("runtime.semawakeup")
		})
	}
}

//go:nosplit
func semacreate(mp *m) {
	if mp.waitsema != 0 {
		return
	}
	mp.waitsema = stdcall4(_CreateEventA, 0, 0, 0, 0)
	if mp.waitsema == 0 {
		systemstack(func() {
			print("runtime: createevent failed; errno=", getlasterror(), "\n")
			throw("runtime.semacreate")
		})
	}
}

// May run with m.p==nil, so write barriers are not allowed. This
// function is called by newosproc0, so it is also required to
// operate without stack guards.
//go:nowritebarrierrec
//go:nosplit
func newosproc(mp *m) {
	// We pass 0 for the stack size to use the default for this binary.
	thandle := stdcall6(_CreateThread, 0, 0,
		funcPC(tstart_stdcall), uintptr(unsafe.Pointer(mp)),
		0, 0)

	if thandle == 0 {
		if atomic.Load(&exiting) != 0 {
			// CreateThread may fail if called
			// concurrently with ExitProcess. If this
			// happens, just freeze this thread and let
			// the process exit. See issue #18253.
			lock(&deadlock)
			lock(&deadlock)
		}
		print("runtime: failed to create new OS thread (have ", mcount(), " already; errno=", getlasterror(), ")\n")
		throw("runtime.newosproc")
	}

	// Close thandle to avoid leaking the thread object if it exits.
	stdcall1(_CloseHandle, thandle)
}

// Used by the C library build mode. On Linux this function would allocate a
// stack, but that's not necessary for Windows. No stack guards are present
// and the GC has not been initialized, so write barriers will fail.
//go:nowritebarrierrec
//go:nosplit
func newosproc0(mp *m, stk unsafe.Pointer) {
	// TODO: this is completely broken. The args passed to newosproc0 (in asm_amd64.s)
	// are stacksize and function, not *m and stack.
	// Check os_linux.go for an implemention that might actually work.
	throw("bad newosproc0")
}

func exitThread(wait *uint32) {
	// We should never reach exitThread on Windows because we let
	// the OS clean up threads.
	throw("exitThread")
}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
}

//go:nosplit
func msigsave(mp *m) {
}

//go:nosplit
func msigrestore(sigmask sigset) {
}

//go:nosplit
//go:nowritebarrierrec
func clearSignalHandlers() {
}

//go:nosplit
func sigblock() {
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, cannot allocate memory.
func minit() {
	var thandle uintptr
	stdcall7(_DuplicateHandle, currentProcess, currentThread, currentProcess, uintptr(unsafe.Pointer(&thandle)), 0, 0, _DUPLICATE_SAME_ACCESS)
	atomic.Storeuintptr(&getg().m.thread, thandle)

	// Query the true stack base from the OS. Currently we're
	// running on a small assumed stack.
	var mbi memoryBasicInformation
	res := stdcall3(_VirtualQuery, uintptr(unsafe.Pointer(&mbi)), uintptr(unsafe.Pointer(&mbi)), unsafe.Sizeof(mbi))
	if res == 0 {
		print("runtime: VirtualQuery failed; errno=", getlasterror(), "\n")
		throw("VirtualQuery for stack base failed")
	}
	// The system leaves an 8K PAGE_GUARD region at the bottom of
	// the stack (in theory VirtualQuery isn't supposed to include
	// that, but it does). Add an additional 8K of slop for
	// calling C functions that don't have stack checks and for
	// lastcontinuehandler. We shouldn't be anywhere near this
	// bound anyway.
	base := mbi.allocationBase + 16<<10
	// Sanity check the stack bounds.
	g0 := getg()
	if base > g0.stack.hi || g0.stack.hi-base > 64<<20 {
		print("runtime: g0 stack [", hex(base), ",", hex(g0.stack.hi), ")\n")
		throw("bad g0 stack")
	}
	g0.stack.lo = base
	g0.stackguard0 = g0.stack.lo + _StackGuard
	g0.stackguard1 = g0.stackguard0
	// Sanity check the SP.
	stackcheck()
}

// Called from dropm to undo the effect of an minit.
//go:nosplit
func unminit() {
	tp := &getg().m.thread
	stdcall1(_CloseHandle, *tp)
	*tp = 0
}

// Calling stdcall on os stack.
// May run during STW, so write barriers are not allowed.
//go:nowritebarrier
//go:nosplit
func stdcall(fn stdFunction) uintptr {
	gp := getg()
	mp := gp.m
	mp.libcall.fn = uintptr(unsafe.Pointer(fn))
	resetLibcall := false
	if mp.profilehz != 0 && mp.libcallsp == 0 {
		// leave pc/sp for cpu profiler
		mp.libcallg.set(gp)
		mp.libcallpc = getcallerpc()
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = getcallersp()
		resetLibcall = true // See comment in sys_darwin.go:libcCall
	}
	asmcgocall(asmstdcallAddr, unsafe.Pointer(&mp.libcall))
	if resetLibcall {
		mp.libcallsp = 0
	}
	return mp.libcall.r1
}

//go:nosplit
func stdcall0(fn stdFunction) uintptr {
	mp := getg().m
	mp.libcall.n = 0
	mp.libcall.args = uintptr(noescape(unsafe.Pointer(&fn))) // it's unused but must be non-nil, otherwise crashes
	return stdcall(fn)
}

//go:nosplit
func stdcall1(fn stdFunction, a0 uintptr) uintptr {
	mp := getg().m
	mp.libcall.n = 1
	mp.libcall.args = uintptr(noescape(unsafe.Pointer(&a0)))
	return stdcall(fn)
}

//go:nosplit
func stdcall2(fn stdFunction, a0, a1 uintptr) uintptr {
	mp := getg().m
	mp.libcall.n = 2
	mp.libcall.args = uintptr(noescape(unsafe.Pointer(&a0)))
	return stdcall(fn)
}

//go:nosplit
func stdcall3(fn stdFunction, a0, a1, a2 uintptr) uintptr {
	mp := getg().m
	mp.libcall.n = 3
	mp.libcall.args = uintptr(noescape(unsafe.Pointer(&a0)))
	return stdcall(fn)
}

//go:nosplit
func stdcall4(fn stdFunction, a0, a1, a2, a3 uintptr) uintptr {
	mp := getg().m
	mp.libcall.n = 4
	mp.libcall.args = uintptr(noescape(unsafe.Pointer(&a0)))
	return stdcall(fn)
}

//go:nosplit
func stdcall5(fn stdFunction, a0, a1, a2, a3, a4 uintptr) uintptr {
	mp := getg().m
	mp.libcall.n = 5
	mp.libcall.args = uintptr(noescape(unsafe.Pointer(&a0)))
	return stdcall(fn)
}

//go:nosplit
func stdcall6(fn stdFunction, a0, a1, a2, a3, a4, a5 uintptr) uintptr {
	mp := getg().m
	mp.libcall.n = 6
	mp.libcall.args = uintptr(noescape(unsafe.Pointer(&a0)))
	return stdcall(fn)
}

//go:nosplit
func stdcall7(fn stdFunction, a0, a1, a2, a3, a4, a5, a6 uintptr) uintptr {
	mp := getg().m
	mp.libcall.n = 7
	mp.libcall.args = uintptr(noescape(unsafe.Pointer(&a0)))
	return stdcall(fn)
}

// in sys_windows_386.s and sys_windows_amd64.s
func onosstack(fn unsafe.Pointer, arg uint32)
func usleep2(usec uint32)
func switchtothread()

var usleep2Addr unsafe.Pointer
var switchtothreadAddr unsafe.Pointer

//go:nosplit
func osyield() {
	onosstack(switchtothreadAddr, 0)
}

//go:nosplit
func usleep(us uint32) {
	// Have 1us units; want 100ns units.
	onosstack(usleep2Addr, 10*us)
}

func ctrlhandler1(_type uint32) uint32 {
	var s uint32

	switch _type {
	case _CTRL_C_EVENT, _CTRL_BREAK_EVENT:
		s = _SIGINT
	default:
		return 0
	}

	if sigsend(s) {
		return 1
	}
	exit(2) // SIGINT, SIGTERM, etc
	return 0
}

// in sys_windows_386.s and sys_windows_amd64.s
func profileloop()

// called from zcallback_windows_*.s to sys_windows_*.s
func callbackasm1()

var profiletimer uintptr

func profilem(mp *m, thread uintptr) {
	var r *context
	rbuf := make([]byte, unsafe.Sizeof(*r)+15)

	// align Context to 16 bytes
	r = (*context)(unsafe.Pointer((uintptr(unsafe.Pointer(&rbuf[15]))) &^ 15))
	r.contextflags = _CONTEXT_CONTROL
	stdcall2(_GetThreadContext, thread, uintptr(unsafe.Pointer(r)))

	var gp *g
	switch GOARCH {
	default:
		panic("unsupported architecture")
	case "arm":
		tls := &mp.tls[0]
		gp = **((***g)(unsafe.Pointer(tls)))
	case "386", "amd64":
		tls := &mp.tls[0]
		gp = *((**g)(unsafe.Pointer(tls)))
	}

	sigprof(r.ip(), r.sp(), r.lr(), gp, mp)
}

func profileloop1(param uintptr) uint32 {
	stdcall2(_SetThreadPriority, currentThread, _THREAD_PRIORITY_HIGHEST)

	for {
		stdcall2(_WaitForSingleObject, profiletimer, _INFINITE)
		first := (*m)(atomic.Loadp(unsafe.Pointer(&allm)))
		for mp := first; mp != nil; mp = mp.alllink {
			thread := atomic.Loaduintptr(&mp.thread)
			// Do not profile threads blocked on Notes,
			// this includes idle worker threads,
			// idle timer thread, idle heap scavenger, etc.
			if thread == 0 || mp.profilehz == 0 || mp.blocked {
				continue
			}
			// mp may exit between the load above and the
			// SuspendThread, so be careful.
			if int32(stdcall1(_SuspendThread, thread)) == -1 {
				// The thread no longer exists.
				continue
			}
			if mp.profilehz != 0 && !mp.blocked {
				// Pass the thread handle in case mp
				// was in the process of shutting down.
				profilem(mp, thread)
			}
			stdcall1(_ResumeThread, thread)
		}
	}
}

func setProcessCPUProfiler(hz int32) {
	if profiletimer == 0 {
		timer := stdcall3(_CreateWaitableTimerA, 0, 0, 0)
		atomic.Storeuintptr(&profiletimer, timer)
		thread := stdcall6(_CreateThread, 0, 0, funcPC(profileloop), 0, 0, 0)
		stdcall2(_SetThreadPriority, thread, _THREAD_PRIORITY_HIGHEST)
		stdcall1(_CloseHandle, thread)
	}
}

func setThreadCPUProfiler(hz int32) {
	ms := int32(0)
	due := ^int64(^uint64(1 << 63))
	if hz > 0 {
		ms = 1000 / hz
		if ms == 0 {
			ms = 1
		}
		due = int64(ms) * -10000
	}
	stdcall6(_SetWaitableTimer, profiletimer, uintptr(unsafe.Pointer(&due)), uintptr(ms), 0, 0, 0)
	atomic.Store((*uint32)(unsafe.Pointer(&getg().m.profilehz)), uint32(hz))
}
