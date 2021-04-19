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
//go:cgo_import_dynamic runtime._VirtualAlloc VirtualAlloc%4 "kernel32.dll"
//go:cgo_import_dynamic runtime._VirtualFree VirtualFree%3 "kernel32.dll"
//go:cgo_import_dynamic runtime._WSAGetOverlappedResult WSAGetOverlappedResult%5 "ws2_32.dll"
//go:cgo_import_dynamic runtime._WaitForSingleObject WaitForSingleObject%2 "kernel32.dll"
//go:cgo_import_dynamic runtime._WriteConsoleW WriteConsoleW%5 "kernel32.dll"
//go:cgo_import_dynamic runtime._WriteFile WriteFile%5 "kernel32.dll"
//go:cgo_import_dynamic runtime._timeBeginPeriod timeBeginPeriod%1 "winmm.dll"

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
	_GetSystemInfo,
	_GetThreadContext,
	_LoadLibraryW,
	_LoadLibraryA,
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
	_VirtualAlloc,
	_VirtualFree,
	_WSAGetOverlappedResult,
	_WaitForSingleObject,
	_WriteConsoleW,
	_WriteFile,
	_timeBeginPeriod,
	_ stdFunction

	// Following syscalls are only available on some Windows PCs.
	// We will load syscalls, if available, before using them.
	_AddDllDirectory,
	_AddVectoredContinueHandler,
	_GetQueuedCompletionStatusEx,
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
)

// Function to be called by windows CreateThread
// to start new os thread.
func tstart_stdcall(newm *m) uint32

func ctrlhandler(_type uint32) uint32

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

func loadOptionalSyscalls() {
	var kernel32dll = []byte("kernel32.dll\000")
	k32 := stdcall1(_LoadLibraryA, uintptr(unsafe.Pointer(&kernel32dll[0])))
	if k32 == 0 {
		throw("kernel32.dll not found")
	}
	_AddDllDirectory = windowsFindfunc(k32, []byte("AddDllDirectory\000"))
	_AddVectoredContinueHandler = windowsFindfunc(k32, []byte("AddVectoredContinueHandler\000"))
	_GetQueuedCompletionStatusEx = windowsFindfunc(k32, []byte("GetQueuedCompletionStatusEx\000"))
	_LoadLibraryExW = windowsFindfunc(k32, []byte("LoadLibraryExW\000"))

	var advapi32dll = []byte("advapi32.dll\000")
	a32 := stdcall1(_LoadLibraryA, uintptr(unsafe.Pointer(&advapi32dll[0])))
	if a32 == 0 {
		throw("advapi32.dll not found")
	}
	_RtlGenRandom = windowsFindfunc(a32, []byte("SystemFunction036\000"))

	var ntdll = []byte("ntdll.dll\000")
	n32 := stdcall1(_LoadLibraryA, uintptr(unsafe.Pointer(&ntdll[0])))
	if n32 == 0 {
		throw("ntdll.dll not found")
	}
	_NtWaitForSingleObject = windowsFindfunc(n32, []byte("NtWaitForSingleObject\000"))
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

func osinit() {
	asmstdcallAddr = unsafe.Pointer(funcPC(asmstdcall))
	usleep2Addr = unsafe.Pointer(funcPC(usleep2))
	switchtothreadAddr = unsafe.Pointer(funcPC(switchtothread))

	setBadSignalMsg()

	loadOptionalSyscalls()

	useLoadLibraryEx = (_LoadLibraryExW != nil && _AddDllDirectory != nil)

	disableWER()

	externalthreadhandlerp = funcPC(externalthreadhandler)

	initExceptionHandler()

	stdcall2(_SetConsoleCtrlHandler, funcPC(ctrlhandler), 1)

	timeBeginPeriodRetValue = uint32(stdcall1(_timeBeginPeriod, 1))

	ncpu = getproccount()

	physPageSize = getPageSize()

	// Windows dynamic priority boosting assumes that a process has different types
	// of dedicated threads -- GUI, IO, computational, etc. Go processes use
	// equivalent threads that all do a mix of GUI, IO, computations, etc.
	// In such context dynamic priority boosting does nothing but harm, so we turn it off.
	stdcall2(_SetProcessPriorityBoost, currentProcess, 1)
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
func newosproc(mp *m, stk unsafe.Pointer) {
	const _STACK_SIZE_PARAM_IS_A_RESERVATION = 0x00010000
	thandle := stdcall6(_CreateThread, 0, 0x20000,
		funcPC(tstart_stdcall), uintptr(unsafe.Pointer(mp)),
		_STACK_SIZE_PARAM_IS_A_RESERVATION, 0)

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
}

// Used by the C library build mode. On Linux this function would allocate a
// stack, but that's not necessary for Windows. No stack guards are present
// and the GC has not been initialized, so write barriers will fail.
//go:nowritebarrierrec
//go:nosplit
func newosproc0(mp *m, stk unsafe.Pointer) {
	newosproc(mp, stk)
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
func sigblock() {
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, cannot allocate memory.
func minit() {
	var thandle uintptr
	stdcall7(_DuplicateHandle, currentProcess, currentThread, currentProcess, uintptr(unsafe.Pointer(&thandle)), 0, 0, _DUPLICATE_SAME_ACCESS)
	atomic.Storeuintptr(&getg().m.thread, thandle)
}

// Called from dropm to undo the effect of an minit.
//go:nosplit
func unminit() {
	tp := &getg().m.thread
	stdcall1(_CloseHandle, *tp)
	*tp = 0
}

// Described in http://www.dcl.hpi.uni-potsdam.de/research/WRK/2007/08/getting-os-information-the-kuser_shared_data-structure/
type _KSYSTEM_TIME struct {
	LowPart   uint32
	High1Time int32
	High2Time int32
}

const (
	_INTERRUPT_TIME = 0x7ffe0008
	_SYSTEM_TIME    = 0x7ffe0014
)

//go:nosplit
func systime(addr uintptr) int64 {
	timeaddr := (*_KSYSTEM_TIME)(unsafe.Pointer(addr))

	var t _KSYSTEM_TIME
	for i := 1; i < 10000; i++ {
		// these fields must be read in that order (see URL above)
		t.High1Time = timeaddr.High1Time
		t.LowPart = timeaddr.LowPart
		t.High2Time = timeaddr.High2Time
		if t.High1Time == t.High2Time {
			return int64(t.High1Time)<<32 | int64(t.LowPart)
		}
		if (i % 100) == 0 {
			osyield()
		}
	}
	systemstack(func() {
		throw("interrupt/system time is changing too fast")
	})
	return 0
}

//go:nosplit
func unixnano() int64 {
	return (systime(_SYSTEM_TIME) - 116444736000000000) * 100
}

//go:nosplit
func nanotime() int64 {
	return systime(_INTERRUPT_TIME) * 100
}

// Calling stdcall on os stack.
// May run during STW, so write barriers are not allowed.
//go:nowritebarrier
//go:nosplit
func stdcall(fn stdFunction) uintptr {
	gp := getg()
	mp := gp.m
	mp.libcall.fn = uintptr(unsafe.Pointer(fn))

	if mp.profilehz != 0 {
		// leave pc/sp for cpu profiler
		mp.libcallg.set(gp)
		mp.libcallpc = getcallerpc(unsafe.Pointer(&fn))
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = getcallersp(unsafe.Pointer(&fn))
	}
	asmcgocall(asmstdcallAddr, unsafe.Pointer(&mp.libcall))
	mp.libcallsp = 0
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

var profiletimer uintptr

func profilem(mp *m) {
	var r *context
	rbuf := make([]byte, unsafe.Sizeof(*r)+15)

	tls := &mp.tls[0]
	gp := *((**g)(unsafe.Pointer(tls)))

	// align Context to 16 bytes
	r = (*context)(unsafe.Pointer((uintptr(unsafe.Pointer(&rbuf[15]))) &^ 15))
	r.contextflags = _CONTEXT_CONTROL
	stdcall2(_GetThreadContext, mp.thread, uintptr(unsafe.Pointer(r)))
	sigprof(r.ip(), r.sp(), 0, gp, mp)
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
			stdcall1(_SuspendThread, thread)
			if mp.profilehz != 0 && !mp.blocked {
				profilem(mp)
			}
			stdcall1(_ResumeThread, thread)
		}
	}
}

var cpuprofilerlock mutex

func resetcpuprofiler(hz int32) {
	lock(&cpuprofilerlock)
	if profiletimer == 0 {
		timer := stdcall3(_CreateWaitableTimerA, 0, 0, 0)
		atomic.Storeuintptr(&profiletimer, timer)
		thread := stdcall6(_CreateThread, 0, 0, funcPC(profileloop), 0, 0, 0)
		stdcall2(_SetThreadPriority, thread, _THREAD_PRIORITY_HIGHEST)
		stdcall1(_CloseHandle, thread)
	}
	unlock(&cpuprofilerlock)

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

func memlimit() uintptr {
	return 0
}
