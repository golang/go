// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"unsafe"
)

//go:cgo_import_dynamic runtime._AddVectoredExceptionHandler AddVectoredExceptionHandler "kernel32.dll"
//go:cgo_import_dynamic runtime._CloseHandle CloseHandle "kernel32.dll"
//go:cgo_import_dynamic runtime._CreateEventA CreateEventA "kernel32.dll"
//go:cgo_import_dynamic runtime._CreateIoCompletionPort CreateIoCompletionPort "kernel32.dll"
//go:cgo_import_dynamic runtime._CreateThread CreateThread "kernel32.dll"
//go:cgo_import_dynamic runtime._CreateWaitableTimerA CreateWaitableTimerA "kernel32.dll"
//go:cgo_import_dynamic runtime._CryptAcquireContextW CryptAcquireContextW "advapi32.dll"
//go:cgo_import_dynamic runtime._CryptGenRandom CryptGenRandom "advapi32.dll"
//go:cgo_import_dynamic runtime._CryptReleaseContext CryptReleaseContext "advapi32.dll"
//go:cgo_import_dynamic runtime._DuplicateHandle DuplicateHandle "kernel32.dll"
//go:cgo_import_dynamic runtime._ExitProcess ExitProcess "kernel32.dll"
//go:cgo_import_dynamic runtime._FreeEnvironmentStringsW FreeEnvironmentStringsW "kernel32.dll"
//go:cgo_import_dynamic runtime._GetEnvironmentStringsW GetEnvironmentStringsW "kernel32.dll"
//go:cgo_import_dynamic runtime._GetProcAddress GetProcAddress "kernel32.dll"
//go:cgo_import_dynamic runtime._GetQueuedCompletionStatus GetQueuedCompletionStatus "kernel32.dll"
//go:cgo_import_dynamic runtime._GetStdHandle GetStdHandle "kernel32.dll"
//go:cgo_import_dynamic runtime._GetSystemInfo GetSystemInfo "kernel32.dll"
//go:cgo_import_dynamic runtime._GetThreadContext GetThreadContext "kernel32.dll"
//go:cgo_import_dynamic runtime._LoadLibraryW LoadLibraryW "kernel32.dll"
//go:cgo_import_dynamic runtime._LoadLibraryA LoadLibraryA "kernel32.dll"
//go:cgo_import_dynamic runtime._NtWaitForSingleObject NtWaitForSingleObject "ntdll.dll"
//go:cgo_import_dynamic runtime._ResumeThread ResumeThread "kernel32.dll"
//go:cgo_import_dynamic runtime._SetConsoleCtrlHandler SetConsoleCtrlHandler "kernel32.dll"
//go:cgo_import_dynamic runtime._SetErrorMode SetErrorMode "kernel32.dll"
//go:cgo_import_dynamic runtime._SetEvent SetEvent "kernel32.dll"
//go:cgo_import_dynamic runtime._SetProcessPriorityBoost SetProcessPriorityBoost "kernel32.dll"
//go:cgo_import_dynamic runtime._SetThreadPriority SetThreadPriority "kernel32.dll"
//go:cgo_import_dynamic runtime._SetUnhandledExceptionFilter SetUnhandledExceptionFilter "kernel32.dll"
//go:cgo_import_dynamic runtime._SetWaitableTimer SetWaitableTimer "kernel32.dll"
//go:cgo_import_dynamic runtime._Sleep Sleep "kernel32.dll"
//go:cgo_import_dynamic runtime._SuspendThread SuspendThread "kernel32.dll"
//go:cgo_import_dynamic runtime._VirtualAlloc VirtualAlloc "kernel32.dll"
//go:cgo_import_dynamic runtime._VirtualFree VirtualFree "kernel32.dll"
//go:cgo_import_dynamic runtime._VirtualProtect VirtualProtect "kernel32.dll"
//go:cgo_import_dynamic runtime._WSAGetOverlappedResult WSAGetOverlappedResult "ws2_32.dll"
//go:cgo_import_dynamic runtime._WaitForSingleObject WaitForSingleObject "kernel32.dll"
//go:cgo_import_dynamic runtime._WriteFile WriteFile "kernel32.dll"
//go:cgo_import_dynamic runtime._timeBeginPeriod timeBeginPeriod "winmm.dll"

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
	_CryptAcquireContextW,
	_CryptGenRandom,
	_CryptReleaseContext,
	_DuplicateHandle,
	_ExitProcess,
	_FreeEnvironmentStringsW,
	_GetEnvironmentStringsW,
	_GetProcAddress,
	_GetQueuedCompletionStatus,
	_GetStdHandle,
	_GetSystemInfo,
	_GetThreadContext,
	_LoadLibraryW,
	_LoadLibraryA,
	_NtWaitForSingleObject,
	_ResumeThread,
	_SetConsoleCtrlHandler,
	_SetErrorMode,
	_SetEvent,
	_SetProcessPriorityBoost,
	_SetThreadPriority,
	_SetUnhandledExceptionFilter,
	_SetWaitableTimer,
	_Sleep,
	_SuspendThread,
	_VirtualAlloc,
	_VirtualFree,
	_VirtualProtect,
	_WSAGetOverlappedResult,
	_WaitForSingleObject,
	_WriteFile,
	_timeBeginPeriod stdFunction

	// Following syscalls are only available on some Windows PCs.
	// We will load syscalls, if available, before using them.
	_AddVectoredContinueHandler,
	_GetQueuedCompletionStatusEx stdFunction
)

func loadOptionalSyscalls() {
	var buf [50]byte // large enough for longest string
	strtoptr := func(s string) uintptr {
		buf[copy(buf[:], s)] = 0 // nil-terminated for OS
		return uintptr(noescape(unsafe.Pointer(&buf[0])))
	}
	l := stdcall1(_LoadLibraryA, strtoptr("kernel32.dll"))
	findfunc := func(name string) stdFunction {
		f := stdcall2(_GetProcAddress, l, strtoptr(name))
		return stdFunction(unsafe.Pointer(f))
	}
	if l != 0 {
		_AddVectoredContinueHandler = findfunc("AddVectoredContinueHandler")
		_GetQueuedCompletionStatusEx = findfunc("GetQueuedCompletionStatusEx")
	}
}

// in sys_windows_386.s and sys_windows_amd64.s
func externalthreadhandler()
func exceptiontramp()
func firstcontinuetramp()
func lastcontinuetramp()

//go:nosplit
func getLoadLibrary() uintptr {
	return uintptr(unsafe.Pointer(_LoadLibraryW))
}

//go:nosplit
func getGetProcAddress() uintptr {
	return uintptr(unsafe.Pointer(_GetProcAddress))
}

func getproccount() int32 {
	var info systeminfo
	stdcall1(_GetSystemInfo, uintptr(unsafe.Pointer(&info)))
	return int32(info.dwnumberofprocessors)
}

const (
	currentProcess = ^uintptr(0) // -1 = current process
	currentThread  = ^uintptr(1) // -2 = current thread
)

func disableWER() {
	// do not display Windows Error Reporting dialogue
	const (
		SEM_FAILCRITICALERRORS     = 0x0001
		SEM_NOGPFAULTERRORBOX      = 0x0002
		SEM_NOALIGNMENTFAULTEXCEPT = 0x0004
		SEM_NOOPENFILEERRORBOX     = 0x8000
	)
	errormode := uint32(stdcall1(_SetErrorMode, SEM_NOGPFAULTERRORBOX))
	stdcall1(_SetErrorMode, uintptr(errormode)|SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX)
}

func osinit() {
	setBadSignalMsg()

	loadOptionalSyscalls()

	disableWER()

	externalthreadhandlerp = funcPC(externalthreadhandler)

	stdcall2(_AddVectoredExceptionHandler, 1, funcPC(exceptiontramp))
	if _AddVectoredContinueHandler == nil || unsafe.Sizeof(&_AddVectoredContinueHandler) == 4 {
		// use SetUnhandledExceptionFilter for windows-386 or
		// if VectoredContinueHandler is unavailable.
		// note: SetUnhandledExceptionFilter handler won't be called, if debugging.
		stdcall1(_SetUnhandledExceptionFilter, funcPC(lastcontinuetramp))
	} else {
		stdcall2(_AddVectoredContinueHandler, 1, funcPC(firstcontinuetramp))
		stdcall2(_AddVectoredContinueHandler, 0, funcPC(lastcontinuetramp))
	}

	stdcall2(_SetConsoleCtrlHandler, funcPC(ctrlhandler), 1)

	stdcall1(_timeBeginPeriod, 1)

	ncpu = getproccount()

	// Windows dynamic priority boosting assumes that a process has different types
	// of dedicated threads -- GUI, IO, computational, etc. Go processes use
	// equivalent threads that all do a mix of GUI, IO, computations, etc.
	// In such context dynamic priority boosting does nothing but harm, so we turn it off.
	stdcall2(_SetProcessPriorityBoost, currentProcess, 1)
}

//go:nosplit
func getRandomData(r []byte) {
	const (
		prov_rsa_full       = 1
		crypt_verifycontext = 0xF0000000
	)
	var handle uintptr
	n := 0
	if stdcall5(_CryptAcquireContextW, uintptr(unsafe.Pointer(&handle)), 0, 0, prov_rsa_full, crypt_verifycontext) != 0 {
		if stdcall3(_CryptGenRandom, handle, uintptr(len(r)), uintptr(unsafe.Pointer(&r[0]))) != 0 {
			n = len(r)
		}
		stdcall2(_CryptReleaseContext, handle, 0)
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
	envs = makeStringSlice(n)

	for i := range envs {
		envs[i] = gostringw(&p[0])
		for p[0] != 0 {
			p = p[1:]
		}
		p = p[1:] // skip nil byte
	}

	stdcall1(_FreeEnvironmentStringsW, uintptr(strings))
}

//go:nosplit
func exit(code int32) {
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
	var written uint32
	stdcall5(_WriteFile, handle, uintptr(buf), uintptr(n), uintptr(unsafe.Pointer(&written)), 0)
	return int32(written)
}

//go:nosplit
func semasleep(ns int64) int32 {
	// store ms in ns to save stack space
	if ns < 0 {
		ns = _INFINITE
	} else {
		ns = int64(timediv(ns, 1000000, nil))
		if ns == 0 {
			ns = 1
		}
	}
	if stdcall2(_WaitForSingleObject, getg().m.waitsema, uintptr(ns)) != 0 {
		return -1 // timeout
	}
	return 0
}

//go:nosplit
func semawakeup(mp *m) {
	stdcall1(_SetEvent, mp.waitsema)
}

//go:nosplit
func semacreate() uintptr {
	return stdcall4(_CreateEventA, 0, 0, 0, 0)
}

func newosproc(mp *m, stk unsafe.Pointer) {
	const _STACK_SIZE_PARAM_IS_A_RESERVATION = 0x00010000
	thandle := stdcall6(_CreateThread, 0, 0x20000,
		funcPC(tstart_stdcall), uintptr(unsafe.Pointer(mp)),
		_STACK_SIZE_PARAM_IS_A_RESERVATION, 0)
	if thandle == 0 {
		println("runtime: failed to create new OS thread (have ", mcount(), " already; errno=", getlasterror(), ")")
		throw("runtime.newosproc")
	}
}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
func minit() {
	var thandle uintptr
	stdcall7(_DuplicateHandle, currentProcess, currentThread, currentProcess, uintptr(unsafe.Pointer(&thandle)), 0, 0, _DUPLICATE_SAME_ACCESS)
	atomicstoreuintptr(&getg().m.thread, thandle)
}

// Called from dropm to undo the effect of an minit.
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
//go:nosplit
func stdcall(fn stdFunction) uintptr {
	gp := getg()
	mp := gp.m
	mp.libcall.fn = uintptr(unsafe.Pointer(fn))

	if mp.profilehz != 0 {
		// leave pc/sp for cpu profiler
		mp.libcallg = gp
		mp.libcallpc = getcallerpc(unsafe.Pointer(&fn))
		// sp must be the last, because once async cpu profiler finds
		// all three values to be non-zero, it will use them
		mp.libcallsp = getcallersp(unsafe.Pointer(&fn))
	}
	asmcgocall(unsafe.Pointer(funcPC(asmstdcall)), unsafe.Pointer(&mp.libcall))
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
func usleep1(usec uint32)

//go:nosplit
func osyield() {
	usleep1(1)
}

//go:nosplit
func usleep(us uint32) {
	// Have 1us units; want 100ns units.
	usleep1(10 * us)
}

func issigpanic(code uint32) uint32 {
	switch code {
	default:
		return 0
	case _EXCEPTION_ACCESS_VIOLATION:
	case _EXCEPTION_INT_DIVIDE_BY_ZERO:
	case _EXCEPTION_INT_OVERFLOW:
	case _EXCEPTION_FLT_DENORMAL_OPERAND:
	case _EXCEPTION_FLT_DIVIDE_BY_ZERO:
	case _EXCEPTION_FLT_INEXACT_RESULT:
	case _EXCEPTION_FLT_OVERFLOW:
	case _EXCEPTION_FLT_UNDERFLOW:
	case _EXCEPTION_BREAKPOINT:
	}
	return 1
}

func initsig() {
	/*
		// TODO(brainman): I don't think we need that bit of code
		// following line keeps these functions alive at link stage
		// if there's a better way please write it here
		void *e = runtime·exceptiontramp;
		void *f = runtime·firstcontinuetramp;
		void *l = runtime·lastcontinuetramp;
		USED(e);
		USED(f);
		USED(l);
	*/
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
	if mp == &m0 {
		tls = &tls0[0]
	}
	gp := *((**g)(unsafe.Pointer(tls)))

	// align Context to 16 bytes
	r = (*context)(unsafe.Pointer((uintptr(unsafe.Pointer(&rbuf[15]))) &^ 15))
	r.contextflags = _CONTEXT_CONTROL
	stdcall2(_GetThreadContext, mp.thread, uintptr(unsafe.Pointer(r)))
	sigprof((*byte)(unsafe.Pointer(r.ip())), (*byte)(unsafe.Pointer(r.sp())), nil, gp, mp)
}

func profileloop1() {
	stdcall2(_SetThreadPriority, currentThread, _THREAD_PRIORITY_HIGHEST)

	for {
		stdcall2(_WaitForSingleObject, profiletimer, _INFINITE)
		first := (*m)(atomicloadp(unsafe.Pointer(&allm)))
		for mp := first; mp != nil; mp = mp.alllink {
			thread := atomicloaduintptr(&mp.thread)
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
		atomicstoreuintptr(&profiletimer, timer)
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
	atomicstore((*uint32)(unsafe.Pointer(&getg().m.profilehz)), uint32(hz))
}

func memlimit() uintptr {
	return 0
}

var (
	badsignalmsg [100]byte
	badsignallen int32
)

func setBadSignalMsg() {
	const msg = "runtime: signal received on thread not created by Go.\n"
	for i, c := range msg {
		badsignalmsg[i] = byte(c)
		badsignallen++
	}
}

const (
	_SIGPROF = 0 // dummy value for badsignal
	_SIGQUIT = 0 // dummy value for sighandler
)

func raiseproc(sig int32) {
}

func crash() {
	// TODO: This routine should do whatever is needed
	// to make the Windows program abort/crash as it
	// would if Go was not intercepting signals.
	// On Unix the routine would remove the custom signal
	// handler and then raise a signal (like SIGABRT).
	// Something like that should happen here.
	// It's okay to leave this empty for now: if crash returns
	// the ordinary exit-after-panic happens.
}
