// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/runtime/atomic"
	"unsafe"
)

// The X versions of syscall expect the libc call to return a 64-bit result.
// Otherwise (the non-X version) expects a 32-bit result.
// This distinction is required because an error is indicated by returning -1,
// and we need to know whether to check 32 or 64 bits of the result.
// (Some libc functions that return 32 bits put junk in the upper 32 bits of AX.)

// golang.org/x/sys linknames syscall_syscall
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
//go:linkname syscall_syscall syscall.syscall
//go:nosplit
func syscall_syscall(fn, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	args := struct{ fn, a1, a2, a3, r1, r2, err uintptr }{fn, a1, a2, a3, r1, r2, err}
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall)), unsafe.Pointer(&args))
	exitsyscall()
	return args.r1, args.r2, args.err
}
func syscall()

//go:linkname syscall_syscallX syscall.syscallX
//go:nosplit
func syscall_syscallX(fn, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	args := struct{ fn, a1, a2, a3, r1, r2, err uintptr }{fn, a1, a2, a3, r1, r2, err}
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscallX)), unsafe.Pointer(&args))
	exitsyscall()
	return args.r1, args.r2, args.err
}
func syscallX()

// golang.org/x/sys linknames syscall.syscall6
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
// syscall.syscall6 is meant for package syscall (and x/sys),
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/tetratelabs/wazero
//
// See go.dev/issue/67401.
//
//go:linkname syscall_syscall6 syscall.syscall6
//go:nosplit
func syscall_syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	args := struct{ fn, a1, a2, a3, a4, a5, a6, r1, r2, err uintptr }{fn, a1, a2, a3, a4, a5, a6, r1, r2, err}
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall6)), unsafe.Pointer(&args))
	exitsyscall()
	return args.r1, args.r2, args.err
}
func syscall6()

// golang.org/x/sys linknames syscall.syscall9
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
//go:linkname syscall_syscall9 syscall.syscall9
//go:nosplit
//go:cgo_unsafe_args
func syscall_syscall9(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2, err uintptr) {
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall9)), unsafe.Pointer(&fn))
	exitsyscall()
	return
}
func syscall9()

//go:linkname syscall_syscall6X syscall.syscall6X
//go:nosplit
func syscall_syscall6X(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	args := struct{ fn, a1, a2, a3, a4, a5, a6, r1, r2, err uintptr }{fn, a1, a2, a3, a4, a5, a6, r1, r2, err}
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall6X)), unsafe.Pointer(&args))
	exitsyscall()
	return args.r1, args.r2, args.err
}
func syscall6X()

// golang.org/x/sys linknames syscall.syscallPtr
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
//go:linkname syscall_syscallPtr syscall.syscallPtr
//go:nosplit
func syscall_syscallPtr(fn, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	args := struct{ fn, a1, a2, a3, r1, r2, err uintptr }{fn, a1, a2, a3, r1, r2, err}
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscallPtr)), unsafe.Pointer(&args))
	exitsyscall()
	return args.r1, args.r2, args.err
}
func syscallPtr()

// golang.org/x/sys linknames syscall_rawSyscall
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
//go:linkname syscall_rawSyscall syscall.rawSyscall
//go:nosplit
func syscall_rawSyscall(fn, a1, a2, a3 uintptr) (r1, r2, err uintptr) {
	args := struct{ fn, a1, a2, a3, r1, r2, err uintptr }{fn, a1, a2, a3, r1, r2, err}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall)), unsafe.Pointer(&args))
	return args.r1, args.r2, args.err
}

// golang.org/x/sys linknames syscall_rawSyscall6
// (in addition to standard package syscall).
// Do not remove or change the type signature.
//
//go:linkname syscall_rawSyscall6 syscall.rawSyscall6
//go:nosplit
func syscall_rawSyscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr) {
	args := struct{ fn, a1, a2, a3, a4, a5, a6, r1, r2, err uintptr }{fn, a1, a2, a3, a4, a5, a6, r1, r2, err}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall6)), unsafe.Pointer(&args))
	return args.r1, args.r2, args.err
}

// crypto_x509_syscall is used in crypto/x509/internal/macos to call into Security.framework and CF.

//go:linkname crypto_x509_syscall crypto/x509/internal/macos.syscall
//go:nosplit
func crypto_x509_syscall(fn, a1, a2, a3, a4, a5 uintptr, f1 float64) (r1 uintptr) {
	args := struct {
		fn, a1, a2, a3, a4, a5 uintptr
		f1                     float64
		r1                     uintptr
	}{fn, a1, a2, a3, a4, a5, f1, r1}
	entersyscall()
	libcCall(unsafe.Pointer(abi.FuncPCABI0(syscall_x509)), unsafe.Pointer(&args))
	exitsyscall()
	return args.r1
}
func syscall_x509()

// The *_trampoline functions convert from the Go calling convention to the C calling convention
// and then call the underlying libc function.  They are defined in sys_darwin_$ARCH.s.

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_init(attr *pthreadattr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_init_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	return ret
}
func pthread_attr_init_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_getstacksize(attr *pthreadattr, size *uintptr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_getstacksize_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	KeepAlive(size)
	return ret
}
func pthread_attr_getstacksize_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_attr_setdetachstate(attr *pthreadattr, state int) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_attr_setdetachstate_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	return ret
}
func pthread_attr_setdetachstate_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_create(attr *pthreadattr, start uintptr, arg unsafe.Pointer) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_create_trampoline)), unsafe.Pointer(&attr))
	KeepAlive(attr)
	KeepAlive(arg) // Just for consistency. Arg of course needs to be kept alive for the start function.
	return ret
}
func pthread_create_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func raise(sig uint32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(raise_trampoline)), unsafe.Pointer(&sig))
}
func raise_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_self() (t pthread) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_self_trampoline)), unsafe.Pointer(&t))
	return
}
func pthread_self_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_kill(t pthread, sig uint32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_kill_trampoline)), unsafe.Pointer(&t))
	return
}
func pthread_kill_trampoline()

// osinit_hack is a clumsy hack to work around Apple libc bugs
// causing fork+exec to hang in the child process intermittently.
// See go.dev/issue/33565 and go.dev/issue/56784 for a few reports.
//
// The stacks obtained from the hung child processes are in
// libSystem_atfork_child, which is supposed to reinitialize various
// parts of the C library in the new process.
//
// One common stack dies in _notify_fork_child calling _notify_globals
// (inlined) calling _os_alloc_once, because _os_alloc_once detects that
// the once lock is held by the parent process and then calls
// _os_once_gate_corruption_abort. The allocation is setting up the
// globals for the notification subsystem. See the source code at [1].
// To work around this, we can allocate the globals earlier in the Go
// program's lifetime, before any execs are involved, by calling any
// notify routine that is exported, calls _notify_globals, and doesn't do
// anything too expensive otherwise. notify_is_valid_token(0) fits the bill.
//
// The other common stack dies in xpc_atfork_child calling
// _objc_msgSend_uncached which ends up in
// WAITING_FOR_ANOTHER_THREAD_TO_FINISH_CALLING_+initialize. Of course,
// whatever thread the child is waiting for is in the parent process and
// is not going to finish anything in the child process. There is no
// public source code for these routines, so it is unclear exactly what
// the problem is. An Apple engineer suggests using xpc_date_create_from_current,
// which empirically does fix the problem.
//
// So osinit_hack_trampoline (in sys_darwin_$GOARCH.s) calls
// notify_is_valid_token(0) and xpc_date_create_from_current(), which makes the
// fork+exec hangs stop happening. If Apple fixes the libc bug in
// some future version of macOS, then we can remove this awful code.
//
//go:nosplit
func osinit_hack() {
	if GOOS == "darwin" { // not ios
		libcCall(unsafe.Pointer(abi.FuncPCABI0(osinit_hack_trampoline)), nil)
	}
	return
}
func osinit_hack_trampoline()

// mmap is used to do low-level memory allocation via mmap. Don't allow stack
// splits, since this function (used by sysAlloc) is called in a lot of low-level
// parts of the runtime and callers often assume it won't acquire any locks.
//
//go:nosplit
func mmap(addr unsafe.Pointer, n uintptr, prot, flags, fd int32, off uint32) (unsafe.Pointer, int) {
	args := struct {
		addr            unsafe.Pointer
		n               uintptr
		prot, flags, fd int32
		off             uint32
		ret1            unsafe.Pointer
		ret2            int
	}{addr, n, prot, flags, fd, off, nil, 0}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(mmap_trampoline)), unsafe.Pointer(&args))
	return args.ret1, args.ret2
}
func mmap_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func munmap(addr unsafe.Pointer, n uintptr) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(munmap_trampoline)), unsafe.Pointer(&addr))
	KeepAlive(addr) // Just for consistency. Hopefully addr is not a Go address.
}
func munmap_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func madvise(addr unsafe.Pointer, n uintptr, flags int32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(madvise_trampoline)), unsafe.Pointer(&addr))
	KeepAlive(addr) // Just for consistency. Hopefully addr is not a Go address.
}
func madvise_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func mlock(addr unsafe.Pointer, n uintptr) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(mlock_trampoline)), unsafe.Pointer(&addr))
	KeepAlive(addr) // Just for consistency. Hopefully addr is not a Go address.
}
func mlock_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func read(fd int32, p unsafe.Pointer, n int32) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(read_trampoline)), unsafe.Pointer(&fd))
	KeepAlive(p)
	return ret
}
func read_trampoline()

func pipe() (r, w int32, errno int32) {
	var p [2]int32
	errno = libcCall(unsafe.Pointer(abi.FuncPCABI0(pipe_trampoline)), noescape(unsafe.Pointer(&p)))
	return p[0], p[1], errno
}
func pipe_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func closefd(fd int32) int32 {
	return libcCall(unsafe.Pointer(abi.FuncPCABI0(close_trampoline)), unsafe.Pointer(&fd))
}
func close_trampoline()

// This is exported via linkname to assembly in runtime/cgo.
//
//go:nosplit
//go:cgo_unsafe_args
//go:linkname exit
func exit(code int32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(exit_trampoline)), unsafe.Pointer(&code))
}
func exit_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func usleep(usec uint32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(usleep_trampoline)), unsafe.Pointer(&usec))
}
func usleep_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func usleep_no_g(usec uint32) {
	asmcgocall_no_g(unsafe.Pointer(abi.FuncPCABI0(usleep_trampoline)), unsafe.Pointer(&usec))
}

//go:nosplit
//go:cgo_unsafe_args
func write1(fd uintptr, p unsafe.Pointer, n int32) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(write_trampoline)), unsafe.Pointer(&fd))
	KeepAlive(p)
	return ret
}
func write_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func open(name *byte, mode, perm int32) (ret int32) {
	ret = libcCall(unsafe.Pointer(abi.FuncPCABI0(open_trampoline)), unsafe.Pointer(&name))
	KeepAlive(name)
	return
}
func open_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func nanotime1() int64 {
	var r struct {
		t            int64  // raw timer
		numer, denom uint32 // conversion factors. nanoseconds = t * numer / denom.
	}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(nanotime_trampoline)), unsafe.Pointer(&r))
	// Note: Apple seems unconcerned about overflow here. See
	// https://developer.apple.com/library/content/qa/qa1398/_index.html
	// Note also, numer == denom == 1 is common.
	t := r.t
	if r.numer != 1 {
		t *= int64(r.numer)
	}
	if r.denom != 1 {
		t /= int64(r.denom)
	}
	return t
}
func nanotime_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func walltime() (int64, int32) {
	var t timespec
	libcCall(unsafe.Pointer(abi.FuncPCABI0(walltime_trampoline)), unsafe.Pointer(&t))
	return t.tv_sec, int32(t.tv_nsec)
}
func walltime_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sigaction(sig uint32, new *usigactiont, old *usigactiont) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(sigaction_trampoline)), unsafe.Pointer(&sig))
	KeepAlive(new)
	KeepAlive(old)
}
func sigaction_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sigprocmask(how uint32, new *sigset, old *sigset) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(sigprocmask_trampoline)), unsafe.Pointer(&how))
	KeepAlive(new)
	KeepAlive(old)
}
func sigprocmask_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sigaltstack(new *stackt, old *stackt) {
	if new != nil && new.ss_flags&_SS_DISABLE != 0 && new.ss_size == 0 {
		// Despite the fact that Darwin's sigaltstack man page says it ignores the size
		// when SS_DISABLE is set, it doesn't. sigaltstack returns ENOMEM
		// if we don't give it a reasonable size.
		// ref: http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20140421/214296.html
		new.ss_size = 32768
	}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(sigaltstack_trampoline)), unsafe.Pointer(&new))
	KeepAlive(new)
	KeepAlive(old)
}
func sigaltstack_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func raiseproc(sig uint32) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(raiseproc_trampoline)), unsafe.Pointer(&sig))
}
func raiseproc_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func setitimer(mode int32, new, old *itimerval) {
	libcCall(unsafe.Pointer(abi.FuncPCABI0(setitimer_trampoline)), unsafe.Pointer(&mode))
	KeepAlive(new)
	KeepAlive(old)
}
func setitimer_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sysctl(mib *uint32, miblen uint32, oldp *byte, oldlenp *uintptr, newp *byte, newlen uintptr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(sysctl_trampoline)), unsafe.Pointer(&mib))
	KeepAlive(mib)
	KeepAlive(oldp)
	KeepAlive(oldlenp)
	KeepAlive(newp)
	return ret
}
func sysctl_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func sysctlbyname(name *byte, oldp *byte, oldlenp *uintptr, newp *byte, newlen uintptr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(sysctlbyname_trampoline)), unsafe.Pointer(&name))
	KeepAlive(name)
	KeepAlive(oldp)
	KeepAlive(oldlenp)
	KeepAlive(newp)
	return ret
}
func sysctlbyname_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func fcntl(fd, cmd, arg int32) (ret int32, errno int32) {
	args := struct {
		fd, cmd, arg int32
		ret, errno   int32
	}{fd, cmd, arg, 0, 0}
	libcCall(unsafe.Pointer(abi.FuncPCABI0(fcntl_trampoline)), unsafe.Pointer(&args))
	return args.ret, args.errno
}
func fcntl_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func kqueue() int32 {
	v := libcCall(unsafe.Pointer(abi.FuncPCABI0(kqueue_trampoline)), nil)
	return v
}
func kqueue_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func kevent(kq int32, ch *keventt, nch int32, ev *keventt, nev int32, ts *timespec) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(kevent_trampoline)), unsafe.Pointer(&kq))
	KeepAlive(ch)
	KeepAlive(ev)
	KeepAlive(ts)
	return ret
}
func kevent_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_mutex_init(m *pthreadmutex, attr *pthreadmutexattr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_mutex_init_trampoline)), unsafe.Pointer(&m))
	KeepAlive(m)
	KeepAlive(attr)
	return ret
}
func pthread_mutex_init_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_mutex_lock(m *pthreadmutex) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_mutex_lock_trampoline)), unsafe.Pointer(&m))
	KeepAlive(m)
	return ret
}
func pthread_mutex_lock_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_mutex_unlock(m *pthreadmutex) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_mutex_unlock_trampoline)), unsafe.Pointer(&m))
	KeepAlive(m)
	return ret
}
func pthread_mutex_unlock_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_cond_init(c *pthreadcond, attr *pthreadcondattr) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_cond_init_trampoline)), unsafe.Pointer(&c))
	KeepAlive(c)
	KeepAlive(attr)
	return ret
}
func pthread_cond_init_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_cond_wait(c *pthreadcond, m *pthreadmutex) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_cond_wait_trampoline)), unsafe.Pointer(&c))
	KeepAlive(c)
	KeepAlive(m)
	return ret
}
func pthread_cond_wait_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_cond_timedwait_relative_np(c *pthreadcond, m *pthreadmutex, t *timespec) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_cond_timedwait_relative_np_trampoline)), unsafe.Pointer(&c))
	KeepAlive(c)
	KeepAlive(m)
	KeepAlive(t)
	return ret
}
func pthread_cond_timedwait_relative_np_trampoline()

//go:nosplit
//go:cgo_unsafe_args
func pthread_cond_signal(c *pthreadcond) int32 {
	ret := libcCall(unsafe.Pointer(abi.FuncPCABI0(pthread_cond_signal_trampoline)), unsafe.Pointer(&c))
	KeepAlive(c)
	return ret
}
func pthread_cond_signal_trampoline()

// Not used on Darwin, but must be defined.
func exitThread(wait *atomic.Uint32) {
	throw("exitThread")
}

//go:nosplit
func setNonblock(fd int32) {
	flags, _ := fcntl(fd, _F_GETFL, 0)
	if flags != -1 {
		fcntl(fd, _F_SETFL, flags|_O_NONBLOCK)
	}
}

func issetugid() int32 {
	return libcCall(unsafe.Pointer(abi.FuncPCABI0(issetugid_trampoline)), nil)
}
func issetugid_trampoline()

// mach_vm_region is used to obtain virtual memory mappings for use by the
// profiling system and is only exported to runtime/pprof. It is restricted
// to obtaining mappings for the current process.
//
//go:linkname mach_vm_region runtime/pprof.mach_vm_region
func mach_vm_region(address, region_size *uint64, info unsafe.Pointer) int32 {
	// kern_return_t mach_vm_region(
	// 	vm_map_read_t target_task,
	// 	mach_vm_address_t *address,
	// 	mach_vm_size_t *size,
	// 	vm_region_flavor_t flavor,
	// 	vm_region_info_t info,
	// 	mach_msg_type_number_t *infoCnt,
	// 	mach_port_t *object_name);
	var count machMsgTypeNumber = _VM_REGION_BASIC_INFO_COUNT_64
	var object_name machPort
	args := struct {
		address     *uint64
		size        *uint64
		flavor      machVMRegionFlavour
		info        unsafe.Pointer
		count       *machMsgTypeNumber
		object_name *machPort
	}{
		address:     address,
		size:        region_size,
		flavor:      _VM_REGION_BASIC_INFO_64,
		info:        info,
		count:       &count,
		object_name: &object_name,
	}
	return libcCall(unsafe.Pointer(abi.FuncPCABI0(mach_vm_region_trampoline)), unsafe.Pointer(&args))
}
func mach_vm_region_trampoline()

//go:linkname proc_regionfilename runtime/pprof.proc_regionfilename
func proc_regionfilename(pid int, address uint64, buf *byte, buflen int64) int32 {
	args := struct {
		pid     int
		address uint64
		buf     *byte
		bufSize int64
	}{
		pid:     pid,
		address: address,
		buf:     buf,
		bufSize: buflen,
	}
	return libcCall(unsafe.Pointer(abi.FuncPCABI0(proc_regionfilename_trampoline)), unsafe.Pointer(&args))
}
func proc_regionfilename_trampoline()

// Tell the linker that the libc_* functions are to be found
// in a system library, with the libc_ prefix missing.

//go:cgo_import_dynamic libc_pthread_attr_init pthread_attr_init "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_attr_getstacksize pthread_attr_getstacksize "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_create pthread_create "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_self pthread_self "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_kill pthread_kill "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_exit _exit "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_raise raise "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_open open "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_close close "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_read read "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_write write "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pipe pipe "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_mmap mmap "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_munmap munmap "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_madvise madvise "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_mlock mlock "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_error __error "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_usleep usleep "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_proc_regionfilename proc_regionfilename "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_mach_task_self_ mach_task_self_ "/usr/lib/libSystem.B.dylib""
//go:cgo_import_dynamic libc_mach_vm_region mach_vm_region "/usr/lib/libSystem.B.dylib""
//go:cgo_import_dynamic libc_mach_timebase_info mach_timebase_info "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_mach_absolute_time mach_absolute_time "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_clock_gettime clock_gettime "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_sigaction sigaction "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_sigmask pthread_sigmask "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_sigaltstack sigaltstack "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_getpid getpid "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_kill kill "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_setitimer setitimer "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_sysctl sysctl "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_sysctlbyname sysctlbyname "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_fcntl fcntl "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_kqueue kqueue "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_kevent kevent "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_pthread_mutex_init pthread_mutex_init "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_mutex_lock pthread_mutex_lock "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_mutex_unlock pthread_mutex_unlock "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_cond_init pthread_cond_init "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_cond_wait pthread_cond_wait "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_cond_timedwait_relative_np pthread_cond_timedwait_relative_np "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_cond_signal pthread_cond_signal "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_notify_is_valid_token notify_is_valid_token "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_xpc_date_create_from_current xpc_date_create_from_current "/usr/lib/libSystem.B.dylib"

//go:cgo_import_dynamic libc_issetugid issetugid "/usr/lib/libSystem.B.dylib"
