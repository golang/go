// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//extern SigTabTT runtime·sigtab[];

var sigset_none = uint32(0)
var sigset_all = ^uint32(0)

func unimplemented(name string) {
	println(name, "not implemented")
	*(*int)(unsafe.Pointer(uintptr(1231))) = 1231
}

//go:nosplit
func semawakeup(mp *m) {
	mach_semrelease(uint32(mp.waitsema))
}

//go:nosplit
func semacreate() uintptr {
	var x uintptr
	systemstack(func() {
		x = uintptr(mach_semcreate())
	})
	return x
}

// BSD interface for threading.
func osinit() {
	// bsdthread_register delayed until end of goenvs so that we
	// can look at the environment first.

	// Use sysctl to fetch hw.ncpu.
	mib := [2]uint32{6, 3}
	out := uint32(0)
	nout := unsafe.Sizeof(out)
	ret := sysctl(&mib[0], 2, (*byte)(unsafe.Pointer(&out)), &nout, nil, 0)
	if ret >= 0 {
		ncpu = int32(out)
	}
}

var urandom_dev = []byte("/dev/urandom\x00")

//go:nosplit
func getRandomData(r []byte) {
	fd := open(&urandom_dev[0], 0 /* O_RDONLY */, 0)
	n := read(fd, unsafe.Pointer(&r[0]), int32(len(r)))
	close(fd)
	extendRandom(r, int(n))
}

func goenvs() {
	goenvs_unix()

	// Register our thread-creation callback (see sys_darwin_{amd64,386}.s)
	// but only if we're not using cgo.  If we are using cgo we need
	// to let the C pthread library install its own thread-creation callback.
	if !iscgo {
		if bsdthread_register() != 0 {
			if gogetenv("DYLD_INSERT_LIBRARIES") != "" {
				throw("runtime: bsdthread_register error (unset DYLD_INSERT_LIBRARIES)")
			}
			throw("runtime: bsdthread_register error")
		}
	}
}

func newosproc(mp *m, stk unsafe.Pointer) {
	mp.tls[0] = uintptr(mp.id) // so 386 asm can find it
	if false {
		print("newosproc stk=", stk, " m=", mp, " g=", mp.g0, " id=", mp.id, "/", int(mp.tls[0]), " ostk=", &mp, "\n")
	}

	var oset uint32
	sigprocmask(_SIG_SETMASK, &sigset_all, &oset)
	errno := bsdthread_create(stk, mp, mp.g0, funcPC(mstart))
	sigprocmask(_SIG_SETMASK, &oset, nil)

	if errno < 0 {
		print("runtime: failed to create new OS thread (have ", mcount(), " already; errno=", -errno, ")\n")
		throw("runtime.newosproc")
	}
}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
	mp.gsignal = malg(32 * 1024) // OS X wants >= 8K
	mp.gsignal.m = mp
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
func minit() {
	// Initialize signal handling.
	_g_ := getg()
	signalstack((*byte)(unsafe.Pointer(_g_.m.gsignal.stack.lo)), 32*1024)
	sigprocmask(_SIG_SETMASK, &sigset_none, nil)
}

// Called from dropm to undo the effect of an minit.
func unminit() {
	signalstack(nil, 0)
}

// Mach IPC, to get at semaphores
// Definitions are in /usr/include/mach on a Mac.

func macherror(r int32, fn string) {
	print("mach error ", fn, ": ", r, "\n")
	throw("mach error")
}

const _DebugMach = false

var zerondr machndr

func mach_msgh_bits(a, b uint32) uint32 {
	return a | b<<8
}

func mach_msg(h *machheader, op int32, send_size, rcv_size, rcv_name, timeout, notify uint32) int32 {
	// TODO: Loop on interrupt.
	return mach_msg_trap(unsafe.Pointer(h), op, send_size, rcv_size, rcv_name, timeout, notify)
}

// Mach RPC (MIG)
const (
	_MinMachMsg = 48
	_MachReply  = 100
)

type codemsg struct {
	h    machheader
	ndr  machndr
	code int32
}

func machcall(h *machheader, maxsize int32, rxsize int32) int32 {
	_g_ := getg()
	port := _g_.m.machport
	if port == 0 {
		port = mach_reply_port()
		_g_.m.machport = port
	}

	h.msgh_bits |= mach_msgh_bits(_MACH_MSG_TYPE_COPY_SEND, _MACH_MSG_TYPE_MAKE_SEND_ONCE)
	h.msgh_local_port = port
	h.msgh_reserved = 0
	id := h.msgh_id

	if _DebugMach {
		p := (*[10000]unsafe.Pointer)(unsafe.Pointer(h))
		print("send:\t")
		var i uint32
		for i = 0; i < h.msgh_size/uint32(unsafe.Sizeof(p[0])); i++ {
			print(" ", p[i])
			if i%8 == 7 {
				print("\n\t")
			}
		}
		if i%8 != 0 {
			print("\n")
		}
	}
	ret := mach_msg(h, _MACH_SEND_MSG|_MACH_RCV_MSG, h.msgh_size, uint32(maxsize), port, 0, 0)
	if ret != 0 {
		if _DebugMach {
			print("mach_msg error ", ret, "\n")
		}
		return ret
	}
	if _DebugMach {
		p := (*[10000]unsafe.Pointer)(unsafe.Pointer(h))
		var i uint32
		for i = 0; i < h.msgh_size/uint32(unsafe.Sizeof(p[0])); i++ {
			print(" ", p[i])
			if i%8 == 7 {
				print("\n\t")
			}
		}
		if i%8 != 0 {
			print("\n")
		}
	}
	if h.msgh_id != id+_MachReply {
		if _DebugMach {
			print("mach_msg _MachReply id mismatch ", h.msgh_id, " != ", id+_MachReply, "\n")
		}
		return -303 // MIG_REPLY_MISMATCH
	}
	// Look for a response giving the return value.
	// Any call can send this back with an error,
	// and some calls only have return values so they
	// send it back on success too.  I don't quite see how
	// you know it's one of these and not the full response
	// format, so just look if the message is right.
	c := (*codemsg)(unsafe.Pointer(h))
	if uintptr(h.msgh_size) == unsafe.Sizeof(*c) && h.msgh_bits&_MACH_MSGH_BITS_COMPLEX == 0 {
		if _DebugMach {
			print("mig result ", c.code, "\n")
		}
		return c.code
	}
	if h.msgh_size != uint32(rxsize) {
		if _DebugMach {
			print("mach_msg _MachReply size mismatch ", h.msgh_size, " != ", rxsize, "\n")
		}
		return -307 // MIG_ARRAY_TOO_LARGE
	}
	return 0
}

// Semaphores!

const (
	tmach_semcreate = 3418
	rmach_semcreate = tmach_semcreate + _MachReply

	tmach_semdestroy = 3419
	rmach_semdestroy = tmach_semdestroy + _MachReply

	_KERN_ABORTED             = 14
	_KERN_OPERATION_TIMED_OUT = 49
)

type tmach_semcreatemsg struct {
	h      machheader
	ndr    machndr
	policy int32
	value  int32
}

type rmach_semcreatemsg struct {
	h         machheader
	body      machbody
	semaphore machport
}

type tmach_semdestroymsg struct {
	h         machheader
	body      machbody
	semaphore machport
}

func mach_semcreate() uint32 {
	var m [256]uint8
	tx := (*tmach_semcreatemsg)(unsafe.Pointer(&m))
	rx := (*rmach_semcreatemsg)(unsafe.Pointer(&m))

	tx.h.msgh_bits = 0
	tx.h.msgh_size = uint32(unsafe.Sizeof(*tx))
	tx.h.msgh_remote_port = mach_task_self()
	tx.h.msgh_id = tmach_semcreate
	tx.ndr = zerondr

	tx.policy = 0 // 0 = SYNC_POLICY_FIFO
	tx.value = 0

	for {
		r := machcall(&tx.h, int32(unsafe.Sizeof(m)), int32(unsafe.Sizeof(*rx)))
		if r == 0 {
			break
		}
		if r == _KERN_ABORTED { // interrupted
			continue
		}
		macherror(r, "semaphore_create")
	}
	if rx.body.msgh_descriptor_count != 1 {
		unimplemented("mach_semcreate desc count")
	}
	return rx.semaphore.name
}

func mach_semdestroy(sem uint32) {
	var m [256]uint8
	tx := (*tmach_semdestroymsg)(unsafe.Pointer(&m))

	tx.h.msgh_bits = _MACH_MSGH_BITS_COMPLEX
	tx.h.msgh_size = uint32(unsafe.Sizeof(*tx))
	tx.h.msgh_remote_port = mach_task_self()
	tx.h.msgh_id = tmach_semdestroy
	tx.body.msgh_descriptor_count = 1
	tx.semaphore.name = sem
	tx.semaphore.disposition = _MACH_MSG_TYPE_MOVE_SEND
	tx.semaphore._type = 0

	for {
		r := machcall(&tx.h, int32(unsafe.Sizeof(m)), 0)
		if r == 0 {
			break
		}
		if r == _KERN_ABORTED { // interrupted
			continue
		}
		macherror(r, "semaphore_destroy")
	}
}

// The other calls have simple system call traps in sys_darwin_{amd64,386}.s

func mach_semaphore_wait(sema uint32) int32
func mach_semaphore_timedwait(sema, sec, nsec uint32) int32
func mach_semaphore_signal(sema uint32) int32
func mach_semaphore_signal_all(sema uint32) int32

func semasleep1(ns int64) int32 {
	_g_ := getg()

	if ns >= 0 {
		var nsecs int32
		secs := timediv(ns, 1000000000, &nsecs)
		r := mach_semaphore_timedwait(uint32(_g_.m.waitsema), uint32(secs), uint32(nsecs))
		if r == _KERN_ABORTED || r == _KERN_OPERATION_TIMED_OUT {
			return -1
		}
		if r != 0 {
			macherror(r, "semaphore_wait")
		}
		return 0
	}

	for {
		r := mach_semaphore_wait(uint32(_g_.m.waitsema))
		if r == 0 {
			break
		}
		if r == _KERN_ABORTED { // interrupted
			continue
		}
		macherror(r, "semaphore_wait")
	}
	return 0
}

//go:nosplit
func semasleep(ns int64) int32 {
	var r int32
	systemstack(func() {
		r = semasleep1(ns)
	})
	return r
}

//go:nosplit
func mach_semrelease(sem uint32) {
	for {
		r := mach_semaphore_signal(sem)
		if r == 0 {
			break
		}
		if r == _KERN_ABORTED { // interrupted
			continue
		}

		// mach_semrelease must be completely nosplit,
		// because it is called from Go code.
		// If we're going to die, start that process on the system stack
		// to avoid a Go stack split.
		systemstack(func() { macherror(r, "semaphore_signal") })
	}
}

//go:nosplit
func osyield() {
	usleep(1)
}

func memlimit() uintptr {
	// NOTE(rsc): Could use getrlimit here,
	// like on FreeBSD or Linux, but Darwin doesn't enforce
	// ulimit -v, so it's unclear why we'd try to stay within
	// the limit.
	return 0
}

func setsig(i int32, fn uintptr, restart bool) {
	var sa sigactiont
	memclr(unsafe.Pointer(&sa), unsafe.Sizeof(sa))
	sa.sa_flags = _SA_SIGINFO | _SA_ONSTACK
	if restart {
		sa.sa_flags |= _SA_RESTART
	}
	sa.sa_mask = ^uint32(0)
	sa.sa_tramp = unsafe.Pointer(funcPC(sigtramp)) // runtime·sigtramp's job is to call into real handler
	*(*uintptr)(unsafe.Pointer(&sa.__sigaction_u)) = fn
	sigaction(uint32(i), &sa, nil)
}

func setsigstack(i int32) {
	throw("setsigstack")
}

func getsig(i int32) uintptr {
	var sa sigactiont
	memclr(unsafe.Pointer(&sa), unsafe.Sizeof(sa))
	sigaction(uint32(i), nil, &sa)
	return *(*uintptr)(unsafe.Pointer(&sa.__sigaction_u))
}

func signalstack(p *byte, n int32) {
	var st stackt
	st.ss_sp = p
	st.ss_size = uintptr(n)
	st.ss_flags = 0
	if p == nil {
		st.ss_flags = _SS_DISABLE
	}
	sigaltstack(&st, nil)
}

func unblocksignals() {
	sigprocmask(_SIG_SETMASK, &sigset_none, nil)
}
