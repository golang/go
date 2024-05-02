// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/runtime/atomic"
	"unsafe"
)

type mOS struct {
	waitsemacount uint32
	notesig       *int8
	errstr        *byte
	ignoreHangup  bool
}

func closefd(fd int32) int32

//go:noescape
func open(name *byte, mode, perm int32) int32

//go:noescape
func pread(fd int32, buf unsafe.Pointer, nbytes int32, offset int64) int32

//go:noescape
func pwrite(fd int32, buf unsafe.Pointer, nbytes int32, offset int64) int32

func seek(fd int32, offset int64, whence int32) int64

//go:noescape
func exits(msg *byte)

//go:noescape
func brk_(addr unsafe.Pointer) int32

func sleep(ms int32) int32

func rfork(flags int32) int32

//go:noescape
func plan9_semacquire(addr *uint32, block int32) int32

//go:noescape
func plan9_tsemacquire(addr *uint32, ms int32) int32

//go:noescape
func plan9_semrelease(addr *uint32, count int32) int32

//go:noescape
func notify(fn unsafe.Pointer) int32

func noted(mode int32) int32

//go:noescape
func nsec(*int64) int64

//go:noescape
func sigtramp(ureg, note unsafe.Pointer)

func setfpmasks()

//go:noescape
func tstart_plan9(newm *m)

func errstr() string

type _Plink uintptr

func sigpanic() {
	gp := getg()
	if !canpanic() {
		throw("unexpected signal during runtime execution")
	}

	note := gostringnocopy((*byte)(unsafe.Pointer(gp.m.notesig)))
	switch gp.sig {
	case _SIGRFAULT, _SIGWFAULT:
		i := indexNoFloat(note, "addr=")
		if i >= 0 {
			i += 5
		} else if i = indexNoFloat(note, "va="); i >= 0 {
			i += 3
		} else {
			panicmem()
		}
		addr := note[i:]
		gp.sigcode1 = uintptr(atolwhex(addr))
		if gp.sigcode1 < 0x1000 {
			panicmem()
		}
		if gp.paniconfault {
			panicmemAddr(gp.sigcode1)
		}
		if inUserArenaChunk(gp.sigcode1) {
			// We could check that the arena chunk is explicitly set to fault,
			// but the fact that we faulted on accessing it is enough to prove
			// that it is.
			print("accessed data from freed user arena ", hex(gp.sigcode1), "\n")
		} else {
			print("unexpected fault address ", hex(gp.sigcode1), "\n")
		}
		throw("fault")
	case _SIGTRAP:
		if gp.paniconfault {
			panicmem()
		}
		throw(note)
	case _SIGINTDIV:
		panicdivide()
	case _SIGFLOAT:
		panicfloat()
	default:
		panic(errorString(note))
	}
}

// indexNoFloat is bytealg.IndexString but safe to use in a note
// handler.
func indexNoFloat(s, t string) int {
	if len(t) == 0 {
		return 0
	}
	for i := 0; i < len(s); i++ {
		if s[i] == t[0] && hasPrefix(s[i:], t) {
			return i
		}
	}
	return -1
}

func atolwhex(p string) int64 {
	for hasPrefix(p, " ") || hasPrefix(p, "\t") {
		p = p[1:]
	}
	neg := false
	if hasPrefix(p, "-") || hasPrefix(p, "+") {
		neg = p[0] == '-'
		p = p[1:]
		for hasPrefix(p, " ") || hasPrefix(p, "\t") {
			p = p[1:]
		}
	}
	var n int64
	switch {
	case hasPrefix(p, "0x"), hasPrefix(p, "0X"):
		p = p[2:]
		for ; len(p) > 0; p = p[1:] {
			if '0' <= p[0] && p[0] <= '9' {
				n = n*16 + int64(p[0]-'0')
			} else if 'a' <= p[0] && p[0] <= 'f' {
				n = n*16 + int64(p[0]-'a'+10)
			} else if 'A' <= p[0] && p[0] <= 'F' {
				n = n*16 + int64(p[0]-'A'+10)
			} else {
				break
			}
		}
	case hasPrefix(p, "0"):
		for ; len(p) > 0 && '0' <= p[0] && p[0] <= '7'; p = p[1:] {
			n = n*8 + int64(p[0]-'0')
		}
	default:
		for ; len(p) > 0 && '0' <= p[0] && p[0] <= '9'; p = p[1:] {
			n = n*10 + int64(p[0]-'0')
		}
	}
	if neg {
		n = -n
	}
	return n
}

type sigset struct{}

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
	// Initialize stack and goroutine for note handling.
	mp.gsignal = malg(32 * 1024)
	mp.gsignal.m = mp
	mp.notesig = (*int8)(mallocgc(_ERRMAX, nil, true))
	// Initialize stack for handling strings from the
	// errstr system call, as used in package syscall.
	mp.errstr = (*byte)(mallocgc(_ERRMAX, nil, true))
}

func sigsave(p *sigset) {
}

func msigrestore(sigmask sigset) {
}

//go:nosplit
//go:nowritebarrierrec
func clearSignalHandlers() {
}

func sigblock(exiting bool) {
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, cannot allocate memory.
func minit() {
	if atomic.Load(&exiting) != 0 {
		exits(&emptystatus[0])
	}
	// Mask all SSE floating-point exceptions
	// when running on the 64-bit kernel.
	setfpmasks()
}

// Called from dropm to undo the effect of an minit.
func unminit() {
}

// Called from exitm, but not from drop, to undo the effect of thread-owned
// resources in minit, semacreate, or elsewhere. Do not take locks after calling this.
func mdestroy(mp *m) {
}

var sysstat = []byte("/dev/sysstat\x00")

func getproccount() int32 {
	var buf [2048]byte
	fd := open(&sysstat[0], _OREAD, 0)
	if fd < 0 {
		return 1
	}
	ncpu := int32(0)
	for {
		n := read(fd, unsafe.Pointer(&buf), int32(len(buf)))
		if n <= 0 {
			break
		}
		for i := int32(0); i < n; i++ {
			if buf[i] == '\n' {
				ncpu++
			}
		}
	}
	closefd(fd)
	if ncpu == 0 {
		ncpu = 1
	}
	return ncpu
}

var devswap = []byte("/dev/swap\x00")
var pagesize = []byte(" pagesize\n")

func getPageSize() uintptr {
	var buf [2048]byte
	var pos int
	fd := open(&devswap[0], _OREAD, 0)
	if fd < 0 {
		// There's not much we can do if /dev/swap doesn't
		// exist. However, nothing in the memory manager uses
		// this on Plan 9, so it also doesn't really matter.
		return minPhysPageSize
	}
	for pos < len(buf) {
		n := read(fd, unsafe.Pointer(&buf[pos]), int32(len(buf)-pos))
		if n <= 0 {
			break
		}
		pos += int(n)
	}
	closefd(fd)
	text := buf[:pos]
	// Find "<n> pagesize" line.
	bol := 0
	for i, c := range text {
		if c == '\n' {
			bol = i + 1
		}
		if bytesHasPrefix(text[i:], pagesize) {
			// Parse number at the beginning of this line.
			return uintptr(_atoi(text[bol:]))
		}
	}
	// Again, the page size doesn't really matter, so use a fallback.
	return minPhysPageSize
}

func bytesHasPrefix(s, prefix []byte) bool {
	if len(s) < len(prefix) {
		return false
	}
	for i, p := range prefix {
		if s[i] != p {
			return false
		}
	}
	return true
}

var pid = []byte("#c/pid\x00")

func getpid() uint64 {
	var b [20]byte
	fd := open(&pid[0], 0, 0)
	if fd >= 0 {
		read(fd, unsafe.Pointer(&b), int32(len(b)))
		closefd(fd)
	}
	c := b[:]
	for c[0] == ' ' || c[0] == '\t' {
		c = c[1:]
	}
	return uint64(_atoi(c))
}

func osinit() {
	physPageSize = getPageSize()
	initBloc()
	ncpu = getproccount()
	getg().m.procid = getpid()
}

//go:nosplit
func crash() {
	notify(nil)
	*(*int)(nil) = 0
}

//go:nosplit
func readRandom(r []byte) int {
	return 0
}

func initsig(preinit bool) {
	if !preinit {
		notify(unsafe.Pointer(abi.FuncPCABI0(sigtramp)))
	}
}

//go:nosplit
func osyield() {
	sleep(0)
}

//go:nosplit
func osyield_no_g() {
	osyield()
}

//go:nosplit
func usleep(µs uint32) {
	ms := int32(µs / 1000)
	if ms == 0 {
		ms = 1
	}
	sleep(ms)
}

//go:nosplit
func usleep_no_g(usec uint32) {
	usleep(usec)
}

//go:nosplit
func nanotime1() int64 {
	var scratch int64
	ns := nsec(&scratch)
	// TODO(aram): remove hack after I fix _nsec in the pc64 kernel.
	if ns == 0 {
		return scratch
	}
	return ns
}

var goexits = []byte("go: exit ")
var emptystatus = []byte("\x00")
var exiting uint32

func goexitsall(status *byte) {
	var buf [_ERRMAX]byte
	if !atomic.Cas(&exiting, 0, 1) {
		return
	}
	getg().m.locks++
	n := copy(buf[:], goexits)
	n = copy(buf[n:], gostringnocopy(status))
	pid := getpid()
	for mp := (*m)(atomic.Loadp(unsafe.Pointer(&allm))); mp != nil; mp = mp.alllink {
		if mp.procid != 0 && mp.procid != pid {
			postnote(mp.procid, buf[:])
		}
	}
	getg().m.locks--
}

var procdir = []byte("/proc/")
var notefile = []byte("/note\x00")

func postnote(pid uint64, msg []byte) int {
	var buf [128]byte
	var tmp [32]byte
	n := copy(buf[:], procdir)
	n += copy(buf[n:], itoa(tmp[:], pid))
	copy(buf[n:], notefile)
	fd := open(&buf[0], _OWRITE, 0)
	if fd < 0 {
		return -1
	}
	len := findnull(&msg[0])
	if write1(uintptr(fd), unsafe.Pointer(&msg[0]), int32(len)) != int32(len) {
		closefd(fd)
		return -1
	}
	closefd(fd)
	return 0
}

//go:nosplit
func exit(e int32) {
	var status []byte
	if e == 0 {
		status = emptystatus
	} else {
		// build error string
		var tmp [32]byte
		sl := itoa(tmp[:len(tmp)-1], uint64(e))
		// Don't append, rely on the existing data being zero.
		status = sl[:len(sl)+1]
	}
	goexitsall(&status[0])
	exits(&status[0])
}

// May run with m.p==nil, so write barriers are not allowed.
//
//go:nowritebarrier
func newosproc(mp *m) {
	if false {
		print("newosproc mp=", mp, " ostk=", &mp, "\n")
	}
	pid := rfork(_RFPROC | _RFMEM | _RFNOWAIT)
	if pid < 0 {
		throw("newosproc: rfork failed")
	}
	if pid == 0 {
		tstart_plan9(mp)
	}
}

func exitThread(wait *atomic.Uint32) {
	// We should never reach exitThread on Plan 9 because we let
	// the OS clean up threads.
	throw("exitThread")
}

//go:nosplit
func semacreate(mp *m) {
}

//go:nosplit
func semasleep(ns int64) int {
	gp := getg()
	if ns >= 0 {
		ms := timediv(ns, 1000000, nil)
		if ms == 0 {
			ms = 1
		}
		ret := plan9_tsemacquire(&gp.m.waitsemacount, ms)
		if ret == 1 {
			return 0 // success
		}
		return -1 // timeout or interrupted
	}
	for plan9_semacquire(&gp.m.waitsemacount, 1) < 0 {
		// interrupted; try again (c.f. lock_sema.go)
	}
	return 0 // success
}

//go:nosplit
func semawakeup(mp *m) {
	plan9_semrelease(&mp.waitsemacount, 1)
}

//go:nosplit
func read(fd int32, buf unsafe.Pointer, n int32) int32 {
	return pread(fd, buf, n, -1)
}

//go:nosplit
func write1(fd uintptr, buf unsafe.Pointer, n int32) int32 {
	return pwrite(int32(fd), buf, n, -1)
}

var _badsignal = []byte("runtime: signal received on thread not created by Go.\n")

// This runs on a foreign stack, without an m or a g. No stack split.
//
//go:nosplit
func badsignal2() {
	pwrite(2, unsafe.Pointer(&_badsignal[0]), int32(len(_badsignal)), -1)
	exits(&_badsignal[0])
}

func raisebadsignal(sig uint32) {
	badsignal2()
}

func _atoi(b []byte) int {
	n := 0
	for len(b) > 0 && '0' <= b[0] && b[0] <= '9' {
		n = n*10 + int(b[0]) - '0'
		b = b[1:]
	}
	return n
}

func signame(sig uint32) string {
	if sig >= uint32(len(sigtable)) {
		return ""
	}
	return sigtable[sig].name
}

const preemptMSupported = false

func preemptM(mp *m) {
	// Not currently supported.
	//
	// TODO: Use a note like we use signals on POSIX OSes
}
