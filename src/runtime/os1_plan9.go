// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
func mpreinit(mp *m) {
	// Initialize stack and goroutine for note handling.
	mp.gsignal = malg(32 * 1024)
	mp.gsignal.m = mp
	mp.notesig = (*int8)(mallocgc(_ERRMAX, nil, _FlagNoScan))
	// Initialize stack for handling strings from the
	// errstr system call, as used in package syscall.
	mp.errstr = (*byte)(mallocgc(_ERRMAX, nil, _FlagNoScan))
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
func minit() {
	// Mask all SSE floating-point exceptions
	// when running on the 64-bit kernel.
	setfpmasks()
}

// Called from dropm to undo the effect of an minit.
func unminit() {
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
	close(fd)
	if ncpu == 0 {
		ncpu = 1
	}
	return ncpu
}

var pid = []byte("#c/pid\x00")

func getpid() uint64 {
	var b [20]byte
	fd := open(&pid[0], 0, 0)
	if fd >= 0 {
		read(fd, unsafe.Pointer(&b), int32(len(b)))
		close(fd)
	}
	c := b[:]
	for c[0] == ' ' || c[0] == '\t' {
		c = c[1:]
	}
	return uint64(atoi(c))
}

func osinit() {
	initBloc()
	ncpu = getproccount()
	getg().m.procid = getpid()
	notify(unsafe.Pointer(funcPC(sigtramp)))
}

func crash() {
	notify(nil)
	*(*int)(nil) = 0
}

var random_dev = []byte("/dev/random\x00")

//go:nosplit
func getRandomData(r []byte) {
	fd := open(&random_dev[0], 0 /* O_RDONLY */, 0)
	n := read(fd, unsafe.Pointer(&r[0]), int32(len(r)))
	close(fd)
	extendRandom(r, int(n))
}

func goenvs() {
}

func initsig() {
}

//go:nosplit
func osyield() {
	sleep(0)
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
func nanotime() int64 {
	var scratch int64
	ns := nsec(&scratch)
	// TODO(aram): remove hack after I fix _nsec in the pc64 kernel.
	if ns == 0 {
		return scratch
	}
	return ns
}

//go:nosplit
func itoa(buf []byte, val uint64) []byte {
	i := len(buf) - 1
	for val >= 10 {
		buf[i] = byte(val%10 + '0')
		i--
		val /= 10
	}
	buf[i] = byte(val + '0')
	return buf[i:]
}

var goexits = []byte("go: exit ")

func goexitsall(status *byte) {
	var buf [_ERRMAX]byte
	n := copy(buf[:], goexits)
	n = copy(buf[n:], gostringnocopy(status))
	pid := getpid()
	for mp := (*m)(atomicloadp(unsafe.Pointer(&allm))); mp != nil; mp = mp.alllink {
		if mp.procid != pid {
			postnote(mp.procid, buf[:])
		}
	}
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
	if write(uintptr(fd), (unsafe.Pointer)(&msg[0]), int32(len)) != int64(len) {
		close(fd)
		return -1
	}
	close(fd)
	return 0
}

//go:nosplit
func exit(e int) {
	var status []byte
	if e == 0 {
		status = []byte("\x00")
	} else {
		// build error string
		var tmp [32]byte
		status = []byte(gostringnocopy(&itoa(tmp[:len(tmp)-1], uint64(e))[0]))
	}
	goexitsall(&status[0])
	exits(&status[0])
}

func newosproc(mp *m, stk unsafe.Pointer) {
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

//go:nosplit
func semacreate() uintptr {
	return 1
}

//go:nosplit
func semasleep(ns int64) int {
	_g_ := getg()
	if ns >= 0 {
		ms := timediv(ns, 1000000, nil)
		if ms == 0 {
			ms = 1
		}
		ret := plan9_tsemacquire(&_g_.m.waitsemacount, ms)
		if ret == 1 {
			return 0 // success
		}
		return -1 // timeout or interrupted
	}
	for plan9_semacquire(&_g_.m.waitsemacount, 1) < 0 {
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
func write(fd uintptr, buf unsafe.Pointer, n int32) int64 {
	return int64(pwrite(int32(fd), buf, n, -1))
}

func memlimit() uint64 {
	return 0
}

var _badsignal = []byte("runtime: signal received on thread not created by Go.\n")

// This runs on a foreign stack, without an m or a g.  No stack split.
//go:nosplit
func badsignal2() {
	pwrite(2, unsafe.Pointer(&_badsignal[0]), int32(len(_badsignal)), -1)
	exits(&_badsignal[0])
}

func atoi(b []byte) int {
	n := 0
	for len(b) > 0 && '0' <= b[0] && b[0] <= '9' {
		n = n*10 + int(b[0]) - '0'
		b = b[1:]
	}
	return n
}
