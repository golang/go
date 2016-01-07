// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// Keep a cached value to make gotraceback fast,
// since we call it on every call to gentraceback.
// The cached value is a uint32 in which the low bits
// are the "crash" and "all" settings and the remaining
// bits are the traceback value (0 off, 1 on, 2 include system).
const (
	tracebackCrash = 1 << iota
	tracebackAll
	tracebackShift = iota
)

var traceback_cache uint32 = 2 << tracebackShift
var traceback_env uint32

// gotraceback returns the current traceback settings.
//
// If level is 0, suppress all tracebacks.
// If level is 1, show tracebacks, but exclude runtime frames.
// If level is 2, show tracebacks including runtime frames.
// If all is set, print all goroutine stacks. Otherwise, print just the current goroutine.
// If crash is set, crash (core dump, etc) after tracebacking.
//
//go:nosplit
func gotraceback() (level int32, all, crash bool) {
	_g_ := getg()
	all = _g_.m.throwing > 0
	if _g_.m.traceback != 0 {
		level = int32(_g_.m.traceback)
		return
	}
	t := atomic.Load(&traceback_cache)
	crash = t&tracebackCrash != 0
	all = all || t&tracebackAll != 0
	level = int32(t >> tracebackShift)
	return
}

var (
	argc int32
	argv **byte
)

// nosplit for use in linux/386 startup linux_setup_vdso
//go:nosplit
func argv_index(argv **byte, i int32) *byte {
	return *(**byte)(add(unsafe.Pointer(argv), uintptr(i)*sys.PtrSize))
}

func args(c int32, v **byte) {
	argc = c
	argv = v
	sysargs(c, v)
}

func goargs() {
	if GOOS == "windows" {
		return
	}

	argslice = make([]string, argc)
	for i := int32(0); i < argc; i++ {
		argslice[i] = gostringnocopy(argv_index(argv, i))
	}
}

func goenvs_unix() {
	// TODO(austin): ppc64 in dynamic linking mode doesn't
	// guarantee env[] will immediately follow argv.  Might cause
	// problems.
	n := int32(0)
	for argv_index(argv, argc+1+n) != nil {
		n++
	}

	envs = make([]string, n)
	for i := int32(0); i < n; i++ {
		envs[i] = gostring(argv_index(argv, argc+1+i))
	}
}

func environ() []string {
	return envs
}

// TODO: These should be locals in testAtomic64, but we don't 8-byte
// align stack variables on 386.
var test_z64, test_x64 uint64

func testAtomic64() {
	test_z64 = 42
	test_x64 = 0
	prefetcht0(uintptr(unsafe.Pointer(&test_z64)))
	prefetcht1(uintptr(unsafe.Pointer(&test_z64)))
	prefetcht2(uintptr(unsafe.Pointer(&test_z64)))
	prefetchnta(uintptr(unsafe.Pointer(&test_z64)))
	if atomic.Cas64(&test_z64, test_x64, 1) {
		throw("cas64 failed")
	}
	if test_x64 != 0 {
		throw("cas64 failed")
	}
	test_x64 = 42
	if !atomic.Cas64(&test_z64, test_x64, 1) {
		throw("cas64 failed")
	}
	if test_x64 != 42 || test_z64 != 1 {
		throw("cas64 failed")
	}
	if atomic.Load64(&test_z64) != 1 {
		throw("load64 failed")
	}
	atomic.Store64(&test_z64, (1<<40)+1)
	if atomic.Load64(&test_z64) != (1<<40)+1 {
		throw("store64 failed")
	}
	if atomic.Xadd64(&test_z64, (1<<40)+1) != (2<<40)+2 {
		throw("xadd64 failed")
	}
	if atomic.Load64(&test_z64) != (2<<40)+2 {
		throw("xadd64 failed")
	}
	if atomic.Xchg64(&test_z64, (3<<40)+3) != (2<<40)+2 {
		throw("xchg64 failed")
	}
	if atomic.Load64(&test_z64) != (3<<40)+3 {
		throw("xchg64 failed")
	}
}

func check() {
	var (
		a     int8
		b     uint8
		c     int16
		d     uint16
		e     int32
		f     uint32
		g     int64
		h     uint64
		i, i1 float32
		j, j1 float64
		k, k1 unsafe.Pointer
		l     *uint16
		m     [4]byte
	)
	type x1t struct {
		x uint8
	}
	type y1t struct {
		x1 x1t
		y  uint8
	}
	var x1 x1t
	var y1 y1t

	if unsafe.Sizeof(a) != 1 {
		throw("bad a")
	}
	if unsafe.Sizeof(b) != 1 {
		throw("bad b")
	}
	if unsafe.Sizeof(c) != 2 {
		throw("bad c")
	}
	if unsafe.Sizeof(d) != 2 {
		throw("bad d")
	}
	if unsafe.Sizeof(e) != 4 {
		throw("bad e")
	}
	if unsafe.Sizeof(f) != 4 {
		throw("bad f")
	}
	if unsafe.Sizeof(g) != 8 {
		throw("bad g")
	}
	if unsafe.Sizeof(h) != 8 {
		throw("bad h")
	}
	if unsafe.Sizeof(i) != 4 {
		throw("bad i")
	}
	if unsafe.Sizeof(j) != 8 {
		throw("bad j")
	}
	if unsafe.Sizeof(k) != sys.PtrSize {
		throw("bad k")
	}
	if unsafe.Sizeof(l) != sys.PtrSize {
		throw("bad l")
	}
	if unsafe.Sizeof(x1) != 1 {
		throw("bad unsafe.Sizeof x1")
	}
	if unsafe.Offsetof(y1.y) != 1 {
		throw("bad offsetof y1.y")
	}
	if unsafe.Sizeof(y1) != 2 {
		throw("bad unsafe.Sizeof y1")
	}

	if timediv(12345*1000000000+54321, 1000000000, &e) != 12345 || e != 54321 {
		throw("bad timediv")
	}

	var z uint32
	z = 1
	if !atomic.Cas(&z, 1, 2) {
		throw("cas1")
	}
	if z != 2 {
		throw("cas2")
	}

	z = 4
	if atomic.Cas(&z, 5, 6) {
		throw("cas3")
	}
	if z != 4 {
		throw("cas4")
	}

	z = 0xffffffff
	if !atomic.Cas(&z, 0xffffffff, 0xfffffffe) {
		throw("cas5")
	}
	if z != 0xfffffffe {
		throw("cas6")
	}

	k = unsafe.Pointer(uintptr(0xfedcb123))
	if sys.PtrSize == 8 {
		k = unsafe.Pointer(uintptr(unsafe.Pointer(k)) << 10)
	}
	if casp(&k, nil, nil) {
		throw("casp1")
	}
	k1 = add(k, 1)
	if !casp(&k, k, k1) {
		throw("casp2")
	}
	if k != k1 {
		throw("casp3")
	}

	m = [4]byte{1, 1, 1, 1}
	atomic.Or8(&m[1], 0xf0)
	if m[0] != 1 || m[1] != 0xf1 || m[2] != 1 || m[3] != 1 {
		throw("atomicor8")
	}

	*(*uint64)(unsafe.Pointer(&j)) = ^uint64(0)
	if j == j {
		throw("float64nan")
	}
	if !(j != j) {
		throw("float64nan1")
	}

	*(*uint64)(unsafe.Pointer(&j1)) = ^uint64(1)
	if j == j1 {
		throw("float64nan2")
	}
	if !(j != j1) {
		throw("float64nan3")
	}

	*(*uint32)(unsafe.Pointer(&i)) = ^uint32(0)
	if i == i {
		throw("float32nan")
	}
	if i == i {
		throw("float32nan1")
	}

	*(*uint32)(unsafe.Pointer(&i1)) = ^uint32(1)
	if i == i1 {
		throw("float32nan2")
	}
	if i == i1 {
		throw("float32nan3")
	}

	testAtomic64()

	if _FixedStack != round2(_FixedStack) {
		throw("FixedStack is not power-of-2")
	}

	if !checkASM() {
		throw("assembly checks failed")
	}
}

type dbgVar struct {
	name  string
	value *int32
}

// Holds variables parsed from GODEBUG env var,
// except for "memprofilerate" since there is an
// existing int var for that value, which may
// already have an initial value.
var debug struct {
	allocfreetrace    int32
	cgocheck          int32
	efence            int32
	gccheckmark       int32
	gcpacertrace      int32
	gcshrinkstackoff  int32
	gcstackbarrieroff int32
	gcstackbarrierall int32
	gcstoptheworld    int32
	gctrace           int32
	invalidptr        int32
	sbrk              int32
	scavenge          int32
	scheddetail       int32
	schedtrace        int32
	wbshadow          int32
}

var dbgvars = []dbgVar{
	{"allocfreetrace", &debug.allocfreetrace},
	{"cgocheck", &debug.cgocheck},
	{"efence", &debug.efence},
	{"gccheckmark", &debug.gccheckmark},
	{"gcpacertrace", &debug.gcpacertrace},
	{"gcshrinkstackoff", &debug.gcshrinkstackoff},
	{"gcstackbarrieroff", &debug.gcstackbarrieroff},
	{"gcstackbarrierall", &debug.gcstackbarrierall},
	{"gcstoptheworld", &debug.gcstoptheworld},
	{"gctrace", &debug.gctrace},
	{"invalidptr", &debug.invalidptr},
	{"sbrk", &debug.sbrk},
	{"scavenge", &debug.scavenge},
	{"scheddetail", &debug.scheddetail},
	{"schedtrace", &debug.schedtrace},
	{"wbshadow", &debug.wbshadow},
}

func parsedebugvars() {
	// defaults
	debug.cgocheck = 1
	debug.invalidptr = 1

	for p := gogetenv("GODEBUG"); p != ""; {
		field := ""
		i := index(p, ",")
		if i < 0 {
			field, p = p, ""
		} else {
			field, p = p[:i], p[i+1:]
		}
		i = index(field, "=")
		if i < 0 {
			continue
		}
		key, value := field[:i], field[i+1:]

		// Update MemProfileRate directly here since it
		// is int, not int32, and should only be updated
		// if specified in GODEBUG.
		if key == "memprofilerate" {
			MemProfileRate = atoi(value)
		} else {
			for _, v := range dbgvars {
				if v.name == key {
					*v.value = int32(atoi(value))
				}
			}
		}
	}

	setTraceback(gogetenv("GOTRACEBACK"))
	traceback_env = traceback_cache

	if debug.gcstackbarrierall > 0 {
		firstStackBarrierOffset = 0
	}

	// For cgocheck > 1, we turn on the write barrier at all times
	// and check all pointer writes.
	if debug.cgocheck > 1 {
		writeBarrier.cgo = true
		writeBarrier.enabled = true
	}
}

//go:linkname setTraceback runtime/debug.SetTraceback
func setTraceback(level string) {
	var t uint32
	switch level {
	case "none":
		t = 0
	case "single", "":
		t = 1 << tracebackShift
	case "all":
		t = 1<<tracebackShift | tracebackAll
	case "system":
		t = 2<<tracebackShift | tracebackAll
	case "crash":
		t = 2<<tracebackShift | tracebackAll | tracebackCrash
	default:
		t = uint32(atoi(level))<<tracebackShift | tracebackAll
	}
	// when C owns the process, simply exit'ing the process on fatal errors
	// and panics is surprising. Be louder and abort instead.
	if islibrary || isarchive {
		t |= tracebackCrash
	}

	t |= traceback_env

	atomic.Store(&traceback_cache, t)
}

// Poor mans 64-bit division.
// This is a very special function, do not use it if you are not sure what you are doing.
// int64 division is lowered into _divv() call on 386, which does not fit into nosplit functions.
// Handles overflow in a time-specific manner.
//go:nosplit
func timediv(v int64, div int32, rem *int32) int32 {
	res := int32(0)
	for bit := 30; bit >= 0; bit-- {
		if v >= int64(div)<<uint(bit) {
			v = v - (int64(div) << uint(bit))
			res += 1 << uint(bit)
		}
	}
	if v >= int64(div) {
		if rem != nil {
			*rem = 0
		}
		return 0x7fffffff
	}
	if rem != nil {
		*rem = int32(v)
	}
	return res
}

// Helpers for Go. Must be NOSPLIT, must only call NOSPLIT functions, and must not block.

//go:nosplit
func acquirem() *m {
	_g_ := getg()
	_g_.m.locks++
	return _g_.m
}

//go:nosplit
func releasem(mp *m) {
	_g_ := getg()
	mp.locks--
	if mp.locks == 0 && _g_.preempt {
		// restore the preemption request in case we've cleared it in newstack
		_g_.stackguard0 = stackPreempt
	}
}

//go:nosplit
func gomcache() *mcache {
	return getg().m.mcache
}

//go:linkname reflect_typelinks reflect.typelinks
func reflect_typelinks() [][]*_type {
	ret := [][]*_type{firstmoduledata.typelinks}
	for datap := firstmoduledata.next; datap != nil; datap = datap.next {
		ret = append(ret, datap.typelinks)
	}
	return ret
}
