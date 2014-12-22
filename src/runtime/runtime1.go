// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Keep a cached value to make gotraceback fast,
// since we call it on every call to gentraceback.
// The cached value is a uint32 in which the low bit
// is the "crash" setting and the top 31 bits are the
// gotraceback value.
var traceback_cache uint32 = 2 << 1

// The GOTRACEBACK environment variable controls the
// behavior of a Go program that is crashing and exiting.
//	GOTRACEBACK=0   suppress all tracebacks
//	GOTRACEBACK=1   default behavior - show tracebacks but exclude runtime frames
//	GOTRACEBACK=2   show tracebacks including runtime frames
//	GOTRACEBACK=crash   show tracebacks including runtime frames, then crash (core dump etc)
//go:nosplit
func gotraceback(crash *bool) int32 {
	_g_ := getg()
	if crash != nil {
		*crash = false
	}
	if _g_.m.traceback != 0 {
		return int32(_g_.m.traceback)
	}
	if crash != nil {
		*crash = traceback_cache&1 != 0
	}
	return int32(traceback_cache >> 1)
}

var (
	argc int32
	argv **byte
)

// nosplit for use in linux/386 startup linux_setup_vdso
//go:nosplit
func argv_index(argv **byte, i int32) *byte {
	return *(**byte)(add(unsafe.Pointer(argv), uintptr(i)*ptrSize))
}

func args(c int32, v **byte) {
	argc = c
	argv = v
	sysargs(c, v)
}

var (
	// TODO: Retire in favor of GOOS== checks.
	isplan9   int32
	issolaris int32
	iswindows int32
)

// Information about what cpu features are available.
// Set on startup in asm_{x86/amd64}.s.
var (
//cpuid_ecx uint32
//cpuid_edx uint32
)

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
	n := int32(0)
	for argv_index(argv, argc+1+n) != nil {
		n++
	}

	envs = make([]string, n)
	for i := int32(0); i < n; i++ {
		envs[i] = gostringnocopy(argv_index(argv, argc+1+i))
	}
}

func environ() []string {
	return envs
}

func testAtomic64() {
	var z64, x64 uint64

	z64 = 42
	x64 = 0
	prefetcht0(uintptr(unsafe.Pointer(&z64)))
	prefetcht1(uintptr(unsafe.Pointer(&z64)))
	prefetcht2(uintptr(unsafe.Pointer(&z64)))
	prefetchnta(uintptr(unsafe.Pointer(&z64)))
	if cas64(&z64, x64, 1) {
		gothrow("cas64 failed")
	}
	if x64 != 0 {
		gothrow("cas64 failed")
	}
	x64 = 42
	if !cas64(&z64, x64, 1) {
		gothrow("cas64 failed")
	}
	if x64 != 42 || z64 != 1 {
		gothrow("cas64 failed")
	}
	if atomicload64(&z64) != 1 {
		gothrow("load64 failed")
	}
	atomicstore64(&z64, (1<<40)+1)
	if atomicload64(&z64) != (1<<40)+1 {
		gothrow("store64 failed")
	}
	if xadd64(&z64, (1<<40)+1) != (2<<40)+2 {
		gothrow("xadd64 failed")
	}
	if atomicload64(&z64) != (2<<40)+2 {
		gothrow("xadd64 failed")
	}
	if xchg64(&z64, (3<<40)+3) != (2<<40)+2 {
		gothrow("xchg64 failed")
	}
	if atomicload64(&z64) != (3<<40)+3 {
		gothrow("xchg64 failed")
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
		gothrow("bad a")
	}
	if unsafe.Sizeof(b) != 1 {
		gothrow("bad b")
	}
	if unsafe.Sizeof(c) != 2 {
		gothrow("bad c")
	}
	if unsafe.Sizeof(d) != 2 {
		gothrow("bad d")
	}
	if unsafe.Sizeof(e) != 4 {
		gothrow("bad e")
	}
	if unsafe.Sizeof(f) != 4 {
		gothrow("bad f")
	}
	if unsafe.Sizeof(g) != 8 {
		gothrow("bad g")
	}
	if unsafe.Sizeof(h) != 8 {
		gothrow("bad h")
	}
	if unsafe.Sizeof(i) != 4 {
		gothrow("bad i")
	}
	if unsafe.Sizeof(j) != 8 {
		gothrow("bad j")
	}
	if unsafe.Sizeof(k) != ptrSize {
		gothrow("bad k")
	}
	if unsafe.Sizeof(l) != ptrSize {
		gothrow("bad l")
	}
	if unsafe.Sizeof(x1) != 1 {
		gothrow("bad unsafe.Sizeof x1")
	}
	if unsafe.Offsetof(y1.y) != 1 {
		gothrow("bad offsetof y1.y")
	}
	if unsafe.Sizeof(y1) != 2 {
		gothrow("bad unsafe.Sizeof y1")
	}

	if timediv(12345*1000000000+54321, 1000000000, &e) != 12345 || e != 54321 {
		gothrow("bad timediv")
	}

	var z uint32
	z = 1
	if !cas(&z, 1, 2) {
		gothrow("cas1")
	}
	if z != 2 {
		gothrow("cas2")
	}

	z = 4
	if cas(&z, 5, 6) {
		gothrow("cas3")
	}
	if z != 4 {
		gothrow("cas4")
	}

	z = 0xffffffff
	if !cas(&z, 0xffffffff, 0xfffffffe) {
		gothrow("cas5")
	}
	if z != 0xfffffffe {
		gothrow("cas6")
	}

	k = unsafe.Pointer(uintptr(0xfedcb123))
	if ptrSize == 8 {
		k = unsafe.Pointer(uintptr(unsafe.Pointer(k)) << 10)
	}
	if casp(&k, nil, nil) {
		gothrow("casp1")
	}
	k1 = add(k, 1)
	if !casp(&k, k, k1) {
		gothrow("casp2")
	}
	if k != k1 {
		gothrow("casp3")
	}

	m = [4]byte{1, 1, 1, 1}
	atomicor8(&m[1], 0xf0)
	if m[0] != 1 || m[1] != 0xf1 || m[2] != 1 || m[3] != 1 {
		gothrow("atomicor8")
	}

	*(*uint64)(unsafe.Pointer(&j)) = ^uint64(0)
	if j == j {
		gothrow("float64nan")
	}
	if !(j != j) {
		gothrow("float64nan1")
	}

	*(*uint64)(unsafe.Pointer(&j1)) = ^uint64(1)
	if j == j1 {
		gothrow("float64nan2")
	}
	if !(j != j1) {
		gothrow("float64nan3")
	}

	*(*uint32)(unsafe.Pointer(&i)) = ^uint32(0)
	if i == i {
		gothrow("float32nan")
	}
	if i == i {
		gothrow("float32nan1")
	}

	*(*uint32)(unsafe.Pointer(&i1)) = ^uint32(1)
	if i == i1 {
		gothrow("float32nan2")
	}
	if i == i1 {
		gothrow("float32nan3")
	}

	testAtomic64()

	if _FixedStack != round2(_FixedStack) {
		gothrow("FixedStack is not power-of-2")
	}
}

type dbgVar struct {
	name  string
	value *int32
}

// Do we report invalid pointers found during stack or heap scans?
//var invalidptr int32 = 1

var dbgvars = []dbgVar{
	{"allocfreetrace", &debug.allocfreetrace},
	{"invalidptr", &invalidptr},
	{"efence", &debug.efence},
	{"gctrace", &debug.gctrace},
	{"gcdead", &debug.gcdead},
	{"scheddetail", &debug.scheddetail},
	{"schedtrace", &debug.schedtrace},
	{"scavenge", &debug.scavenge},
}

func parsedebugvars() {
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
		for _, v := range dbgvars {
			if v.name == key {
				*v.value = int32(goatoi(value))
			}
		}
	}

	switch p := gogetenv("GOTRACEBACK"); p {
	case "":
		traceback_cache = 1 << 1
	case "crash":
		traceback_cache = 2<<1 | 1
	default:
		traceback_cache = uint32(goatoi(p)) << 1
	}
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

var typelink, etypelink [0]byte

//go:linkname reflect_typelinks reflect.typelinks
//go:nosplit
func reflect_typelinks() []*_type {
	var ret []*_type
	sp := (*slice)(unsafe.Pointer(&ret))
	sp.array = (*byte)(unsafe.Pointer(&typelink))
	sp.len = uint((uintptr(unsafe.Pointer(&etypelink)) - uintptr(unsafe.Pointer(&typelink))) / unsafe.Sizeof(ret[0]))
	sp.cap = sp.len
	return ret
}

// TODO: move back into mgc0.c when converted to Go
func readgogc() int32 {
	p := gogetenv("GOGC")
	if p == "" {
		return 100
	}
	if p == "off" {
		return -1
	}
	return int32(goatoi(p))
}
