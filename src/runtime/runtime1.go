// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/bytealg"
	"internal/goarch"
	"internal/runtime/atomic"
	"internal/runtime/strconv"
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
	gp := getg()
	t := atomic.Load(&traceback_cache)
	crash = t&tracebackCrash != 0
	all = gp.m.throwing >= throwTypeUser || t&tracebackAll != 0
	if gp.m.traceback != 0 {
		level = int32(gp.m.traceback)
	} else if gp.m.throwing >= throwTypeRuntime {
		// Always include runtime frames in runtime throws unless
		// otherwise overridden by m.traceback.
		level = 2
	} else {
		level = int32(t >> tracebackShift)
	}
	return
}

var (
	argc int32
	argv **byte
)

// nosplit for use in linux startup sysargs.
//
//go:nosplit
func argv_index(argv **byte, i int32) *byte {
	return *(**byte)(add(unsafe.Pointer(argv), uintptr(i)*goarch.PtrSize))
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
	// guarantee env[] will immediately follow argv. Might cause
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
		k     unsafe.Pointer
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
	if unsafe.Sizeof(k) != goarch.PtrSize {
		throw("bad k")
	}
	if unsafe.Sizeof(l) != goarch.PtrSize {
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

	m = [4]byte{1, 1, 1, 1}
	atomic.Or8(&m[1], 0xf0)
	if m[0] != 1 || m[1] != 0xf1 || m[2] != 1 || m[3] != 1 {
		throw("atomicor8")
	}

	m = [4]byte{0xff, 0xff, 0xff, 0xff}
	atomic.And8(&m[1], 0x1)
	if m[0] != 0xff || m[1] != 0x1 || m[2] != 0xff || m[3] != 0xff {
		throw("atomicand8")
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

	if fixedStack != round2(fixedStack) {
		throw("FixedStack is not power-of-2")
	}

	if !checkASM() {
		throw("assembly checks failed")
	}
}

type dbgVar struct {
	name   string
	value  *int32        // for variables that can only be set at startup
	atomic *atomic.Int32 // for variables that can be changed during execution
	def    int32         // default value (ideally zero)
}

// Holds variables parsed from GODEBUG env var,
// except for "memprofilerate" since there is an
// existing int var for that value, which may
// already have an initial value.
var debug struct {
	cgocheck                 int32
	clobberfree              int32
	containermaxprocs        int32
	decoratemappings         int32
	disablethp               int32
	dontfreezetheworld       int32
	efence                   int32
	gccheckmark              int32
	gcpacertrace             int32
	gcshrinkstackoff         int32
	gcstoptheworld           int32
	gctrace                  int32
	invalidptr               int32
	madvdontneed             int32 // for Linux; issue 28466
	scavtrace                int32
	scheddetail              int32
	schedtrace               int32
	tracebackancestors       int32
	updatemaxprocs           int32
	asyncpreemptoff          int32
	harddecommit             int32
	adaptivestackstart       int32
	tracefpunwindoff         int32
	traceadvanceperiod       int32
	traceCheckStackOwnership int32
	profstackdepth           int32
	dataindependenttiming    int32

	// debug.malloc is used as a combined debug check
	// in the malloc function and should be set
	// if any of the below debug options is != 0.
	malloc          bool
	inittrace       int32
	sbrk            int32
	checkfinalizers int32
	// traceallocfree controls whether execution traces contain
	// detailed trace data about memory allocation. This value
	// affects debug.malloc only if it is != 0 and the execution
	// tracer is enabled, in which case debug.malloc will be
	// set to "true" if it isn't already while tracing is enabled.
	// It will be set while the world is stopped, so it's safe.
	// The value of traceallocfree can be changed any time in response
	// to os.Setenv("GODEBUG").
	traceallocfree atomic.Int32

	panicnil atomic.Int32

	// asynctimerchan controls whether timer channels
	// behave asynchronously (as in Go 1.22 and earlier)
	// instead of their Go 1.23+ synchronous behavior.
	// The value can change at any time (in response to os.Setenv("GODEBUG"))
	// and affects all extant timer channels immediately.
	// Programs wouldn't normally change over an execution,
	// but allowing it is convenient for testing and for programs
	// that do an os.Setenv in main.init or main.main.
	asynctimerchan atomic.Int32
}

var dbgvars = []*dbgVar{
	{name: "adaptivestackstart", value: &debug.adaptivestackstart},
	{name: "asyncpreemptoff", value: &debug.asyncpreemptoff},
	{name: "asynctimerchan", atomic: &debug.asynctimerchan},
	{name: "cgocheck", value: &debug.cgocheck},
	{name: "clobberfree", value: &debug.clobberfree},
	{name: "containermaxprocs", value: &debug.containermaxprocs, def: 1},
	{name: "dataindependenttiming", value: &debug.dataindependenttiming},
	{name: "decoratemappings", value: &debug.decoratemappings, def: 1},
	{name: "disablethp", value: &debug.disablethp},
	{name: "dontfreezetheworld", value: &debug.dontfreezetheworld},
	{name: "checkfinalizers", value: &debug.checkfinalizers},
	{name: "efence", value: &debug.efence},
	{name: "gccheckmark", value: &debug.gccheckmark},
	{name: "gcpacertrace", value: &debug.gcpacertrace},
	{name: "gcshrinkstackoff", value: &debug.gcshrinkstackoff},
	{name: "gcstoptheworld", value: &debug.gcstoptheworld},
	{name: "gctrace", value: &debug.gctrace},
	{name: "harddecommit", value: &debug.harddecommit},
	{name: "inittrace", value: &debug.inittrace},
	{name: "invalidptr", value: &debug.invalidptr},
	{name: "madvdontneed", value: &debug.madvdontneed},
	{name: "panicnil", atomic: &debug.panicnil},
	{name: "profstackdepth", value: &debug.profstackdepth, def: 128},
	{name: "sbrk", value: &debug.sbrk},
	{name: "scavtrace", value: &debug.scavtrace},
	{name: "scheddetail", value: &debug.scheddetail},
	{name: "schedtrace", value: &debug.schedtrace},
	{name: "traceadvanceperiod", value: &debug.traceadvanceperiod},
	{name: "traceallocfree", atomic: &debug.traceallocfree},
	{name: "tracecheckstackownership", value: &debug.traceCheckStackOwnership},
	{name: "tracebackancestors", value: &debug.tracebackancestors},
	{name: "tracefpunwindoff", value: &debug.tracefpunwindoff},
	{name: "updatemaxprocs", value: &debug.updatemaxprocs, def: 1},
}

func parsedebugvars() {
	// defaults
	debug.cgocheck = 1
	debug.invalidptr = 1
	debug.adaptivestackstart = 1 // set this to 0 to turn larger initial goroutine stacks off
	if GOOS == "linux" {
		// On Linux, MADV_FREE is faster than MADV_DONTNEED,
		// but doesn't affect many of the statistics that
		// MADV_DONTNEED does until the memory is actually
		// reclaimed. This generally leads to poor user
		// experience, like confusing stats in top and other
		// monitoring tools; and bad integration with
		// management systems that respond to memory usage.
		// Hence, default to MADV_DONTNEED.
		debug.madvdontneed = 1
	}
	debug.traceadvanceperiod = defaultTraceAdvancePeriod

	godebug := gogetenv("GODEBUG")

	p := new(string)
	*p = godebug
	godebugEnv.Store(p)

	// apply runtime defaults, if any
	for _, v := range dbgvars {
		if v.def != 0 {
			// Every var should have either v.value or v.atomic set.
			if v.value != nil {
				*v.value = v.def
			} else if v.atomic != nil {
				v.atomic.Store(v.def)
			}
		}
	}

	// apply compile-time GODEBUG settings
	parsegodebug(godebugDefault, nil)

	// apply environment settings
	parsegodebug(godebug, nil)

	debug.malloc = (debug.inittrace | debug.sbrk | debug.checkfinalizers) != 0
	debug.profstackdepth = min(debug.profstackdepth, maxProfStackDepth)

	// Disable async preemption in checkmark mode. The following situation is
	// problematic with checkmark mode:
	//
	// - The GC doesn't mark object A because it is truly dead.
	// - The GC stops the world, asynchronously preempting G1 which has a reference
	//   to A in its top stack frame
	// - During the stop the world, we run the second checkmark GC. It marks the roots
	//   and discovers A through G1.
	// - Checkmark mode reports a failure since there's a discrepancy in mark metadata.
	//
	// We could disable just conservative scanning during the checkmark scan, which is
	// safe but makes checkmark slightly less powerful, but that's a lot more invasive
	// than just disabling async preemption altogether.
	if debug.gccheckmark > 0 {
		debug.asyncpreemptoff = 1
	}

	setTraceback(gogetenv("GOTRACEBACK"))
	traceback_env = traceback_cache
}

// reparsedebugvars reparses the runtime's debug variables
// because the environment variable has been changed to env.
func reparsedebugvars(env string) {
	seen := make(map[string]bool)
	// apply environment settings
	parsegodebug(env, seen)
	// apply compile-time GODEBUG settings for as-yet-unseen variables
	parsegodebug(godebugDefault, seen)
	// apply defaults for as-yet-unseen variables
	for _, v := range dbgvars {
		if v.atomic != nil && !seen[v.name] {
			v.atomic.Store(0)
		}
	}
}

// parsegodebug parses the godebug string, updating variables listed in dbgvars.
// If seen == nil, this is startup time and we process the string left to right
// overwriting older settings with newer ones.
// If seen != nil, $GODEBUG has changed and we are doing an
// incremental update. To avoid flapping in the case where a value is
// set multiple times (perhaps in the default and the environment,
// or perhaps twice in the environment), we process the string right-to-left
// and only change values not already seen. After doing this for both
// the environment and the default settings, the caller must also call
// cleargodebug(seen) to reset any now-unset values back to their defaults.
func parsegodebug(godebug string, seen map[string]bool) {
	for p := godebug; p != ""; {
		var field string
		if seen == nil {
			// startup: process left to right, overwriting older settings with newer
			i := bytealg.IndexByteString(p, ',')
			if i < 0 {
				field, p = p, ""
			} else {
				field, p = p[:i], p[i+1:]
			}
		} else {
			// incremental update: process right to left, updating and skipping seen
			i := len(p) - 1
			for i >= 0 && p[i] != ',' {
				i--
			}
			if i < 0 {
				p, field = "", p
			} else {
				p, field = p[:i], p[i+1:]
			}
		}
		i := bytealg.IndexByteString(field, '=')
		if i < 0 {
			continue
		}
		key, value := field[:i], field[i+1:]
		if seen[key] {
			continue
		}
		if seen != nil {
			seen[key] = true
		}

		// Update MemProfileRate directly here since it
		// is int, not int32, and should only be updated
		// if specified in GODEBUG.
		if seen == nil && key == "memprofilerate" {
			if n, ok := strconv.Atoi(value); ok {
				MemProfileRate = n
			}
		} else {
			for _, v := range dbgvars {
				if v.name == key {
					if n, ok := strconv.Atoi32(value); ok {
						if seen == nil && v.value != nil {
							*v.value = n
						} else if v.atomic != nil {
							v.atomic.Store(n)
						}
					}
				}
			}
		}
	}

	if debug.cgocheck > 1 {
		throw("cgocheck > 1 mode is no longer supported at runtime. Use GOEXPERIMENT=cgocheck2 at build time instead.")
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
	case "wer":
		if GOOS == "windows" {
			t = 2<<tracebackShift | tracebackAll | tracebackCrash
			enableWER()
			break
		}
		fallthrough
	default:
		t = tracebackAll
		if n, ok := strconv.Atoi(level); ok && n == int(uint32(n)) {
			t |= uint32(n) << tracebackShift
		}
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
// This keeps us within no-split stack limits on 32-bit processors.
//
//go:nosplit
func timediv(v int64, div int32, rem *int32) int32 {
	res := int32(0)
	for bit := 30; bit >= 0; bit-- {
		if v >= int64(div)<<uint(bit) {
			v = v - (int64(div) << uint(bit))
			// Before this for loop, res was 0, thus all these
			// power of 2 increments are now just bitsets.
			res |= 1 << uint(bit)
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
	gp := getg()
	gp.m.locks++
	return gp.m
}

//go:nosplit
func releasem(mp *m) {
	gp := getg()
	mp.locks--
	if mp.locks == 0 && gp.preempt {
		// restore the preemption request in case we've cleared it in newstack
		gp.stackguard0 = stackPreempt
	}
}

// reflect_typelinks is meant for package reflect,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - gitee.com/quant1x/gox
//   - github.com/goccy/json
//   - github.com/modern-go/reflect2
//   - github.com/vmware/govmomi
//   - github.com/pinpoint-apm/pinpoint-go-agent
//   - github.com/timandy/routine
//   - github.com/v2pro/plz
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname reflect_typelinks reflect.typelinks
func reflect_typelinks() ([]unsafe.Pointer, [][]int32) {
	modules := activeModules()
	sections := []unsafe.Pointer{unsafe.Pointer(modules[0].types)}
	ret := [][]int32{modules[0].typelinks}
	for _, md := range modules[1:] {
		sections = append(sections, unsafe.Pointer(md.types))
		ret = append(ret, md.typelinks)
	}
	return sections, ret
}

// reflect_resolveNameOff resolves a name offset from a base pointer.
//
// reflect_resolveNameOff is for package reflect,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/agiledragon/gomonkey/v2
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname reflect_resolveNameOff reflect.resolveNameOff
func reflect_resolveNameOff(ptrInModule unsafe.Pointer, off int32) unsafe.Pointer {
	return unsafe.Pointer(resolveNameOff(ptrInModule, nameOff(off)).Bytes)
}

// reflect_resolveTypeOff resolves an *rtype offset from a base type.
//
// reflect_resolveTypeOff is meant for package reflect,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - gitee.com/quant1x/gox
//   - github.com/modern-go/reflect2
//   - github.com/v2pro/plz
//   - github.com/timandy/routine
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname reflect_resolveTypeOff reflect.resolveTypeOff
func reflect_resolveTypeOff(rtype unsafe.Pointer, off int32) unsafe.Pointer {
	return unsafe.Pointer(toRType((*_type)(rtype)).typeOff(typeOff(off)))
}

// reflect_resolveTextOff resolves a function pointer offset from a base type.
//
// reflect_resolveTextOff is for package reflect,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/agiledragon/gomonkey/v2
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname reflect_resolveTextOff reflect.resolveTextOff
func reflect_resolveTextOff(rtype unsafe.Pointer, off int32) unsafe.Pointer {
	return toRType((*_type)(rtype)).textOff(textOff(off))
}

// reflectlite_resolveNameOff resolves a name offset from a base pointer.
//
//go:linkname reflectlite_resolveNameOff internal/reflectlite.resolveNameOff
func reflectlite_resolveNameOff(ptrInModule unsafe.Pointer, off int32) unsafe.Pointer {
	return unsafe.Pointer(resolveNameOff(ptrInModule, nameOff(off)).Bytes)
}

// reflectlite_resolveTypeOff resolves an *rtype offset from a base type.
//
//go:linkname reflectlite_resolveTypeOff internal/reflectlite.resolveTypeOff
func reflectlite_resolveTypeOff(rtype unsafe.Pointer, off int32) unsafe.Pointer {
	return unsafe.Pointer(toRType((*_type)(rtype)).typeOff(typeOff(off)))
}

// reflect_addReflectOff adds a pointer to the reflection offset lookup map.
//
//go:linkname reflect_addReflectOff reflect.addReflectOff
func reflect_addReflectOff(ptr unsafe.Pointer) int32 {
	reflectOffsLock()
	if reflectOffs.m == nil {
		reflectOffs.m = make(map[int32]unsafe.Pointer)
		reflectOffs.minv = make(map[unsafe.Pointer]int32)
		reflectOffs.next = -1
	}
	id, found := reflectOffs.minv[ptr]
	if !found {
		id = reflectOffs.next
		reflectOffs.next-- // use negative offsets as IDs to aid debugging
		reflectOffs.m[id] = ptr
		reflectOffs.minv[ptr] = id
	}
	reflectOffsUnlock()
	return id
}

//go:linkname fips_getIndicator crypto/internal/fips140.getIndicator
func fips_getIndicator() uint8 {
	return getg().fipsIndicator
}

//go:linkname fips_setIndicator crypto/internal/fips140.setIndicator
func fips_setIndicator(indicator uint8) {
	getg().fipsIndicator = indicator
}
