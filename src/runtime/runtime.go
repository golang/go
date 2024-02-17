// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

//go:generate go run wincallback.go
//go:generate go run mkduff.go
//go:generate go run mkfastlog2table.go
//go:generate go run mklockrank.go -o lockrank.go

var ticks ticksType

type ticksType struct {
	// lock protects access to start* and val.
	lock       mutex
	startTicks int64
	startTime  int64
	val        atomic.Int64
}

// init initializes ticks to maximize the chance that we have a good ticksPerSecond reference.
//
// Must not run concurrently with ticksPerSecond.
func (t *ticksType) init() {
	lock(&ticks.lock)
	t.startTime = nanotime()
	t.startTicks = cputicks()
	unlock(&ticks.lock)
}

// minTimeForTicksPerSecond is the minimum elapsed time we require to consider our ticksPerSecond
// measurement to be of decent enough quality for profiling.
//
// There's a linear relationship here between minimum time and error from the true value.
// The error from the true ticks-per-second in a linux/amd64 VM seems to be:
// -   1 ms -> ~0.02% error
// -   5 ms -> ~0.004% error
// -  10 ms -> ~0.002% error
// -  50 ms -> ~0.0003% error
// - 100 ms -> ~0.0001% error
//
// We're willing to take 0.004% error here, because ticksPerSecond is intended to be used for
// converting durations, not timestamps. Durations are usually going to be much larger, and so
// the tiny error doesn't matter. The error is definitely going to be a problem when trying to
// use this for timestamps, as it'll make those timestamps much less likely to line up.
const minTimeForTicksPerSecond = 5_000_000*(1-osHasLowResClockInt) + 100_000_000*osHasLowResClockInt

// ticksPerSecond returns a conversion rate between the cputicks clock and the nanotime clock.
//
// Note: Clocks are hard. Using this as an actual conversion rate for timestamps is ill-advised
// and should be avoided when possible. Use only for durations, where a tiny error term isn't going
// to make a meaningful difference in even a 1ms duration. If an accurate timestamp is needed,
// use nanotime instead. (The entire Windows platform is a broad exception to this rule, where nanotime
// produces timestamps on such a coarse granularity that the error from this conversion is actually
// preferable.)
//
// The strategy for computing the conversion rate is to write down nanotime and cputicks as
// early in process startup as possible. From then, we just need to wait until we get values
// from nanotime that we can use (some platforms have a really coarse system time granularity).
// We require some amount of time to pass to ensure that the conversion rate is fairly accurate
// in aggregate. But because we compute this rate lazily, there's a pretty good chance a decent
// amount of time has passed by the time we get here.
//
// Must be called from a normal goroutine context (running regular goroutine with a P).
//
// Called by runtime/pprof in addition to runtime code.
//
// TODO(mknyszek): This doesn't account for things like CPU frequency scaling. Consider
// a more sophisticated and general approach in the future.
func ticksPerSecond() int64 {
	// Get the conversion rate if we've already computed it.
	r := ticks.val.Load()
	if r != 0 {
		return r
	}

	// Compute the conversion rate.
	for {
		lock(&ticks.lock)
		r = ticks.val.Load()
		if r != 0 {
			unlock(&ticks.lock)
			return r
		}

		// Grab the current time in both clocks.
		nowTime := nanotime()
		nowTicks := cputicks()

		// See if we can use these times.
		if nowTicks > ticks.startTicks && nowTime-ticks.startTime > minTimeForTicksPerSecond {
			// Perform the calculation with floats. We don't want to risk overflow.
			r = int64(float64(nowTicks-ticks.startTicks) * 1e9 / float64(nowTime-ticks.startTime))
			if r == 0 {
				// Zero is both a sentinel value and it would be bad if callers used this as
				// a divisor. We tried out best, so just make it 1.
				r++
			}
			ticks.val.Store(r)
			unlock(&ticks.lock)
			break
		}
		unlock(&ticks.lock)

		// Sleep in one millisecond increments until we have a reliable time.
		timeSleep(1_000_000)
	}
	return r
}

var envs []string
var argslice []string

//go:linkname syscall_runtime_envs syscall.runtime_envs
func syscall_runtime_envs() []string { return append([]string{}, envs...) }

//go:linkname syscall_Getpagesize syscall.Getpagesize
func syscall_Getpagesize() int { return int(physPageSize) }

//go:linkname os_runtime_args os.runtime_args
func os_runtime_args() []string { return append([]string{}, argslice...) }

//go:linkname syscall_Exit syscall.Exit
//go:nosplit
func syscall_Exit(code int) {
	exit(int32(code))
}

var godebugDefault string
var godebugUpdate atomic.Pointer[func(string, string)]
var godebugEnv atomic.Pointer[string] // set by parsedebugvars
var godebugNewIncNonDefault atomic.Pointer[func(string) func()]

//go:linkname godebug_setUpdate internal/godebug.setUpdate
func godebug_setUpdate(update func(string, string)) {
	p := new(func(string, string))
	*p = update
	godebugUpdate.Store(p)
	godebugNotify(false)
}

//go:linkname godebug_setNewIncNonDefault internal/godebug.setNewIncNonDefault
func godebug_setNewIncNonDefault(newIncNonDefault func(string) func()) {
	p := new(func(string) func())
	*p = newIncNonDefault
	godebugNewIncNonDefault.Store(p)
}

// A godebugInc provides access to internal/godebug's IncNonDefault function
// for a given GODEBUG setting.
// Calls before internal/godebug registers itself are dropped on the floor.
type godebugInc struct {
	name string
	inc  atomic.Pointer[func()]
}

func (g *godebugInc) IncNonDefault() {
	inc := g.inc.Load()
	if inc == nil {
		newInc := godebugNewIncNonDefault.Load()
		if newInc == nil {
			return
		}
		inc = new(func())
		*inc = (*newInc)(g.name)
		if raceenabled {
			racereleasemerge(unsafe.Pointer(&g.inc))
		}
		if !g.inc.CompareAndSwap(nil, inc) {
			inc = g.inc.Load()
		}
	}
	if raceenabled {
		raceacquire(unsafe.Pointer(&g.inc))
	}
	(*inc)()
}

func godebugNotify(envChanged bool) {
	update := godebugUpdate.Load()
	var env string
	if p := godebugEnv.Load(); p != nil {
		env = *p
	}
	if envChanged {
		reparsedebugvars(env)
	}
	if update != nil {
		(*update)(godebugDefault, env)
	}
}

//go:linkname syscall_runtimeSetenv syscall.runtimeSetenv
func syscall_runtimeSetenv(key, value string) {
	setenv_c(key, value)
	if key == "GODEBUG" {
		p := new(string)
		*p = value
		godebugEnv.Store(p)
		godebugNotify(true)
	}
}

//go:linkname syscall_runtimeUnsetenv syscall.runtimeUnsetenv
func syscall_runtimeUnsetenv(key string) {
	unsetenv_c(key)
	if key == "GODEBUG" {
		godebugEnv.Store(nil)
		godebugNotify(true)
	}
}

// writeErrStr writes a string to descriptor 2.
// If SetCrashOutput(f) was called, it also writes to f.
//
//go:nosplit
func writeErrStr(s string) {
	writeErrData(unsafe.StringData(s), int32(len(s)))
}

// writeErrData is the common parts of writeErr{,Str}.
//
//go:nosplit
func writeErrData(data *byte, n int32) {
	write(2, unsafe.Pointer(data), n)

	// If crashing, print a copy to the SetCrashOutput fd.
	gp := getg()
	if gp != nil && gp.m.dying > 0 ||
		gp == nil && panicking.Load() > 0 {
		if fd := crashFD.Load(); fd != ^uintptr(0) {
			write(fd, unsafe.Pointer(data), n)
		}
	}
}

// crashFD is an optional file descriptor to use for fatal panics, as
// set by debug.SetCrashOutput (see #42888). If it is a valid fd (not
// all ones), writeErr and related functions write to it in addition
// to standard error.
//
// Initialized to -1 in schedinit.
var crashFD atomic.Uintptr

//go:linkname setCrashFD
func setCrashFD(fd uintptr) uintptr {
	return crashFD.Swap(fd)
}

// auxv is populated on relevant platforms but defined here for all platforms
// so x/sys/cpu can assume the getAuxv symbol exists without keeping its list
// of auxv-using GOOS build tags in sync.
//
// It contains an even number of elements, (tag, value) pairs.
var auxv []uintptr

func getAuxv() []uintptr { return auxv } // accessed from x/sys/cpu; see issue 57336
