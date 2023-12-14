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
	lock mutex
	val  atomic.Int64
}

// Note: Called by runtime/pprof in addition to runtime code.
func tickspersecond() int64 {
	r := ticks.val.Load()
	if r != 0 {
		return r
	}
	lock(&ticks.lock)
	r = ticks.val.Load()
	if r == 0 {
		t0 := nanotime()
		c0 := cputicks()
		usleep(100 * 1000)
		t1 := nanotime()
		c1 := cputicks()
		if t1 == t0 {
			t1++
		}
		r = (c1 - c0) * 1000 * 1000 * 1000 / (t1 - t0)
		if r == 0 {
			r++
		}
		ticks.val.Store(r)
	}
	unlock(&ticks.lock)
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
//
//go:nosplit
func writeErrStr(s string) {
	write(2, unsafe.Pointer(unsafe.StringData(s)), int32(len(s)))
}

// auxv is populated on relevant platforms but defined here for all platforms
// so x/sys/cpu can assume the getAuxv symbol exists without keeping its list
// of auxv-using GOOS build tags in sync.
//
// It contains an even number of elements, (tag, value) pairs.
var auxv []uintptr

func getAuxv() []uintptr { return auxv } // accessed from x/sys/cpu; see issue 57336
