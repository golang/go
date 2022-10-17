// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	_ "unsafe" // for go:linkname
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

var godebugenv atomic.Pointer[string] // set by parsedebugvars

//go:linkname godebug_getGODEBUG internal/godebug.getGODEBUG
func godebug_getGODEBUG() string {
	if p := godebugenv.Load(); p != nil {
		return *p
	}
	return ""
}

//go:linkname syscall_runtimeSetenv syscall.runtimeSetenv
func syscall_runtimeSetenv(key, value string) {
	setenv_c(key, value)
	if key == "GODEBUG" {
		p := new(string)
		*p = value
		godebugenv.Store(p)
	}
}

//go:linkname syscall_runtimeUnsetenv syscall.runtimeUnsetenv
func syscall_runtimeUnsetenv(key string) {
	unsetenv_c(key)
	if key == "GODEBUG" {
		godebugenv.Store(nil)
	}
}
