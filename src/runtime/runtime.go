// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

var ticks struct {
	lock mutex
	val  uint64
}

var tls0 [8]uintptr // available storage for m0's TLS; not necessarily used; opaque to GC

// Note: Called by runtime/pprof in addition to runtime code.
func tickspersecond() int64 {
	r := int64(atomicload64(&ticks.val))
	if r != 0 {
		return r
	}
	lock(&ticks.lock)
	r = int64(ticks.val)
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
		atomicstore64(&ticks.val, uint64(r))
	}
	unlock(&ticks.lock)
	return r
}

func makeStringSlice(n int) []string {
	return make([]string, n)
}

// TODO: Move to parfor.go when parfor.c becomes parfor.go.
func parforalloc(nthrmax uint32) *parfor {
	return &parfor{
		thr:     &make([]parforthread, nthrmax)[0],
		nthrmax: nthrmax,
	}
}

var envs []string
var argslice []string

// called from syscall
func runtime_envs() []string { return envs }

// called from os
func runtime_args() []string { return argslice }
