// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Called from C. Returns the Go type *m.
func gc_m_ptr(ret *interface{}) {
	*ret = (*m)(nil)
}

// Called from C. Returns the Go type *g.
func gc_g_ptr(ret *interface{}) {
	*ret = (*g)(nil)
}

// Called from C. Returns the Go type *itab.
func gc_itab_ptr(ret *interface{}) {
	*ret = (*itab)(nil)
}

// Type used for "conservative" allocations in C code.
type notype [8]*byte

// Called from C. Returns the Go type used for C allocations w/o type.
func gc_notype_ptr(ret *interface{}) {
	var x notype
	*ret = x
}

func timenow() (sec int64, nsec int32)

func gc_unixnanotime(now *int64) {
	sec, nsec := timenow()
	*now = sec*1e9 + int64(nsec)
}

func freeOSMemory() {
	gogc(2) // force GC and do eager sweep
	onM(&scavenge_m)
}

var poolcleanup func()

func registerPoolCleanup(f func()) {
	poolcleanup = f
}

func clearpools() {
	// clear sync.Pools
	if poolcleanup != nil {
		poolcleanup()
	}

	for _, p := range &allp {
		if p == nil {
			break
		}
		// clear tinyalloc pool
		if c := p.mcache; c != nil {
			c.tiny = nil
			c.tinysize = 0
			c.sudogcache = nil
		}
		// clear defer pools
		for i := range p.deferpool {
			p.deferpool[i] = nil
		}
	}
}
