// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package scratch provides the scratch memory used by the FIPS 140-3 CPU
// jitter entropy source. It's outside the module boundary and the certified
// entropy source, so we can change how the memory is allocated.
package scratch

import (
	"internal/bytealg"
	"sync"
)

var shared struct {
	sync.Mutex
	buf    []byte
	mapped bool
	refs   int
}

// Acquire returns a buffer of n bytes with unspecified contents, which must be
// passed to [Release] when no longer in use. n must be the same for all calls.
//
// Concurrent callers may get the same buffer, so the contents may change at
// any time, and must only be accessed with atomic operations.
func Acquire(n int) []byte {
	shared.Lock()
	defer shared.Unlock()
	if shared.refs == 0 {
		// Get the memory directly from the operating system if possible, so
		// that it doesn't count towards the garbage collector's heap goal,
		// and is returned as soon as it's released.
		if b, err := mmap(n); err == nil {
			shared.buf = b
			shared.mapped = true
		} else {
			// The contents are never used, so skip zeroing.
			shared.buf = bytealg.MakeNoZero(n)
			shared.mapped = false
		}
	}
	if len(shared.buf) != n {
		panic("scratch: inconsistent buffer size")
	}
	shared.refs++
	return shared.buf
}

// Release releases a buffer returned by [Acquire].
// The buffer must not be used after it's released.
func Release(b []byte) {
	shared.Lock()
	defer shared.Unlock()
	if shared.refs--; shared.refs == 0 {
		if shared.mapped {
			munmap(shared.buf)
		}
		shared.buf = nil
	}
}
