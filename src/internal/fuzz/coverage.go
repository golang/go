// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"internal/unsafeheader"
	"unsafe"
)

// coverage returns a []byte containing unique 8-bit counters for each edge of
// the instrumented source code. This coverage data will only be generated if
// `-d=libfuzzer` is set at build time. This can be used to understand the code
// coverage of a test execution.
func coverage() []byte {
	addr := unsafe.Pointer(&_counters)
	size := uintptr(unsafe.Pointer(&_ecounters)) - uintptr(addr)

	var res []byte
	*(*unsafeheader.Slice)(unsafe.Pointer(&res)) = unsafeheader.Slice{
		Data: addr,
		Len:  int(size),
		Cap:  int(size),
	}
	return res
}

// coverageCopy returns a copy of the current bytes provided by coverage().
// TODO(jayconrod,katiehockman): consider using a shared buffer instead, to
// make fewer costly allocations.
func coverageCopy() []byte {
	cov := coverage()
	ret := make([]byte, len(cov))
	copy(ret, cov)
	return ret
}

// resetCovereage sets all of the counters for each edge of the instrumented
// source code to 0.
func resetCoverage() {
	cov := coverage()
	for i := range cov {
		cov[i] = 0
	}
}

// _counters and _ecounters mark the start and end, respectively, of where
// the 8-bit coverage counters reside in memory. They're known to cmd/link,
// which specially assigns their addresses for this purpose.
var _counters, _ecounters [0]byte
