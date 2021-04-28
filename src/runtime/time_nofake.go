// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !faketime
// +build !faketime

package runtime

import "unsafe"

// faketime is the simulated time in nanoseconds since 1970 for the
// playground.
//
// Zero means not to use faketime.
var faketime int64

//go:nosplit
func nanotime() int64 {
	return nanotime1()
}

// write must be nosplit on Windows (see write1)
//
//go:nosplit
func write(fd uintptr, p unsafe.Pointer, n int32) int32 {
	return write1(fd, p, n)
}
