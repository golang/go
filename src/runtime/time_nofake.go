// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !faketime

package runtime

import "unsafe"

// faketime is the simulated time in nanoseconds since 1970 for the
// playground.
//
// Zero means not to use faketime.
var faketime int64

// Exported via linkname for use by time and internal/poll.
//
// Many external packages also linkname nanotime for a fast monotonic time.
// Such code should be updated to use:
//
//	var start = time.Now() // at init time
//
// and then replace nanotime() with time.Since(start), which is equally fast.
//
// However, all the code linknaming nanotime is never going to go away.
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname nanotime
//go:nosplit
func nanotime() int64 {
	return nanotime1()
}

// overrideWrite allows write to be redirected externally, by
// linkname'ing this and set it to a write function.
//
// overrideWrite should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - golang.zx2c4.com/wireguard/windows
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname overrideWrite
var overrideWrite func(fd uintptr, p unsafe.Pointer, n int32) int32

// write must be nosplit on Windows (see write1)
//
//go:nosplit
func write(fd uintptr, p unsafe.Pointer, n int32) int32 {
	if overrideWrite != nil {
		return overrideWrite(fd, noescape(p), n)
	}
	return write1(fd, p, n)
}
