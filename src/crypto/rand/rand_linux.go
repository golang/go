// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"internal/syscall/unix"
)

func init() {
	altGetRandom = batched(getRandomLinux, maxGetRandomRead)
}

// maxGetRandomRead is the maximum number of bytes to ask for in one call to the
// getrandom() syscall. In linux at most 2^25-1 bytes will be returned per call.
// From the manpage
//
//	*  When reading from the urandom source, a maximum of 33554431 bytes
//	   is returned by a single call to getrandom() on systems where int
//	   has a size of 32 bits.
const maxGetRandomRead = (1 << 25) - 1

// batched returns a function that calls f to populate a []byte by chunking it
// into subslices of, at most, readMax bytes.
func batched(f func([]byte) bool, readMax int) func([]byte) bool {
	return func(buf []byte) bool {
		for len(buf) > readMax {
			if !f(buf[:readMax]) {
				return false
			}
			buf = buf[readMax:]
		}
		return len(buf) == 0 || f(buf)
	}
}

// If the kernel is too old (before 3.17) to support the getrandom syscall(),
// unix.GetRandom will immediately return ENOSYS and we will then fall back to
// reading from /dev/urandom in rand_unix.go. unix.GetRandom caches the ENOSYS
// result so we only suffer the syscall overhead once in this case.
// If the kernel supports the getrandom() syscall, unix.GetRandom will block
// until the kernel has sufficient randomness (as we don't use GRND_NONBLOCK).
// In this case, unix.GetRandom will not return an error.
func getRandomLinux(p []byte) (ok bool) {
	n, err := unix.GetRandom(p, 0)
	return n == len(p) && err == nil
}
