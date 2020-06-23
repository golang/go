// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux freebsd

package rand

import (
	"internal/syscall/unix"
)

// maxGetRandomRead is platform dependent.
func init() {
	altGetRandom = batched(getRandomBatch, maxGetRandomRead)
}

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

// If the kernel is too old to support the getrandom syscall(),
// unix.GetRandom will immediately return ENOSYS and we will then fall back to
// reading from /dev/urandom in rand_unix.go. unix.GetRandom caches the ENOSYS
// result so we only suffer the syscall overhead once in this case.
// If the kernel supports the getrandom() syscall, unix.GetRandom will block
// until the kernel has sufficient randomness (as we don't use GRND_NONBLOCK).
// In this case, unix.GetRandom will not return an error.
func getRandomBatch(p []byte) (ok bool) {
	n, err := unix.GetRandom(p, 0)
	return n == len(p) && err == nil
}
