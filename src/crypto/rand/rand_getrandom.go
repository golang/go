// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || solaris

package rand

import (
	"internal/syscall/unix"
	"runtime"
	"syscall"
)

func init() {
	var maxGetRandomRead int
	switch runtime.GOOS {
	case "linux", "android":
		// Per the manpage:
		//     When reading from the urandom source, a maximum of 33554431 bytes
		//     is returned by a single call to getrandom() on systems where int
		//     has a size of 32 bits.
		maxGetRandomRead = (1 << 25) - 1
	case "dragonfly", "freebsd", "illumos", "solaris":
		maxGetRandomRead = 1 << 8
	default:
		panic("no maximum specified for GetRandom")
	}
	altGetRandom = batched(getRandom, maxGetRandomRead)
}

// If the kernel is too old to support the getrandom syscall(),
// unix.GetRandom will immediately return ENOSYS and we will then fall back to
// reading from /dev/urandom in rand_unix.go. unix.GetRandom caches the ENOSYS
// result so we only suffer the syscall overhead once in this case.
// If the kernel supports the getrandom() syscall, unix.GetRandom will block
// until the kernel has sufficient randomness (as we don't use GRND_NONBLOCK).
// In this case, unix.GetRandom will not return an error.
func getRandom(p []byte) error {
	n, err := unix.GetRandom(p, 0)
	if err != nil {
		return err
	}
	if n != len(p) {
		return syscall.EIO
	}
	return nil
}
