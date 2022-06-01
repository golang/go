// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || freebsd || dragonfly || solaris

package rand

import (
	"errors"
	"internal/syscall/unix"
)

// maxGetRandomRead is platform dependent.
func init() {
	altGetRandom = batched(getRandomBatch, maxGetRandomRead)
}

// If the kernel is too old to support the getrandom syscall(),
// unix.GetRandom will immediately return ENOSYS and we will then fall back to
// reading from /dev/urandom in rand_unix.go. unix.GetRandom caches the ENOSYS
// result so we only suffer the syscall overhead once in this case.
// If the kernel supports the getrandom() syscall, unix.GetRandom will block
// until the kernel has sufficient randomness (as we don't use GRND_NONBLOCK).
// In this case, unix.GetRandom will not return an error.
func getRandomBatch(p []byte) (err error) {
	n, err := unix.GetRandom(p, 0)
	if n != len(p) {
		return errors.New("short read")
	}
	return err
}
