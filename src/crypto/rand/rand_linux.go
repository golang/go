// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"internal/syscall/unix"
	"sync"
)

func init() {
	altGetRandom = getRandomLinux
}

var (
	once       sync.Once
	useSyscall bool
)

func pickStrategy() {
	// Test whether we should use the system call or /dev/urandom.
	// We'll fall back to urandom if:
	// - the kernel is too old (before 3.17)
	// - the machine has no entropy available (early boot + no hardware
	//   entropy source?) and we want to avoid blocking later.
	var buf [1]byte
	n, err := unix.GetRandom(buf[:], unix.GRND_NONBLOCK)
	useSyscall = n == 1 && err == nil
}

func getRandomLinux(p []byte) (ok bool) {
	once.Do(pickStrategy)
	if !useSyscall {
		return false
	}
	n, err := unix.GetRandom(p, 0)
	return n == len(p) && err == nil
}
