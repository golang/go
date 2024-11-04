// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux || solaris

package sysrand

import (
	"errors"
	"internal/syscall/unix"
	"math"
	"runtime"
	"syscall"
)

func read(b []byte) error {
	// Linux, DragonFly, and illumos don't have a limit on the buffer size.
	// FreeBSD has a limit of IOSIZE_MAX, which seems to be either INT_MAX or
	// SSIZE_MAX. 2^31-1 is a safe and high enough value to use for all of them.
	//
	// Note that Linux returns "a maximum of 32Mi-1 bytes", but that will only
	// result in a short read, not an error. Short reads can also happen above
	// 256 bytes due to signals. Reads up to 256 bytes are guaranteed not to
	// return short (and not to return an error IF THE POOL IS INITIALIZED) on
	// at least Linux, FreeBSD, DragonFly, and Oracle Solaris, but we don't make
	// use of that.
	maxSize := math.MaxInt32

	// Oracle Solaris has a limit of 133120 bytes. Very specific.
	//
	//    The getrandom() and getentropy() functions fail if: [...]
	//
	//    - bufsz is <= 0 or > 133120, when GRND_RANDOM is not set
	//
	// https://docs.oracle.com/cd/E88353_01/html/E37841/getrandom-2.html
	if runtime.GOOS == "solaris" {
		maxSize = 133120
	}

	for len(b) > 0 {
		size := len(b)
		if size > maxSize {
			size = maxSize
		}
		n, err := unix.GetRandom(b[:size], 0)
		if errors.Is(err, syscall.ENOSYS) {
			// If getrandom(2) is not available, presumably on Linux versions
			// earlier than 3.17, fall back to reading from /dev/urandom.
			return urandomRead(b)
		}
		if errors.Is(err, syscall.EINTR) {
			// If getrandom(2) is blocking, either because it is waiting for the
			// entropy pool to become initialized or because we requested more
			// than 256 bytes, it might get interrupted by a signal.
			continue
		}
		if err != nil {
			return err
		}
		b = b[n:]
	}
	return nil
}
