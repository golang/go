// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

// FreeBSD getrandom system call number.
const getrandomTrap uintptr = 563

const (
	// GRND_NONBLOCK means return EAGAIN rather than blocking.
	GRND_NONBLOCK GetRandomFlag = 0x0001

	// GRND_RANDOM is only set for portability purpose, no-op on FreeBSD.
	GRND_RANDOM GetRandomFlag = 0x0002
)
