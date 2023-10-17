// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

// DragonFlyBSD getrandom system call number.
const getrandomTrap uintptr = 550

const (
	// GRND_RANDOM is only set for portability purpose, no-op on DragonFlyBSD.
	GRND_RANDOM GetRandomFlag = 0x0001

	// GRND_NONBLOCK means return EAGAIN rather than blocking.
	GRND_NONBLOCK GetRandomFlag = 0x0002
)
