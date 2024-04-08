// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import "syscall"

const unlinkatTrap uintptr = syscall.SYS_UNLINKAT
const openatTrap uintptr = syscall.SYS_OPENAT
const fstatatTrap uintptr = syscall.SYS_FSTATAT

const (
	AT_EACCESS          = 0x1
	AT_FDCWD            = -0x64
	AT_REMOVEDIR        = 0x08
	AT_SYMLINK_NOFOLLOW = 0x02

	UTIME_OMIT = -0x1
)
