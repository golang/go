// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import "syscall"

const (
	unlinkatTrap uintptr = syscall.SYS_UNLINKAT
	openatTrap   uintptr = syscall.SYS_OPENAT
	fstatatTrap  uintptr = syscall.SYS_FSTATAT

	AT_EACCESS          = 0x4
	AT_FDCWD            = 0xfffafdcd
	AT_REMOVEDIR        = 0x2
	AT_SYMLINK_NOFOLLOW = 0x1

	UTIME_OMIT = -0x2
)
