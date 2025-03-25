// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import "syscall"

const (
	unlinkatTrap   uintptr = syscall.SYS_UNLINKAT
	openatTrap     uintptr = syscall.SYS_OPENAT
	fstatatTrap    uintptr = syscall.SYS_FSTATAT
	readlinkatTrap uintptr = syscall.SYS_READLINKAT
	mkdiratTrap    uintptr = syscall.SYS_MKDIRAT
	fchmodatTrap   uintptr = syscall.SYS_FCHMODAT
	fchownatTrap   uintptr = syscall.SYS_FCHOWNAT
	renameatTrap   uintptr = syscall.SYS_RENAMEAT
	linkatTrap     uintptr = syscall.SYS_LINKAT
	symlinkatTrap  uintptr = syscall.SYS_SYMLINKAT
)

const (
	AT_EACCESS          = 0x100
	AT_FDCWD            = -0x64
	AT_REMOVEDIR        = 0x800
	AT_SYMLINK_NOFOLLOW = 0x200

	UTIME_OMIT = (1 << 30) - 2
)
