// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm mips mipsle 386

package unix

import "syscall"

const fstatatTrap uintptr = syscall.SYS_FSTATAT64
