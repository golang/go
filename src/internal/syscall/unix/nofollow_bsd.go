// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd

package unix

import "syscall"

// References:
// - https://man.freebsd.org/cgi/man.cgi?open(2)
// - https://man.dragonflybsd.org/?command=open&section=2
const noFollowErrno = syscall.EMLINK
