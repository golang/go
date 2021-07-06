// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || linux || netbsd || openbsd
// +build dragonfly linux netbsd openbsd

package net

import "syscall"

const readMsgFlags = syscall.MSG_CMSG_CLOEXEC

func setReadMsgCloseOnExec(oob []byte) {}
