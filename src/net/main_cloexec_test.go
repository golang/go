// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd linux

package net

import "internal/poll"

func init() {
	extraTestHookInstallers = append(extraTestHookInstallers, installAccept4TestHook)
	extraTestHookUninstallers = append(extraTestHookUninstallers, uninstallAccept4TestHook)
}

var (
	// Placeholders for saving original socket system calls.
	origAccept4 = poll.Accept4Func
)

func installAccept4TestHook() {
	poll.Accept4Func = sw.Accept4
}

func uninstallAccept4TestHook() {
	poll.Accept4Func = origAccept4
}
