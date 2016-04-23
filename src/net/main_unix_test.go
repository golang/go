// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package net

var (
	// Placeholders for saving original socket system calls.
	origSocket        = socketFunc
	origClose         = closeFunc
	origConnect       = connectFunc
	origListen        = listenFunc
	origAccept        = acceptFunc
	origGetsockoptInt = getsockoptIntFunc

	extraTestHookInstallers   []func()
	extraTestHookUninstallers []func()
)

func installTestHooks() {
	socketFunc = sw.Socket
	closeFunc = sw.Close
	connectFunc = sw.Connect
	listenFunc = sw.Listen
	acceptFunc = sw.Accept
	getsockoptIntFunc = sw.GetsockoptInt

	for _, fn := range extraTestHookInstallers {
		fn()
	}
}

func uninstallTestHooks() {
	socketFunc = origSocket
	closeFunc = origClose
	connectFunc = origConnect
	listenFunc = origListen
	acceptFunc = origAccept
	getsockoptIntFunc = origGetsockoptInt

	for _, fn := range extraTestHookUninstallers {
		fn()
	}
}

func forceCloseSockets() {
	for s := range sw.Sockets() {
		closeFunc(s)
	}
}
