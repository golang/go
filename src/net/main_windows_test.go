// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

var (
	// Placeholders for saving original socket system calls.
	origSocket      = socketFunc
	origClosesocket = closeFunc
	origConnect     = connectFunc
	origConnectEx   = connectExFunc
	origListen      = listenFunc
	origAccept      = acceptFunc
)

func installTestHooks() {
	socketFunc = sw.Socket
	closeFunc = sw.Closesocket
	connectFunc = sw.Connect
	connectExFunc = sw.ConnectEx
	listenFunc = sw.Listen
	acceptFunc = sw.AcceptEx
}

func uninstallTestHooks() {
	socketFunc = origSocket
	closeFunc = origClosesocket
	connectFunc = origConnect
	connectExFunc = origConnectEx
	listenFunc = origListen
	acceptFunc = origAccept
}

// forceCloseSockets must be called only from TestMain.
func forceCloseSockets() {
	for s := range sw.Sockets() {
		closeFunc(s)
	}
}
