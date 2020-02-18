// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package lsprpc

// autoNetworkAddress returns the default network and address for the
// automatically-started gopls remote. See autostart_posix.go for more
// information.
func autoNetworkAddress(goplsPath, id string) (network string, address string) {
	if id != "" {
		panic("identified remotes are not supported on windows")
	}
	return "tcp", ":37374"
}
