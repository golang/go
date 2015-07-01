// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

var (
	testHookDialTCP      = dialTCP
	testHookHostsPath    = "/etc/hosts"
	testHookLookupIP     = func(fn func(string) ([]IPAddr, error), host string) ([]IPAddr, error) { return fn(host) }
	testHookSetKeepAlive = func() {}
)
