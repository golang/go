// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm nacl plan9 windows

package net

import "runtime"

// See main_conf_test.go for what these (don't) do.
func forceGoDNS() func() {
	switch runtime.GOOS {
	case "plan9", "windows":
		return func() {}
	default:
		return nil
	}
}

// See main_conf_test.go for what these (don't) do.
func forceCgoDNS() func() { return nil }
