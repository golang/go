// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !cgo netgo
// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import "testing"

func TestGoLookupIP(t *testing.T) {
	host := "localhost"
	_, err, ok := cgoLookupIP(host)
	if ok {
		t.Errorf("cgoLookupIP must be a placeholder")
	}
	if err != nil {
		t.Errorf("cgoLookupIP failed: %v", err)
	}
	if _, err := goLookupIP(host); err != nil {
		t.Errorf("goLookupIP failed: %v", err)
	}
}
