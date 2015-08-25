// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl plan9 windows

package net

// See unix_test.go for what these (don't) do.
func forceGoDNS() func() { return func() {} }
func forceCgoDNS() bool  { return false }
