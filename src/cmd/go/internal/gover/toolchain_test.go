// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gover

import "testing"

func TestFromToolchain(t *testing.T) { test1(t, fromToolchainTests, "FromToolchain", FromToolchain) }

var fromToolchainTests = []testCase1[string, string]{
	{"go1.2.3", "1.2.3"},
	{"1.2.3", ""},
	{"go1.2.3+bigcorp", ""},
	{"go1.2.3-bigcorp", "1.2.3"},
	{"go1.2.3-bigcorp more text", "1.2.3"},
	{"gccgo-go1.23rc4", ""},
	{"gccgo-go1.23rc4-bigdwarf", ""},
}
