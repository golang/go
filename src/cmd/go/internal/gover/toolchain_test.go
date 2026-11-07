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

func TestToolchainForGoVersion(t *testing.T) {
	test1(t, toolchainForGoVersionTests, "ToolchainForGoVersion", ToolchainForGoVersion)
}

var toolchainForGoVersionTests = []testCase1[string, string]{
	{"1.20", "go1.20"},
	{"1.21", "go1.21.0"},
	{"1.22", "go1.22.0"},
	{"1.23", "go1.23.0"},
	{"1.23.4", "go1.23.4"},
	{"1.24rc1", "go1.24rc1"},
}
