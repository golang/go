// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "testing"

var toolchainCmpTests = []struct {
	x   string
	y   string
	out int
}{
	{"", "", 0},
	{"x", "x", 0},
	{"", "x", -1},
	{"go1.5", "go1.6", -1},
	{"go1.5", "go1.10", -1},
	{"go1.6", "go1.6.1", -1},
	{"go1.999", "devel go1.4", -1},
	{"devel go1.5", "devel go1.6", 0}, // devels are all +infinity
	{"go1.19", "go1.19.1", -1},
	{"go1.19rc1", "go1.19", -1},
	{"go1.19rc1", "go1.19.1", -1},
	{"go1.19rc1", "go1.19rc2", -1},
	{"go1.19.0", "go1.19.1", -1},
	{"go1.19rc1", "go1.19.0", -1},
	{"go1.19alpha3", "go1.19beta2", -1},
	{"go1.19beta2", "go1.19rc1", -1},

	// Syntax we don't ever plan to use, but just in case we do.
	{"go1.19.0-rc.1", "go1.19.0-rc.2", -1},
	{"go1.19.0-rc.1", "go1.19.0", -1},
	{"go1.19.0-alpha.3", "go1.19.0-beta.2", -1},
	{"go1.19.0-beta.2", "go1.19.0-rc.1", -1},
}

func TestToolchainCmp(t *testing.T) {
	for _, tt := range toolchainCmpTests {
		out := toolchainCmp(tt.x, tt.y)
		if out != tt.out {
			t.Errorf("toolchainCmp(%q, %q) = %d, want %d", tt.x, tt.y, out, tt.out)
		}
		out = toolchainCmp(tt.y, tt.x)
		if out != -tt.out {
			t.Errorf("toolchainCmp(%q, %q) = %d, want %d", tt.y, tt.x, out, -tt.out)
		}
	}
}
