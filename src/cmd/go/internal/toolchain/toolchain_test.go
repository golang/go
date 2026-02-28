// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package toolchain

import (
	"strings"
	"testing"
)

func TestNewerToolchain(t *testing.T) {
	for _, tt := range newerToolchainTests {
		out, err := newerToolchain(tt.need, tt.list)
		if (err != nil) != (out == "") {
			t.Errorf("newerToolchain(%v, %v) = %v, %v, want error", tt.need, tt.list, out, err)
			continue
		}
		if out != tt.out {
			t.Errorf("newerToolchain(%v, %v) = %v, %v want %v, nil", tt.need, tt.list, out, err, tt.out)
		}
	}
}

var f = strings.Fields

var relRC = []string{"1.39.0", "1.39.1", "1.39.2", "1.40.0", "1.40.1", "1.40.2", "1.41rc1"}
var rel2 = []string{"1.39.0", "1.39.1", "1.39.2", "1.40.0", "1.40.1", "1.40.2"}
var rel0 = []string{"1.39.0", "1.39.1", "1.39.2", "1.40.0"}
var newerToolchainTests = []struct {
	need string
	list []string
	out  string
}{
	{"1.30", rel0, "go1.39.2"},
	{"1.30", rel2, "go1.39.2"},
	{"1.30", relRC, "go1.39.2"},
	{"1.38", rel0, "go1.39.2"},
	{"1.38", rel2, "go1.39.2"},
	{"1.38", relRC, "go1.39.2"},
	{"1.38.1", rel0, "go1.39.2"},
	{"1.38.1", rel2, "go1.39.2"},
	{"1.38.1", relRC, "go1.39.2"},
	{"1.39", rel0, "go1.39.2"},
	{"1.39", rel2, "go1.39.2"},
	{"1.39", relRC, "go1.39.2"},
	{"1.39.2", rel0, "go1.39.2"},
	{"1.39.2", rel2, "go1.39.2"},
	{"1.39.2", relRC, "go1.39.2"},
	{"1.39.3", rel0, "go1.40.0"},
	{"1.39.3", rel2, "go1.40.2"},
	{"1.39.3", relRC, "go1.40.2"},
	{"1.40", rel0, "go1.40.0"},
	{"1.40", rel2, "go1.40.2"},
	{"1.40", relRC, "go1.40.2"},
	{"1.40.1", rel0, ""},
	{"1.40.1", rel2, "go1.40.2"},
	{"1.40.1", relRC, "go1.40.2"},
	{"1.41", rel0, ""},
	{"1.41", rel2, ""},
	{"1.41", relRC, "go1.41rc1"},
	{"1.41.0", rel0, ""},
	{"1.41.0", rel2, ""},
	{"1.41.0", relRC, ""},
	{"1.40", nil, ""},
}
