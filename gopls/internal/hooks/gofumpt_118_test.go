// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package hooks

import "testing"

func TestFixLangVersion(t *testing.T) {
	tests := []struct {
		input, want string
		wantErr     bool
	}{
		{"", "", false},
		{"1.18", "1.18", false},
		{"v1.18", "v1.18", false},
		{"1.21", "1.21", false},
		{"1.21rc3", "1.21", false},
		{"1.21.0", "1.21.0", false},
		{"1.21.1", "1.21.1", false},
		{"v1.21.1", "v1.21.1", false},
		{"v1.21.0rc1", "v1.21.0", false}, // not technically valid, but we're flexible
		{"v1.21.0.0", "v1.21.0", false},  // also technically invalid
		{"1.1", "1.1", false},
		{"v1", "v1", false},
		{"1", "1", false},
		{"v1.21.", "v1.21", false}, // also invalid
		{"1.21.", "1.21", false},

		// Error cases.
		{"rc1", "", true},
		{"x1.2.3", "", true},
	}

	for _, test := range tests {
		got, err := fixLangVersion(test.input)
		if test.wantErr {
			if err == nil {
				t.Errorf("fixLangVersion(%q) succeeded unexpectedly", test.input)
			}
			continue
		}
		if err != nil {
			t.Fatalf("fixLangVersion(%q) failed: %v", test.input, err)
		}
		if got != test.want {
			t.Errorf("fixLangVersion(%q) = %s, want %s", test.input, got, test.want)
		}
	}
}
