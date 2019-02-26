// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"testing"
)

var notaryShouldVerifyTests = []struct {
	modPath    string
	GONOVERIFY string
	result     int // -1 = bad GONOVERIFY, 0 = wantNotary=false, 1 = wantNotary=true
}{
	{"anything", "off", 0},
	{"anything", "", 1},
	{"anything", ",", 1},
	{"anything", ",foo,", 1},
	{"anything", "[malformed", -1},
	{"anything", "malformed[", 1},
	{"my.corp.example.com", "*.[c]orp.*", 0},
	{"my.corp.example.com/foo", "*.c[^a]rp.*", 0},
	{"my.corp.example.com", "*.corp.*,bar.com", 0},
	{"my.corp.example.com/foo", "*.corp.*,bar.com", 0},
	{"my.corp.example.com", "bar.com,*.corp.*", 0},
	{"my.corp.example.com/foo", "bar.com,*.corp.*", 0},
	{"bar.com", "*.corp.*", 1},
	{"bar.com/foo", "*.corp.*", 1},
	{"bar.com", "*.corp.*,bar.com", 0},
	{"bar.com/foo", "*.corp.*,bar.com", 0},
	{"bar.com", "bar.com,*.corp.*", 0},
	{"bar.com/foo", "bar.com,*.corp.*", 0},
}

func TestNotaryShouldVerify(t *testing.T) {
	for _, tt := range notaryShouldVerifyTests {
		wantNotary, err := notaryShouldVerify(tt.modPath, tt.GONOVERIFY)
		if wantNotary != (tt.result > 0) || (err != nil) != (tt.result < 0) {
			wantErr := "nil"
			if tt.result < 0 {
				wantErr = "non-nil error"
			}
			t.Errorf("notaryShouldVerify(%q, %q) = %v, %v, want %v, %s", tt.modPath, tt.GONOVERIFY, wantNotary, err, tt.result > 0, wantErr)
		}
	}
}
