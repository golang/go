// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package constraint

import (
	"fmt"
	"testing"
)

var tests = []struct {
	in  string
	out int
}{
	{"//go:build linux && go1.60", 60},
	{"//go:build ignore && go1.60", 60},
	{"//go:build ignore || go1.60", -1},
	{"//go:build go1.50 || (ignore && go1.60)", 50},
	{"// +build go1.60,linux", 60},
	{"// +build go1.60 linux", -1},
	{"//go:build go1.50 && !go1.60", 50},
	{"//go:build !go1.60", -1},
	{"//go:build linux && go1.50 || darwin && go1.60", 50},
	{"//go:build linux && go1.50 || !(!darwin || !go1.60)", 50},
}

func TestGoVersion(t *testing.T) {
	for _, tt := range tests {
		x, err := Parse(tt.in)
		if err != nil {
			t.Fatal(err)
		}
		v := GoVersion(x)
		want := ""
		if tt.out == 0 {
			want = "go1"
		} else if tt.out > 0 {
			want = fmt.Sprintf("go1.%d", tt.out)
		}
		if v != want {
			t.Errorf("GoVersion(%q) = %q, want %q, nil", tt.in, v, want)
		}
	}
}
