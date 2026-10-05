// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import "testing"

func TestIsGoType(t *testing.T) {
	for _, tc := range []struct {
		name string
		want bool
	}{
		{"type:int", true},
		{"type:main.Info", true},
		{"type:*main.Info", true},
		{"type:.eqfunc.EE", false},
		{"type:.hashfunc.8", false},
		{"type:.namedata.foo", false},
		{"type:", false},
		{"type", false},
		{"go:itab.main.T,main.I", false},
		{"main.f", false},
		{"", false},
	} {
		s := &LSym{Name: tc.name}
		if got := s.IsGoType(); got != tc.want {
			t.Errorf("(&LSym{Name: %q}).IsGoType() = %v, want %v", tc.name, got, tc.want)
		}
	}
}
