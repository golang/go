// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import "testing"

func TestResetRestoresAllImplementations(t *testing.T) {
	defer func(saved []implementation) { allImplementations = saved }(allImplementations)

	first, second, unavailable := true, true, false
	Register("testpkg", "First", &first)
	Register("testpkg", "Second", &second)
	Register("testpkg", "Unavailable", &unavailable)

	if !Select("testpkg", "First") {
		t.Fatal("Select reported an available implementation as unavailable")
	}
	if second {
		t.Error("Select left another implementation enabled")
	}
	Select("testpkg", "")

	Reset("testpkg")
	for _, tc := range []struct {
		name string
		got  bool
		want bool
	}{
		{"First", first, true},
		{"Second", second, true},
		{"Unavailable", unavailable, false},
	} {
		if tc.got != tc.want {
			t.Errorf("after Reset, %s is %v, want %v", tc.name, tc.got, tc.want)
		}
	}
}
