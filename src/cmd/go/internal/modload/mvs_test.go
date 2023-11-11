// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"testing"
)

func TestReqsMax(t *testing.T) {
	type testCase struct {
		a, b, want string
	}
	reqs := new(mvsReqs)
	for _, tc := range []testCase{
		{a: "v0.1.0", b: "v0.2.0", want: "v0.2.0"},
		{a: "v0.2.0", b: "v0.1.0", want: "v0.2.0"},
		{a: "", b: "v0.1.0", want: ""}, // "" is Target.Version
		{a: "v0.1.0", b: "", want: ""},
		{a: "none", b: "v0.1.0", want: "v0.1.0"},
		{a: "v0.1.0", b: "none", want: "v0.1.0"},
		{a: "none", b: "", want: ""},
		{a: "", b: "none", want: ""},
	} {
		max := reqs.Max("", tc.a, tc.b)
		if max != tc.want {
			t.Errorf("(%T).Max(%q, %q) = %q; want %q", reqs, tc.a, tc.b, max, tc.want)
		}
	}
}
