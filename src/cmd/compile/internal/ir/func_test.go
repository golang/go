// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"testing"
)

func TestSplitPkg(t *testing.T) {
	tests := []struct {
		in  string
		pkg string
		sym string
	}{
		{
			in:  "foo.Bar",
			pkg: "foo",
			sym: "Bar",
		},
		{
			in:  "foo/bar.Baz",
			pkg: "foo/bar",
			sym: "Baz",
		},
		{
			in:  "memeqbody",
			pkg: "",
			sym: "memeqbody",
		},
		{
			in:  `example%2ecom.Bar`,
			pkg: `example%2ecom`,
			sym: "Bar",
		},
		{
			// Not a real generated symbol name, but easier to catch the general parameter form.
			in:  `foo.Bar[sync/atomic.Uint64]`,
			pkg: `foo`,
			sym: "Bar[sync/atomic.Uint64]",
		},
		{
			in:  `example%2ecom.Bar[sync/atomic.Uint64]`,
			pkg: `example%2ecom`,
			sym: "Bar[sync/atomic.Uint64]",
		},
		{
			in:  `gopkg.in/yaml%2ev3.Bar[sync/atomic.Uint64]`,
			pkg: `gopkg.in/yaml%2ev3`,
			sym: "Bar[sync/atomic.Uint64]",
		},
		{
			// This one is a real symbol name.
			in:  `foo.Bar[go.shape.struct { sync/atomic._ sync/atomic.noCopy; sync/atomic._ sync/atomic.align64; sync/atomic.v uint64 }]`,
			pkg: `foo`,
			sym: "Bar[go.shape.struct { sync/atomic._ sync/atomic.noCopy; sync/atomic._ sync/atomic.align64; sync/atomic.v uint64 }]",
		},
		{
			in:  `example%2ecom.Bar[go.shape.struct { sync/atomic._ sync/atomic.noCopy; sync/atomic._ sync/atomic.align64; sync/atomic.v uint64 }]`,
			pkg: `example%2ecom`,
			sym: "Bar[go.shape.struct { sync/atomic._ sync/atomic.noCopy; sync/atomic._ sync/atomic.align64; sync/atomic.v uint64 }]",
		},
		{
			in:  `gopkg.in/yaml%2ev3.Bar[go.shape.struct { sync/atomic._ sync/atomic.noCopy; sync/atomic._ sync/atomic.align64; sync/atomic.v uint64 }]`,
			pkg: `gopkg.in/yaml%2ev3`,
			sym: "Bar[go.shape.struct { sync/atomic._ sync/atomic.noCopy; sync/atomic._ sync/atomic.align64; sync/atomic.v uint64 }]",
		},
	}

	for _, tc := range tests {
		t.Run(tc.in, func { t ->
			pkg, sym := splitPkg(tc.in)
			if pkg != tc.pkg {
				t.Errorf("splitPkg(%q) got pkg %q want %q", tc.in, pkg, tc.pkg)
			}
			if sym != tc.sym {
				t.Errorf("splitPkg(%q) got sym %q want %q", tc.in, sym, tc.sym)
			}
		})
	}
}
