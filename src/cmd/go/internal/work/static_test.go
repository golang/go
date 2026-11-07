// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"slices"
	"testing"
)

func TestAddStaticExtldflag(t *testing.T) {
	for _, tt := range []struct {
		ldflags []string
		want    []string
	}{
		{nil, []string{"-extldflags=-static"}},
		{[]string{"-s", "-w"}, []string{"-s", "-w", "-extldflags=-static"}},

		// The flags that the user gave are kept.
		{[]string{"-extldflags=-pthread"}, []string{"-extldflags=-pthread -static"}},
		{[]string{"--extldflags=-pthread"}, []string{"--extldflags=-pthread -static"}},
		{[]string{"-extldflags", "-pthread -lm", "-s"}, []string{"-extldflags", "-pthread -lm -static", "-s"}},
		{[]string{"-extldflags="}, []string{"-extldflags=-static"}},
		{[]string{"-extldflags", ""}, []string{"-extldflags", "-static"}},
		{[]string{"-extldflags", `-L "/some dir"`}, []string{"-extldflags", `-L "/some dir" -static`}},

		// The linker uses the last instance.
		{[]string{"-extldflags=-lm", "-extldflags=-pthread"}, []string{"-extldflags=-lm", "-extldflags=-pthread -static"}},
		{[]string{"-extldflags=-static", "-extldflags", "-pthread"}, []string{"-extldflags=-static", "-extldflags", "-pthread -static"}},

		// -static is not added twice.
		{[]string{"-extldflags=-static"}, []string{"-extldflags=-static"}},
		{[]string{"-extldflags", "-static -pthread"}, []string{"-extldflags", "-static -pthread"}},

		// Other flags that mention -extldflags are not it.
		{[]string{"-X", "main.flag=-extldflags"}, []string{"-X", "main.flag=-extldflags", "-extldflags=-static"}},
		{[]string{"-X=main.flag=-extldflags=x"}, []string{"-X=main.flag=-extldflags=x", "-extldflags=-static"}},

		// A value that the linker will reject is left for it to report.
		{[]string{"-s", "-extldflags"}, []string{"-s", "-extldflags"}},
		{[]string{"-extldflags", `"-pthread`}, []string{"-extldflags", `"-pthread`}},
	} {
		got := addStaticExtldflag(slices.Clone(tt.ldflags))
		if !slices.Equal(got, tt.want) {
			t.Errorf("addStaticExtldflag(%q) = %q, want %q", tt.ldflags, got, tt.want)
		}
	}
}
