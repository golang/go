// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package semver

import (
	"strings"
	"testing"
)

var tests = []struct {
	in  string
	out string
}{
	{"bad", ""},
	{"v1-alpha.beta.gamma", ""},
	{"v1-pre", ""},
	{"v1+meta", ""},
	{"v1-pre+meta", ""},
	{"v1.2-pre", ""},
	{"v1.2+meta", ""},
	{"v1.2-pre+meta", ""},
	{"v1.0.0-alpha", "v1.0.0-alpha"},
	{"v1.0.0-alpha.1", "v1.0.0-alpha.1"},
	{"v1.0.0-alpha.beta", "v1.0.0-alpha.beta"},
	{"v1.0.0-beta", "v1.0.0-beta"},
	{"v1.0.0-beta.2", "v1.0.0-beta.2"},
	{"v1.0.0-beta.11", "v1.0.0-beta.11"},
	{"v1.0.0-rc.1", "v1.0.0-rc.1"},
	{"v1", "v1.0.0"},
	{"v1.0", "v1.0.0"},
	{"v1.0.0", "v1.0.0"},
	{"v1.2", "v1.2.0"},
	{"v1.2.0", "v1.2.0"},
	{"v1.2.3-456", "v1.2.3-456"},
	{"v1.2.3-456.789", "v1.2.3-456.789"},
	{"v1.2.3-456-789", "v1.2.3-456-789"},
	{"v1.2.3-456a", "v1.2.3-456a"},
	{"v1.2.3-pre", "v1.2.3-pre"},
	{"v1.2.3-pre+meta", "v1.2.3-pre"},
	{"v1.2.3-pre.1", "v1.2.3-pre.1"},
	{"v1.2.3-zzz", "v1.2.3-zzz"},
	{"v1.2.3", "v1.2.3"},
	{"v1.2.3+meta", "v1.2.3"},
	{"v1.2.3+meta-pre", "v1.2.3"},
}

func TestIsValid(t *testing.T) {
	for _, tt := range tests {
		ok := IsValid(tt.in)
		if ok != (tt.out != "") {
			t.Errorf("IsValid(%q) = %v, want %v", tt.in, ok, !ok)
		}
	}
}

func TestCanonical(t *testing.T) {
	for _, tt := range tests {
		out := Canonical(tt.in)
		if out != tt.out {
			t.Errorf("Canonical(%q) = %q, want %q", tt.in, out, tt.out)
		}
	}
}

func TestMajor(t *testing.T) {
	for _, tt := range tests {
		out := Major(tt.in)
		want := ""
		if i := strings.Index(tt.out, "."); i >= 0 {
			want = tt.out[:i]
		}
		if out != want {
			t.Errorf("Major(%q) = %q, want %q", tt.in, out, want)
		}
	}
}

func TestMajorMinor(t *testing.T) {
	for _, tt := range tests {
		out := MajorMinor(tt.in)
		var want string
		if tt.out != "" {
			want = tt.in
			if i := strings.Index(want, "+"); i >= 0 {
				want = want[:i]
			}
			if i := strings.Index(want, "-"); i >= 0 {
				want = want[:i]
			}
			switch strings.Count(want, ".") {
			case 0:
				want += ".0"
			case 1:
				// ok
			case 2:
				want = want[:strings.LastIndex(want, ".")]
			}
		}
		if out != want {
			t.Errorf("MajorMinor(%q) = %q, want %q", tt.in, out, want)
		}
	}
}

func TestPrerelease(t *testing.T) {
	for _, tt := range tests {
		pre := Prerelease(tt.in)
		var want string
		if tt.out != "" {
			if i := strings.Index(tt.out, "-"); i >= 0 {
				want = tt.out[i:]
			}
		}
		if pre != want {
			t.Errorf("Prerelease(%q) = %q, want %q", tt.in, pre, want)
		}
	}
}

func TestBuild(t *testing.T) {
	for _, tt := range tests {
		build := Build(tt.in)
		var want string
		if tt.out != "" {
			if i := strings.Index(tt.in, "+"); i >= 0 {
				want = tt.in[i:]
			}
		}
		if build != want {
			t.Errorf("Build(%q) = %q, want %q", tt.in, build, want)
		}
	}
}

func TestCompare(t *testing.T) {
	for i, ti := range tests {
		for j, tj := range tests {
			cmp := Compare(ti.in, tj.in)
			var want int
			if ti.out == tj.out {
				want = 0
			} else if i < j {
				want = -1
			} else {
				want = +1
			}
			if cmp != want {
				t.Errorf("Compare(%q, %q) = %d, want %d", ti.in, tj.in, cmp, want)
			}
		}
	}
}

func TestMax(t *testing.T) {
	for i, ti := range tests {
		for j, tj := range tests {
			max := Max(ti.in, tj.in)
			want := Canonical(ti.in)
			if i < j {
				want = Canonical(tj.in)
			}
			if max != want {
				t.Errorf("Max(%q, %q) = %q, want %q", ti.in, tj.in, max, want)
			}
		}
	}
}

var (
	v1 = "v1.0.0+metadata-dash"
	v2 = "v1.0.0+metadata-dash1"
)

func BenchmarkCompare(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if Compare(v1, v2) != 0 {
			b.Fatalf("bad compare")
		}
	}
}
