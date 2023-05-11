// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.16
// +build go1.16

package cache

import (
	"testing"
)

func TestIsStandaloneFile(t *testing.T) {
	tests := []struct {
		desc           string
		contents       string
		standaloneTags []string
		want           bool
	}{
		{
			"new syntax",
			"//go:build ignore\n\npackage main\n",
			[]string{"ignore"},
			true,
		},
		{
			"legacy syntax",
			"// +build ignore\n\npackage main\n",
			[]string{"ignore"},
			true,
		},
		{
			"multiple tags",
			"//go:build ignore\n\npackage main\n",
			[]string{"exclude", "ignore"},
			true,
		},
		{
			"invalid tag",
			"// +build ignore\n\npackage main\n",
			[]string{"script"},
			false,
		},
		{
			"non-main package",
			"//go:build ignore\n\npackage p\n",
			[]string{"ignore"},
			false,
		},
		{
			"alternate tag",
			"// +build script\n\npackage main\n",
			[]string{"script"},
			true,
		},
		{
			"both syntax",
			"//go:build ignore\n// +build ignore\n\npackage main\n",
			[]string{"ignore"},
			true,
		},
		{
			"after comments",
			"// A non-directive comment\n//go:build ignore\n\npackage main\n",
			[]string{"ignore"},
			true,
		},
		{
			"after package decl",
			"package main //go:build ignore\n",
			[]string{"ignore"},
			false,
		},
		{
			"on line after package decl",
			"package main\n\n//go:build ignore\n",
			[]string{"ignore"},
			false,
		},
		{
			"combined with other expressions",
			"\n\n//go:build ignore || darwin\n\npackage main\n",
			[]string{"ignore"},
			false,
		},
	}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if got := isStandaloneFile([]byte(test.contents), test.standaloneTags); got != test.want {
				t.Errorf("isStandaloneFile(%q, %v) = %t, want %t", test.contents, test.standaloneTags, got, test.want)
			}
		})
	}
}
