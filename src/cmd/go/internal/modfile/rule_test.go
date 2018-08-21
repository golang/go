// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfile

import (
	"bytes"
	"fmt"
	"testing"
)

var addRequireTests = []struct {
	in   string
	path string
	vers string
	out  string
}{
	{
		`
		module m
		require x.y/z v1.2.3
		`,
		"x.y/z", "v1.5.6",
		`
		module m
		require x.y/z v1.5.6
		`,
	},
	{
		`
		module m
		require x.y/z v1.2.3
		`,
		"x.y/w", "v1.5.6",
		`
		module m
		require (
			x.y/z v1.2.3
			x.y/w v1.5.6
		)
		`,
	},
	{
		`
		module m
		require x.y/z v1.2.3
		require x.y/q/v2 v2.3.4
		`,
		"x.y/w", "v1.5.6",
		`
		module m
		require x.y/z v1.2.3
		require (
			x.y/q/v2 v2.3.4
			x.y/w v1.5.6
		)
		`,
	},
}

func TestAddRequire(t *testing.T) {
	for i, tt := range addRequireTests {
		t.Run(fmt.Sprintf("#%d", i), func(t *testing.T) {
			f, err := Parse("in", []byte(tt.in), nil)
			if err != nil {
				t.Fatal(err)
			}
			g, err := Parse("out", []byte(tt.out), nil)
			if err != nil {
				t.Fatal(err)
			}
			golden, err := g.Format()
			if err != nil {
				t.Fatal(err)
			}

			if err := f.AddRequire(tt.path, tt.vers); err != nil {
				t.Fatal(err)
			}
			out, err := f.Format()
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(out, golden) {
				t.Errorf("have:\n%s\nwant:\n%s", out, golden)
			}
		})
	}
}
