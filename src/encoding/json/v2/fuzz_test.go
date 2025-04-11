// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"bytes"
	"testing"
)

func FuzzEqualFold(f *testing.F) {
	for _, tt := range equalFoldTestdata {
		f.Add([]byte(tt.in1), []byte(tt.in2))
	}

	equalFoldSimple := func(x, y []byte) bool {
		strip := func(b []byte) []byte {
			return bytes.Map(func(r rune) rune {
				if r == '_' || r == '-' {
					return -1 // ignore underscores and dashes
				}
				return r
			}, b)
		}
		return bytes.EqualFold(strip(x), strip(y))
	}

	f.Fuzz(func(t *testing.T, s1, s2 []byte) {
		// Compare the optimized and simplified implementations.
		got := equalFold(s1, s2)
		want := equalFoldSimple(s1, s2)
		if got != want {
			t.Errorf("equalFold(%q, %q) = %v, want %v", s1, s2, got, want)
		}
	})
}
