// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import "testing"

const arrayIndexPreamble = `package p

type token uint8

const (
	tokenLight token = iota
	tokenDark
	tokenCount
)

`

func TestArrayIndexOutOfBounds(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "guardedReceiver",
			body: `func (t token) String() string {
		names := [...]string{"light", "dark"}
		if t >= tokenCount { return "unknown" }
		return names[t]
	}`,
		},
		{
			name: "offByOneReceiver",
			body: `func (t token) String() string {
		names := [...]string{"light", "dark"}
		if t > tokenCount { return "unknown" }
		return names[t] // want
	}`,
		},
		{
			name: "signedBothBounds",
			body: `func f(i int8) string {
		names := [...]string{"light", "dark"}
		if i < 0 || i >= 2 { return "unknown" }
		return names[i]
	}`,
		},
		{
			name: "signedMissingLowerBound",
			body: `func f(i int8) string {
		names := [...]string{"light", "dark"}
		if i >= 2 { return "unknown" }
		return names[i] // want
	}`,
		},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			for name, kind := range map[string]Kind{"x.go": KindGo, "x.lgo": KindLgo} {
				t.Run(name, func(t *testing.T) {
					expect(t, kind, nil, testSource{name, arrayIndexPreamble + tt.body}, finding{"correctness::array_index_out_of_bounds", Deny})
				})
			}
		})
	}
}
