// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ascii

import "testing"

func TestEqualFold(t *testing.T) {
	var tests = []struct {
		name string
		a, b string
		want bool
	}{
		{
			name: "empty",
			want: true,
		},
		{
			name: "simple match",
			a:    "CHUNKED",
			b:    "chunked",
			want: true,
		},
		{
			name: "same string",
			a:    "chunked",
			b:    "chunked",
			want: true,
		},
		{
			name: "Unicode Kelvin symbol",
			a:    "chunKed", // This "K" is 'KELVIN SIGN' (\u212A)
			b:    "chunked",
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func { t ->
			if got := EqualFold(tt.a, tt.b); got != tt.want {
				t.Errorf("AsciiEqualFold(%q,%q): got %v want %v", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestIsPrint(t *testing.T) {
	var tests = []struct {
		name string
		in   string
		want bool
	}{
		{
			name: "empty",
			want: true,
		},
		{
			name: "ASCII low",
			in:   "This is a space: ' '",
			want: true,
		},
		{
			name: "ASCII high",
			in:   "This is a tilde: '~'",
			want: true,
		},
		{
			name: "ASCII low non-print",
			in:   "This is a unit separator: \x1F",
			want: false,
		},
		{
			name: "Ascii high non-print",
			in:   "This is a Delete: \x7F",
			want: false,
		},
		{
			name: "Unicode letter",
			in:   "Today it's 280K outside: it's freezing!", // This "K" is 'KELVIN SIGN' (\u212A)
			want: false,
		},
		{
			name: "Unicode emoji",
			in:   "Gophers like 🧀",
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func { t ->
			if got := IsPrint(tt.in); got != tt.want {
				t.Errorf("IsASCIIPrint(%q): got %v want %v", tt.in, got, tt.want)
			}
		})
	}
}
