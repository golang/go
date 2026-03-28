// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quoted

import (
	"reflect"
	"strings"
	"testing"
)

func TestSplit(t *testing.T) {
	for _, test := range []struct {
		name    string
		in      string
		want    []string
		wantErr string
	}{
		{name: "empty", in: "", want: nil},
		{name: "space", in: " ", want: nil},
		{name: "one", in: "a", want: []string{"a"}},
		{name: "leading_space", in: " a", want: []string{"a"}},
		{name: "trailing_space", in: "a ", want: []string{"a"}},
		{name: "two", in: "a b", want: []string{"a", "b"}},
		{name: "two_multi_space", in: "a  b", want: []string{"a", "b"}},
		{name: "two_tab", in: "a\tb", want: []string{"a", "b"}},
		{name: "two_newline", in: "a\nb", want: []string{"a", "b"}},
		{name: "quote_single", in: `'a b'`, want: []string{"a b"}},
		{name: "quote_double", in: `"a b"`, want: []string{"a b"}},
		{name: "quote_both", in: `'a '"b "`, want: []string{"a ", "b "}},
		{name: "quote_contains", in: `'a "'"'b"`, want: []string{`a "`, `'b`}},
		{name: "escape", in: `\'`, want: []string{`\'`}},
		{name: "quote_unclosed", in: `'a`, wantErr: "unterminated ' string"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := Split(test.in)
			if err != nil {
				if test.wantErr == "" {
					t.Fatalf("unexpected error: %v", err)
				} else if errMsg := err.Error(); !strings.Contains(errMsg, test.wantErr) {
					t.Fatalf("error %q does not contain %q", errMsg, test.wantErr)
				}
				return
			}
			if test.wantErr != "" {
				t.Fatalf("unexpected success; wanted error containing %q", test.wantErr)
			}
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("Split(%q) = %q, want %q", test.in, got, test.want)
			}
		})
	}
}

func TestJoin(t *testing.T) {
	for _, test := range []struct {
		name          string
		in            []string
		want, wantErr string
	}{
		{name: "empty", in: nil, want: ""},
		{name: "one", in: []string{"a"}, want: "a"},
		{name: "two", in: []string{"a", "b"}, want: "a b"},
		{name: "space", in: []string{"a ", "b"}, want: "'a ' b"},
		{name: "newline", in: []string{"a\n", "b"}, want: "'a\n' b"},
		{name: "quote", in: []string{`'a `, "b"}, want: `"'a " b`},
		{name: "unquoteable", in: []string{`'"`}, wantErr: "contains both single and double quotes and cannot be quoted"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := Join(test.in)
			if err != nil {
				if test.wantErr == "" {
					t.Fatalf("unexpected error: %v", err)
				} else if errMsg := err.Error(); !strings.Contains(errMsg, test.wantErr) {
					t.Fatalf("error %q does not contain %q", errMsg, test.wantErr)
				}
				return
			}
			if test.wantErr != "" {
				t.Fatalf("unexpected success; wanted error containing %q", test.wantErr)
			}
			if got != test.want {
				t.Errorf("Join(%v) = %s, want %s", test.in, got, test.want)
			}
		})
	}
}
