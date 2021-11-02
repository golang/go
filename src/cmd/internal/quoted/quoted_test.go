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
		value   string
		want    []string
		wantErr string
	}{
		{name: "empty", value: "", want: nil},
		{name: "space", value: " ", want: nil},
		{name: "one", value: "a", want: []string{"a"}},
		{name: "leading_space", value: " a", want: []string{"a"}},
		{name: "trailing_space", value: "a ", want: []string{"a"}},
		{name: "two", value: "a b", want: []string{"a", "b"}},
		{name: "two_multi_space", value: "a  b", want: []string{"a", "b"}},
		{name: "two_tab", value: "a\tb", want: []string{"a", "b"}},
		{name: "two_newline", value: "a\nb", want: []string{"a", "b"}},
		{name: "quote_single", value: `'a b'`, want: []string{"a b"}},
		{name: "quote_double", value: `"a b"`, want: []string{"a b"}},
		{name: "quote_both", value: `'a '"b "`, want: []string{"a ", "b "}},
		{name: "quote_contains", value: `'a "'"'b"`, want: []string{`a "`, `'b`}},
		{name: "escape", value: `\'`, want: []string{`\'`}},
		{name: "quote_unclosed", value: `'a`, wantErr: "unterminated ' string"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := Split(test.value)
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
				t.Errorf("got %q; want %q", got, test.want)
			}
		})
	}
}

func TestJoin(t *testing.T) {
	for _, test := range []struct {
		name          string
		args          []string
		want, wantErr string
	}{
		{name: "empty", args: nil, want: ""},
		{name: "one", args: []string{"a"}, want: "a"},
		{name: "two", args: []string{"a", "b"}, want: "a b"},
		{name: "space", args: []string{"a ", "b"}, want: "'a ' b"},
		{name: "newline", args: []string{"a\n", "b"}, want: "'a\n' b"},
		{name: "quote", args: []string{`'a `, "b"}, want: `"'a " b`},
		{name: "unquoteable", args: []string{`'"`}, wantErr: "contains both single and double quotes and cannot be quoted"},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := Join(test.args)
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
				t.Errorf("got %s; want %s", got, test.want)
			}
		})
	}
}
