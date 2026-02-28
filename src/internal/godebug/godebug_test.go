// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godebug

import "testing"

func TestGet(t *testing.T) {
	tests := []struct {
		godebug string
		key     string
		want    string
	}{
		{"", "", ""},
		{"", "foo", ""},
		{"foo=bar", "foo", "bar"},
		{"foo=bar,after=x", "foo", "bar"},
		{"before=x,foo=bar,after=x", "foo", "bar"},
		{"before=x,foo=bar", "foo", "bar"},
		{",,,foo=bar,,,", "foo", "bar"},
		{"foodecoy=wrong,foo=bar", "foo", "bar"},
		{"foo=", "foo", ""},
		{"foo", "foo", ""},
		{",foo", "foo", ""},
		{"foo=bar,baz", "loooooooong", ""},
	}
	for _, tt := range tests {
		got := get(tt.godebug, tt.key)
		if got != tt.want {
			t.Errorf("get(%q, %q) = %q; want %q", tt.godebug, tt.key, got, tt.want)
		}
	}
}
