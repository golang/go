// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godebug_test

import (
	. "internal/godebug"
	"testing"
)

func TestGet(t *testing.T) {
	foo := New("foo")
	tests := []struct {
		godebug string
		setting *Setting
		want    string
	}{
		{"", New(""), ""},
		{"", foo, ""},
		{"foo=bar", foo, "bar"},
		{"foo=bar,after=x", foo, "bar"},
		{"before=x,foo=bar,after=x", foo, "bar"},
		{"before=x,foo=bar", foo, "bar"},
		{",,,foo=bar,,,", foo, "bar"},
		{"foodecoy=wrong,foo=bar", foo, "bar"},
		{"foo=", foo, ""},
		{"foo", foo, ""},
		{",foo", foo, ""},
		{"foo=bar,baz", New("loooooooong"), ""},
	}
	for _, tt := range tests {
		t.Setenv("GODEBUG", tt.godebug)
		got := tt.setting.Value()
		if got != tt.want {
			t.Errorf("get(%q, %q) = %q; want %q", tt.godebug, tt.setting.Name(), got, tt.want)
		}
	}
}
