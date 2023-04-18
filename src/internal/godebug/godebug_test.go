// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godebug_test

import (
	. "internal/godebug"
	"runtime/metrics"
	"testing"
)

func TestGet(t *testing.T) {
	foo := New("#foo")
	tests := []struct {
		godebug string
		setting *Setting
		want    string
	}{
		{"", New("#"), ""},
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
		{"foo=bar,baz", New("#loooooooong"), ""},
	}
	for _, tt := range tests {
		t.Setenv("GODEBUG", tt.godebug)
		got := tt.setting.Value()
		if got != tt.want {
			t.Errorf("get(%q, %q) = %q; want %q", tt.godebug, tt.setting.Name(), got, tt.want)
		}
	}
}

func TestMetrics(t *testing.T) {
	const name = "http2client" // must be a real name so runtime will accept it

	var m [1]metrics.Sample
	m[0].Name = "/godebug/non-default-behavior/" + name + ":events"
	metrics.Read(m[:])
	if kind := m[0].Value.Kind(); kind != metrics.KindUint64 {
		t.Fatalf("NonDefault kind = %v, want uint64", kind)
	}

	s := New(name)
	s.Value()
	s.IncNonDefault()
	s.IncNonDefault()
	s.IncNonDefault()
	metrics.Read(m[:])
	if kind := m[0].Value.Kind(); kind != metrics.KindUint64 {
		t.Fatalf("NonDefault kind = %v, want uint64", kind)
	}
	if count := m[0].Value.Uint64(); count != 3 {
		t.Fatalf("NonDefault value = %d, want 3", count)
	}
}
