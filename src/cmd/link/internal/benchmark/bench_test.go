// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package benchmark

import (
	"testing"
)

func TestMakeBenchString(t *testing.T) {
	tests := []struct {
		have, want string
	}{
		{"foo", "BenchmarkFoo"},
		{"  foo  ", "BenchmarkFoo"},
		{"foo bar", "BenchmarkFooBar"},
	}
	for i, test := range tests {
		if v := makeBenchString(test.have); test.want != v {
			t.Errorf("test[%d] makeBenchString(%q) == %q, want %q", i, test.have, v, test.want)
		}
	}
}

func TestPProfFlag(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"", false},
		{"foo", true},
	}
	for i, test := range tests {
		b := New(GC, test.name)
		if v := b.shouldPProf(); test.want != v {
			t.Errorf("test[%d] shouldPProf() == %v, want %v", i, v, test.want)
		}
	}
}

func TestPProfNames(t *testing.T) {
	want := "foo_BenchmarkTest.cpuprof"
	if v := makePProfFilename("foo", "test", "cpuprof"); v != want {
		t.Errorf("makePProfFilename() == %q, want %q", v, want)
	}
}

// Ensure that public APIs work with a nil Metrics object.
func TestNilBenchmarkObject(t *testing.T) {
	var b *Metrics
	b.Start("TEST")
	b.Report(nil)
}
