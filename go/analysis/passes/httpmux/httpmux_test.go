// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httpmux

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
)

func Test(t *testing.T) {
	testdata := analysistest.TestData()
	tests := []string{"a"}
	inTest = true
	analysistest.Run(t, testdata, Analyzer, tests...)
}

func TestGoVersion(t *testing.T) {
	for _, test := range []struct {
		in   string
		want bool
	}{
		{"", true},
		{"go1", false},
		{"go1.21", false},
		{"go1.21rc3", false},
		{"go1.22", true},
		{"go1.22rc1", true},
	} {
		got := goVersionAfter121(test.in)
		if got != test.want {
			t.Errorf("%q: got %t, want %t", test.in, got, test.want)
		}
	}
}
