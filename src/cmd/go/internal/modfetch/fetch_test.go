// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"io/ioutil"
	"path/filepath"
	"testing"

	"golang.org/x/mod/module"
)

func BenchmarkReadGoSum(b *testing.B) {
	var testGoSum = "testdata/go.sum"
	data, err := ioutil.ReadFile(filepath.FromSlash(testGoSum))
	if len(data) == 0 || err != nil {
		b.Fatalf("Failed to read test file: %s, err: %v", testGoSum, err)
	}

	var goSum map[module.Version][]string
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		goSum = make(map[module.Version][]string)
		if err := readGoSum(goSum, testGoSum, data); err != nil {
			b.Fatalf("unexpected error happends when reading go sum: %v", err)
		}
	}
}
