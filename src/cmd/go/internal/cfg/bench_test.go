// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"cmd/internal/pathcache"
	"internal/testenv"
	"testing"
)

func BenchmarkLookPath(b *testing.B) {
	testenv.MustHaveExecPath(b, "go")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := pathcache.LookPath("go")
		if err != nil {
			b.Fatal(err)
		}
	}
}
