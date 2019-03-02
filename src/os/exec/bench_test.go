// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"testing"
)

func BenchmarkExecEcho(b *testing.B) {
	b.ReportAllocs()
	path, err := LookPath("echo")
	if err != nil {
		b.Fatalf("could not find echo: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Command(path).Run(); err != nil {
			b.Fatalf("echo: %v", err)
		}
	}
}
