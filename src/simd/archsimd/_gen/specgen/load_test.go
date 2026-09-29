// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import "testing"

func BenchmarkLoad(b *testing.B) {
	for b.Loop() {
		funcs, err := Load("../../../internal/spec", &LoadOptions{})
		if err != nil {
			b.Fatalf("Load failed: %s", err)
		}
		if len(funcs) < 100 {
			b.Fatalf("Load only loaded %d Funcs, expect >= 100", len(funcs))
		}
	}
}
