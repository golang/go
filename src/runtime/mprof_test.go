// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"testing"
)

func BenchmarkBlocksampled(b *testing.B) {
	for b.Loop() {
		Blocksampled(42, 1337)
	}
}
