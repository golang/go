// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"testing"
	"time"
)

var sink time.Time
var sinkHPT testing.HighPrecisionTime

func BenchmarkTimeNow(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sink = time.Now()
	}
}

func BenchmarkHighPrecisionTimeNow(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkHPT = testing.HighPrecisionTimeNow()
	}
}
