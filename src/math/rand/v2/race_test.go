// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	. "math/rand/v2"
	"sync"
	"testing"
)

// TestConcurrent exercises the rand API concurrently, triggering situations
// where the race detector is likely to detect issues.
func TestConcurrent(t *testing.T) {
	const (
		numRoutines = 10
		numCycles   = 10
	)
	var wg sync.WaitGroup
	defer wg.Wait()
	wg.Add(numRoutines)
	for i := 0; i < numRoutines; i++ {
		go func(i int) {
			defer wg.Done()
			var seed int64
			for j := 0; j < numCycles; j++ {
				seed += int64(ExpFloat64())
				seed += int64(Float32())
				seed += int64(Float64())
				seed += int64(IntN(Int()))
				seed += int64(Int32N(Int32()))
				seed += int64(Int64N(Int64()))
				seed += int64(NormFloat64())
				seed += int64(Uint32())
				seed += int64(Uint64())
				for _, p := range Perm(10) {
					seed += int64(p)
				}
			}
			_ = seed
		}(i)
	}
}
