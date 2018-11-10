// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
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
			buf := make([]byte, 997)
			for j := 0; j < numCycles; j++ {
				var seed int64
				seed += int64(ExpFloat64())
				seed += int64(Float32())
				seed += int64(Float64())
				seed += int64(Intn(Int()))
				seed += int64(Int31n(Int31()))
				seed += int64(Int63n(Int63()))
				seed += int64(NormFloat64())
				seed += int64(Uint32())
				seed += int64(Uint64())
				for _, p := range Perm(10) {
					seed += int64(p)
				}
				Read(buf)
				for _, b := range buf {
					seed += int64(b)
				}
				Seed(int64(i*j) * seed)
			}
		}(i)
	}
}
