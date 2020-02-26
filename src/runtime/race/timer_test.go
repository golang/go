// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

package race_test

import (
	"sync"
	"testing"
	"time"
)

func TestTimers(t *testing.T) {
	const goroutines = 8
	var wg sync.WaitGroup
	wg.Add(goroutines)
	var mu sync.Mutex
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			ticker := time.NewTicker(1)
			defer ticker.Stop()
			for c := 0; c < 1000; c++ {
				<-ticker.C
				mu.Lock()
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
}
