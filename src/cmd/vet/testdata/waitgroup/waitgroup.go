// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the waitgroup checker.

package waitgroup

import "sync"

func _() {
	var wg *sync.WaitGroup
	wg.Add(1)
	go func() {
		wg.Add(1) // ERROR "WaitGroup.Add called from inside new goroutine"
		defer wg.Done()
		// ...
	}()
	wg.Wait()
}
