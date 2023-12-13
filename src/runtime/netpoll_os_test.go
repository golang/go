// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"sync"
	"testing"
)

var wg sync.WaitGroup

func init() {
	runtime.NetpollGenericInit()
}

func BenchmarkNetpollBreak(b *testing.B) {
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10; j++ {
			wg.Add(1)
			go func() {
				runtime.NetpollBreak()
				wg.Done()
			}()
		}
	}
	wg.Wait()
	b.StopTimer()
}
