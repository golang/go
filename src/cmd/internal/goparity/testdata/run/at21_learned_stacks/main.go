// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"sync"
	"sync/atomic"
)

type small struct {
	a int64
	b int64
	c int64
	d int64
}

const (
	shallowBatches = 9
	batchSize      = 100
	deepTotal      = 100
	heapTotal      = 512
	deepFrames     = 2000
)

func fillFrame(buf *[1024]byte, seed byte) {
	for i := range buf {
		buf[i] = seed ^ byte(i)
	}
}

func checkFrame(buf *[1024]byte, seed byte) bool {
	for i := range buf {
		if buf[i] != seed^byte(i) {
			return false
		}
	}
	return true
}

func deep(n int, seed byte) bool {
	var buf [1024]byte
	fillFrame(&buf, seed)
	ok := true
	if n > 0 {
		ok = deep(n-1, seed+1)
	}
	return ok && checkFrame(&buf, seed)
}

func main() {
	heap := make([]small, heapTotal)
	for i := range heap {
		heap[i] = small{int64(i*3 + 1), int64(i*5 + 2), int64(i*7 + 3), int64(i*11 + 4)}
	}
	var before int64
	for _, h := range heap {
		before += h.a + h.b + h.c + h.d
	}
	var wg sync.WaitGroup
	var bad atomic.Int32
	for range shallowBatches {
		for range batchSize {
			wg.Go(func() {
				var buf [512]byte
				for j := range buf {
					buf[j] = byte(j)
				}
				var s int64
				for j := range buf {
					s += int64(buf[j])
				}
				if s < 0 {
					bad.Add(1)
				}
			})
		}
		wg.Wait()
	}
	for range deepTotal {
		wg.Go(func() {
			if !deep(deepFrames, 1) {
				bad.Add(1)
			}
		})
	}
	wg.Wait()
	var after int64
	for _, h := range heap {
		after += h.a + h.b + h.c + h.d
	}
	if before != after {
		fmt.Println("LAW heap changed")
		os.Exit(1)
	}
	if bad.Load() != 0 {
		fmt.Println("LAW frames changed")
		os.Exit(1)
	}
	fmt.Println("ok")
}
