// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"runtime"
	"sync/atomic"
	"weak"
)

type point struct {
	v int64
	w int64
}

var dynN = 64

var finalized atomic.Int32

func finalizer(*point) {
	finalized.Store(1)
}
func checkArenaGC() int {
	n := dynN
	s := make([]*point, n)
	for i := range s {
		p := &point{v: int64(i*1000 + 7), w: int64(i*1000 + 13)}
		s[i] = p
	}
	ptrs := make([]weak.Pointer[point], n)
	for i, p := range s {
		ptrs[i] = weak.Make(p)
	}
	for i := 0; i < n; i += 8 {
		runtime.SetFinalizer(s[i], finalizer)
	}
	for range 3 {
		runtime.GC()
	}
	for range 1000 {
		if finalized.Load() != 0 {
			break
		}
		runtime.Gosched()
	}
	if finalized.Load() != 0 {
		fmt.Println("LAW finalizer ran while reachable")
		return 1
	}
	for i, w := range ptrs {
		if w.Value() == nil {
			fmt.Println("LAW weak nil while reachable at", i)
			return 1
		}
	}
	var sum int64
	for i, p := range s {
		if p.v != int64(i*1000+7) || p.w != int64(i*1000+13) {
			fmt.Println("LAW pointee changed at", i)
			return 1
		}
		sum += p.v
	}
	fmt.Println("arena_gc ok", sum)
	runtime.KeepAlive(s)
	return 0
}

func main() {
	os.Exit(checkArenaGC())
}
