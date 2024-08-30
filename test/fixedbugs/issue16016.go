// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"time"
)

type T struct{}

func (*T) Foo(vals []interface{}) {
	switch v := vals[0].(type) {
	case string:
		_ = v
	}
}

type R struct{ *T }

type Q interface {
	Foo([]interface{})
}

func main() {
	var count = 10000
	if runtime.Compiler == "gccgo" {
		// On targets without split-stack libgo allocates
		// a large stack for each goroutine. On 32-bit
		// systems this test can run out of memory.
		const intSize = 32 << (^uint(0) >> 63) // 32 or 64
		if intSize < 64 {
			count = 100
		}
	}

	var q Q = &R{&T{}}
	for i := 0; i < count; i++ {
		go func() {
			defer q.Foo([]interface{}{"meow"})
			time.Sleep(100 * time.Millisecond)
		}()
	}
	time.Sleep(1 * time.Second)
}
