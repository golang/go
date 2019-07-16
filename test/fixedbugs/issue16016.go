// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "time"

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
	var q Q = &R{&T{}}
	for i := 0; i < 10000; i++ {
		go func() {
			defer q.Foo([]interface{}{"meow"})
			time.Sleep(100 * time.Millisecond)
		}()
	}
	time.Sleep(1 * time.Second)
}
