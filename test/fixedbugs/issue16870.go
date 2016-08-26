// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"log"
	"reflect"
)

func test(got, want interface{}) {
	if !reflect.DeepEqual(got, want) {
		log.Fatalf("got %v, want %v", got, want)
	}
}

func main() {
	var i int
	var ip *int
	var ok interface{}

	// Channel receives.
	c := make(chan int, 1)
	c2 := make(chan int)

	c <- 42
	i, ok = <-c
	test(i, 42)
	test(ok, true)

	c <- 42
	_, ok = <-c
	test(ok, true)

	c <- 42
	select {
	case i, ok = <-c:
		test(i, 42)
		test(ok, true)
	}

	c <- 42
	select {
	case _, ok = <-c:
		test(ok, true)
	}

	c <- 42
	select {
	case i, ok = <-c:
		test(i, 42)
		test(ok, true)
	default:
		log.Fatal("bad select")
	}

	c <- 42
	select {
	case _, ok = <-c:
		test(ok, true)
	default:
		log.Fatal("bad select")
	}

	c <- 42
	select {
	case i, ok = <-c:
		test(i, 42)
		test(ok, true)
	case <-c2:
		log.Fatal("bad select")
	}

	c <- 42
	select {
	case _, ok = <-c:
		test(ok, true)
	case <-c2:
		log.Fatal("bad select")
	}

	close(c)
	i, ok = <-c
	test(i, 0)
	test(ok, false)

	_, ok = <-c
	test(ok, false)

	// Map indexing.
	m := make(map[int]int)

	i, ok = m[0]
	test(i, 0)
	test(ok, false)

	_, ok = m[0]
	test(ok, false)

	m[0] = 42
	i, ok = m[0]
	test(i, 42)
	test(ok, true)

	_, ok = m[0]
	test(ok, true)

	// Type assertions.
	var u interface{}

	i, ok = u.(int)
	test(i, 0)
	test(ok, false)

	ip, ok = u.(*int)
	test(ip, (*int)(nil))
	test(ok, false)

	_, ok = u.(int)
	test(ok, false)

	u = 42
	i, ok = u.(int)
	test(i, 42)
	test(ok, true)

	_, ok = u.(int)
	test(ok, true)

	u = &i
	ip, ok = u.(*int)
	test(ip, &i)
	test(ok, true)

	_, ok = u.(*int)
	test(ok, true)
}
