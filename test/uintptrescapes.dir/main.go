// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"sync"
	"unsafe"

	"./a"
)

func F1() int {
	var buf [1024]int
	a.F1(uintptr(unsafe.Pointer(&buf[0])))
	return buf[0]
}

func F2() int {
	var buf [1024]int
	a.F2(uintptr(unsafe.Pointer(&buf[0])))
	return buf[0]
}

var t = a.GetT()

func M1() int {
	var buf [1024]int
	t.M1(uintptr(unsafe.Pointer(&buf[0])))
	return buf[0]
}

func M2() int {
	var buf [1024]int
	t.M2(uintptr(unsafe.Pointer(&buf[0])))
	return buf[0]
}

func main() {
	// Use different goroutines to force stack growth.
	var wg sync.WaitGroup
	wg.Add(4)
	c := make(chan bool, 4)

	go func() {
		defer wg.Done()
		b := F1()
		if b != 42 {
			fmt.Printf("F1: got %d, expected 42\n", b)
			c <- false
		}
	}()

	go func() {
		defer wg.Done()
		b := F2()
		if b != 42 {
			fmt.Printf("F2: got %d, expected 42\n", b)
			c <- false
		}
	}()

	go func() {
		defer wg.Done()
		b := M1()
		if b != 42 {
			fmt.Printf("M1: got %d, expected 42\n", b)
			c <- false
		}
	}()

	go func() {
		defer wg.Done()
		b := M2()
		if b != 42 {
			fmt.Printf("M2: got %d, expected 42\n", b)
			c <- false
		}
	}()

	wg.Wait()

	select {
	case <-c:
		os.Exit(1)
	default:
	}
}
