// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

func init() {
	register("concurrentMapWrites", concurrentMapWrites)
	register("concurrentMapReadWrite", concurrentMapReadWrite)
	register("concurrentMapIterateWrite", concurrentMapIterateWrite)
}

func concurrentMapWrites() {
	m := map[int]int{}
	c := make(chan struct{})
	go func() {
		for i := 0; i < 10000; i++ {
			m[5] = 0
			runtime.Gosched()
		}
		c <- struct{}{}
	}()
	go func() {
		for i := 0; i < 10000; i++ {
			m[6] = 0
			runtime.Gosched()
		}
		c <- struct{}{}
	}()
	<-c
	<-c
}

func concurrentMapReadWrite() {
	m := map[int]int{}
	c := make(chan struct{})
	go func() {
		for i := 0; i < 10000; i++ {
			m[5] = 0
			runtime.Gosched()
		}
		c <- struct{}{}
	}()
	go func() {
		for i := 0; i < 10000; i++ {
			_ = m[6]
			runtime.Gosched()
		}
		c <- struct{}{}
	}()
	<-c
	<-c
}

func concurrentMapIterateWrite() {
	m := map[int]int{}
	c := make(chan struct{})
	go func() {
		for i := 0; i < 10000; i++ {
			m[5] = 0
			runtime.Gosched()
		}
		c <- struct{}{}
	}()
	go func() {
		for i := 0; i < 10000; i++ {
			for range m {
			}
			runtime.Gosched()
		}
		c <- struct{}{}
	}()
	<-c
	<-c
}
