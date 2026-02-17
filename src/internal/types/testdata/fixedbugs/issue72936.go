// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[C chan<- int | chan int](c C)   { c <- 0 }
func _[C chan int | chan<- int](c C)   { c <- 0 }
func _[C <-chan int | chan<- int](c C) { c <- /* ERROR "receive-only channel <-chan int" */ 0 }

func _[C <-chan int | chan int](c C)   { <-c }
func _[C chan int | <-chan int](c C)   { <-c }
func _[C chan<- int | <-chan int](c C) { <-c /* ERROR "send-only channel chan<- int" */ }

// from issue report

func send[C interface{ ~chan<- V | ~chan V }, V any](c C, v V) {
	c <- v
}

func receive[C interface{ ~<-chan V | ~chan V }, V any](c C) V {
	return <-c
}
