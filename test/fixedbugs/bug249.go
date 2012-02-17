// errorcheck

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var c1 chan <- chan int = (chan<- (chan int))(nil)
var c2 chan <- chan int = (chan (<-chan int))(nil)  // ERROR "chan|incompatible"
var c3 <- chan chan int = (<-chan (chan int))(nil)
var c4 chan chan <- int = (chan (chan<- int))(nil)

var c5 <- chan <- chan int = (<-chan (<-chan int))(nil)
var c6 chan <- <- chan int = (chan<- (<-chan int))(nil)
var c7 chan <- chan <- int = (chan<- (chan<- int))(nil)

var c8 <- chan <- chan chan int = (<-chan (<-chan (chan int)))(nil)
var c9 <- chan chan <- chan int = (<-chan (chan<- (chan int)))(nil)
var c10 chan <- <- chan chan int = (chan<- (<-chan (chan int)))(nil)
var c11 chan <- chan <- chan int = (chan<- (chan<- (chan int)))(nil)
var c12 chan chan <- <- chan int = (chan (chan<- (<-chan int)))(nil)
var c13 chan chan <- chan <- int = (chan (chan<- (chan<- int)))(nil)

var r1 chan<- (chan int) = (chan <- chan int)(nil)
var r2 chan (<-chan int) = (chan <- chan int)(nil)  // ERROR "chan|incompatible"
var r3 <-chan (chan int) = (<- chan chan int)(nil)
var r4 chan (chan<- int) = (chan chan <- int)(nil)

var r5 <-chan (<-chan int) = (<- chan <- chan int)(nil)
var r6 chan<- (<-chan int) = (chan <- <- chan int)(nil)
var r7 chan<- (chan<- int) = (chan <- chan <- int)(nil)

var r8 <-chan (<-chan (chan int)) = (<- chan <- chan chan int)(nil)
var r9 <-chan (chan<- (chan int)) = (<- chan chan <- chan int)(nil)
var r10 chan<- (<-chan (chan int)) = (chan <- <- chan chan int)(nil)
var r11 chan<- (chan<- (chan int)) = (chan <- chan <- chan int)(nil)
var r12 chan (chan<- (<-chan int)) = (chan chan <- <- chan int)(nil)
var r13 chan (chan<- (chan<- int)) = (chan chan <- chan <- int)(nil)
