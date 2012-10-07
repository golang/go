// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Various declarations of exported variables and functions.

package p

var C1 chan <- chan int = (chan<- (chan int))(nil)
var C2 chan (<- chan int) = (chan (<-chan int))(nil)
var C3 <- chan chan int = (<-chan (chan int))(nil)
var C4 chan chan <- int = (chan (chan<- int))(nil)

var C5 <- chan <- chan int = (<-chan (<-chan int))(nil)
var C6 chan <- <- chan int = (chan<- (<-chan int))(nil)
var C7 chan <- chan <- int = (chan<- (chan<- int))(nil)

var C8 <- chan <- chan chan int = (<-chan (<-chan (chan int)))(nil)
var C9 <- chan chan <- chan int = (<-chan (chan<- (chan int)))(nil)
var C10 chan <- <- chan chan int = (chan<- (<-chan (chan int)))(nil)
var C11 chan <- chan <- chan int = (chan<- (chan<- (chan int)))(nil)
var C12 chan chan <- <- chan int = (chan (chan<- (<-chan int)))(nil)
var C13 chan chan <- chan <- int = (chan (chan<- (chan<- int)))(nil)

var R1 chan<- (chan int) = (chan <- chan int)(nil)
var R3 <-chan (chan int) = (<- chan chan int)(nil)
var R4 chan (chan<- int) = (chan chan <- int)(nil)

var R5 <-chan (<-chan int) = (<- chan <- chan int)(nil)
var R6 chan<- (<-chan int) = (chan <- <- chan int)(nil)
var R7 chan<- (chan<- int) = (chan <- chan <- int)(nil)

var R8 <-chan (<-chan (chan int)) = (<- chan <- chan chan int)(nil)
var R9 <-chan (chan<- (chan int)) = (<- chan chan <- chan int)(nil)
var R10 chan<- (<-chan (chan int)) = (chan <- <- chan chan int)(nil)
var R11 chan<- (chan<- (chan int)) = (chan <- chan <- chan int)(nil)
var R12 chan (chan<- (<-chan int)) = (chan chan <- <- chan int)(nil)
var R13 chan (chan<- (chan<- int)) = (chan chan <- chan <- int)(nil)

var F1 func() func() int
func F2() func() func() int
func F3(func() func() int)
