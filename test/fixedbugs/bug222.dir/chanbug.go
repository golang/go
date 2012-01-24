// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file

package chanbug
var C chan<- (chan int)
var D chan<- func()
var E func() chan int
var F func() (func())
