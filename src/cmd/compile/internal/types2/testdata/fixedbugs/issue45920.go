// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f1[T any, C chan T | <-chan T](ch C) {}

func _(ch chan int)   { f1(ch) }
func _(ch <-chan int) { f1(ch) }
func _(ch chan<- int) { f1( /* ERROR chan<- int does not implement chan int\|<-chan int */ ch) }

func f2[T any, C chan T | chan<- T](ch C) {}

func _(ch chan int)   { f2(ch) }
func _(ch <-chan int) { f2( /* ERROR <-chan int does not implement chan int\|chan<- int */ ch) }
func _(ch chan<- int) { f2(ch) }
