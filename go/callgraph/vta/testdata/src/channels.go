// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

func foo(c chan interface{}, j int) {
	c <- j + 1
}

func Baz(i int) {
	c := make(chan interface{})
	go foo(c, i)
	x := <-c
	print(x)
}

// Relevant SSA:
//  func foo(c chan interface{}, j int):
//  t0 = j + 1:int
//  t1 = make interface{} <- int (t0)
//  send c <- t1                        // t1 -> chan {}interface
//  return
//
// func Baz(i int):
//  t0 = make chan interface{} 0:int
//  go foo(t0, i)
//  t1 = <-t0                           // chan {}interface -> t1
//  t2 = print(t1)
//  return

// WANT:
// Channel(chan interface{}) -> Local(t1)
// Local(t1) -> Channel(chan interface{})
