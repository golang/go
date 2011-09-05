// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// zero length structs.
// used to not be evaluated.
// issue 2232.

package main

func recv(c chan interface{}) struct{} {
	return (<-c).(struct{})
}

var m = make(map[interface{}]int)

func recv1(c chan interface{}) {
	defer rec()
	m[(<-c).(struct{})] = 0
}

func rec() {
	recover()
}

func main() {
	c := make(chan interface{})
	go recv(c)
	c <- struct{}{}
	go recv1(c)
	c <- struct{}{}
}
