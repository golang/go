// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Service struct {
	rpc [2]int
}

func (s *Service) Serve(a int64) {
	if a != 1234 {
		print(a, " not 1234\n")
		panic("fail")
	}
}

var arith Service

func main() {
	c := make(chan string)
	a := new(Service)
	go a.Serve(1234)
	_ = c
}
