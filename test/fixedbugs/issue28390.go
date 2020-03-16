// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 28390/28430: Function call arguments were not
// converted correctly under some circumstances.

package main

import "fmt"

type A struct {
	K int
	S string
	M map[string]string
}

func newA(k int, s string) (a A) {
	a.K = k
	a.S = s
	a.M = make(map[string]string)
	a.M[s] = s
	return
}

func proxy() (x int, a A) {
	return 1, newA(2, "3")
}

func consume(x int, a interface{}) {
	fmt.Println(x)
	fmt.Println(a) // used to panic here
}

func main() {
	consume(proxy())
}
