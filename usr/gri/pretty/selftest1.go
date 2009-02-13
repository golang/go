// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import P0 /* ERROR expected */ ; /* SYNC */
import P1 /* ERROR expected */ Flags /* SYNC */
import P2 /* ERROR expected */ 42 /* SYNC */


type S0 struct {
	f0, f1, f2;
}


func /* ERROR receiver */ () f0() {} /* SYNC */
func /* ERROR receiver */ (*S0, *S0) f1() {} /* SYNC */


func f0(a b, c /* ERROR type */ ) /* SYNC */ {}


func f1() {
}


func CompositeLiterals() {
	a1 := []int();
	a2 := []int(0, 1, 2, );
	a3 := []int(0, 1, 2, /* ERROR single value expected */ 3 : 4, 5); /* SYNC */
	a1 := []int(0 : 1, 2 : 3, /* ERROR key:value pair expected */ 4, ); /* SYNC */
}


func main () {
}


func /* ERROR EOF */
