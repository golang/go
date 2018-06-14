// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func f0() string {
	const f = 3.141592;
	return fmt.Sprintf("%v", float64(f));
}


func f1() string {
	const f = 3.141592;
	x := float64(float32(f));  // appears to change the precision of f
	_ = x;
	return fmt.Sprintf("%v", float64(f));
}


func main() {
	r0 := f0();
	r1 := f1();
	if r0 != r1 {
		println("r0 =", r0);
		println("r1 =", r1);
		panic("r0 and r1 should be the same");
	}
}
