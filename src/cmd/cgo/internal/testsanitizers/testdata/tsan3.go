// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// The stubs for the C functions read and write the same slot on the
// g0 stack when copying arguments in and out.

/*
#cgo CFLAGS: -fsanitize=thread
#cgo LDFLAGS: -fsanitize=thread

int Func1() {
	return 0;
}

void Func2(int x) {
	(void)x;
}
*/
import "C"

func main() {
	const N = 10000
	done := make(chan bool, N)
	for i := 0; i < N; i++ {
		go func() {
			C.Func1()
			done <- true
		}()
		go func() {
			C.Func2(0)
			done <- true
		}()
	}
	for i := 0; i < 2*N; i++ {
		<-done
	}
}
