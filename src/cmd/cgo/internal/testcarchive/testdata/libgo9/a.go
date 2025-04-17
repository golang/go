// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

// extern void callGoWithVariousStack(int);
import "C"

func main() {}

//export GoF
func GoF(p int32) {
	runtime.GC()
	if p != 0 {
		panic("panic")
	}
}

//export callGoWithVariousStackAndGoFrame
func callGoWithVariousStackAndGoFrame(p int32) {
	if p != 0 {
		defer func() {
			e := recover()
			if e == nil {
				panic("did not panic")
			}
			runtime.GC()
		}()
	}
	C.callGoWithVariousStack(C.int(p));
}
