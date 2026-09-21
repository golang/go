// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimesecret

package main

/*
static int cAdd(int a, int b) { return a + b; }
*/
import "C"

import (
	"fmt"
	"runtime/secret"
)

func init() {
	register("SecretCgo", SecretCgo)
}

func SecretCgo() {
	secret.Do(func() {
		r := C.cAdd(1, 2)
		if r != 3 {
			panic(fmt.Sprintf("got %d, want 3", r))
		}
	})
	fmt.Println("OK")
}
