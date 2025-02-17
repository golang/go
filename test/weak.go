// errorcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test weak pointers.

package p

import (
	"runtime"
	"weak"
)

// Adapted from example in https://github.com/golang/go/issues/67552#issuecomment-2639661220
func conversion() {
	p := "hello"
	a := weak.Make(&p)
	b := (weak.Pointer[*byte])(a) // ERROR "cannot convert a \(variable of struct type weak\.Pointer\[string\]\) to type weak.Pointer\[\*byte\]"
	c := b.Value()
	println(**c)
	runtime.KeepAlive(p)
}
