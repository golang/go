// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Type struct {
	any
}

type typeObject struct {
	e struct{}
	b *byte
}

func f(b *byte) Type {
	return Type{
		typeObject{
			b: b,
		},
	}

}
