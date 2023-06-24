// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.20
// +build !go1.20

package abi

import "unsafe"

type (
	stringHeader struct {
		Data *byte
		Len  int
	}
	sliceHeader struct {
		Data *byte
		Len  int
		Cap  int
	}
)

func unsafeStringFor(b *byte, l int) string {
	h := stringHeader{Data: b, Len: l}
	return *(*string)(unsafe.Pointer(&h))
}

func unsafeSliceFor(b *byte, l int) []byte {
	h := sliceHeader{Data: b, Len: l, Cap: l}
	return *(*[]byte)(unsafe.Pointer(&h))
}
