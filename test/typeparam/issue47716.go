// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"unsafe"
)

// size returns the size of type T
func size[T any](x T) uintptr {
	return unsafe.Sizeof(x)
}

// size returns the alignment of type T
func align[T any](x T) uintptr {
	return unsafe.Alignof(x)
}

type Tstruct[T any] struct {
	f1 T
	f2 int
}

// offset returns the offset of field f2 in the generic type Tstruct
func (r *Tstruct[T]) offset() uintptr {
	return unsafe.Offsetof(r.f2)
}

func main() {
	v1 := int(5)
	if got, want := size(v1), unsafe.Sizeof(v1); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	if got, want := align(v1), unsafe.Alignof(v1); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	v2 := "abc"
	if got, want := size(v2), unsafe.Sizeof(v2); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	if got, want := align(v2), unsafe.Alignof(v2); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	var v3 Tstruct[int]
	if got, want := unsafe.Offsetof(v3.f2), unsafe.Sizeof(v1); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	var v4 Tstruct[interface{}]
	var v5 interface{}
	if got, want := unsafe.Offsetof(v4.f2), unsafe.Sizeof(v5); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	if got, want := v3.offset(), unsafe.Offsetof(v3.f2); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	if got, want := v4.offset(), unsafe.Offsetof(v4.f2); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
}
