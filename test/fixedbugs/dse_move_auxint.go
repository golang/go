// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Dead-store elimination must use the AuxInt byte count of OpMove,
// not the size of its Aux type. The "inline runtime.memmove" rewrite
// emits Move {uint8} [N]; if DSE treats that as a 1-byte write, a
// trailing 1-byte store appears to fully shadow it and the move is
// incorrectly deleted.

package main

import "fmt"

//go:noinline
func f() [8]byte {
	str := "ABCDEFGH"
	dst := make([]byte, len(str))
	copy(dst, str)
	dst[0] = 99
	return [8]byte(dst)
}

func main() {
	got := f()
	want := [8]byte{99, 'B', 'C', 'D', 'E', 'F', 'G', 'H'}
	if got != want {
		panic(fmt.Sprintf("got %v, want %v", got, want))
	}
}
