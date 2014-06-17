// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// Declarations for runtime services implemented in C or assembly.
// C implementations of these functions are in stubs.goc.
// Assembly implementations are in various files, see comments with
// each function.

// rawstring allocates storage for a new string. The returned
// string and byte slice both refer to the same storage.
// The storage is not zeroed. Callers should use
// b to set the string contents and then drop b.
func rawstring(size int) (string, []byte)

// rawbyteslice allocates a new byte slice. The byte slice is not zeroed.
func rawbyteslice(size int) []byte

// rawruneslice allocates a new rune slice. The rune slice is not zeroed.
func rawruneslice(size int) []rune

//go:noescape
func gogetcallerpc(p unsafe.Pointer) uintptr

//go:noescape
func racereadrangepc(addr unsafe.Pointer, len int, callpc, pc uintptr)
