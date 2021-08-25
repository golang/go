// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64 || mips64 || mips64le || ppc64 || ppc64le || riscv64 || s390x || wasm

package atomic

// LoadAcquire is a partially unsynchronized version
// of Load that relaxes ordering constraints. Other threads
// may observe operations that precede this operation to
// occur after it, but no operation that occurs after it
// on this thread can be observed to occur before it.
//
// WARNING: Use sparingly and with great care.
func (u *Uint64) LoadAcquire() uint64 {
	return LoadAcq64(&u.value)
}

// StoreRelease is a partially unsynchronized version
// of Store that relaxes ordering constraints. Other threads
// may observe operations that occur after this operation to
// precede it, but no operation that precedes it
// on this thread can be observed to occur after it.
//
// WARNING: Use sparingly and with great care.
func (u *Uint64) StoreRelease(value uint64) {
	StoreRel64(&u.value, value)
}
