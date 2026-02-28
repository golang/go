// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bbig

import (
	"crypto/internal/boring"
	"math/big"
	"unsafe"
)

func Enc(b *big.Int) boring.BigInt {
	if b == nil {
		return nil
	}
	x := b.Bits()
	if len(x) == 0 {
		return boring.BigInt{}
	}
	return unsafe.Slice((*uint)(&x[0]), len(x))
}

func Dec(b boring.BigInt) *big.Int {
	if b == nil {
		return nil
	}
	if len(b) == 0 {
		return new(big.Int)
	}
	x := unsafe.Slice((*big.Word)(&b[0]), len(b))
	return new(big.Int).SetBits(x)
}
