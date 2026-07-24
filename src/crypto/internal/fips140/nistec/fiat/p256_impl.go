// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !riscv64

package fiat

func p256Mul(out, a, b *p256MontgomeryDomainFieldElement) {
	p256MulGeneric(out, a, b)
}
