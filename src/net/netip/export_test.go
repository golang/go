// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package netip

import "internal/intern"

var (
	Z0    = z0
	Z4    = z4
	Z6noz = z6noz
)

type Uint128 = uint128

func Mk128(hi, lo uint64) Uint128 {
	return uint128{hi, lo}
}

func MkAddr(u Uint128, z *intern.Value) Addr {
	return Addr{u, z}
}

func IPv4(a, b, c, d uint8) Addr { return AddrFrom4([4]byte{a, b, c, d}) }

var TestAppendToMarshal = testAppendToMarshal

func (a Addr) IsZero() bool   { return a.isZero() }
func (p Prefix) IsZero() bool { return p.isZero() }

func (p Prefix) Compare(p2 Prefix) int { return p.compare(p2) }
