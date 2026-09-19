// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssaop

import (
	"fmt"
	"math/bits"
)

// A RegMask encodes a set of machine registers.
type RegMask struct {
	V1, V2 uint64
}

type Register uint8

func (r RegMask) Intersect(s RegMask) RegMask {
	return RegMask{r.V1 & s.V1, r.V2 & s.V2}
}

func (r RegMask) Union(s RegMask) RegMask {
	return RegMask{r.V1 | s.V1, r.V2 | s.V2}
}

func (r RegMask) Minus(s RegMask) RegMask {
	return RegMask{r.V1 &^ s.V1, r.V2 &^ s.V2}
}

func (r RegMask) Empty() bool {
	return r.V1 == 0 && r.V2 == 0
}

func (r RegMask) PickReg() Register {
	if r.Empty() {
		panic("can't pick a register from an empty set")
	}
	// pick the lowest one
	if r.V1 != 0 {
		return Register(bits.TrailingZeros64(r.V1))
	}
	return Register(bits.TrailingZeros64(r.V2) + 64)
}

func (r RegMask) AddReg(i Register) RegMask {
	if i < 64 {
		return RegMask{r.V1 | 1<<i, r.V2}
	}
	return RegMask{r.V1, r.V2 | 1<<(i-64)}
}

func (r RegMask) RemoveReg(i Register) RegMask {
	if i < 64 {
		return RegMask{r.V1 &^ (1 << i), r.V2}
	}
	return RegMask{r.V1, r.V2 &^ (1 << (i - 64))}
}

func (r RegMask) HasReg(i Register) bool {
	if i < 64 {
		return (r.V1>>i)&1 != 0
	}
	return (r.V2>>(i-64))&1 != 0
}

func (m RegMask) String() string {
	s := ""
	for r := Register(0); !m.Empty(); r++ {
		if !m.HasReg(r) {
			continue
		}
		m = m.RemoveReg(r)
		if s != "" {
			s += " "
		}
		s += fmt.Sprintf("r%d", r)
	}
	return s
}
