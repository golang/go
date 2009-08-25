// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmp

type Int struct {
	hidden *byte
}

func addInt(z, x, y *Int) *Int

func (z *Int) Add(x, y *Int) *Int {
	return addInt(z, x, y)
}

func stringInt(z *Int) string

func (z *Int) String() string {
	return stringInt(z)
}

func NewInt(n uint64) *Int

