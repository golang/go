// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fiat_test

import (
	"crypto/elliptic/internal/fiat"
	"crypto/rand"
	"testing"
)

func p521Random(t *testing.T) *fiat.P521Element {
	buf := make([]byte, 66)
	if _, err := rand.Read(buf); err != nil {
		t.Fatal(err)
	}
	buf[65] &= 1
	e, err := new(fiat.P521Element).SetBytes(buf)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestP521Invert(t *testing.T) {
	a := p521Random(t)
	inv := new(fiat.P521Element).Invert(a)
	one := new(fiat.P521Element).Mul(a, inv)
	if new(fiat.P521Element).One().Equal(one) != 1 {
		t.Errorf("a * 1/a != 1; got %x for %x", one.Bytes(), a.Bytes())
	}
	inv.Invert(new(fiat.P521Element))
	if new(fiat.P521Element).Equal(inv) != 1 {
		t.Errorf("1/0 != 0; got %x", inv.Bytes())
	}
}
