// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"eval";
	"fmt";
	"go/token";
)

// TODO(austin): Maybe add to bignum in more general form
func ratToString(rat *bignum.Rational) string {
	n, dnat := rat.Value();
	d := bignum.MakeInt(false, dnat);
	w, frac := n.QuoRem(d);
	out := w.String();
	if frac.IsZero() {
		return out;
	}

	r := frac.Abs();
	r = r.Mul(bignum.Nat(1e6));
	dec, tail := r.DivMod(dnat);
	// Round last digit
	if tail.Cmp(dnat.Div(bignum.Nat(2))) >= 0 {
		dec = dec.Add(bignum.Nat(1));
	}
	// Strip zeros
	ten := bignum.Nat(10);
	for !dec.IsZero() {
		dec2, r2 := dec.DivMod(ten);
		if !r2.IsZero() {
			break;
		}
		dec = dec2;
	}
	out += "." + dec.String();
	return out;
}
