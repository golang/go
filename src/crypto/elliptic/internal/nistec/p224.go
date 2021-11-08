// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nistec

import (
	"crypto/elliptic/internal/fiat"
	"crypto/subtle"
	"errors"
)

var p224B, _ = new(fiat.P224Element).SetBytes([]byte{0xb4, 0x05, 0x0a, 0x85,
	0x0c, 0x04, 0xb3, 0xab, 0xf5, 0x41, 0x32, 0x56, 0x50, 0x44, 0xb0, 0xb7,
	0xd7, 0xbf, 0xd8, 0xba, 0x27, 0x0b, 0x39, 0x43, 0x23, 0x55, 0xff, 0xb4})

var p224G, _ = NewP224Point().SetBytes([]byte{0x04,
	0xb7, 0x0e, 0x0c, 0xbd, 0x6b, 0xb4, 0xbf, 0x7f, 0x32, 0x13, 0x90, 0xb9,
	0x4a, 0x03, 0xc1, 0xd3, 0x56, 0xc2, 0x11, 0x22, 0x34, 0x32, 0x80, 0xd6,
	0x11, 0x5c, 0x1d, 0x21, 0xbd, 0x37, 0x63, 0x88, 0xb5, 0xf7, 0x23, 0xfb,
	0x4c, 0x22, 0xdf, 0xe6, 0xcd, 0x43, 0x75, 0xa0, 0x5a, 0x07, 0x47, 0x64,
	0x44, 0xd5, 0x81, 0x99, 0x85, 0x0, 0x7e, 0x34})

const p224ElementLength = 28

// P224Point is a P-224 point. The zero value is NOT valid.
type P224Point struct {
	// The point is represented in projective coordinates (X:Y:Z),
	// where x = X/Z and y = Y/Z.
	x, y, z *fiat.P224Element
}

// NewP224Point returns a new P224Point representing the point at infinity point.
func NewP224Point() *P224Point {
	return &P224Point{
		x: new(fiat.P224Element),
		y: new(fiat.P224Element).One(),
		z: new(fiat.P224Element),
	}
}

// NewP224Generator returns a new P224Point set to the canonical generator.
func NewP224Generator() *P224Point {
	return (&P224Point{
		x: new(fiat.P224Element),
		y: new(fiat.P224Element),
		z: new(fiat.P224Element),
	}).Set(p224G)
}

// Set sets p = q and returns p.
func (p *P224Point) Set(q *P224Point) *P224Point {
	p.x.Set(q.x)
	p.y.Set(q.y)
	p.z.Set(q.z)
	return p
}

// SetBytes sets p to the compressed, uncompressed, or infinity value encoded in
// b, as specified in SEC 1, Version 2.0, Section 2.3.4. If the point is not on
// the curve, it returns nil and an error, and the receiver is unchanged.
// Otherwise, it returns p.
func (p *P224Point) SetBytes(b []byte) (*P224Point, error) {
	switch {
	// Point at infinity.
	case len(b) == 1 && b[0] == 0:
		return p.Set(NewP224Point()), nil

	// Uncompressed form.
	case len(b) == 1+2*p224ElementLength && b[0] == 4:
		x, err := new(fiat.P224Element).SetBytes(b[1 : 1+p224ElementLength])
		if err != nil {
			return nil, err
		}
		y, err := new(fiat.P224Element).SetBytes(b[1+p224ElementLength:])
		if err != nil {
			return nil, err
		}
		if err := p224CheckOnCurve(x, y); err != nil {
			return nil, err
		}
		p.x.Set(x)
		p.y.Set(y)
		p.z.One()
		return p, nil

	// Compressed form
	case len(b) == 1+p224ElementLength && b[0] == 0:
		return nil, errors.New("unimplemented") // TODO(filippo)

	default:
		return nil, errors.New("invalid P224 point encoding")
	}
}

func p224CheckOnCurve(x, y *fiat.P224Element) error {
	// x³ - 3x + b.
	x3 := new(fiat.P224Element).Square(x)
	x3.Mul(x3, x)

	threeX := new(fiat.P224Element).Add(x, x)
	threeX.Add(threeX, x)

	x3.Sub(x3, threeX)
	x3.Add(x3, p224B)

	// y² = x³ - 3x + b
	y2 := new(fiat.P224Element).Square(y)

	if x3.Equal(y2) != 1 {
		return errors.New("P224 point not on curve")
	}
	return nil
}

// Bytes returns the uncompressed or infinity encoding of p, as specified in
// SEC 1, Version 2.0, Section 2.3.3. Note that the encoding of the point at
// infinity is shorter than all other encodings.
func (p *P224Point) Bytes() []byte {
	// This function is outlined to make the allocations inline in the caller
	// rather than happen on the heap.
	var out [133]byte
	return p.bytes(&out)
}

func (p *P224Point) bytes(out *[133]byte) []byte {
	if p.z.IsZero() == 1 {
		return append(out[:0], 0)
	}

	zinv := new(fiat.P224Element).Invert(p.z)
	xx := new(fiat.P224Element).Mul(p.x, zinv)
	yy := new(fiat.P224Element).Mul(p.y, zinv)

	buf := append(out[:0], 4)
	buf = append(buf, xx.Bytes()...)
	buf = append(buf, yy.Bytes()...)
	return buf
}

// Add sets q = p1 + p2, and returns q. The points may overlap.
func (q *P224Point) Add(p1, p2 *P224Point) *P224Point {
	// Complete addition formula for a = -3 from "Complete addition formulas for
	// prime order elliptic curves" (https://eprint.iacr.org/2015/1060), §A.2.

	t0 := new(fiat.P224Element).Mul(p1.x, p2.x) // t0 := X1 * X2
	t1 := new(fiat.P224Element).Mul(p1.y, p2.y) // t1 := Y1 * Y2
	t2 := new(fiat.P224Element).Mul(p1.z, p2.z) // t2 := Z1 * Z2
	t3 := new(fiat.P224Element).Add(p1.x, p1.y) // t3 := X1 + Y1
	t4 := new(fiat.P224Element).Add(p2.x, p2.y) // t4 := X2 + Y2
	t3.Mul(t3, t4)                              // t3 := t3 * t4
	t4.Add(t0, t1)                              // t4 := t0 + t1
	t3.Sub(t3, t4)                              // t3 := t3 - t4
	t4.Add(p1.y, p1.z)                          // t4 := Y1 + Z1
	x3 := new(fiat.P224Element).Add(p2.y, p2.z) // X3 := Y2 + Z2
	t4.Mul(t4, x3)                              // t4 := t4 * X3
	x3.Add(t1, t2)                              // X3 := t1 + t2
	t4.Sub(t4, x3)                              // t4 := t4 - X3
	x3.Add(p1.x, p1.z)                          // X3 := X1 + Z1
	y3 := new(fiat.P224Element).Add(p2.x, p2.z) // Y3 := X2 + Z2
	x3.Mul(x3, y3)                              // X3 := X3 * Y3
	y3.Add(t0, t2)                              // Y3 := t0 + t2
	y3.Sub(x3, y3)                              // Y3 := X3 - Y3
	z3 := new(fiat.P224Element).Mul(p224B, t2)  // Z3 := b * t2
	x3.Sub(y3, z3)                              // X3 := Y3 - Z3
	z3.Add(x3, x3)                              // Z3 := X3 + X3
	x3.Add(x3, z3)                              // X3 := X3 + Z3
	z3.Sub(t1, x3)                              // Z3 := t1 - X3
	x3.Add(t1, x3)                              // X3 := t1 + X3
	y3.Mul(p224B, y3)                           // Y3 := b * Y3
	t1.Add(t2, t2)                              // t1 := t2 + t2
	t2.Add(t1, t2)                              // t2 := t1 + t2
	y3.Sub(y3, t2)                              // Y3 := Y3 - t2
	y3.Sub(y3, t0)                              // Y3 := Y3 - t0
	t1.Add(y3, y3)                              // t1 := Y3 + Y3
	y3.Add(t1, y3)                              // Y3 := t1 + Y3
	t1.Add(t0, t0)                              // t1 := t0 + t0
	t0.Add(t1, t0)                              // t0 := t1 + t0
	t0.Sub(t0, t2)                              // t0 := t0 - t2
	t1.Mul(t4, y3)                              // t1 := t4 * Y3
	t2.Mul(t0, y3)                              // t2 := t0 * Y3
	y3.Mul(x3, z3)                              // Y3 := X3 * Z3
	y3.Add(y3, t2)                              // Y3 := Y3 + t2
	x3.Mul(t3, x3)                              // X3 := t3 * X3
	x3.Sub(x3, t1)                              // X3 := X3 - t1
	z3.Mul(t4, z3)                              // Z3 := t4 * Z3
	t1.Mul(t3, t0)                              // t1 := t3 * t0
	z3.Add(z3, t1)                              // Z3 := Z3 + t1

	q.x.Set(x3)
	q.y.Set(y3)
	q.z.Set(z3)
	return q
}

// Double sets q = p + p, and returns q. The points may overlap.
func (q *P224Point) Double(p *P224Point) *P224Point {
	// Complete addition formula for a = -3 from "Complete addition formulas for
	// prime order elliptic curves" (https://eprint.iacr.org/2015/1060), §A.2.

	t0 := new(fiat.P224Element).Square(p.x)    // t0 := X ^ 2
	t1 := new(fiat.P224Element).Square(p.y)    // t1 := Y ^ 2
	t2 := new(fiat.P224Element).Square(p.z)    // t2 := Z ^ 2
	t3 := new(fiat.P224Element).Mul(p.x, p.y)  // t3 := X * Y
	t3.Add(t3, t3)                             // t3 := t3 + t3
	z3 := new(fiat.P224Element).Mul(p.x, p.z)  // Z3 := X * Z
	z3.Add(z3, z3)                             // Z3 := Z3 + Z3
	y3 := new(fiat.P224Element).Mul(p224B, t2) // Y3 := b * t2
	y3.Sub(y3, z3)                             // Y3 := Y3 - Z3
	x3 := new(fiat.P224Element).Add(y3, y3)    // X3 := Y3 + Y3
	y3.Add(x3, y3)                             // Y3 := X3 + Y3
	x3.Sub(t1, y3)                             // X3 := t1 - Y3
	y3.Add(t1, y3)                             // Y3 := t1 + Y3
	y3.Mul(x3, y3)                             // Y3 := X3 * Y3
	x3.Mul(x3, t3)                             // X3 := X3 * t3
	t3.Add(t2, t2)                             // t3 := t2 + t2
	t2.Add(t2, t3)                             // t2 := t2 + t3
	z3.Mul(p224B, z3)                          // Z3 := b * Z3
	z3.Sub(z3, t2)                             // Z3 := Z3 - t2
	z3.Sub(z3, t0)                             // Z3 := Z3 - t0
	t3.Add(z3, z3)                             // t3 := Z3 + Z3
	z3.Add(z3, t3)                             // Z3 := Z3 + t3
	t3.Add(t0, t0)                             // t3 := t0 + t0
	t0.Add(t3, t0)                             // t0 := t3 + t0
	t0.Sub(t0, t2)                             // t0 := t0 - t2
	t0.Mul(t0, z3)                             // t0 := t0 * Z3
	y3.Add(y3, t0)                             // Y3 := Y3 + t0
	t0.Mul(p.y, p.z)                           // t0 := Y * Z
	t0.Add(t0, t0)                             // t0 := t0 + t0
	z3.Mul(t0, z3)                             // Z3 := t0 * Z3
	x3.Sub(x3, z3)                             // X3 := X3 - Z3
	z3.Mul(t0, t1)                             // Z3 := t0 * t1
	z3.Add(z3, z3)                             // Z3 := Z3 + Z3
	z3.Add(z3, z3)                             // Z3 := Z3 + Z3

	q.x.Set(x3)
	q.y.Set(y3)
	q.z.Set(z3)
	return q
}

// Select sets q to p1 if cond == 1, and to p2 if cond == 0.
func (q *P224Point) Select(p1, p2 *P224Point, cond int) *P224Point {
	q.x.Select(p1.x, p2.x, cond)
	q.y.Select(p1.y, p2.y, cond)
	q.z.Select(p1.z, p2.z, cond)
	return q
}

// ScalarMult sets p = scalar * q, and returns p.
func (p *P224Point) ScalarMult(q *P224Point, scalar []byte) *P224Point {
	// table holds the first 16 multiples of q. The explicit newP224Point calls
	// get inlined, letting the allocations live on the stack.
	var table = [16]*P224Point{
		NewP224Point(), NewP224Point(), NewP224Point(), NewP224Point(),
		NewP224Point(), NewP224Point(), NewP224Point(), NewP224Point(),
		NewP224Point(), NewP224Point(), NewP224Point(), NewP224Point(),
		NewP224Point(), NewP224Point(), NewP224Point(), NewP224Point(),
	}
	for i := 1; i < 16; i++ {
		table[i].Add(table[i-1], q)
	}

	// Instead of doing the classic double-and-add chain, we do it with a
	// four-bit window: we double four times, and then add [0-15]P.
	t := NewP224Point()
	p.Set(NewP224Point())
	for _, byte := range scalar {
		p.Double(p)
		p.Double(p)
		p.Double(p)
		p.Double(p)

		for i := uint8(0); i < 16; i++ {
			cond := subtle.ConstantTimeByteEq(byte>>4, i)
			t.Select(table[i], t, cond)
		}
		p.Add(p, t)

		p.Double(p)
		p.Double(p)
		p.Double(p)
		p.Double(p)

		for i := uint8(0); i < 16; i++ {
			cond := subtle.ConstantTimeByteEq(byte&0b1111, i)
			t.Select(table[i], t, cond)
		}
		p.Add(p, t)
	}

	return p
}
