// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gen

type Uint8x64 struct {
	valAny
}

var kindUint8x64 = &kind{typ: "Uint8x64", reg: regClassZ}

func ConstUint8x64(c [64]uint8, name string) (y Uint8x64) {
	y.initOp(&op{op: "const", kind: y.kind(), c: c, name: name})
	return y
}

func (Uint8x64) kind() *kind {
	return kindUint8x64
}

func (Uint8x64) wrap(x *op) Uint8x64 {
	var y Uint8x64
	y.initOp(x)
	return y
}

func (x Uint8x64) ToUint64x8() (z Uint64x8) {
	z.op = x.op
	return z
}

func (x Uint8x64) Shuffle(shuf Uint8x64) (y Uint8x64) {
	if shuf.op.op == "const" {
		// TODO: There are often patterns we can take advantage of here. Sometimes
		// we can do a broadcast. Sometimes we can at least do a quadword
		// permutation instead of a full byte permutation.

		// Range check the shuffle
		for i, inp := range shuf.op.c.([64]uint8) {
			// 0xff is a special "don't care" value
			if !(inp == 0xff || inp < 64) {
				fatalf("shuffle[%d] = %d out of range [0, %d) or 0xff", i, inp, 64)
			}
		}
	}

	args := []*op{x.op, shuf.op}
	y.initOp(&op{op: "VPERMB", kind: y.kind(), args: args})
	return y
}

func (x Uint8x64) ShuffleZeroed(shuf Uint8x64, mask Mask64) (y Uint8x64) {
	args := []*op{x.op, shuf.op, mask.op}
	y.initOp(&op{op: "VPERMB.Z", kind: y.kind(), args: args})
	return y
}

func (x Uint8x64) ShuffleMasked(shuf Uint8x64, mask Mask64) (y Uint8x64) {
	args := []*op{x.op, shuf.op, mask.op}
	y.initOp(&op{op: "VPERMB.mask", kind: y.kind(), args: args})
	return y
}

// TODO: The two-argument shuffle is a little weird. You almost want the
// receiver to be the shuffle and the two arguments to be the two inputs, but
// that's almost certainly *not* what you want for the single input shuffle.

func (x Uint8x64) Shuffle2(y Uint8x64, shuf Uint8x64) (z Uint8x64) {
	// Confusingly, the inputs are in the opposite order from what you'd expect.
	args := []*op{y.op, x.op, shuf.op}
	z.initOp(&op{op: "VPERMI2B", kind: z.kind(), args: args})
	return z
}

func (x Uint8x64) Shuffle2Zeroed(y Uint8x64, shuf Uint8x64, mask Mask64) (z Uint8x64) {
	// Confusingly, the inputs are in the opposite order from what you'd expect.
	args := []*op{y.op, x.op, mask.op, shuf.op}
	z.initOp(&op{op: "VPERMI2B.Z", kind: z.kind(), args: args})
	return z
}

func (x Uint8x64) Shuffle2Masked(y Uint8x64, shuf Uint8x64, mask Mask64) (z Uint8x64) {
	// Confusingly, the inputs are in the opposite order from what you'd expect.
	args := []*op{y.op, x.op, mask.op, shuf.op}
	z.initOp(&op{op: "VPERMI2B.mask", kind: z.kind(), args: args})
	return z
}

type Uint64x8 struct {
	valAny
}

var kindUint64x8 = &kind{typ: "Uint64x8", reg: regClassZ}

func ConstUint64x8(c [8]uint64, name string) (y Uint64x8) {
	// TODO: Sometimes these can be optimized into broadcast loads.
	y.initOp(&op{op: "const", kind: y.kind(), c: c, name: name})
	return y
}

func BroadcastUint64x8Zeroed(src Uint64, mask Mask8) (z Uint64x8) {
	z.initOp(&op{op: "VPBROADCASTQ.Z", kind: z.kind(), args: []*op{src.op, mask.op}})
	return z
}

func (x Uint64x8) BroadcastMasked(src Uint64, mask Mask8) (z Uint64x8) {
	z.initOp(&op{op: "VPBROADCASTQ.mask", kind: z.kind(), args: []*op{src.op, mask.op, x.op}})
	return z
}

func (Uint64x8) kind() *kind {
	return kindUint64x8
}

func (Uint64x8) wrap(x *op) Uint64x8 {
	var y Uint64x8
	y.initOp(x)
	return y
}

func (x Uint64x8) Or(y Uint64x8) (z Uint64x8) {
	z.initOp(&op{op: "VPORQ", kind: z.kind(), args: []*op{y.op, x.op}})
	return z
}

func (x Uint64x8) Sub(y Uint64x8) (z Uint64x8) {
	// Arguments are backwards
	z.initOp(&op{op: "VPSUBQ", kind: z.kind(), args: []*op{y.op, x.op}})
	return z
}

func (x Uint64x8) ToUint8x64() (z Uint8x64) {
	z.op = x.op
	return z
}

func (x Uint64x8) GF2P8Affine(y Uint8x64) (z Uint8x64) {
	// matrix, vector
	z.initOp(&op{op: "VGF2P8AFFINEQB", kind: z.kind(), args: []*op{x.op, y.op}})
	return z
}

func (x Uint64x8) ShuffleBits(y Uint8x64) (z Mask64) {
	z.initOp(&op{op: "VPSHUFBITQMB", kind: z.kind(), args: []*op{y.op, x.op}})
	return z
}

func (x Uint64x8) ShuffleBitsMasked(y Uint8x64, mask Mask64) (z Mask64) {
	// This is always zeroing if the mask is provided.
	z.initOp(&op{op: "VPSHUFBITQMB", kind: z.kind(), args: []*op{y.op, x.op, mask.op}})
	return z
}

type Mask8 struct {
	valAny
}

var kindMask8 = &kind{typ: "Mask8", reg: regClassK}

func ConstMask8(c uint8) (y Mask8) {
	var tmp Uint64
	tmp.initOp(&op{op: "MOVQ", kind: tmp.kind(), args: []*op{imm(c)}})
	y.initOp(&op{op: "KMOVB", kind: y.kind(), args: []*op{tmp.op}})
	return y
}

func (Mask8) kind() *kind {
	return kindMask8
}

func (Mask8) wrap(x *op) Mask8 {
	var y Mask8
	y.initOp(x)
	return y
}

func (x Mask8) ToUint8() (z Uint64) {
	z.initOp(&op{op: "KMOVB", kind: z.kind(), args: []*op{x.op}})
	return z
}

func (x Mask8) Or(y Mask8) (z Mask8) {
	z.initOp(&op{op: "KORQ", kind: z.kind(), args: []*op{y.op, x.op}})
	return z
}

func (x Mask8) ShiftLeft(c uint8) (z Mask8) {
	if c == 0 {
		z = x
	} else {
		z.initOp(&op{op: "KSHIFTLB", kind: z.kind(), args: []*op{imm(c), x.op}})
	}
	return z
}

type Mask64 struct {
	valAny
}

var kindMask64 = &kind{typ: "Mask64", reg: regClassK}

func ConstMask64(c uint64) (y Mask64) {
	var tmp Uint64
	tmp.initOp(&op{op: "MOVQ", kind: tmp.kind(), args: []*op{imm(c)}})
	y.initOp(&op{op: "KMOVQ", kind: y.kind(), args: []*op{tmp.op}})
	return y
}

func (Mask64) kind() *kind {
	return kindMask64
}

func (Mask64) wrap(x *op) Mask64 {
	var y Mask64
	y.initOp(x)
	return y
}

func (x Mask64) ToUint64() (z Uint64) {
	z.initOp(&op{op: "KMOVQ", kind: z.kind(), args: []*op{x.op}})
	return z
}

func (x Mask64) Or(y Mask64) (z Mask64) {
	z.initOp(&op{op: "KORQ", kind: z.kind(), args: []*op{y.op, x.op}})
	return z
}

func (x Mask64) ShiftLeft(c uint8) (z Mask64) {
	if c == 0 {
		z = x
	} else {
		z.initOp(&op{op: "KSHIFTLQ", kind: z.kind(), args: []*op{imm(c), x.op}})
	}
	return z
}

func (x Mask64) ShiftRight(c uint8) (z Mask64) {
	if c == 0 {
		z = x
	} else {
		z.initOp(&op{op: "KSHIFTRQ", kind: z.kind(), args: []*op{imm(c), x.op}})
	}
	return z
}
