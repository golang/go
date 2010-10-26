// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Software IEEE754 64-bit floating point.
// Only referred to (and thus linked in) by arm port
// and by gotest in this directory.

package runtime

const (
	mantbits64 uint = 52
	expbits64  uint = 11
	bias64     = -1<<(expbits64-1) + 1

	nan64 uint64 = (1<<expbits64-1)<<mantbits64 + 1
	inf64 uint64 = (1<<expbits64 - 1) << mantbits64
	neg64 uint64 = 1 << (expbits64 + mantbits64)

	mantbits32 uint = 23
	expbits32  uint = 8
	bias32     = -1<<(expbits32-1) + 1

	nan32 uint32 = (1<<expbits32-1)<<mantbits32 + 1
	inf32 uint32 = (1<<expbits32 - 1) << mantbits32
	neg32 uint32 = 1 << (expbits32 + mantbits32)
)

func funpack64(f uint64) (sign, mant uint64, exp int, inf, nan bool) {
	sign = f & (1 << (mantbits64 + expbits64))
	mant = f & (1<<mantbits64 - 1)
	exp = int(f>>mantbits64) & (1<<expbits64 - 1)

	switch exp {
	case 1<<expbits64 - 1:
		if mant != 0 {
			nan = true
			return
		}
		inf = true
		return

	case 0:
		// denormalized
		if mant != 0 {
			exp += bias64 + 1
			for mant < 1<<mantbits64 {
				mant <<= 1
				exp--
			}
		}

	default:
		// add implicit top bit
		mant |= 1 << mantbits64
		exp += bias64
	}
	return
}

func funpack32(f uint32) (sign, mant uint32, exp int, inf, nan bool) {
	sign = f & (1 << (mantbits32 + expbits32))
	mant = f & (1<<mantbits32 - 1)
	exp = int(f>>mantbits32) & (1<<expbits32 - 1)

	switch exp {
	case 1<<expbits32 - 1:
		if mant != 0 {
			nan = true
			return
		}
		inf = true
		return

	case 0:
		// denormalized
		if mant != 0 {
			exp += bias32 + 1
			for mant < 1<<mantbits32 {
				mant <<= 1
				exp--
			}
		}

	default:
		// add implicit top bit
		mant |= 1 << mantbits32
		exp += bias32
	}
	return
}

func fpack64(sign, mant uint64, exp int, trunc uint64) uint64 {
	mant0, exp0, trunc0 := mant, exp, trunc
	if mant == 0 {
		return sign
	}
	for mant < 1<<mantbits64 {
		mant <<= 1
		exp--
	}
	for mant >= 4<<mantbits64 {
		trunc |= mant & 1
		mant >>= 1
		exp++
	}
	if mant >= 2<<mantbits64 {
		if mant&1 != 0 && (trunc != 0 || mant&2 != 0) {
			mant++
			if mant >= 4<<mantbits64 {
				mant >>= 1
				exp++
			}
		}
		mant >>= 1
		exp++
	}
	if exp >= 1<<expbits64-1+bias64 {
		return sign ^ inf64
	}
	if exp < bias64+1 {
		if exp < bias64-int(mantbits64) {
			return sign | 0
		}
		// repeat expecting denormal
		mant, exp, trunc = mant0, exp0, trunc0
		for exp < bias64 {
			trunc |= mant & 1
			mant >>= 1
			exp++
		}
		if mant&1 != 0 && (trunc != 0 || mant&2 != 0) {
			mant++
		}
		mant >>= 1
		exp++
		if mant < 1<<mantbits64 {
			return sign | mant
		}
	}
	return sign | uint64(exp-bias64)<<mantbits64 | mant&(1<<mantbits64-1)
}

func fpack32(sign, mant uint32, exp int, trunc uint32) uint32 {
	mant0, exp0, trunc0 := mant, exp, trunc
	if mant == 0 {
		return sign
	}
	for mant < 1<<mantbits32 {
		mant <<= 1
		exp--
	}
	for mant >= 4<<mantbits32 {
		trunc |= mant & 1
		mant >>= 1
		exp++
	}
	if mant >= 2<<mantbits32 {
		if mant&1 != 0 && (trunc != 0 || mant&2 != 0) {
			mant++
			if mant >= 4<<mantbits32 {
				mant >>= 1
				exp++
			}
		}
		mant >>= 1
		exp++
	}
	if exp >= 1<<expbits32-1+bias32 {
		return sign ^ inf32
	}
	if exp < bias32+1 {
		if exp < bias32-int(mantbits32) {
			return sign | 0
		}
		// repeat expecting denormal
		mant, exp, trunc = mant0, exp0, trunc0
		for exp < bias32 {
			trunc |= mant & 1
			mant >>= 1
			exp++
		}
		if mant&1 != 0 && (trunc != 0 || mant&2 != 0) {
			mant++
		}
		mant >>= 1
		exp++
		if mant < 1<<mantbits32 {
			return sign | mant
		}
	}
	return sign | uint32(exp-bias32)<<mantbits32 | mant&(1<<mantbits32-1)
}

func fadd64(f, g uint64) uint64 {
	fs, fm, fe, fi, fn := funpack64(f)
	gs, gm, ge, gi, gn := funpack64(g)

	// Special cases.
	switch {
	case fn || gn: // NaN + x or x + NaN = NaN
		return nan64

	case fi && gi && fs != gs: // +Inf + -Inf or -Inf + +Inf = NaN
		return nan64

	case fi: // ±Inf + g = ±Inf
		return f

	case gi: // f + ±Inf = ±Inf
		return g

	case fm == 0 && gm == 0 && fs != 0 && gs != 0: // -0 + -0 = -0
		return f

	case fm == 0: // 0 + g = g but 0 + -0 = +0
		if gm == 0 {
			g ^= gs
		}
		return g

	case gm == 0: // f + 0 = f
		return f

	}

	if fe < ge || fe == ge && fm < gm {
		f, g, fs, fm, fe, gs, gm, ge = g, f, gs, gm, ge, fs, fm, fe
	}

	shift := uint(fe - ge)
	fm <<= 2
	gm <<= 2
	trunc := gm & (1<<shift - 1)
	gm >>= shift
	if fs == gs {
		fm += gm
	} else {
		fm -= gm
		if trunc != 0 {
			fm--
		}
	}
	if fm == 0 {
		fs = 0
	}
	return fpack64(fs, fm, fe-2, trunc)
}

func fsub64(f, g uint64) uint64 {
	return fadd64(f, fneg64(g))
}

func fneg64(f uint64) uint64 {
	return f ^ (1 << (mantbits64 + expbits64))
}

func fmul64(f, g uint64) uint64 {
	fs, fm, fe, fi, fn := funpack64(f)
	gs, gm, ge, gi, gn := funpack64(g)

	// Special cases.
	switch {
	case fn || gn: // NaN * g or f * NaN = NaN
		return nan64

	case fi && gi: // Inf * Inf = Inf (with sign adjusted)
		return f ^ gs

	case fi && gm == 0, fm == 0 && gi: // 0 * Inf = Inf * 0 = NaN
		return nan64

	case fm == 0: // 0 * x = 0 (with sign adjusted)
		return f ^ gs

	case gm == 0: // x * 0 = 0 (with sign adjusted)
		return g ^ fs
	}

	// 53-bit * 53-bit = 107- or 108-bit
	lo, hi := mullu(fm, gm)
	shift := mantbits64 - 1
	trunc := lo & (1<<shift - 1)
	mant := hi<<(64-shift) | lo>>shift
	return fpack64(fs^gs, mant, fe+ge-1, trunc)
}

func fdiv64(f, g uint64) uint64 {
	fs, fm, fe, fi, fn := funpack64(f)
	gs, gm, ge, gi, gn := funpack64(g)

	// Special cases.
	switch {
	case fn || gn: // NaN / g = f / NaN = NaN
		return nan64

	case fi && gi: // ±Inf / ±Inf = NaN
		return nan64

	case !fi && !gi && fm == 0 && gm == 0: // 0 / 0 = NaN
		return nan64

	case fi, !gi && gm == 0: // Inf / g = f / 0 = Inf
		return fs ^ gs ^ inf64

	case gi, fm == 0: // f / Inf = 0 / g = Inf
		return fs ^ gs ^ 0
	}
	_, _, _, _ = fi, fn, gi, gn

	// 53-bit<<54 / 53-bit = 53- or 54-bit.
	shift := mantbits64 + 2
	q, r := divlu(fm>>(64-shift), fm<<shift, gm)
	return fpack64(fs^gs, q, fe-ge-2, r)
}

func f64to32(f uint64) uint32 {
	fs, fm, fe, fi, fn := funpack64(f)
	if fn {
		return nan32
	}
	fs32 := uint32(fs >> 32)
	if fi {
		return fs32 ^ inf32
	}
	const d = mantbits64 - mantbits32 - 1
	return fpack32(fs32, uint32(fm>>d), fe-1, uint32(fm&(1<<d-1)))
}

func f32to64(f uint32) uint64 {
	const d = mantbits64 - mantbits32
	fs, fm, fe, fi, fn := funpack32(f)
	if fn {
		return nan64
	}
	fs64 := uint64(fs) << 32
	if fi {
		return fs64 ^ inf64
	}
	return fpack64(fs64, uint64(fm)<<d, fe, 0)
}

func fcmp64(f, g uint64) (cmp int, isnan bool) {
	fs, fm, _, fi, fn := funpack64(f)
	gs, gm, _, gi, gn := funpack64(g)

	switch {
	case fn, gn: // flag NaN
		return 0, true

	case !fi && !gi && fm == 0 && gm == 0: // ±0 == ±0
		return 0, false

	case fs > gs: // f < 0, g > 0
		return -1, false

	case fs < gs: // f > 0, g < 0
		return +1, false

	// Same sign, not NaN.
	// Can compare encodings directly now.
	// Reverse for sign.
	case fs == 0 && f < g, fs != 0 && f > g:
		return -1, false

	case fs == 0 && f > g, fs != 0 && f < g:
		return +1, false
	}

	// f == g
	return 0, false
}

func f64toint(f uint64) (val int64, ok bool) {
	fs, fm, fe, fi, fn := funpack64(f)

	switch {
	case fi, fn: // NaN
		return 0, false

	case fe < -1: // f < 0.5
		return 0, false

	case fe > 63: // f >= 2^63
		if fs != 0 && fm == 0 { // f == -2^63
			return -1 << 63, true
		}
		if fs != 0 {
			return 0, false
		}
		return 0, false
	}

	for fe > int(mantbits64) {
		fe--
		fm <<= 1
	}
	for fe < int(mantbits64) {
		fe++
		fm >>= 1
	}
	val = int64(fm)
	if fs != 0 {
		val = -val
	}
	return val, true
}

func fintto64(val int64) (f uint64) {
	fs := uint64(val) & (1 << 63)
	mant := uint64(val)
	if fs != 0 {
		mant = -mant
	}
	return fpack64(fs, mant, int(mantbits64), 0)
}

// 64x64 -> 128 multiply.
// adapted from hacker's delight.
func mullu(u, v uint64) (lo, hi uint64) {
	const (
		s    = 32
		mask = 1<<s - 1
	)
	u0 := u & mask
	u1 := u >> s
	v0 := v & mask
	v1 := v >> s
	w0 := u0 * v0
	t := u1*v0 + w0>>s
	w1 := t & mask
	w2 := t >> s
	w1 += u0 * v1
	return u * v, u1*v1 + w2 + w1>>s
}

// 128/64 -> 64 quotient, 64 remainder.
// adapted from hacker's delight
func divlu(u1, u0, v uint64) (q, r uint64) {
	const b = 1 << 32

	if u1 >= v {
		return 1<<64 - 1, 1<<64 - 1
	}

	// s = nlz(v); v <<= s
	s := uint(0)
	for v&(1<<63) == 0 {
		s++
		v <<= 1
	}

	vn1 := v >> 32
	vn0 := v & (1<<32 - 1)
	un32 := u1<<s | u0>>(64-s)
	un10 := u0 << s
	un1 := un10 >> 32
	un0 := un10 & (1<<32 - 1)
	q1 := un32 / vn1
	rhat := un32 - q1*vn1

again1:
	if q1 >= b || q1*vn0 > b*rhat+un1 {
		q1--
		rhat += vn1
		if rhat < b {
			goto again1
		}
	}

	un21 := un32*b + un1 - q1*v
	q0 := un21 / vn1
	rhat = un21 - q0*vn1

again2:
	if q0 >= b || q0*vn0 > b*rhat+un0 {
		q0--
		rhat += vn1
		if rhat < b {
			goto again2
		}
	}

	return q1*b + q0, (un21*b + un0 - q0*v) >> s
}

// callable from C

func fadd64c(f, g uint64, ret *uint64)            { *ret = fadd64(f, g) }
func fsub64c(f, g uint64, ret *uint64)            { *ret = fsub64(f, g) }
func fmul64c(f, g uint64, ret *uint64)            { *ret = fmul64(f, g) }
func fdiv64c(f, g uint64, ret *uint64)            { *ret = fdiv64(f, g) }
func fneg64c(f uint64, ret *uint64)               { *ret = fneg64(f) }
func f32to64c(f uint32, ret *uint64)              { *ret = f32to64(f) }
func f64to32c(f uint64, ret *uint32)              { *ret = f64to32(f) }
func fcmp64c(f, g uint64, ret *int, retnan *bool) { *ret, *retnan = fcmp64(f, g) }
func fintto64c(val int64, ret *uint64)            { *ret = fintto64(val) }
func f64tointc(f uint64, ret *int64, retok *bool) { *ret, *retok = f64toint(f) }
