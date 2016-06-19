// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// A copy of the code in ../gc/subr.go.
// We can't use it directly because it would generate
// an import cycle. TODO: move to a common support package.

// argument passing to/from
// smagic and umagic
type magic struct {
	W   int // input for both - width
	S   int // output for both - shift
	Bad int // output for both - unexpected failure

	// magic multiplier for signed literal divisors
	Sd int64 // input - literal divisor
	Sm int64 // output - multiplier

	// magic multiplier for unsigned literal divisors
	Ud uint64 // input - literal divisor
	Um uint64 // output - multiplier
	Ua int    // output - adder
}

// magic number for signed division
// see hacker's delight chapter 10
func smagic(m *magic) {
	var mask uint64

	m.Bad = 0
	switch m.W {
	default:
		m.Bad = 1
		return

	case 8:
		mask = 0xff

	case 16:
		mask = 0xffff

	case 32:
		mask = 0xffffffff

	case 64:
		mask = 0xffffffffffffffff
	}

	two31 := mask ^ (mask >> 1)

	p := m.W - 1
	ad := uint64(m.Sd)
	if m.Sd < 0 {
		ad = -uint64(m.Sd)
	}

	// bad denominators
	if ad == 0 || ad == 1 || ad == two31 {
		m.Bad = 1
		return
	}

	t := two31
	ad &= mask

	anc := t - 1 - t%ad
	anc &= mask

	q1 := two31 / anc
	r1 := two31 - q1*anc
	q1 &= mask
	r1 &= mask

	q2 := two31 / ad
	r2 := two31 - q2*ad
	q2 &= mask
	r2 &= mask

	var delta uint64
	for {
		p++
		q1 <<= 1
		r1 <<= 1
		q1 &= mask
		r1 &= mask
		if r1 >= anc {
			q1++
			r1 -= anc
			q1 &= mask
			r1 &= mask
		}

		q2 <<= 1
		r2 <<= 1
		q2 &= mask
		r2 &= mask
		if r2 >= ad {
			q2++
			r2 -= ad
			q2 &= mask
			r2 &= mask
		}

		delta = ad - r2
		delta &= mask
		if q1 < delta || (q1 == delta && r1 == 0) {
			continue
		}

		break
	}

	m.Sm = int64(q2 + 1)
	if uint64(m.Sm)&two31 != 0 {
		m.Sm |= ^int64(mask)
	}
	m.S = p - m.W
}

// magic number for unsigned division
// see hacker's delight chapter 10
func umagic(m *magic) {
	var mask uint64

	m.Bad = 0
	m.Ua = 0

	switch m.W {
	default:
		m.Bad = 1
		return

	case 8:
		mask = 0xff

	case 16:
		mask = 0xffff

	case 32:
		mask = 0xffffffff

	case 64:
		mask = 0xffffffffffffffff
	}

	two31 := mask ^ (mask >> 1)

	m.Ud &= mask
	if m.Ud == 0 || m.Ud == two31 {
		m.Bad = 1
		return
	}

	nc := mask - (-m.Ud&mask)%m.Ud
	p := m.W - 1

	q1 := two31 / nc
	r1 := two31 - q1*nc
	q1 &= mask
	r1 &= mask

	q2 := (two31 - 1) / m.Ud
	r2 := (two31 - 1) - q2*m.Ud
	q2 &= mask
	r2 &= mask

	var delta uint64
	for {
		p++
		if r1 >= nc-r1 {
			q1 <<= 1
			q1++
			r1 <<= 1
			r1 -= nc
		} else {
			q1 <<= 1
			r1 <<= 1
		}

		q1 &= mask
		r1 &= mask
		if r2+1 >= m.Ud-r2 {
			if q2 >= two31-1 {
				m.Ua = 1
			}

			q2 <<= 1
			q2++
			r2 <<= 1
			r2++
			r2 -= m.Ud
		} else {
			if q2 >= two31 {
				m.Ua = 1
			}

			q2 <<= 1
			r2 <<= 1
			r2++
		}

		q2 &= mask
		r2 &= mask

		delta = m.Ud - 1 - r2
		delta &= mask

		if p < m.W+m.W {
			if q1 < delta || (q1 == delta && r1 == 0) {
				continue
			}
		}

		break
	}

	m.Um = q2 + 1
	m.S = p - m.W
}

// adaptors for use by rewrite rules
func smagic64ok(d int64) bool {
	m := magic{W: 64, Sd: d}
	smagic(&m)
	return m.Bad == 0
}
func smagic64m(d int64) int64 {
	m := magic{W: 64, Sd: d}
	smagic(&m)
	return m.Sm
}
func smagic64s(d int64) int64 {
	m := magic{W: 64, Sd: d}
	smagic(&m)
	return int64(m.S)
}

func umagic64ok(d int64) bool {
	m := magic{W: 64, Ud: uint64(d)}
	umagic(&m)
	return m.Bad == 0
}
func umagic64m(d int64) int64 {
	m := magic{W: 64, Ud: uint64(d)}
	umagic(&m)
	return int64(m.Um)
}
func umagic64s(d int64) int64 {
	m := magic{W: 64, Ud: uint64(d)}
	umagic(&m)
	return int64(m.S)
}
func umagic64a(d int64) bool {
	m := magic{W: 64, Ud: uint64(d)}
	umagic(&m)
	return m.Ua != 0
}
