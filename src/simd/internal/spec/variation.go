// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spec

import (
	"fmt"
	"math"
	"strings"
)

// Platform identifies a specific platform (generally a GOARCH) for
// platform-defined behavior.
type Platform int8

const (
	PlatformAMD64 Platform = iota
	PlatformARM64
	PlatformWASM

	numPlatforms

	// PlatformAny permits the behavior of any platform (for example, when
	// testing the portable simd package or emulated execution).
	PlatformAny Platform = -1
)

func (p Platform) String() string {
	switch p {
	case PlatformAMD64:
		return "amd64"
	case PlatformARM64:
		return "arm64"
	case PlatformWASM:
		return "wasm"
	case PlatformAny:
		return "any"
	}
	return fmt.Sprintf("Platform(%d)", p)
}

// Variation represents a value of type E that may vary between platforms.
//
// Its zero value is the exact zero value of E.
//
// This type is known to specgen.
type Variation[E EltOrMask] struct {
	canon E
	v     variation[E] // nil means exact match on canon
}

type variation[E EltOrMask] interface {
	permits(x E, plat Platform) bool
	String() string
}

func (Variation[E]) elemBits() int {
	return elemBits[E]()
}

// Permits reports whether this variation permits value x on plat.
func (v Variation[E]) Permits(x E, plat Platform) bool {
	if v.v == nil {
		return equalExact(x, v.canon)
	}
	return v.v.permits(x, plat)
}

// Canon returns a single canonical value for this variation.
func (v Variation[E]) Canon() E {
	return v.canon
}

// String returns this variation as a human-readable string.
func (v Variation[E]) String() string {
	if v.v == nil {
		return fmt.Sprint(v.canon)
	}
	return v.v.String()
}

// exact returns a [Variation] that permits only x.
func exact[E EltOrMask](x E) Variation[E] {
	return Variation[E]{canon: x}
}

// perPlatform returns a [Variation] with a specific, but different value per
// platform. The canonical value is the value for the smallest present Platform
// value.
func perPlatform[E EltOrMask](vals map[Platform]E) Variation[E] {
	if len(vals) == 0 {
		panic("perPlatform requires at least one platform value")
	}
	var pp perPlatformVar[E]
	var canon E
	haveCanon := false
	for p := range numPlatforms {
		if val, ok := vals[p]; ok {
			pp.values[p] = val
			pp.present |= 1 << p
			if !haveCanon {
				canon = val
				haveCanon = true
			}
		}
	}
	if pp.present == 0 {
		panic("perPlatform contains no recognized platforms")
	}
	return Variation[E]{canon: canon, v: pp}
}

type perPlatformVar[E EltOrMask] struct {
	values  [numPlatforms]E
	present uint8 // Bit mask over 1<<Platform
}

func (pp perPlatformVar[E]) permits(x E, plat Platform) bool {
	if plat == PlatformAny {
		for p := range numPlatforms {
			if pp.present&(1<<p) != 0 && equalExact(x, pp.values[p]) {
				return true
			}
		}
		return false
	}
	if plat < 0 || plat >= numPlatforms || pp.present&(1<<plat) == 0 {
		return false
	}
	return equalExact(x, pp.values[plat])
}

func (pp perPlatformVar[E]) String() string {
	var b strings.Builder
	b.WriteByte('{')
	for p := range numPlatforms {
		if pp.present&(1<<p) != 0 {
			if b.Len() > 1 {
				b.WriteByte(' ')
			}
			fmt.Fprintf(&b, "%s:%v", p, pp.values[p])
		}
	}
	b.WriteByte('}')
	return b.String()
}

// ulps returns a [Variation] that permits any value within n ULPs of x.
func ulps[E Floats](x E, n uint64) Variation[E] {
	if n == 0 {
		return exact(x)
	}
	return Variation[E]{canon: x, v: ulpsVar[E]{x: x, ulps: n}}
}

type ulpsVar[E Floats] struct {
	x    E
	ulps uint64
}

func (u ulpsVar[E]) permits(got E, _ Platform) bool {
	if equalExact(got, u.x) {
		return true
	}
	// Require NaN and +/-inf to match exactly, and require the same sign.
	got64, want64 := float64(got), float64(u.x) // Preserves NaN, +/-inf, and sign
	if math.IsNaN(got64) || math.IsNaN(want64) ||
		math.IsInf(got64, 0) || math.IsInf(want64, 0) ||
		math.Signbit(got64) != math.Signbit(want64) {
		return false
	}
	// By design, IEEE 754 binary representations of finite floats with the same
	// sign bit are ordered monotonically by magnitude, with adjacent representable
	// values differing by 1 (including across subnormals and the subnormal/normal
	// boundary). Thus, the ULP distance is simply the difference between their
	// integer bit patterns.
	switch want := any(u.x).(type) {
	case float32:
		gb, wb := math.Float32bits(any(got).(float32)), math.Float32bits(want)
		return uint64(max(gb, wb)-min(gb, wb)) <= u.ulps
	case float64:
		gb, wb := math.Float64bits(any(got).(float64)), math.Float64bits(want)
		return max(gb, wb)-min(gb, wb) <= u.ulps
	}
	panic("unreachable")
}

func (u ulpsVar[E]) String() string {
	return fmt.Sprintf("%v±%dulp", u.x, u.ulps)
}

// anyOf returns a [Variation] that permits any of the values in vals.
// Value 0 is the canonical value.
//
// Prefer [perPlatform].
func anyOf[E EltOrMask](vals ...E) Variation[E] {
	if len(vals) == 0 {
		panic("anyOf requires at least one value")
	}
	if len(vals) == 1 {
		return exact(vals[0])
	}
	return Variation[E]{canon: vals[0], v: anyOfVar[E](vals)}
}

type anyOfVar[E EltOrMask] []E

func (a anyOfVar[E]) permits(x E, _ Platform) bool {
	for _, v := range a {
		if equalExact(x, v) {
			return true
		}
	}
	return false
}

func (a anyOfVar[E]) String() string {
	var b strings.Builder
	b.WriteByte('{')
	for i, v := range a {
		if i > 0 {
			b.WriteByte('|')
		}
		fmt.Fprint(&b, v)
	}
	b.WriteByte('}')
	return b.String()
}

// undef returns a [Variation] that is fully undefined and permits any value.
// The canonical value is the zero value of E.
//
// Try really hard not to use this. Even undefined results should be
// deterministic on a given platform, and it's better to model that as
// [perPlatform], or [anyOf].
func undef[E EltOrMask]() Variation[E] {
	return Variation[E]{v: undefVar[E]{}}
}

type undefVar[E EltOrMask] struct{}

func (undefVar[E]) permits(_ E, _ Platform) bool { return true }
func (undefVar[E]) String() string               { return "undef" }

// equalExact reports whether got and want are bitwise identical, except that
// all NaN bit patterns are considered equal to each other.
func equalExact[E EltOrMask](got, want E) bool {
	switch got := any(got).(type) {
	case float32:
		want := any(want).(float32)
		if math.IsNaN(float64(got)) && math.IsNaN(float64(want)) {
			return true
		}
		return math.Float32bits(got) == math.Float32bits(want)
	case float64:
		want := any(want).(float64)
		if math.IsNaN(got) && math.IsNaN(want) {
			return true
		}
		return math.Float64bits(got) == math.Float64bits(want)
	}
	return got == want
}
