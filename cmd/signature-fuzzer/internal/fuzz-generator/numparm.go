// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generator

import (
	"bytes"
	"fmt"
	"math"
)

// numparm describes a numeric parameter type; it implements the
// "parm" interface.
type numparm struct {
	tag         string
	widthInBits uint32
	ctl         bool
	isBlank
	addrTakenHow
	isGenValFunc
	skipCompare
}

var f32parm *numparm = &numparm{
	tag:         "float",
	widthInBits: uint32(32),
	ctl:         false,
}
var f64parm *numparm = &numparm{
	tag:         "float",
	widthInBits: uint32(64),
	ctl:         false,
}

func (p numparm) TypeName() string {
	if p.tag == "byte" {
		return "byte"
	}
	return fmt.Sprintf("%s%d", p.tag, p.widthInBits)
}

func (p numparm) QualName() string {
	return p.TypeName()
}

func (p numparm) String() string {
	if p.tag == "byte" {
		return "byte"
	}
	ctl := ""
	if p.ctl {
		ctl = " [ctl=yes]"
	}
	return fmt.Sprintf("%s%s", p.TypeName(), ctl)
}

func (p numparm) NumElements() int {
	return 1
}

func (p numparm) IsControl() bool {
	return p.ctl
}

func (p numparm) GenElemRef(elidx int, path string) (string, parm) {
	return path, &p
}

func (p numparm) Declare(b *bytes.Buffer, prefix string, suffix string, caller bool) {
	t := fmt.Sprintf("%s%d%s", p.tag, p.widthInBits, suffix)
	if p.tag == "byte" {
		t = fmt.Sprintf("%s%s", p.tag, suffix)
	}
	b.WriteString(prefix + " " + t)
}

func (p numparm) genRandNum(s *genstate, value int) (string, int) {
	which := uint8(s.wr.Intn(int64(100)))
	if p.tag == "int" {
		var v int64
		if which < 3 {
			// max
			v = (1 << (p.widthInBits - 1)) - 1

		} else if which < 5 {
			// min
			v = (-1 << (p.widthInBits - 1))
		} else {
			nrange := int64(1 << (p.widthInBits - 2))
			v = s.wr.Intn(nrange)
			if value%2 != 0 {
				v = -v
			}
		}
		return fmt.Sprintf("%s%d(%d)", p.tag, p.widthInBits, v), value + 1
	}
	if p.tag == "uint" || p.tag == "byte" {
		nrange := int64(1 << (p.widthInBits - 2))
		v := s.wr.Intn(nrange)
		if p.tag == "byte" {
			return fmt.Sprintf("%s(%d)", p.tag, v), value + 1
		}
		return fmt.Sprintf("%s%d(0x%x)", p.tag, p.widthInBits, v), value + 1
	}
	if p.tag == "float" {
		if p.widthInBits == 32 {
			rf := s.wr.Float32() * (math.MaxFloat32 / 4)
			if value%2 != 0 {
				rf = -rf
			}
			return fmt.Sprintf("%s%d(%v)", p.tag, p.widthInBits, rf), value + 1
		}
		if p.widthInBits == 64 {
			return fmt.Sprintf("%s%d(%v)", p.tag, p.widthInBits,
				s.wr.NormFloat64()), value + 1
		}
		panic("unknown float type")
	}
	if p.tag == "complex" {
		if p.widthInBits == 64 {
			f1, v2 := f32parm.genRandNum(s, value)
			f2, v3 := f32parm.genRandNum(s, v2)
			return fmt.Sprintf("complex(%s,%s)", f1, f2), v3
		}
		if p.widthInBits == 128 {
			f1, v2 := f64parm.genRandNum(s, value)
			f2, v3 := f64parm.genRandNum(s, v2)
			return fmt.Sprintf("complex(%v,%v)", f1, f2), v3
		}
		panic("unknown complex type")
	}
	panic("unknown numeric type")
}

func (p numparm) GenValue(s *genstate, f *funcdef, value int, caller bool) (string, int) {
	r, nv := p.genRandNum(s, value)
	verb(5, "numparm.GenValue(%d) = %s", value, r)
	return r, nv
}

func (p numparm) HasPointer() bool {
	return false
}
