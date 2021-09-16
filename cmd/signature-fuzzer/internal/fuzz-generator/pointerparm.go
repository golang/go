// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generator

import (
	"bytes"
	"fmt"
)

// pointerparm describes a parameter of pointer type; it implements the
// "parm" interface.
type pointerparm struct {
	tag    string
	totype parm
	isBlank
	addrTakenHow
	isGenValFunc
	skipCompare
}

func (p pointerparm) Declare(b *bytes.Buffer, prefix string, suffix string, caller bool) {
	n := p.totype.TypeName()
	if caller {
		n = p.totype.QualName()
	}
	b.WriteString(fmt.Sprintf("%s *%s%s", prefix, n, suffix))
}

func (p pointerparm) GenElemRef(elidx int, path string) (string, parm) {
	return path, &p
}

func (p pointerparm) GenValue(s *genstate, f *funcdef, value int, caller bool) (string, int) {
	pref := ""
	if caller {
		pref = s.checkerPkg(s.pkidx) + "."
	}
	var valstr string
	valstr, value = s.GenValue(f, p.totype, value, caller)
	fname := s.genAllocFunc(p.totype)
	return fmt.Sprintf("%s%s(%s)", pref, fname, valstr), value
}

func (p pointerparm) IsControl() bool {
	return false
}

func (p pointerparm) NumElements() int {
	return 1
}

func (p pointerparm) String() string {
	return fmt.Sprintf("*%s", p.totype)
}

func (p pointerparm) TypeName() string {
	return fmt.Sprintf("*%s", p.totype.TypeName())
}

func (p pointerparm) QualName() string {
	return fmt.Sprintf("*%s", p.totype.QualName())
}

func mkPointerParm(to parm) pointerparm {
	var pp pointerparm
	pp.tag = "pointer"
	pp.totype = to
	return pp
}

func (p pointerparm) HasPointer() bool {
	return true
}
