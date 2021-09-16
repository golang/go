// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generator

import (
	"bytes"
	"fmt"
)

// arrayparm describes a parameter of array type; it implements the
// "parm" interface.
type arrayparm struct {
	aname     string
	qname     string
	nelements uint8
	eltype    parm
	slice     bool
	isBlank
	addrTakenHow
	isGenValFunc
	skipCompare
}

func (p arrayparm) IsControl() bool {
	return false
}

func (p arrayparm) TypeName() string {
	return p.aname
}

func (p arrayparm) QualName() string {
	return p.qname
}

func (p arrayparm) Declare(b *bytes.Buffer, prefix string, suffix string, caller bool) {
	n := p.aname
	if caller {
		n = p.qname
	}
	b.WriteString(fmt.Sprintf("%s %s%s", prefix, n, suffix))
}

func (p arrayparm) String() string {
	return fmt.Sprintf("%s %d-element array of %s", p.aname, p.nelements, p.eltype.String())
}

func (p arrayparm) GenValue(s *genstate, f *funcdef, value int, caller bool) (string, int) {
	var buf bytes.Buffer

	verb(5, "arrayparm.GenValue(%d)", value)

	n := p.aname
	if caller {
		n = p.qname
	}
	buf.WriteString(fmt.Sprintf("%s{", n))
	for i := 0; i < int(p.nelements); i++ {
		var valstr string
		valstr, value = s.GenValue(f, p.eltype, value, caller)
		writeCom(&buf, i)
		buf.WriteString(valstr)
	}
	buf.WriteString("}")
	return buf.String(), value
}

func (p arrayparm) GenElemRef(elidx int, path string) (string, parm) {
	ene := p.eltype.NumElements()
	verb(4, "begin GenElemRef(%d,%s) on %s ene %d", elidx, path, p.String(), ene)

	// For empty arrays, convention is to return empty string
	if ene == 0 {
		return "", &p
	}

	// Find slot within array of element of interest
	slot := elidx / ene

	// If this is the element we're interested in, return it
	if ene == 1 {
		verb(4, "hit scalar element")
		epath := fmt.Sprintf("%s[%d]", path, slot)
		if path == "_" || p.IsBlank() {
			epath = "_"
		}
		return epath, p.eltype
	}

	verb(4, "recur slot=%d GenElemRef(%d,...)", slot, elidx-(slot*ene))

	// Otherwise our victim is somewhere inside the slot
	ppath := fmt.Sprintf("%s[%d]", path, slot)
	if p.IsBlank() {
		ppath = "_"
	}
	return p.eltype.GenElemRef(elidx-(slot*ene), ppath)
}

func (p arrayparm) NumElements() int {
	return p.eltype.NumElements() * int(p.nelements)
}

func (p arrayparm) HasPointer() bool {
	return p.eltype.HasPointer() || p.slice
}
