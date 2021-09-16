// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generator

import (
	"bytes"
	"fmt"
	"strings"
)

// structparm describes a parameter of struct type; it implements the
// "parm" interface.
type structparm struct {
	sname  string
	qname  string
	fields []parm
	isBlank
	addrTakenHow
	isGenValFunc
	skipCompare
}

func (p structparm) TypeName() string {
	return p.sname
}

func (p structparm) QualName() string {
	return p.qname
}

func (p structparm) Declare(b *bytes.Buffer, prefix string, suffix string, caller bool) {
	n := p.sname
	if caller {
		n = p.qname
	}
	b.WriteString(fmt.Sprintf("%s %s%s", prefix, n, suffix))
}

func (p structparm) FieldName(i int) string {
	if p.fields[i].IsBlank() {
		return "_"
	}
	return fmt.Sprintf("F%d", i)
}

func (p structparm) String() string {
	var buf bytes.Buffer

	buf.WriteString(fmt.Sprintf("struct %s {\n", p.sname))
	for fi, f := range p.fields {
		buf.WriteString(fmt.Sprintf("%s %s\n", p.FieldName(fi), f.String()))
	}
	buf.WriteString("}")
	return buf.String()
}

func (p structparm) GenValue(s *genstate, f *funcdef, value int, caller bool) (string, int) {
	var buf bytes.Buffer

	verb(5, "structparm.GenValue(%d)", value)

	n := p.sname
	if caller {
		n = p.qname
	}
	buf.WriteString(fmt.Sprintf("%s{", n))
	nbfi := 0
	for fi, fld := range p.fields {
		var valstr string
		valstr, value = s.GenValue(f, fld, value, caller)
		if p.fields[fi].IsBlank() {
			buf.WriteString("/* ")
			valstr = strings.ReplaceAll(valstr, "/*", "[[")
			valstr = strings.ReplaceAll(valstr, "*/", "]]")
		} else {
			writeCom(&buf, nbfi)
		}
		buf.WriteString(p.FieldName(fi) + ": ")
		buf.WriteString(valstr)
		if p.fields[fi].IsBlank() {
			buf.WriteString(" */")
		} else {
			nbfi++
		}
	}
	buf.WriteString("}")
	return buf.String(), value
}

func (p structparm) IsControl() bool {
	return false
}

func (p structparm) NumElements() int {
	ne := 0
	for _, f := range p.fields {
		ne += f.NumElements()
	}
	return ne
}

func (p structparm) GenElemRef(elidx int, path string) (string, parm) {
	ct := 0
	verb(4, "begin GenElemRef(%d,%s) on %s", elidx, path, p.String())

	for fi, f := range p.fields {
		fne := f.NumElements()

		//verb(4, "+ examining field %d fne %d ct %d", fi, fne, ct)

		// Empty field. Continue on.
		if elidx == ct && fne == 0 {
			continue
		}

		// Is this field the element we're interested in?
		if fne == 1 && elidx == ct {

			// The field in question may be a composite that has only
			// multiple elements but a single non-zero-sized element.
			// If this is the case, keep going.
			if sp, ok := f.(*structparm); ok {
				if len(sp.fields) > 1 {
					ppath := fmt.Sprintf("%s.F%d", path, fi)
					if p.fields[fi].IsBlank() || path == "_" {
						ppath = "_"
					}
					return f.GenElemRef(elidx-ct, ppath)
				}
			}

			verb(4, "found field %d type %s in GenElemRef(%d,%s)", fi, f.TypeName(), elidx, path)
			ppath := fmt.Sprintf("%s.F%d", path, fi)
			if p.fields[fi].IsBlank() || path == "_" {
				ppath = "_"
			}
			return ppath, f
		}

		// Is the element we want somewhere inside this field?
		if fne > 1 && elidx >= ct && elidx < ct+fne {
			ppath := fmt.Sprintf("%s.F%d", path, fi)
			if p.fields[fi].IsBlank() || path == "_" {
				ppath = "_"
			}
			return f.GenElemRef(elidx-ct, ppath)
		}

		ct += fne
	}
	panic(fmt.Sprintf("GenElemRef failed for struct %s elidx %d", p.TypeName(), elidx))
}

func (p structparm) HasPointer() bool {
	for _, f := range p.fields {
		if f.HasPointer() {
			return true
		}
	}
	return false
}
