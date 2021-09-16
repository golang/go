// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generator

import (
	"bytes"
	"fmt"
)

// mapparm describes a parameter of map type; it implements the
// "parm" interface.
type mapparm struct {
	aname   string
	qname   string
	keytype parm
	valtype parm
	keytmp  string
	isBlank
	addrTakenHow
	isGenValFunc
	skipCompare
}

func (p mapparm) IsControl() bool {
	return false
}

func (p mapparm) TypeName() string {
	return p.aname
}

func (p mapparm) QualName() string {
	return p.qname
}

func (p mapparm) Declare(b *bytes.Buffer, prefix string, suffix string, caller bool) {
	n := p.aname
	if caller {
		n = p.qname
	}
	b.WriteString(fmt.Sprintf("%s %s%s", prefix, n, suffix))
}

func (p mapparm) String() string {
	return fmt.Sprintf("%s map[%s]%s", p.aname,
		p.keytype.String(), p.valtype.String())
}

func (p mapparm) GenValue(s *genstate, f *funcdef, value int, caller bool) (string, int) {
	var buf bytes.Buffer

	verb(5, "mapparm.GenValue(%d)", value)

	n := p.aname
	if caller {
		n = p.qname
	}
	buf.WriteString(fmt.Sprintf("%s{", n))
	buf.WriteString(p.keytmp + ": ")

	var valstr string
	valstr, value = s.GenValue(f, p.valtype, value, caller)
	buf.WriteString(valstr + "}")
	return buf.String(), value
}

func (p mapparm) GenElemRef(elidx int, path string) (string, parm) {
	vne := p.valtype.NumElements()
	verb(4, "begin GenElemRef(%d,%s) on %s %d", elidx, path, p.String(), vne)

	ppath := fmt.Sprintf("%s[mkt.%s]", path, p.keytmp)

	// otherwise dig into the value
	verb(4, "recur GenElemRef(%d,...)", elidx)

	// Otherwise our victim is somewhere inside the value
	if p.IsBlank() {
		ppath = "_"
	}
	return p.valtype.GenElemRef(elidx, ppath)
}

func (p mapparm) NumElements() int {
	return p.valtype.NumElements()
}

func (p mapparm) HasPointer() bool {
	return true
}
