// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generator

import (
	"bytes"
	"fmt"
)

// typedefparm describes a parameter that is a typedef of some other
// type; it implements the "parm" interface
type typedefparm struct {
	aname  string
	qname  string
	target parm
	isBlank
	addrTakenHow
	isGenValFunc
	skipCompare
}

func (p typedefparm) Declare(b *bytes.Buffer, prefix string, suffix string, caller bool) {
	n := p.aname
	if caller {
		n = p.qname
	}
	b.WriteString(fmt.Sprintf("%s %s%s", prefix, n, suffix))
}

func (p typedefparm) GenElemRef(elidx int, path string) (string, parm) {
	_, isarr := p.target.(*arrayparm)
	_, isstruct := p.target.(*structparm)
	_, ismap := p.target.(*mapparm)
	rv, rp := p.target.GenElemRef(elidx, path)
	// this is hacky, but I don't see a nicer way to do this
	if isarr || isstruct || ismap {
		return rv, rp
	}
	rp = &p
	return rv, rp
}

func (p typedefparm) GenValue(s *genstate, f *funcdef, value int, caller bool) (string, int) {
	n := p.aname
	if caller {
		n = p.qname
	}
	rv, v := s.GenValue(f, p.target, value, caller)
	rv = n + "(" + rv + ")"
	return rv, v
}

func (p typedefparm) IsControl() bool {
	return false
}

func (p typedefparm) NumElements() int {
	return p.target.NumElements()
}

func (p typedefparm) String() string {
	return fmt.Sprintf("%s typedef of %s", p.aname, p.target.String())

}

func (p typedefparm) TypeName() string {
	return p.aname

}

func (p typedefparm) QualName() string {
	return p.qname
}

func (p typedefparm) HasPointer() bool {
	return p.target.HasPointer()
}

func (s *genstate) makeTypedefParm(f *funcdef, target parm, pidx int) parm {
	var tdp typedefparm
	ns := len(f.typedefs)
	tdp.aname = fmt.Sprintf("MyTypeF%dS%d", f.idx, ns)
	tdp.qname = fmt.Sprintf("%s.MyTypeF%dS%d", s.checkerPkg(pidx), f.idx, ns)
	tdp.target = target
	tdp.SetBlank(uint8(s.wr.Intn(100)) < tunables.blankPerc)
	f.typedefs = append(f.typedefs, tdp)
	return &tdp
}
