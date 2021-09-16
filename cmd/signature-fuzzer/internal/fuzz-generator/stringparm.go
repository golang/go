// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generator

import (
	"bytes"
)

// stringparm describes a parameter of string type; it implements the
// "parm" interface
type stringparm struct {
	tag string
	isBlank
	addrTakenHow
	isGenValFunc
	skipCompare
}

func (p stringparm) Declare(b *bytes.Buffer, prefix string, suffix string, caller bool) {
	b.WriteString(prefix + " string" + suffix)
}

func (p stringparm) GenElemRef(elidx int, path string) (string, parm) {
	return path, &p
}

var letters = []rune("�꿦3򂨃f6ꂅ8ˋ<􂊇񊶿(z̽|ϣᇊ񁗇򟄼q񧲥筁{ЂƜĽ")

func (p stringparm) GenValue(s *genstate, f *funcdef, value int, caller bool) (string, int) {
	ns := len(letters) - 9
	nel := int(s.wr.Intn(8))
	st := int(s.wr.Intn(int64(ns)))
	en := st + nel
	if en > ns {
		en = ns
	}
	return "\"" + string(letters[st:en]) + "\"", value + 1
}

func (p stringparm) IsControl() bool {
	return false
}

func (p stringparm) NumElements() int {
	return 1
}

func (p stringparm) String() string {
	return "string"
}

func (p stringparm) TypeName() string {
	return "string"
}

func (p stringparm) QualName() string {
	return "string"
}

func (p stringparm) HasPointer() bool {
	return false
}
