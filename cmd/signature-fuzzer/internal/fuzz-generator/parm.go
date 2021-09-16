// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generator

import (
	"bytes"
	"fmt"
	"os"
	"sort"
)

// parm is an interface describing an abstract parameter var or return
// var; there will be concrete types of various sorts that implement
// this interface.
type parm interface {

	// Declare emits text containing a declaration of this param
	// or return var into the specified buffer. Prefix is a tag to
	// prepend before the declaration (for example a variable
	// name) followed by a space; suffix is an arbitrary string to
	// tack onto the end of the param's type text. Here 'caller'
	// is set to true if we're emitting the caller part of a test
	// pair as opposed to the checker.
	Declare(b *bytes.Buffer, prefix string, suffix string, caller bool)

	// GenElemRef returns a pair [X,Y] corresponding to a
	// component piece of some composite parm, where X is a string
	// forming the reference (ex: ".field" if we're picking out a
	// struct field) and Y is a parm object corresponding to the
	// type of the element.
	GenElemRef(elidx int, path string) (string, parm)

	// GenValue constructs a new concrete random value appropriate
	// for the type in question and returns it, along with a
	// sequence number indicating how many random decisions we had
	// to make. Here "s" is the current generator state, "f" is
	// the current function we're emitting, value is a sequence
	// number indicating how many random decisions have been made
	// up until this point, and 'caller' is set to true if we're
	// emitting the caller part of a test pair as opposed to the
	// checker.  Return value is a pair [V,I] where V is the text
	// if the value, and I is a new sequence number reflecting any
	// additional random choices we had to make.  For example, if
	// the parm is something like "type Foo struct { f1 int32; f2
	// float64 }" then we might expect GenValue to emit something
	// like "Foo{int32(-9), float64(123.123)}".
	GenValue(s *genstate, f *funcdef, value int, caller bool) (string, int)

	// IsControl returns true if this specific param has been marked
	// as the single param that controls recursion for a recursive
	// checker function. The test code doesn't check this param for a specific
	// value, but instead returns early if it has value 0 or decrements it
	// on a recursive call.
	IsControl() bool

	// NumElements returns the total number of discrete elements contained
	// in this parm. For non-composite types, this will always be 1.
	NumElements() int

	// String returns a descriptive string for this parm.
	String() string

	// TypeName returns the non-qualified type name for this parm.
	TypeName() string

	// QualName returns a package-qualified type name for this parm.
	QualName() string

	// HasPointer returns true if this parm is of pointer type, or
	// if it is a composite that has a pointer element somewhere inside.
	// Strings and slices return true for this hook.
	HasPointer() bool

	// IsBlank() returns true if the name of this parm is "_" (that is,
	// if we randomly chose to make it a blank). SetBlank() is used
	// to set the 'blank' property for this parm.
	IsBlank() bool
	SetBlank(v bool)

	// AddrTaken() return a token indicating whether this parm should
	// be address taken or not, the nature of the address-taken-ness (see
	// below at the def of addrTakenHow). SetAddrTaken is used to set
	// the address taken property of the parm.
	AddrTaken() addrTakenHow
	SetAddrTaken(val addrTakenHow)

	// IsGenVal() returns true if the values of this type should
	// be obtained by calling a helper func, as opposed to
	// emitting code inline (as one would for things like numeric
	// types). SetIsGenVal is used to set the gen-val property of
	// the parm.
	IsGenVal() bool
	SetIsGenVal(val bool)

	// SkipCompare() returns true if we've randomly decided that
	// we don't want to compare the value for this param or
	// return.  SetSkipCompare is used to set the skip-compare
	// property of the parm.
	SkipCompare() skipCompare
	SetSkipCompare(val skipCompare)
}

type addrTakenHow uint8

const (
	// Param not address taken.
	notAddrTaken addrTakenHow = 0

	// Param address is taken and used for simple reads/writes.
	addrTakenSimple addrTakenHow = 1

	// Param address is taken and passed to a well-behaved function.
	addrTakenPassed addrTakenHow = 2

	// Param address is taken and stored to a global var.
	addrTakenHeap addrTakenHow = 3
)

func (a *addrTakenHow) AddrTaken() addrTakenHow {
	return *a
}

func (a *addrTakenHow) SetAddrTaken(val addrTakenHow) {
	*a = val
}

type isBlank bool

func (b *isBlank) IsBlank() bool {
	return bool(*b)
}

func (b *isBlank) SetBlank(val bool) {
	*b = isBlank(val)
}

type isGenValFunc bool

func (g *isGenValFunc) IsGenVal() bool {
	return bool(*g)
}

func (g *isGenValFunc) SetIsGenVal(val bool) {
	*g = isGenValFunc(val)
}

type skipCompare int

const (
	// Param not address taken.
	SkipAll     = -1
	SkipNone    = 0
	SkipPayload = 1
)

func (s *skipCompare) SkipCompare() skipCompare {
	return skipCompare(*s)
}

func (s *skipCompare) SetSkipCompare(val skipCompare) {
	*s = skipCompare(val)
}

// containedParms takes an arbitrary param 'p' and returns a slice
// with 'p' itself plus any component parms contained within 'p'.
func containedParms(p parm) []parm {
	visited := make(map[string]parm)
	worklist := []parm{p}

	addToWork := func(p parm) {
		if p == nil {
			panic("not expected")
		}
		if _, ok := visited[p.TypeName()]; !ok {
			worklist = append(worklist, p)
		}
	}

	for len(worklist) != 0 {
		cp := worklist[0]
		worklist = worklist[1:]
		if _, ok := visited[cp.TypeName()]; ok {
			continue
		}
		visited[cp.TypeName()] = cp
		switch x := cp.(type) {
		case *mapparm:
			addToWork(x.keytype)
			addToWork(x.valtype)
		case *structparm:
			for _, fld := range x.fields {
				addToWork(fld)
			}
		case *arrayparm:
			addToWork(x.eltype)
		case *pointerparm:
			addToWork(x.totype)
		case *typedefparm:
			addToWork(x.target)
		}
	}
	rv := []parm{}
	for _, v := range visited {
		rv = append(rv, v)
	}
	sort.Slice(rv, func(i, j int) bool {
		if rv[i].TypeName() == rv[j].TypeName() {
			fmt.Fprintf(os.Stderr, "%d %d %+v %+v %s %s\n", i, j, rv[i], rv[i].String(), rv[j], rv[j].String())
			panic("unexpected")
		}
		return rv[i].TypeName() < rv[j].TypeName()
	})
	return rv
}
