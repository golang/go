// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

import (
	"fmt"
	"go/token"
	"strings"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/ssa"
)

// A Label is an entity that may be pointed to by a pointer, map,
// channel, 'func', slice or interface.  Labels include:
//
// Labels include:
// 	- functions
//      - globals
//      - tagged objects, representing interfaces and reflect.Values
//      - arrays created by literals (e.g. []byte("foo")) and conversions ([]byte(s))
// 	- stack- and heap-allocated variables (including composite literals)
// 	- channels, maps and arrays created by make()
//	- instrinsic or reflective operations that allocate (e.g. append, reflect.New)
// 	- and their subelements, e.g. "alloc.y[*].z"
//
// Labels are so varied that they defy good generalizations;
// some have no value, no callgraph node, or no position.
// Many objects have types that are inexpressible in Go:
// maps, channels, functions, tagged objects.
//
type Label struct {
	obj        *object    // the addressable memory location containing this label
	subelement *fieldInfo // subelement path within obj, e.g. ".a.b[*].c"
}

// Value returns the ssa.Value that allocated this label's object,
// or nil if it was allocated by an intrinsic.
//
func (l Label) Value() ssa.Value {
	return l.obj.val
}

// Context returns the analytic context in which this label's object was allocated,
// or nil for global objects: global, const, and shared contours for functions.
//
func (l Label) Context() CallGraphNode {
	return l.obj.cgn
}

// Path returns the path to the subelement of the object containing
// this label.  For example, ".x[*].y".
//
func (l Label) Path() string {
	return l.subelement.path()
}

// Pos returns the position of this label, if known, zero otherwise.
func (l Label) Pos() token.Pos {
	if v := l.Value(); v != nil {
		return v.Pos()
	}
	if l.obj.rtype != nil {
		if nt, ok := deref(l.obj.rtype).(*types.Named); ok {
			return nt.Obj().Pos()
		}
	}
	if cgn := l.obj.cgn; cgn != nil {
		return cgn.Func().Pos()
	}
	return token.NoPos
}

// String returns the printed form of this label.
//
// Examples:					Object type:
// 	(sync.Mutex).Lock			(a function)
// 	"foo":[]byte				(a slice constant)
//	makemap					(map allocated via make)
//	makechan				(channel allocated via make)
//	makeinterface				(tagged object allocated by makeinterface)
//      <alloc in reflect.Zero>			(allocation in instrinsic)
//      sync.Mutex				(a reflect.rtype instance)
//
// Labels within compound objects have subelement paths:
//	x.y[*].z				(a struct variable, x)
//	append.y[*].z				(array allocated by append)
//	makeslice.y[*].z			(array allocated via make)
//
func (l Label) String() string {
	var s string
	switch v := l.obj.val.(type) {
	case nil:
		if l.obj.rtype != nil {
			return l.obj.rtype.String()
		}
		if l.obj.cgn != nil {
			// allocation by intrinsic or reflective operation
			return fmt.Sprintf("<alloc in %s>", l.obj.cgn.Func())
		}
		return "<unknown>" // should be unreachable

	case *ssa.Function, *ssa.Global:
		s = v.String()

	case *ssa.Const:
		s = v.Name()

	case *ssa.Alloc:
		s = v.Comment
		if s == "" {
			s = "alloc"
		}

	case *ssa.Call:
		// Currently only calls to append can allocate objects.
		if v.Call.Value.(*ssa.Builtin).Object().Name() != "append" {
			panic("unhandled *ssa.Call label: " + v.Name())
		}
		s = "append"

	case *ssa.MakeMap, *ssa.MakeChan, *ssa.MakeSlice, *ssa.Convert:
		s = strings.ToLower(strings.TrimPrefix(fmt.Sprintf("%T", v), "*ssa."))

	case *ssa.MakeInterface:
		// MakeInterface is usually implicit in Go source (so
		// Pos()==0), and tagged objects may be allocated
		// synthetically (so no *MakeInterface data).
		s = "makeinterface:" + v.X.Type().String()

	default:
		panic(fmt.Sprintf("unhandled Label.val type: %T", v))
	}

	return s + l.subelement.path()
}
