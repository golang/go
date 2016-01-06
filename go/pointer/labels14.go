// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

package pointer

import (
	"fmt"
	"go/token"
	"strings"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/types"
)

// A Label is an entity that may be pointed to by a pointer, map,
// channel, 'func', slice or interface.
//
// Labels include:
//      - functions
//      - globals
//      - tagged objects, representing interfaces and reflect.Values
//      - arrays created by conversions (e.g. []byte("foo"), []byte(s))
//      - stack- and heap-allocated variables (including composite literals)
//      - channels, maps and arrays created by make()
//      - intrinsic or reflective operations that allocate (e.g. append, reflect.New)
//      - intrinsic objects, e.g. the initial array behind os.Args.
//      - and their subelements, e.g. "alloc.y[*].z"
//
// Labels are so varied that they defy good generalizations;
// some have no value, no callgraph node, or no position.
// Many objects have types that are inexpressible in Go:
// maps, channels, functions, tagged objects.
//
// At most one of Value() or ReflectType() may return non-nil.
//
type Label struct {
	obj        *object    // the addressable memory location containing this label
	subelement *fieldInfo // subelement path within obj, e.g. ".a.b[*].c"
}

// Value returns the ssa.Value that allocated this label's object, if any.
func (l Label) Value() ssa.Value {
	val, _ := l.obj.data.(ssa.Value)
	return val
}

// ReflectType returns the type represented by this label if it is an
// reflect.rtype instance object or *reflect.rtype-tagged object.
//
func (l Label) ReflectType() types.Type {
	rtype, _ := l.obj.data.(types.Type)
	return rtype
}

// Path returns the path to the subelement of the object containing
// this label.  For example, ".x[*].y".
//
func (l Label) Path() string {
	return l.subelement.path()
}

// Pos returns the position of this label, if known, zero otherwise.
func (l Label) Pos() token.Pos {
	switch data := l.obj.data.(type) {
	case ssa.Value:
		return data.Pos()
	case types.Type:
		if nt, ok := deref(data).(*types.Named); ok {
			return nt.Obj().Pos()
		}
	}
	if cgn := l.obj.cgn; cgn != nil {
		return cgn.fn.Pos()
	}
	return token.NoPos
}

// String returns the printed form of this label.
//
// Examples:                                    Object type:
//      x                                       (a variable)
//      (sync.Mutex).Lock                       (a function)
//      convert                                 (array created by conversion)
//      makemap                                 (map allocated via make)
//      makechan                                (channel allocated via make)
//      makeinterface                           (tagged object allocated by makeinterface)
//      <alloc in reflect.Zero>                 (allocation in instrinsic)
//      sync.Mutex                              (a reflect.rtype instance)
//      <command-line arguments>                (an intrinsic object)
//
// Labels within compound objects have subelement paths:
//      x.y[*].z                                (a struct variable, x)
//      append.y[*].z                           (array allocated by append)
//      makeslice.y[*].z                        (array allocated via make)
//
// TODO(adonovan): expose func LabelString(*types.Package, Label).
//
func (l Label) String() string {
	var s string
	switch v := l.obj.data.(type) {
	case types.Type:
		return v.String()

	case string:
		s = v // an intrinsic object (e.g. os.Args[*])

	case nil:
		if l.obj.cgn != nil {
			// allocation by intrinsic or reflective operation
			s = fmt.Sprintf("<alloc in %s>", l.obj.cgn.fn)
		} else {
			s = "<unknown>" // should be unreachable
		}

	case *ssa.Function:
		s = v.String()

	case *ssa.Global:
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
		panic(fmt.Sprintf("unhandled object data type: %T", v))
	}

	return s + l.subelement.path()
}
