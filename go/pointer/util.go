// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

import (
	"bytes"
	"fmt"
	"go/types"
	exec "golang.org/x/sys/execabs"
	"log"
	"os"
	"runtime"
	"time"

	"golang.org/x/tools/container/intsets"
)

// CanPoint reports whether the type T is pointerlike,
// for the purposes of this analysis.
func CanPoint(T types.Type) bool {
	switch T := T.(type) {
	case *types.Named:
		if obj := T.Obj(); obj.Name() == "Value" && obj.Pkg().Path() == "reflect" {
			return true // treat reflect.Value like interface{}
		}
		return CanPoint(T.Underlying())
	case *types.Pointer, *types.Interface, *types.Map, *types.Chan, *types.Signature, *types.Slice:
		return true
	}

	return false // array struct tuple builtin basic
}

// CanHaveDynamicTypes reports whether the type T can "hold" dynamic types,
// i.e. is an interface (incl. reflect.Type) or a reflect.Value.
func CanHaveDynamicTypes(T types.Type) bool {
	switch T := T.(type) {
	case *types.Named:
		if obj := T.Obj(); obj.Name() == "Value" && obj.Pkg().Path() == "reflect" {
			return true // reflect.Value
		}
		return CanHaveDynamicTypes(T.Underlying())
	case *types.Interface:
		return true
	}
	return false
}

func isInterface(T types.Type) bool { return types.IsInterface(T) }

// mustDeref returns the element type of its argument, which must be a
// pointer; panic ensues otherwise.
func mustDeref(typ types.Type) types.Type {
	return typ.Underlying().(*types.Pointer).Elem()
}

// deref returns a pointer's element type; otherwise it returns typ.
func deref(typ types.Type) types.Type {
	if p, ok := typ.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
}

// A fieldInfo describes one subelement (node) of the flattening-out
// of a type T: the subelement's type and its path from the root of T.
//
// For example, for this type:
//
//	type line struct{ points []struct{x, y int} }
//
// flatten() of the inner struct yields the following []fieldInfo:
//
//	struct{ x, y int }                      ""
//	int                                     ".x"
//	int                                     ".y"
//
// and flatten(line) yields:
//
//	struct{ points []struct{x, y int} }     ""
//	struct{ x, y int }                      ".points[*]"
//	int                                     ".points[*].x
//	int                                     ".points[*].y"
type fieldInfo struct {
	typ types.Type

	// op and tail describe the path to the element (e.g. ".a#2.b[*].c").
	op   interface{} // *Array: true; *Tuple: int; *Struct: *types.Var; *Named: nil
	tail *fieldInfo
}

// path returns a user-friendly string describing the subelement path.
func (fi *fieldInfo) path() string {
	var buf bytes.Buffer
	for p := fi; p != nil; p = p.tail {
		switch op := p.op.(type) {
		case bool:
			fmt.Fprintf(&buf, "[*]")
		case int:
			fmt.Fprintf(&buf, "#%d", op)
		case *types.Var:
			fmt.Fprintf(&buf, ".%s", op.Name())
		}
	}
	return buf.String()
}

// flatten returns a list of directly contained fields in the preorder
// traversal of the type tree of t.  The resulting elements are all
// scalars (basic types or pointerlike types), except for struct/array
// "identity" nodes, whose type is that of the aggregate.
//
// reflect.Value is considered pointerlike, similar to interface{}.
//
// Callers must not mutate the result.
func (a *analysis) flatten(t types.Type) []*fieldInfo {
	fl, ok := a.flattenMemo[t]
	if !ok {
		switch t := t.(type) {
		case *types.Named:
			u := t.Underlying()
			if isInterface(u) {
				// Debuggability hack: don't remove
				// the named type from interfaces as
				// they're very verbose.
				fl = append(fl, &fieldInfo{typ: t})
			} else {
				fl = a.flatten(u)
			}

		case *types.Basic,
			*types.Signature,
			*types.Chan,
			*types.Map,
			*types.Interface,
			*types.Slice,
			*types.Pointer:
			fl = append(fl, &fieldInfo{typ: t})

		case *types.Array:
			fl = append(fl, &fieldInfo{typ: t}) // identity node
			for _, fi := range a.flatten(t.Elem()) {
				fl = append(fl, &fieldInfo{typ: fi.typ, op: true, tail: fi})
			}

		case *types.Struct:
			fl = append(fl, &fieldInfo{typ: t}) // identity node
			for i, n := 0, t.NumFields(); i < n; i++ {
				f := t.Field(i)
				for _, fi := range a.flatten(f.Type()) {
					fl = append(fl, &fieldInfo{typ: fi.typ, op: f, tail: fi})
				}
			}

		case *types.Tuple:
			// No identity node: tuples are never address-taken.
			n := t.Len()
			if n == 1 {
				// Don't add a fieldInfo link for singletons,
				// e.g. in params/results.
				fl = append(fl, a.flatten(t.At(0).Type())...)
			} else {
				for i := 0; i < n; i++ {
					f := t.At(i)
					for _, fi := range a.flatten(f.Type()) {
						fl = append(fl, &fieldInfo{typ: fi.typ, op: i, tail: fi})
					}
				}
			}

		default:
			panic(fmt.Sprintf("cannot flatten unsupported type %T", t))
		}

		a.flattenMemo[t] = fl
	}

	return fl
}

// sizeof returns the number of pointerlike abstractions (nodes) in the type t.
func (a *analysis) sizeof(t types.Type) uint32 {
	return uint32(len(a.flatten(t)))
}

// shouldTrack reports whether object type T contains (recursively)
// any fields whose addresses should be tracked.
func (a *analysis) shouldTrack(T types.Type) bool {
	if a.track == trackAll {
		return true // fast path
	}
	track, ok := a.trackTypes[T]
	if !ok {
		a.trackTypes[T] = true // break cycles conservatively
		// NB: reflect.Value, reflect.Type are pre-populated to true.
		for _, fi := range a.flatten(T) {
			switch ft := fi.typ.Underlying().(type) {
			case *types.Interface, *types.Signature:
				track = true // needed for callgraph
			case *types.Basic:
				// no-op
			case *types.Chan:
				track = a.track&trackChan != 0 || a.shouldTrack(ft.Elem())
			case *types.Map:
				track = a.track&trackMap != 0 || a.shouldTrack(ft.Key()) || a.shouldTrack(ft.Elem())
			case *types.Slice:
				track = a.track&trackSlice != 0 || a.shouldTrack(ft.Elem())
			case *types.Pointer:
				track = a.track&trackPtr != 0 || a.shouldTrack(ft.Elem())
			case *types.Array, *types.Struct:
				// No need to look at field types since they will follow (flattened).
			default:
				// Includes *types.Tuple, which are never address-taken.
				panic(ft)
			}
			if track {
				break
			}
		}
		a.trackTypes[T] = track
		if !track && a.log != nil {
			fmt.Fprintf(a.log, "\ttype not tracked: %s\n", T)
		}
	}
	return track
}

// offsetOf returns the (abstract) offset of field index within struct
// or tuple typ.
func (a *analysis) offsetOf(typ types.Type, index int) uint32 {
	var offset uint32
	switch t := typ.Underlying().(type) {
	case *types.Tuple:
		for i := 0; i < index; i++ {
			offset += a.sizeof(t.At(i).Type())
		}
	case *types.Struct:
		offset++ // the node for the struct itself
		for i := 0; i < index; i++ {
			offset += a.sizeof(t.Field(i).Type())
		}
	default:
		panic(fmt.Sprintf("offsetOf(%s : %T)", typ, typ))
	}
	return offset
}

// sliceToArray returns the type representing the arrays to which
// slice type slice points.
func sliceToArray(slice types.Type) *types.Array {
	return types.NewArray(slice.Underlying().(*types.Slice).Elem(), 1)
}

// Node set -------------------------------------------------------------------

type nodeset struct {
	intsets.Sparse
}

func (ns *nodeset) String() string {
	var buf bytes.Buffer
	buf.WriteRune('{')
	var space [50]int
	for i, n := range ns.AppendTo(space[:0]) {
		if i > 0 {
			buf.WriteString(", ")
		}
		buf.WriteRune('n')
		fmt.Fprintf(&buf, "%d", n)
	}
	buf.WriteRune('}')
	return buf.String()
}

func (ns *nodeset) add(n nodeid) bool {
	return ns.Sparse.Insert(int(n))
}

func (ns *nodeset) addAll(y *nodeset) bool {
	return ns.UnionWith(&y.Sparse)
}

// Profiling & debugging -------------------------------------------------------

var timers = make(map[string]time.Time)

func start(name string) {
	if debugTimers {
		timers[name] = time.Now()
		log.Printf("%s...\n", name)
	}
}

func stop(name string) {
	if debugTimers {
		log.Printf("%s took %s\n", name, time.Since(timers[name]))
	}
}

// diff runs the command "diff a b" and reports its success.
func diff(a, b string) bool {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "plan9":
		cmd = exec.Command("/bin/diff", "-c", a, b)
	default:
		cmd = exec.Command("/usr/bin/diff", "-u", a, b)
	}
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run() == nil
}
