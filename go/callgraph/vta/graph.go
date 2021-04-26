// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vta

import (
	"fmt"
	"go/types"

	"golang.org/x/tools/go/ssa"
)

// node interface for VTA nodes.
type node interface {
	Type() types.Type
	String() string
}

// constant node for VTA.
type constant struct {
	typ types.Type
}

func (c constant) Type() types.Type {
	return c.typ
}

func (c constant) String() string {
	return fmt.Sprintf("Constant(%v)", c.Type())
}

// pointer node for VTA.
type pointer struct {
	typ *types.Pointer
}

func (p pointer) Type() types.Type {
	return p.typ
}

func (p pointer) String() string {
	return fmt.Sprintf("Pointer(%v)", p.Type())
}

// mapKey node for VTA, modeling reachable map key types.
type mapKey struct {
	typ types.Type
}

func (mk mapKey) Type() types.Type {
	return mk.typ
}

func (mk mapKey) String() string {
	return fmt.Sprintf("MapKey(%v)", mk.Type())
}

// mapValue node for VTA, modeling reachable map value types.
type mapValue struct {
	typ types.Type
}

func (mv mapValue) Type() types.Type {
	return mv.typ
}

func (mv mapValue) String() string {
	return fmt.Sprintf("MapValue(%v)", mv.Type())
}

// sliceElem node for VTA, modeling reachable slice element types.
type sliceElem struct {
	typ types.Type
}

func (s sliceElem) Type() types.Type {
	return s.typ
}

func (s sliceElem) String() string {
	return fmt.Sprintf("Slice([]%v)", s.Type())
}

// channelElem node for VTA, modeling reachable channel element types.
type channelElem struct {
	typ types.Type
}

func (c channelElem) Type() types.Type {
	return c.typ
}

func (c channelElem) String() string {
	return fmt.Sprintf("Channel(chan %v)", c.Type())
}

// field node for VTA.
type field struct {
	StructType types.Type
	index      int // index of the field in the struct
}

func (f field) Type() types.Type {
	s := f.StructType.Underlying().(*types.Struct)
	return s.Field(f.index).Type()
}

func (f field) String() string {
	s := f.StructType.Underlying().(*types.Struct)
	return fmt.Sprintf("Field(%v:%s)", f.StructType, s.Field(f.index).Name())
}

// global node for VTA.
type global struct {
	val *ssa.Global
}

func (g global) Type() types.Type {
	return g.val.Type()
}

func (g global) String() string {
	return fmt.Sprintf("Global(%s)", g.val.Name())
}

// local node for VTA modeling local variables
// and function/method parameters.
type local struct {
	val ssa.Value
}

func (l local) Type() types.Type {
	return l.val.Type()
}

func (l local) String() string {
	return fmt.Sprintf("Local(%s)", l.val.Name())
}

// indexedLocal node for VTA node. Models indexed locals
// related to the ssa extract instructions.
type indexedLocal struct {
	val   ssa.Value
	index int
	typ   types.Type
}

func (i indexedLocal) Type() types.Type {
	return i.typ
}

func (i indexedLocal) String() string {
	return fmt.Sprintf("Local(%s[%d])", i.val.Name(), i.index)
}

// function node for VTA.
type function struct {
	f *ssa.Function
}

func (f function) Type() types.Type {
	return f.f.Type()
}

func (f function) String() string {
	return fmt.Sprintf("Function(%s)", f.f.Name())
}

// nestedPtrInterface node represents all references and dereferences
// of locals and globals that have a nested pointer to interface type.
// We merge such constructs into a single node for simplicity and without
// much precision sacrifice as such variables are rare in practice. Both
// a and b would be represented as the same PtrInterface(I) node in:
//   type I interface
//   var a ***I
//   var b **I
type nestedPtrInterface struct {
	typ types.Type
}

func (l nestedPtrInterface) Type() types.Type {
	return l.typ
}

func (l nestedPtrInterface) String() string {
	return fmt.Sprintf("PtrInterface(%v)", l.typ)
}

// panicArg models types of all arguments passed to panic.
type panicArg struct{}

func (p panicArg) Type() types.Type {
	return nil
}

func (p panicArg) String() string {
	return "Panic"
}

// recoverReturn models types of all return values of recover().
type recoverReturn struct{}

func (r recoverReturn) Type() types.Type {
	return nil
}

func (r recoverReturn) String() string {
	return "Recover"
}

// vtaGraph remembers for each VTA node the set of its successors.
// Tailored for VTA, hence does not support singleton (sub)graphs.
type vtaGraph map[node]map[node]bool

// addEdge adds an edge x->y to the graph.
func (g vtaGraph) addEdge(x, y node) {
	succs, ok := g[x]
	if !ok {
		succs = make(map[node]bool)
		g[x] = succs
	}
	succs[y] = true
}

// successors returns all of n's immediate successors in the graph.
// The order of successor nodes is arbitrary.
func (g vtaGraph) successors(n node) []node {
	var succs []node
	for succ := range g[n] {
		succs = append(succs, succ)
	}
	return succs
}
