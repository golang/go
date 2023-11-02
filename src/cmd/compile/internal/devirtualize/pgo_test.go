// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package devirtualize

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/pgo"
	"cmd/compile/internal/types"
	"cmd/compile/internal/typecheck"
	"cmd/internal/obj"
	"cmd/internal/src"
	"testing"
)

func init() {
	// These are the few constants that need to be initialized in order to use
	// the types package without using the typecheck package by calling
	// typecheck.InitUniverse() (the normal way to initialize the types package).
	types.PtrSize = 8
	types.RegSize = 8
	types.MaxWidth = 1 << 50
	typecheck.InitUniverse()
	base.Ctxt = &obj.Link{}
	base.Debug.PGODebug = 3
}

func makePos(b *src.PosBase, line, col uint) src.XPos {
	return base.Ctxt.PosTable.XPos(src.MakePos(b, line, col))
}

func TestFindHotConcreteCallee(t *testing.T) {
	// findHotConcreteCallee only uses pgo.Profile.WeightedCG, so we're
	// going to take a shortcut and only construct that.
	p := &pgo.Profile{
		WeightedCG: &pgo.IRGraph{
			IRNodes: make(map[string]*pgo.IRNode),
		},
	}

	// Create a new IRNode and add it to p.
	//
	// fn may be nil, in which case the node will set LinkerSymbolName.
	newNode := func(name string, fn *ir.Func) *pgo.IRNode {
		n := &pgo.IRNode{
			OutEdges: make(map[pgo.NamedCallEdge]*pgo.IREdge),
		}
		if fn != nil {
			n.AST = fn
		} else {
			n.LinkerSymbolName = name
		}
		p.WeightedCG.IRNodes[name] = n
		return n
	}

	// Add a new call edge from caller to callee.
	addEdge := func(caller, callee *pgo.IRNode, offset int, weight int64) {
		namedEdge := pgo.NamedCallEdge{
			CallerName:     caller.Name(),
			CalleeName:     callee.Name(),
			CallSiteOffset: offset,
		}
		irEdge := &pgo.IREdge{
			Src:            caller,
			Dst:            callee,
			CallSiteOffset: offset,
			Weight:         weight,
		}
		caller.OutEdges[namedEdge] = irEdge
	}

	pkgFoo := types.NewPkg("example.com/foo", "foo")
	basePos := src.NewFileBase("foo.go", "/foo.go")

	// Create a new struct type named structName with a method named methName and
	// return the method.
	makeStructWithMethod := func(structName, methName string) *ir.Func {
		// type structName struct{}
		structType := types.NewStruct(nil)

		// func (structName) methodName()
		recv := types.NewField(src.NoXPos, typecheck.Lookup(structName), structType)
		sig := types.NewSignature(recv, nil, nil)
		fn := ir.NewFunc(src.NoXPos, src.NoXPos, pkgFoo.Lookup(structName + "." + methName), sig)

		// Add the method to the struct.
		structType.SetMethods([]*types.Field{types.NewField(src.NoXPos, typecheck.Lookup(methName), sig)})

		return fn
	}

	const (
		// Caller start line.
		callerStart = 42

		// The line offset of the call we care about.
		callOffset = 1

		// The line offset of some other call we don't care about.
		wrongCallOffset = 2
	)

	// type IFace interface {
	//	Foo()
	// }
	fooSig := types.NewSignature(types.FakeRecv(), nil, nil)
	method := types.NewField(src.NoXPos, typecheck.Lookup("Foo"), fooSig)
	iface := types.NewInterface([]*types.Field{method})

	callerFn := ir.NewFunc(makePos(basePos, callerStart, 1), src.NoXPos, pkgFoo.Lookup("Caller"), types.NewSignature(nil, nil, nil))

	hotCalleeFn := makeStructWithMethod("HotCallee", "Foo")
	coldCalleeFn := makeStructWithMethod("ColdCallee", "Foo")
	wrongLineCalleeFn := makeStructWithMethod("WrongLineCallee", "Foo")
	wrongMethodCalleeFn := makeStructWithMethod("WrongMethodCallee", "Bar")

	callerNode := newNode("example.com/foo.Caller", callerFn)
	hotCalleeNode := newNode("example.com/foo.HotCallee.Foo", hotCalleeFn)
	coldCalleeNode := newNode("example.com/foo.ColdCallee.Foo", coldCalleeFn)
	wrongLineCalleeNode := newNode("example.com/foo.WrongCalleeLine.Foo", wrongLineCalleeFn)
	wrongMethodCalleeNode := newNode("example.com/foo.WrongCalleeMethod.Foo", wrongMethodCalleeFn)

	hotMissingCalleeNode := newNode("example.com/bar.HotMissingCallee.Foo", nil)

	addEdge(callerNode, wrongLineCalleeNode, wrongCallOffset, 100) // Really hot, but wrong line.
	addEdge(callerNode, wrongMethodCalleeNode, callOffset, 100) // Really hot, but wrong method type.
	addEdge(callerNode, hotCalleeNode, callOffset, 10)
	addEdge(callerNode, coldCalleeNode, callOffset, 1)

	// Equal weight, but IR missing.
	//
	// N.B. example.com/bar sorts lexicographically before example.com/foo,
	// so if the IR availability of hotCalleeNode doesn't get precedence,
	// this would be mistakenly selected.
	addEdge(callerNode, hotMissingCalleeNode, callOffset, 10)

	// IFace.Foo()
	sel := typecheck.NewMethodExpr(src.NoXPos, iface, typecheck.Lookup("Foo"))
	call := ir.NewCallExpr(makePos(basePos, callerStart+callOffset, 1), ir.OCALLINTER, sel, nil)

	gotFn, gotWeight := findHotConcreteCallee(p, callerFn, call)
	if gotFn != hotCalleeFn {
		t.Errorf("findHotConcreteInterfaceCallee func got %v want %v", gotFn, hotCalleeFn)
	}
	if gotWeight != 10 {
		t.Errorf("findHotConcreteInterfaceCallee weight got %v want 10", gotWeight)
	}
}
