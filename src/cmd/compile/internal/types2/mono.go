// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// This file implements a check to validate that a Go package doesn't
// have unbounded recursive instantiation, which is not compatible
// with compilers using static instantiation (such as
// monomorphization).
//
// It implements a sort of "type flow" analysis by detecting which
// type parameters are instantiated with other type parameters (or
// types derived thereof). A package cannot be statically instantiated
// if the graph has any cycles involving at least one derived type.
//
// Concretely, we construct a directed, weighted graph. Vertices are
// used to represent type parameters as well as some defined
// types. Edges are used to represent how types depend on each other:
//
// * Everywhere a type-parameterized function or type is instantiated,
//   we add edges to each type parameter from the vertices (if any)
//   representing each type parameter or defined type referenced by
//   the type argument. If the type argument is just the referenced
//   type itself, then the edge has weight 0, otherwise 1.
//
// * For every defined type declared within a type-parameterized
//   function or method, we add an edge of weight 1 to the defined
//   type from each ambient type parameter.
//
// For example, given:
//
//	func f[A, B any]() {
//		type T int
//		f[T, map[A]B]()
//	}
//
// we construct vertices representing types A, B, and T. Because of
// declaration "type T int", we construct edges T<-A and T<-B with
// weight 1; and because of instantiation "f[T, map[A]B]" we construct
// edges A<-T with weight 0, and B<-A and B<-B with weight 1.
//
// Finally, we look for any positive-weight cycles. Zero-weight cycles
// are allowed because static instantiation will reach a fixed point.

type monoGraph struct {
	vertices []monoVertex
	edges    []monoEdge

	// canon maps method receiver type parameters to their respective
	// receiver type's type parameters.
	canon map[*TypeParam]*TypeParam

	// nameIdx maps a defined type or (canonical) type parameter to its
	// vertex index.
	nameIdx map[*TypeName]int
}

type monoVertex struct {
	weight int // weight of heaviest known path to this vertex
	pre    int // previous edge (if any) in the above path
	len    int // length of the above path

	// obj is the defined type or type parameter represented by this
	// vertex.
	obj *TypeName
}

type monoEdge struct {
	dst, src int
	weight   int

	pos syntax.Pos
	typ Type
}

func (check *Checker) monomorph() {
	// We detect unbounded instantiation cycles using a variant of
	// Bellman-Ford's algorithm. Namely, instead of always running |V|
	// iterations, we run until we either reach a fixed point or we've
	// found a path of length |V|. This allows us to terminate earlier
	// when there are no cycles, which should be the common case.

	again := true
	for again {
		again = false

		for i, edge := range check.mono.edges {
			src := &check.mono.vertices[edge.src]
			dst := &check.mono.vertices[edge.dst]

			// N.B., we're looking for the greatest weight paths, unlike
			// typical Bellman-Ford.
			w := src.weight + edge.weight
			if w <= dst.weight {
				continue
			}

			dst.pre = i
			dst.len = src.len + 1
			if dst.len == len(check.mono.vertices) {
				check.reportInstanceLoop(edge.dst)
				return
			}

			dst.weight = w
			again = true
		}
	}
}

func (check *Checker) reportInstanceLoop(v int) {
	var stack []int
	seen := make([]bool, len(check.mono.vertices))

	// We have a path that contains a cycle and ends at v, but v may
	// only be reachable from the cycle, not on the cycle itself. We
	// start by walking backwards along the path until we find a vertex
	// that appears twice.
	for !seen[v] {
		stack = append(stack, v)
		seen[v] = true
		v = check.mono.edges[check.mono.vertices[v].pre].src
	}

	// Trim any vertices we visited before visiting v the first
	// time. Since v is the first vertex we found within the cycle, any
	// vertices we visited earlier cannot be part of the cycle.
	for stack[0] != v {
		stack = stack[1:]
	}

	// TODO(mdempsky): Pivot stack so we report the cycle from the top?

	err := check.newError(InvalidInstanceCycle)
	obj0 := check.mono.vertices[v].obj
	err.addf(obj0, "instantiation cycle:")

	qf := RelativeTo(check.pkg)
	for _, v := range stack {
		edge := check.mono.edges[check.mono.vertices[v].pre]
		obj := check.mono.vertices[edge.dst].obj

		switch obj.Type().(type) {
		default:
			panic("unexpected type")
		case *Named:
			err.addf(atPos(edge.pos), "%s implicitly parameterized by %s", obj.Name(), TypeString(edge.typ, qf)) // secondary error, \t indented
		case *TypeParam:
			err.addf(atPos(edge.pos), "%s instantiated as %s", obj.Name(), TypeString(edge.typ, qf)) // secondary error, \t indented
		}
	}
	err.report()
}

// recordCanon records that tpar is the canonical type parameter
// corresponding to method type parameter mpar.
func (w *monoGraph) recordCanon(mpar, tpar *TypeParam) {
	if w.canon == nil {
		w.canon = make(map[*TypeParam]*TypeParam)
	}
	w.canon[mpar] = tpar
}

// recordInstance records that the given type parameters were
// instantiated with the corresponding type arguments.
func (w *monoGraph) recordInstance(pkg *Package, pos syntax.Pos, tparams []*TypeParam, targs []Type, xlist []syntax.Expr) {
	for i, tpar := range tparams {
		pos := pos
		if i < len(xlist) {
			pos = startPos(xlist[i])
		}
		w.assign(pkg, pos, tpar, targs[i])
	}
}

// assign records that tpar was instantiated as targ at pos.
func (w *monoGraph) assign(pkg *Package, pos syntax.Pos, tpar *TypeParam, targ Type) {
	// Go generics do not have an analog to C++`s template-templates,
	// where a template parameter can itself be an instantiable
	// template. So any instantiation cycles must occur within a single
	// package. Accordingly, we can ignore instantiations of imported
	// type parameters.
	//
	// TODO(mdempsky): Push this check up into recordInstance? All type
	// parameters in a list will appear in the same package.
	if tpar.Obj().Pkg() != pkg {
		return
	}

	// flow adds an edge from vertex src representing that typ flows to tpar.
	flow := func(src int, typ Type) {
		weight := 1
		if typ == targ {
			weight = 0
		}

		w.addEdge(w.typeParamVertex(tpar), src, weight, pos, targ)
	}

	// Recursively walk the type argument to find any defined types or
	// type parameters.
	var do func(typ Type)
	do = func(typ Type) {
		switch typ := Unalias(typ).(type) {
		default:
			panic("unexpected type")

		case *TypeParam:
			assert(typ.Obj().Pkg() == pkg)
			flow(w.typeParamVertex(typ), typ)

		case *Named:
			if src := w.localNamedVertex(pkg, typ.Origin()); src >= 0 {
				flow(src, typ)
			}

			targs := typ.TypeArgs()
			for i := 0; i < targs.Len(); i++ {
				do(targs.At(i))
			}

		case *Array:
			do(typ.Elem())
		case *Basic:
			// ok
		case *Chan:
			do(typ.Elem())
		case *Map:
			do(typ.Key())
			do(typ.Elem())
		case *Pointer:
			do(typ.Elem())
		case *Slice:
			do(typ.Elem())

		case *Interface:
			for i := 0; i < typ.NumMethods(); i++ {
				do(typ.Method(i).Type())
			}
		case *Signature:
			tuple := func(tup *Tuple) {
				for i := 0; i < tup.Len(); i++ {
					do(tup.At(i).Type())
				}
			}
			tuple(typ.Params())
			tuple(typ.Results())
		case *Struct:
			for i := 0; i < typ.NumFields(); i++ {
				do(typ.Field(i).Type())
			}
		}
	}
	do(targ)
}

// localNamedVertex returns the index of the vertex representing
// named, or -1 if named doesn't need representation.
func (w *monoGraph) localNamedVertex(pkg *Package, named *Named) int {
	obj := named.Obj()
	if obj.Pkg() != pkg {
		return -1 // imported type
	}

	root := pkg.Scope()
	if obj.Parent() == root {
		return -1 // package scope, no ambient type parameters
	}

	if idx, ok := w.nameIdx[obj]; ok {
		return idx
	}

	idx := -1

	// Walk the type definition's scope to find any ambient type
	// parameters that it's implicitly parameterized by.
	for scope := obj.Parent(); scope != root; scope = scope.Parent() {
		for _, elem := range scope.elems {
			if elem, ok := elem.(*TypeName); ok && !elem.IsAlias() && cmpPos(elem.Pos(), obj.Pos()) < 0 {
				if tpar, ok := elem.Type().(*TypeParam); ok {
					if idx < 0 {
						idx = len(w.vertices)
						w.vertices = append(w.vertices, monoVertex{obj: obj})
					}

					w.addEdge(idx, w.typeParamVertex(tpar), 1, obj.Pos(), tpar)
				}
			}
		}
	}

	if w.nameIdx == nil {
		w.nameIdx = make(map[*TypeName]int)
	}
	w.nameIdx[obj] = idx
	return idx
}

// typeParamVertex returns the index of the vertex representing tpar.
func (w *monoGraph) typeParamVertex(tpar *TypeParam) int {
	if x, ok := w.canon[tpar]; ok {
		tpar = x
	}

	obj := tpar.Obj()

	if idx, ok := w.nameIdx[obj]; ok {
		return idx
	}

	if w.nameIdx == nil {
		w.nameIdx = make(map[*TypeName]int)
	}

	idx := len(w.vertices)
	w.vertices = append(w.vertices, monoVertex{obj: obj})
	w.nameIdx[obj] = idx
	return idx
}

func (w *monoGraph) addEdge(dst, src, weight int, pos syntax.Pos, typ Type) {
	// TODO(mdempsky): Deduplicate redundant edges?
	w.edges = append(w.edges, monoEdge{
		dst:    dst,
		src:    src,
		weight: weight,

		pos: pos,
		typ: typ,
	})
}
