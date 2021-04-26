// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vta computes the call graph of a Go program using the Variable
// Type Analysis (VTA) algorithm originally described in ``Practical Virtual
// Method Call Resolution for Java," Vijay Sundaresan, Laurie Hendren,
// Chrislain Razafimahefa, Raja Vallée-Rai, Patrick Lam, Etienne Gagnon, and
// Charles Godin.
//
// Note: this package is in experimental phase and its interface is
// subject to change.
// TODO(zpavlinovic): reiterate on documentation.
//
// The VTA algorithm overapproximates the set of types (and function literals)
// a variable can take during runtime by building a global type propagation
// graph and propagating types (and function literals) through the graph.
//
// A type propagation is a directed, labeled graph. A node can represent
// one of the following:
//  - A field of a struct type.
//  - A local (SSA) variable of a method/function.
//  - All pointers to a non-interface type.
//  - The return value of a method.
//  - All elements in an array.
//  - All elements in a slice.
//  - All elements in a map.
//  - All elements in a channel.
//  - A global variable.
// In addition, the implementation used in this package introduces
// a few Go specific kinds of nodes:
//  - (De)references of nested pointers to interfaces are modeled
//    as a unique nestedPtrInterface node in the type propagation graph.
//  - Each function literal is represented as a function node whose
//    internal value is the (SSA) representation of the function. This
//    is done to precisely infer flow of higher-order functions.
//
// Edges in the graph represent flow of types (and function literals) through
// the program. That is, the model 1) typing constraints that are induced by
// assignment statements or function and method calls and 2) higher-order flow
// of functions in the program.
//
// The labeling function maps each node to a set of types and functions that
// can intuitively reach the program construct the node represents. Initially,
// every node is assigned a type corresponding to the program construct it
// represents. Function nodes are also assigned the function they represent.
// The labeling function then propagates types and function through the graph.
//
// The result of VTA is a type propagation graph in which each node is labeled
// with a conservative overapproximation of the set of types (and functions)
// it may have. This information is then used to construct the call graph.
// For each unresolved call site, vta uses the set of types and functions
// reaching the node representing the call site to create a set of callees.

package vta

// TODO(zpavlinovic): add exported VTA library functions.
