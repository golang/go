// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ssa defines a representation of the elements of Go programs
// (packages, types, functions, variables and constants) using a
// static single-assignment (SSA) form intermediate representation
// (IR) for the bodies of functions.
//
// THIS INTERFACE IS EXPERIMENTAL AND IS LIKELY TO CHANGE.
//
// For an introduction to SSA form, see
// http://en.wikipedia.org/wiki/Static_single_assignment_form.
// This page provides a broader reading list:
// http://www.dcs.gla.ac.uk/~jsinger/ssa.html.
//
// The level of abstraction of the SSA form is intentionally close to
// the source language to facilitate construction of source analysis
// tools.  It is not intended for machine code generation.
//
// All looping, branching and switching constructs are replaced with
// unstructured control flow.  We may add higher-level control flow
// primitives in the future to facilitate constant-time dispatch of
// switch statements, for example.
//
// Builder encapsulates the tasks of type-checking (using go/types)
// abstract syntax trees (as defined by go/ast) for the source files
// comprising a Go program, and the conversion of each function from
// Go ASTs to the SSA representation.
//
// By supplying an instance of the SourceLocator function prototype,
// clients may control how the builder locates, loads and parses Go
// sources files for imported packages.  This package provides
// MakeGoBuildLoader, which creates a loader that uses go/build to
// locate packages in the Go source distribution, and go/parser to
// parse them.
//
// The builder initially builds a naive SSA form in which all local
// variables are addresses of stack locations with explicit loads and
// stores.  Registerisation of eligible locals and φ-node insertion
// using dominance and dataflow are then performed as a second pass
// called "lifting" to improve the accuracy and performance of
// subsequent analyses; this pass can be skipped by setting the
// NaiveForm builder flag.
//
// The primary interfaces of this package are:
//
//    - Member: a named member of a Go package.
//    - Value: an expression that yields a value.
//    - Instruction: a statement that consumes values and performs computation.
//
// A computation that yields a result implements both the Value and
// Instruction interfaces.  The following table shows for each
// concrete type which of these interfaces it implements.
//
//                      Value?          Instruction?    Member?
//   *Alloc             ✔               ✔
//   *BinOp             ✔               ✔
//   *Builtin           ✔               ✔
//   *Call              ✔               ✔
//   *Capture           ✔
//   *ChangeInterface   ✔               ✔
//   *ChangeType        ✔               ✔
//   *Const             ✔
//   *Convert           ✔               ✔
//   *DebugRef                          ✔
//   *Defer                             ✔
//   *Extract           ✔               ✔
//   *Field             ✔               ✔
//   *FieldAddr         ✔               ✔
//   *Function          ✔                               ✔ (func)
//   *Global            ✔                               ✔ (var)
//   *Go                                ✔
//   *If                                ✔
//   *Index             ✔               ✔
//   *IndexAddr         ✔               ✔
//   *Jump                              ✔
//   *Lookup            ✔               ✔
//   *MakeChan          ✔               ✔
//   *MakeClosure       ✔               ✔
//   *MakeInterface     ✔               ✔
//   *MakeMap           ✔               ✔
//   *MakeSlice         ✔               ✔
//   *MapUpdate                         ✔
//   *NamedConst                                        ✔ (const)
//   *Next              ✔               ✔
//   *Panic                             ✔
//   *Parameter         ✔
//   *Phi               ✔               ✔
//   *Range             ✔               ✔
//   *Return                            ✔
//   *RunDefers                         ✔
//   *Select            ✔               ✔
//   *Send                              ✔
//   *Slice             ✔               ✔
//   *Store                             ✔
//   *Type                                              ✔ (type)
//   *TypeAssert        ✔               ✔
//   *UnOp              ✔               ✔
//
// Other key types in this package include: Program, Package, Function
// and BasicBlock.
//
// The program representation constructed by this package is fully
// resolved internally, i.e. it does not rely on the names of Values,
// Packages, Functions, Types or BasicBlocks for the correct
// interpretation of the program.  Only the identities of objects and
// the topology of the SSA and type graphs are semantically
// significant.  (There is one exception: Ids, used to identify field
// and method names, contain strings.)  Avoidance of name-based
// operations simplifies the implementation of subsequent passes and
// can make them very efficient.  Many objects are nonetheless named
// to aid in debugging, but it is not essential that the names be
// either accurate or unambiguous.  The public API exposes a number of
// name-based maps for client convenience.
//
// The ssa/ssautil package provides various utilities that depend only
// on the public API of this package.
//
// TODO(adonovan): Consider the exceptional control-flow implications
// of defer and recover().
//
// TODO(adonovan): write a how-to document for all the various cases
// of trying to determine corresponding elements across the four
// domains of source locations, ast.Nodes, types.Objects,
// ssa.Values/Instructions.
//
package ssa
