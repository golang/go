// Package ssa defines a representation of the elements of Go programs
// (packages, types, functions, variables and constants) using a
// static single-assignment (SSA) form intermediate representation
// (IR) for the the bodies of functions.
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
// tools.  It is not primarily intended for machine code generation.
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
// GorootLoader, which uses go/build to locate packages in the Go
// source distribution, and go/parser to parse them.
//
// The builder initially builds a naive SSA form in which all local
// variables are addresses of stack locations with explicit loads and
// stores.  If desired, registerisation and φ-node insertion using
// dominance and dataflow can be performed as a later pass to improve
// the accuracy and performance of subsequent analyses; this pass is
// not yet implemented.
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
// Given a Go source package such as this:
//
//      package main
//
//      import "fmt"
//
//      const message = "Hello, World!"
//
//      func hello() {
//              fmt.Println(message)
//      }
//
// The SSA Builder creates a *Program containing a main *Package such
// as this:
//
//      Package(Name: "main")
//        Members:
//          "message":          *Literal (Type: untyped string, Value: "Hello, World!")
//          "init·guard":       *Global (Type: *bool)
//          "hello":            *Function (Type: func())
//        Init:                 *Function (Type: func())
//
// The printed representation of the function main.hello is shown
// below.  Within the function listing, the name of each BasicBlock
// such as ".0.entry" is printed left-aligned, followed by the block's
// instructions, i.e. implementations of Instruction.
// For each instruction that defines an SSA virtual register
// (i.e. implements Value), the type of that value is shown in the
// right column.
//
//      # Name: main.hello
//      # Declared at hello.go:7:6
//      # Type: func()
//      func hello():
//      .0.entry:
//              t0 = new [1]interface{}                                                 *[1]interface{}
//              t1 = &t0[0:untyped integer]                                             *interface{}
//              t2 = make interface interface{} <- string ("Hello, World!":string)      interface{}
//              *t1 = t2
//              t3 = slice t0[:]                                                        []interface{}
//              t4 = fmt.Println(t3)                                                    (n int, err error)
//              ret
//
// TODO(adonovan): demonstrate more features in the example:
// parameters and control flow at the least.
//
// TODO(adonovan): Consider how token.Pos source location information
// should be made available generally.  Currently it is only present in
// Package, Function and CallCommon.
//
// TODO(adonovan): Provide an example skeleton application that loads
// and dumps the SSA form of a program.  Accommodate package-at-a-time
// vs. whole-program operation.
//
// TODO(adonovan): Consider the exceptional control-flow implications
// of defer and recover().
//
// TODO(adonovan): build tables/functions that relate source variables
// to SSA variables to assist user interfaces that make queries about
// specific source entities.
package ssa
