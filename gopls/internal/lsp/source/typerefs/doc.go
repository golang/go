// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typerefs extracts from Go syntax a graph of symbol-level
// dependencies, for the purpose of precise invalidation of package data.
//
// # Background
//
// The goal of this analysis is to determine, for each package P, a nearly
// minimal set of packages that could affect the type checking of P. This set
// may contain false positives, but the smaller this set the better we can
// invalidate and prune packages in gopls.
//
// More precisely, for each package P we define the set of "reachable" packages
// from P as the set of packages that may affect the (deep) export data of the
// direct dependencies of P. By this definition, the complement of this set
// cannot affect any information derived from type checking P (e.g.
// diagnostics, cross references, or method sets). Therefore we need not
// invalidate any results for P when a package in the complement of this set
// changes.
//
// # Computing references
//
// For a given declaration D, references are computed based on identifiers or
// dotted identifiers referenced in the declaration of D, that may affect
// the type of D. However, these references reflect only local knowledge of the
// package and its dependency metadata, and do not depend on any analysis of
// the dependencies themselves.
//
// Specifically, if a referring identifier I appears in the declaration, we
// record an edge from D to each object possibly referenced by I. We search for
// references within type syntax, but do not actual type-check, so we can't
// reliably determine whether an expression is a type or a term, or whether a
// function is a builtin or generic. For example, the type of x in var x =
// p.F(W) only depends on W if p.F is a builtin or generic function, which we
// cannot know without type-checking package p. So we may over-approximate in
// this way.
//
//   - If I is declared in the current package, record a reference to its
//     declaration.
//   - Else, if there are any dot-imported imports in the current file and I is
//     exported, record a (possibly dangling) edge to the corresponding
//     declaration in each dot-imported package.
//
// If a dotted identifier q.I appears in the declaration, we
// perform a similar operation:
//   - If q is declared in the current package, we record a reference to that
//     object. It may be a var or const that has a field or method I.
//   - Else, if q is a valid import name based on imports in the current file
//     and the provided metadata for dependency package names, record a
//     reference to the object I in that package.
//   - Additionally, handle the case where Q is exported, and Q.I may refer to
//     a field or method in a dot-imported package.
//
// That is essentially the entire algorithm, though there is some subtlety to
// visiting the set of identifiers or dotted identifiers that may affect the
// declaration type. See the visitDeclOrSpec function for the details of this
// analysis. Notably, we also skip identifiers that refer to type parameters in
// generic declarations.
//
// # Graph optimizations
//
// The references extracted from the syntax are used to construct
// edges between declNodes. Edges are of two kinds: internal
// references, from one package-level declaration to another; and
// external references, from a symbol in this package to a symbol
// imported from a direct dependency.
//
// Once the symbol reference graph is constructed, we find its
// strongly connected components (SCCs) using Tarjan's algorithm. A
// node from each SCC is chosen arbitrarily to be its representative,
// and all the edges (internal and external) of the SCC are
// accumulated into the representative, thus forming the strong
// component graph, which is acyclic. This property simplifies the
// logic and improves the efficiency of the reachability query.
//
// TODO(adonovan): opt: subsequent planned optimizations include:
//
//   - The Hash-Value Numbering optimization described in
//     Hardekopf and Lin; see golang.org/x/go/pointer/hvn.go for an
//     implementation. (Like pointer analysis, our problem is
//     fundamentally one of graph reachability.)
//
//     The "pointer equivalence" (PE) portion of this algorithm uses a
//     hash table to create a mapping from unique sets of external
//     references to small integers. Each of the n external symbols
//     referenced by the package is assigned a integer from 1 to n;
//     this number stands for a singleton set. Higher numbers refer to
//     unions of strictly smaller sets. The PE algorithm allows us to
//     coalesce redundant graph nodes. For example, all functions that
//     ultimately reference only {fmt.Println,fmt.Sprintf} would be
//     marked as equivalent to each other, and to the union of
//     the sets of {fmt.Sprint} and {fmt.Println}.
//
//     This reduces the worst-case size of the Refs() result. Consider
//     M decls that each reference type t, which references N imported
//     types. The source code has O(M + N) lines but the Refs result
//     is current O(M*N). Preserving the essential structure of the
//     reference graph (as a DAG of union operations) will reduce the
//     asymptote.
//
//   - Serializing the SC graph obtained each package and saving it in
//     the file cache. Once we have a DAG of unions, we can serialize
//     it easily and amortize the cost of the local preprocessing.
//
// # API
//
// The main entry point for this analysis is the [Refs] function, which
// implements the aforementioned syntactic analysis for a set of files
// constituting a package.
//
// These references use shared state to efficiently represent references, by
// way of the [PackageIndex] and [PackageSet] types.
//
// The [BuildPackageGraph] constructor implements a whole-graph analysis similar
// to that which will be implemented by gopls, but for various reasons the
// logic for this analysis will eventually live in the
// [golang.org/x/tools/gopls/internal/lsp/cache] package. Nevertheless,
// BuildPackageGraph and its test serve to verify the syntactic analysis, and
// may serve as a proving ground for new optimizations of the whole-graph analysis.
//
// # Comparison with export data
//
// At first it may seem that the simplest way to implement this analysis would
// be to consider the types.Packages of the dependencies of P, for example
// during export. After all, it makes sense that the type checked packages
// themselves could describe their dependencies. However, this does not work as
// type information does not describe certain syntactic relationships.
//
// For example, the following scenarios cause type information to miss
// syntactic relationships:
//
// Named type forwarding:
//
//	package a; type A b.B
//	package b; type B int
//
// Aliases:
//
//	package a; func A(f b.B)
//	package b; type B = func()
//
// Initializers:
//
//	package a; var A = b.B()
//	package b; func B() string { return "hi" }
//
// Use of the unsafe package:
//
//	package a; type A [unsafe.Sizeof(B{})]int
//	package b; type B struct { f1, f2, f3 int }
//
// In all of these examples, types do not contain information about the edge
// between the a.A and b.B declarations.
package typerefs
