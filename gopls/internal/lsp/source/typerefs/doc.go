// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typerefs extracts symbol-level reachability information
// from the syntax of a Go package.
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
// cannot affect any information derived from type checking P, such as
// diagnostics, cross references, or method sets. Therefore we need not
// invalidate any results for P when a package in the complement of this set
// changes.
//
// # Computing references
//
// For a given declaration D, references are computed based on identifiers or
// dotted identifiers referenced in the declaration of D, that may affect
// the type of D. However, these references reflect only local knowledge of the
// package and its dependency metadata, and do not depend on any analysis of
// the dependencies themselves. This allows the reference information for
// a package to be cached independent of all others.
//
// Specifically, if a referring identifier I appears in the declaration, we
// record an edge from D to each object possibly referenced by I. We search for
// references within type syntax, but do not actually type-check, so we can't
// reliably determine whether an expression is a type or a term, or whether a
// function is a builtin or generic. For example, the type of x in var x =
// p.F(W) only depends on W if p.F is a builtin or generic function, which we
// cannot know without type-checking package p. So we may over-approximate in
// this way.
//
//   - If I is declared in the current package, record a reference to its
//     declaration.
//   - Otherwise, if there are any dot imports in the current
//     file and I is exported, record a (possibly dangling) edge to
//     the corresponding declaration in each dot-imported package.
//
// If a dotted identifier q.I appears in the declaration, we
// perform a similar operation:
//
//   - If q is declared in the current package, we record a reference to that
//     object. It may be a var or const that has a field or method I.
//   - Otherwise, if q is a valid import name based on imports in the current file
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
// edges between nodes representing declarations. Edges are of two
// kinds: internal references, from one package-level declaration to
// another; and external references, from a symbol in this package to
// a symbol imported from a direct dependency.
//
// Once the symbol reference graph is constructed, we find its
// strongly connected components (SCCs) using Tarjan's algorithm.
// As we coalesce the nodes of each SCC we compute the union of
// external references reached by each package-level declaration.
// The final result is the mapping from each exported package-level
// declaration to the set of external (imported) declarations that it
// reaches.
//
// Because it is common for many package members to have the same
// reachability, the result takes the form of a set of equivalence
// classes, each mapping a set of package-level declarations to a set
// of external symbols. We use a hash table to canonicalize sets so that
// repeated occurrences of the same set (which are common) are only
// represented once in memory or in the file system.
// For example, all declarations that ultimately reference only
// {fmt.Println,strings.Join} would be classed as equivalent.
//
// This approach was inspired by the Hash-Value Numbering (HVN)
// optimization described by Hardekopf and Lin. See
// golang.org/x/tools/go/pointer/hvn.go for an implementation. (Like
// pointer analysis, this problem is fundamentally one of graph
// reachability.) The HVN algorithm takes the compression a step
// further by preserving the topology of the SCC DAG, in which edges
// represent "is a superset of" constraints. Redundant edges that
// don't increase the solution can be deleted. We could apply the same
// technique here to further reduce the worst-case size of the result,
// but the current implementation seems adequate.
//
// # API
//
// The main entry point for this analysis is the [Encode] function,
// which implements the analysis described above for one package, and
// encodes the result as a binary message.
//
// The [Decode] function decodes the message into a usable form: a set
// of equivalence classes. The decoder uses a shared [PackageIndex] to
// enable more compact representations of sets of packages
// ([PackageSet]) during the global reacahability computation.
//
// The [BuildPackageGraph] constructor implements a whole-graph analysis similar
// to that which will be implemented by gopls, but for various reasons the
// logic for this analysis will eventually live in the
// [golang.org/x/tools/gopls/internal/lsp/cache] package. Nevertheless,
// BuildPackageGraph and its test serve to verify the syntactic analysis, and
// may serve as a proving ground for new optimizations of the whole-graph analysis.
//
// # Export data is insufficient
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
