// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package spec describes all possible operations in the SIMD API.
//
// The SIMD spec describes the function and method signatures, documentation
// comments, and behavior (written as a reference Go implementation) of all
// possible Go SIMD APIs. The archsimd and simd packages are subsets of this
// specified API. This approach enforces that "one name means one thing" across
// all platforms and both packages.
//
// The spec is written as buildable and executable Go code, but isn't meant to
// be called directly. Rather, the specgen package interprets this spec package
// into a full API description, which can then be fed into other generators for
// the actual SIMD packages. The executable part of the spec serves to precisely
// specify the semantics of operations, and is intended for conformance testing.
//
// To see the spec-generated API and debug issues with it, use [cmd/specls].
//
// ## Basic operation specifications
//
// Spec operations are written in a stylized form that makes heavy use of type
// parameters so a single function can describe an operation generalized across
// many vector types. This is in contrast with the public SIMD API, where every
// function and method operates on concrete types. The specgen generator bridges
// this gap, instantiating a single parameterized spec function into many
// concrete methods.
//
// Consider a simple example, the Add operation:
//
//	// Add adds corresponding elements of two vectors.
//	func Add[E Nums, W Width](x, y Vec[E, W]) (z Vec[E, W]) {
//	    ...
//	}
//
// All spec operations are written as functions, but if the first parameter has
// type Vec, then they specify a method of a vector type. Since Add's first
// parameter (x) is a Vec, this describes a method on vector types.
//
// The "E Nums" type parameter controls the allowed element types of the three
// vector types. Here, it can be any numeric type of any size (uint8, float64,
// etc). The "W Width" type parameter controls the total bit width of the three
// vector types. The types that implement Width stand in for 128, 256, or 512
// bits, or "scalable", which can represent any power of two >= 128. The number
// of lanes of a vector is derived from the element size and the total vector
// width.
//
// The Add spec expands to all possible types that satisfy the E and W type
// parameters, which are in turn translated to types in the public API:
//
//	func (Int8x16) Add(Int8x16) Int8x16
//	func (Int8x32) Add(Int8x32) Int8x32
//	func (Int8x64) Add(Int8x64) Int8x64
//	func (Int8s) Add(Int8s) Int8s
//	...
//	func (Float64x8) Add(Float64x8) Float64x8
//	func (Float64s) Add(Float64s) Float64s
//
// ## Spec constraints
//
// For many operations, all possible combinations of their type parameters are
// valid, but some need to express constraints between type parameters that
// can't easily be described in the Go type system. For these, we support a
// `//specgen:requires` directive. Consider DotProductPairs:
//
//	// DotProductPairs computes the dot product of x and y.
//	//
//	//specgen:require z={xB}{xN*2}x{xL/2}
//	func DotProductPairs[E Nums, W Width, zE Nums](x, y Vec[E, W]) (z Vec[zE, W]) {
//	    ...
//	}
//
// The require expression refers to the types of each parameter and result by
// name. For Vec (and Array) arguments, each parameter and result also get
// several related variables:
//
//	v  = The whole vector type (e.g., Uint32x8)
//	vE = The element type (e.g., uint32)
//	vB = The base type (e.g., uint)
//	vN = The base type size (e.g., 32)
//	vL = The number of lanes in the vector or elements in the array (e.g., 8)
//	vW = The total bit width of the vector or array (e.g., 256)
//
// The full syntax for constraints is described in the [specgen/specexpr]
// package, but they often describe vector shapes like those in the
// DotProductPairs example. The form of these is:
//
//   - BaseNxL, which describes a vector with L elements of type BaseN;
//   - BaseNwW, which describes a vector of total width W; or
//   - BaseNs, which describes a scalable vector of BaseN elements.
//
// Base, N, L, or W can be either a literal or an expression in {}'s. For
// example, Uint32x{vL} or {zB}{zN}w128.
//
// For DotProductPairs, the constraint "z={vB}{vN*2}x{vL/2}" says the result z
// must have the same base type as v, but z's element type must be twice as
// wide, and z must have half the number of lanes of v.
//
// The DotProductPairs constraint could also have been written in any of the
// following equivalent ways:
//
//	z={vB}{vN*2}w{vW}       Constrain the total vector width
//	zB=vB zN=vN*2 zL=vL/2   Constrain each component separately
//	zE={vB}{vN*2} zW=vW     Constrain the element type and width separately
//
// ## Name and doc templates
//
// For some operations, the API name or documentation depends on type
// parameters. For these we support a simple template system where constraint
// variables can be referenced in curly braces, like {vE}, similar to {}
// expressions in shape constraints. For doc comments, these can be included
// directly in the doc comment. For names, we use a `//specgen:name` directive,
// such as
//
//	//specgen:name Load{z}
//	func LoadZ[E Elt, W Width](s []E) (z Vec[E, W]) {
//
// In this case, the spec function itself can be named anything (as long as it's
// exported), and the API name is generated from the directive. For example,
// when LoadZ is instantiated on uint32 and Width128, the API name generated
// from the template will be LoadUint32x4. This is particularly useful for
// constructor functions and conversion functions where types must appear in the
// name, such as LoadZ.
//
// ## The spec type system
//
// This package defines a set of types that translate to API types. We saw the
// [Vec] type above, which translates to a concrete BNxL (or BNs) vector type in
// the API, where L is determined from E and W.
//
// Masks are also represented using the [Vec] type, but with an element type
// from the [MaskElt] interface, such as Mask8, Mask16, etc. The generator
// translates these to Mask types in the API. Internal to the spec package,
// these are like a wide mask, where only 0 and ^0 are legal values for these
// elements.
//
// Similar to [Vec], there an [Array[E,W]] type that translates into a [L]E Go
// array type in the API.
//
// The type [UintN] stands for a uint type whose bit width is determined by spec
// constraints.
//
// Pointer and slice types translate directly to the API.
package spec
