// -lang=go1.17

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parser accepts type parameters but the type checker
// needs to report any operations that are not permitted
// before Go 1.18.

package p

type T[P /* ERROR type parameters require go1\.18 or later */ any /* ERROR undeclared name: any \(requires version go1\.18 or later\) */ ] struct{}

// for init (and main, but we're not in package main) we should only get one error
func init[P /* ERROR func init must have no type parameters */ any /* ERROR undeclared name: any \(requires version go1\.18 or later\) */ ]()   {}
func main[P /* ERROR type parameters require go1\.18 or later */ any /* ERROR undeclared name: any \(requires version go1\.18 or later\) */ ]() {}

func f[P /* ERROR type parameters require go1\.18 or later */ any /* ERROR undeclared name: any \(requires version go1\.18 or later\) */ ](x P) {
	var _ T[ /* ERROR type instantiation requires go1\.18 or later */ int]
	var _ (T[ /* ERROR type instantiation requires go1\.18 or later */ int])
	_ = T[ /* ERROR type instantiation requires go1\.18 or later */ int]{}
	_ = T[ /* ERROR type instantiation requires go1\.18 or later */ int](struct{}{})
}

func (T[ /* ERROR type instantiation requires go1\.18 or later */ P]) g(x int) {
	f[ /* ERROR function instantiation requires go1\.18 or later */ int](0)     // explicit instantiation
	(f[ /* ERROR function instantiation requires go1\.18 or later */ int])(0)   // parentheses (different code path)
	f( /* ERROR implicit function instantiation requires go1\.18 or later */ x) // implicit instantiation
}

type C1 interface {
	comparable // ERROR undeclared name: comparable \(requires version go1\.18 or later\)
}

type C2 interface {
	comparable // ERROR undeclared name: comparable \(requires version go1\.18 or later\)
	int        // ERROR embedding non-interface type int requires go1\.18 or later
	~ /* ERROR embedding interface element ~int requires go1\.18 or later */ int
	int /* ERROR embedding interface element int\|~string requires go1\.18 or later */ | ~string
}

type _ interface {
	// errors for these were reported with their declaration
	C1
	C2
}

type (
	_ comparable // ERROR undeclared name: comparable \(requires version go1\.18 or later\)
	// errors for these were reported with their declaration
	_ C1
	_ C2

	_ = comparable // ERROR undeclared name: comparable \(requires version go1\.18 or later\)
	// errors for these were reported with their declaration
	_ = C1
	_ = C2
)
