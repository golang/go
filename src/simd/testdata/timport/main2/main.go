// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// main2 validates cross-package SIMD package-variable handling in a package
// that intentionally does not directly import "simd":
//   - Package variables whose imported types (from timport/p) depend on SIMD
//     solely through their method sets (crossStruct, crossIface).
//   - Single-variable storage and address identity for an exported non-SIMD
//     package variable (p.N) initialized via SIMD in timport/p.
package main

import (
	"fmt"
	"os"
	"timport/p"
)

// Package-level variables in a non-"simd"-importing package whose inferred
// types from timport/p are SIMD-dependent solely through their method sets;
// midway must recognize their dependence, inject @simd/@bridge, and specialize
// them per size.
var (
	crossStruct = p.MakeHasSimdMethod()
	crossIface  = p.MakeSimdMethodIface()
)

// checkCrossPkgMethodDepVars validates that package variables initialized from
// imported method-set-SIMD-dependent types are properly initialized.
func checkCrossPkgMethodDepVars() {
	if crossStruct.X != 42 || crossIface == nil {
		fmt.Printf("FAIL, unexpected cross-pkg values: X=%d iface=%v\n", crossStruct.X, crossIface)
		os.Exit(1)
	}
}

// checkSingleVarIdentity validates that p.N (type int, initialized from
// simd.Int8s{}.Len() via init proxies) stays a single variable with a single
// address (p.PN == &p.N) shared between timport/p and external callers.
func checkSingleVarIdentity() {
	fmt.Println("init:", p.N, p.GetN(), *p.PN)
	if p.N < 16 || p.N&(p.N-1) != 0 {
		fmt.Printf("FAIL, unexpected initial p.N = %d\n", p.N)
		os.Exit(1)
	}
	if p.GetN() != p.N || *p.PN != p.N {
		fmt.Printf("FAIL, init mismatch: p.N=%d p.GetN()=%d *p.PN=%d\n", p.N, p.GetN(), *p.PN)
		os.Exit(1)
	}
	p.N = 1000
	fmt.Println("after p.N=1000: p.N =", p.N, "p.GetN() =", p.GetN())
	if p.N != 1000 || p.GetN() != 1000 {
		fmt.Printf("FAIL, after p.N=1000: p.N=%d p.GetN()=%d\n", p.N, p.GetN())
		os.Exit(1)
	}
	p.SetN(7)
	fmt.Println("after p.SetN(7): p.N =", p.N, "p.GetN() =", p.GetN())
	if p.N != 7 || p.GetN() != 7 {
		fmt.Printf("FAIL, after p.SetN(7): p.N=%d p.GetN()=%d\n", p.N, p.GetN())
		os.Exit(1)
	}
	fmt.Println("p.PN == &p.N:", p.PN == &p.N)
	if p.PN != &p.N {
		fmt.Println("FAIL, pointers are not equal")
		os.Exit(1)
	}
}

func main() {
	checkCrossPkgMethodDepVars()
	checkSingleVarIdentity()
}
