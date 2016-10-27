// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test basic restrictions on alias declarations.

package p

import (
	"flag"
	"fmt" // use at most once (to test "imported but not used" error)
	"go/build"
	. "go/build"
	"io"
	"math"
	"unsafe"
)

// helper
var before struct {
	f
}

// aliases must refer to package-qualified identifiers
// TODO(gri) should only see one error for declaration below - fix this
const _ => 0 // ERROR "unexpected literal 0|_ is not a package-qualified identifier"

type _ => _ // ERROR "_ is not a package-qualified identifier"
type t => _ // ERROR "_ is not a package-qualified identifier"

const _ => iota // ERROR "iota is not a package-qualified identifier"
type _ => int   // ERROR "int is not a package-qualified identifier"

const c => iota // ERROR "iota is not a package-qualified identifier"
type t => int   // ERROR "int is not a package-qualified identifier"

// dot-imported identifiers are not qualified identifiers
// TODO(gri) fix error printing - should not print a qualified identifier...
var _ => Default // ERROR "build\.Default is not a package-qualified identifier"

// qualified identifiers must start with a package
var _ => before.f  // ERROR "before is not a package"
func _ => before.f // ERROR "before is not a package"
var _ => after.m   // ERROR "after is not a package"
func _ => after.m  // ERROR "after is not a package"

var v => before.f  // ERROR "before is not a package"
func f => before.f // ERROR "before is not a package"
var v => after.m   // ERROR "after is not a package"
func f => after.m  // ERROR "after is not a package"

// TODO(gri) fix error printing - should print correct qualified identifier...
var _ => Default.ARCH // ERROR "build.Default is not a package"

// aliases may not refer to package unsafe
type ptr => unsafe.Pointer // ERROR "ptr refers to package unsafe"
func size => unsafe.Sizeof // ERROR "size refers to package unsafe"

// aliases must refer to entities of the same kind
const _ => math.Pi
const pi => math.Pi
const pi1 => math.Sin // ERROR "math.Sin is not a constant"

type _ => io.Writer
type writer => io.Writer
type writer1 => math.Sin // ERROR "math.Sin is not a type"

var _ => build.Default
var def => build.Default
var def1 => build.Import // ERROR "build.Import is not a variable"

func _ => math.Sin
func sin => math.Sin
func sin1 => math.Pi // ERROR "math.Pi is not a function"

// aliases may not be called init
func init => flag.Parse // ERROR "cannot declare init"

// alias reference to a package marks package as used
func _ => fmt.Println

// re-exported aliases
const Pi => math.Pi

type Writer => io.Writer

var Def => build.Default

func Sin => math.Sin

// type aliases denote identical types
type myPackage => build.Package

var pkg myPackage
var _ build.Package = pkg   // valid assignment
var _ *build.Package = &pkg // valid assignment

// helper
type after struct{}

func (after) m() {}
