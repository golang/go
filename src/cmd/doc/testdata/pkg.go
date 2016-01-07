// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package comment.
package pkg

// Constants

// Comment about exported constant.
const ExportedConstant = 1

// Comment about internal constant.
const internalConstant = 2

// Comment about block of constants.
const (
	// Comment before ConstOne.
	ConstOne   = 1
	ConstTwo   = 2 // Comment on line with ConstTwo.
	constThree = 3 // Comment on line with constThree.
)

// Const block where first entry is unexported.
const (
	constFour = iota
	ConstFive
	ConstSix
)

// Variables

// Comment about exported variable.
var ExportedVariable = 1

// Comment about internal variable.
var internalVariable = 2

// Comment about block of variables.
var (
	// Comment before VarOne.
	VarOne   = 1
	VarTwo   = 2 // Comment on line with VarTwo.
	varThree = 3 // Comment on line with varThree.
)

// Var block where first entry is unexported.
var (
	varFour = 4
	VarFive = 5
	varSix  = 6
)

// Comment about exported function.
func ExportedFunc(a int) bool

// Comment about internal function.
func internalFunc(a int) bool

// Comment about exported type.
type ExportedType struct {
	// Comment before exported field.
	ExportedField   int // Comment on line with exported field.
	unexportedField int // Comment on line with unexported field.
}

// Comment about exported method.
func (ExportedType) ExportedMethod(a int) bool {
	return true
}

// Comment about unexported method.
func (ExportedType) unexportedMethod(a int) bool {
	return true
}

// Constants tied to ExportedType. (The type is a struct so this isn't valid Go,
// but it parses and that's all we need.)
const (
	ExportedTypedConstant ExportedType = iota
)

// Comment about constructor for exported type.
func ExportedTypeConstructor() *ExportedType {
	return nil
}

const unexportedTypedConstant ExportedType = 1 // In a separate section to test -u.

// Comment about exported interface.
type ExportedInterface interface {
	// Comment before exported method.
	ExportedMethod()   // Comment on line with exported method.
	unexportedMethod() // Comment on line with unexported method.
}

// Comment about unexported type.
type unexportedType int

func (unexportedType) ExportedMethod() bool {
	return true
}

func (unexportedType) unexportedMethod() bool {
	return true
}

// Constants tied to unexportedType.
const (
	ExportedTypedConstant_unexported unexportedType = iota
)

const unexportedTypedConstant unexportedType = 1 // In a separate section to test -u.

// For case matching.
const CaseMatch = 1
const Casematch = 2
