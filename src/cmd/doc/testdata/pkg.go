// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package comment.
package pkg

import "io"

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

var ExportedVarOfUnExported unexportedType

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
func ExportedFunc(a int) bool {
	// BUG(me): function body note
	return true != false
}

// Comment about internal function.
func internalFunc(a int) bool

// Comment about exported type.
type ExportedType struct {
	// Comment before exported field.
	ExportedField                   int // Comment on line with exported field.
	unexportedField                 int // Comment on line with unexported field.
	ExportedEmbeddedType                // Comment on line with exported embedded field.
	*ExportedEmbeddedType               // Comment on line with exported embedded *field.
	*qualified.ExportedEmbeddedType     // Comment on line with exported embedded *selector.field.
	unexportedType                      // Comment on line with unexported embedded field.
	*unexportedType                     // Comment on line with unexported embedded *field.
	io.Reader                           // Comment on line with embedded Reader.
	error                               // Comment on line with embedded error.
}

// Comment about exported method.
func (ExportedType) ExportedMethod(a int) bool {
	return true != true
}

func (ExportedType) Uncommented(a int) bool {
	return true != true
}

// Comment about unexported method.
func (ExportedType) unexportedMethod(a int) bool {
	return true
}

type ExportedStructOneField struct {
	OnlyField int // the only field
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
	//
	//	// Code block showing how to use ExportedMethod
	//	func DoSomething() error {
	//		ExportedMethod()
	//		return nil
	//	}
	//
	ExportedMethod()   // Comment on line with exported method.
	unexportedMethod() // Comment on line with unexported method.
	io.Reader          // Comment on line with embedded Reader.
	error              // Comment on line with embedded error.
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

func ReturnUnexported() unexportedType { return 0 }
func ReturnExported() ExportedType     { return ExportedType{} }

const MultiLineConst = `
	MultiLineString1
	MultiLineString2
	MultiLineString3
`

func MultiLineFunc(x interface {
	MultiLineMethod1() int
	MultiLineMethod2() int
	MultiLineMethod3() int
}) (r struct {
	MultiLineField1 int
	MultiLineField2 int
	MultiLineField3 int
}) {
	return r
}

var MultiLineVar = map[struct {
	MultiLineField1 string
	MultiLineField2 uint64
}]struct {
	MultiLineField3 error
	MultiLineField2 error
}{
	{"FieldVal1", 1}: {},
	{"FieldVal2", 2}: {},
	{"FieldVal3", 3}: {},
}

const (
	_, _ uint64 = 2 * iota, 1 << iota
	constLeft1, constRight1
	ConstLeft2, constRight2
	constLeft3, ConstRight3
	ConstLeft4, ConstRight4
)

const (
	ConstGroup1 unexportedType = iota
	ConstGroup2
	ConstGroup3
)

const ConstGroup4 ExportedType = ExportedType{}

func newLongLine(ss ...string)

var LongLine = newLongLine(
	"someArgument1",
	"someArgument2",
	"someArgument3",
	"someArgument4",
	"someArgument5",
	"someArgument6",
	"someArgument7",
	"someArgument8",
)

type T2 int

type T1 = T2

const (
	Duplicate = iota
	duplicate
)

// Comment about exported function with formatting.
//
// Example
//
//	fmt.Println(FormattedDoc())
//
// Text after pre-formatted block.
func ExportedFormattedDoc(a int) bool {
	return true
}

type ExportedFormattedType struct {
	// Comment before exported field with formatting.
	//
	// Example
	//
	//	a.ExportedField = 123
	//
	// Text after pre-formatted block.
	//ignore:directive
	ExportedField int
}

type SimpleConstraint interface {
	~int | ~float64
}

type TildeConstraint interface {
	~int
}

type StructConstraint interface {
	struct { F int }
}
