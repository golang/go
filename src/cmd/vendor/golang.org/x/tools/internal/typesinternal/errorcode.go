// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

//go:generate stringer -type=ErrorCode

type ErrorCode int

// This file defines the error codes that can be produced during type-checking.
// Collectively, these codes provide an identifier that may be used to
// implement special handling for certain types of errors.
//
// Error codes should be fine-grained enough that the exact nature of the error
// can be easily determined, but coarse enough that they are not an
// implementation detail of the type checking algorithm. As a rule-of-thumb,
// errors should be considered equivalent if there is a theoretical refactoring
// of the type checker in which they are emitted in exactly one place. For
// example, the type checker emits different error messages for "too many
// arguments" and "too few arguments", but one can imagine an alternative type
// checker where this check instead just emits a single "wrong number of
// arguments", so these errors should have the same code.
//
// Error code names should be as brief as possible while retaining accuracy and
// distinctiveness. In most cases names should start with an adjective
// describing the nature of the error (e.g. "invalid", "unused", "misplaced"),
// and end with a noun identifying the relevant language object. For example,
// "DuplicateDecl" or "InvalidSliceExpr". For brevity, naming follows the
// convention that "bad" implies a problem with syntax, and "invalid" implies a
// problem with types.

const (
	// InvalidSyntaxTree occurs if an invalid syntax tree is provided
	// to the type checker. It should never happen.
	InvalidSyntaxTree ErrorCode = -1
)

const (
	_ ErrorCode = iota

	// Test is reserved for errors that only apply while in self-test mode.
	Test

	/* package names */

	// BlankPkgName occurs when a package name is the blank identifier "_".
	//
	// Per the spec:
	//  "The PackageName must not be the blank identifier."
	BlankPkgName

	// MismatchedPkgName occurs when a file's package name doesn't match the
	// package name already established by other files.
	MismatchedPkgName

	// InvalidPkgUse occurs when a package identifier is used outside of a
	// selector expression.
	//
	// Example:
	//  import "fmt"
	//
	//  var _ = fmt
	InvalidPkgUse

	/* imports */

	// BadImportPath occurs when an import path is not valid.
	BadImportPath

	// BrokenImport occurs when importing a package fails.
	//
	// Example:
	//  import "amissingpackage"
	BrokenImport

	// ImportCRenamed occurs when the special import "C" is renamed. "C" is a
	// pseudo-package, and must not be renamed.
	//
	// Example:
	//  import _ "C"
	ImportCRenamed

	// UnusedImport occurs when an import is unused.
	//
	// Example:
	//  import "fmt"
	//
	//  func main() {}
	UnusedImport

	/* initialization */

	// InvalidInitCycle occurs when an invalid cycle is detected within the
	// initialization graph.
	//
	// Example:
	//  var x int = f()
	//
	//  func f() int { return x }
	InvalidInitCycle

	/* decls */

	// DuplicateDecl occurs when an identifier is declared multiple times.
	//
	// Example:
	//  var x = 1
	//  var x = 2
	DuplicateDecl

	// InvalidDeclCycle occurs when a declaration cycle is not valid.
	//
	// Example:
	//  import "unsafe"
	//
	//  type T struct {
	//  	a [n]int
	//  }
	//
	//  var n = unsafe.Sizeof(T{})
	InvalidDeclCycle

	// InvalidTypeCycle occurs when a cycle in type definitions results in a
	// type that is not well-defined.
	//
	// Example:
	//  import "unsafe"
	//
	//  type T [unsafe.Sizeof(T{})]int
	InvalidTypeCycle

	/* decls > const */

	// InvalidConstInit occurs when a const declaration has a non-constant
	// initializer.
	//
	// Example:
	//  var x int
	//  const _ = x
	InvalidConstInit

	// InvalidConstVal occurs when a const value cannot be converted to its
	// target type.
	//
	// TODO(findleyr): this error code and example are not very clear. Consider
	// removing it.
	//
	// Example:
	//  const _ = 1 << "hello"
	InvalidConstVal

	// InvalidConstType occurs when the underlying type in a const declaration
	// is not a valid constant type.
	//
	// Example:
	//  const c *int = 4
	InvalidConstType

	/* decls > var (+ other variable assignment codes) */

	// UntypedNilUse occurs when the predeclared (untyped) value nil is used to
	// initialize a variable declared without an explicit type.
	//
	// Example:
	//  var x = nil
	UntypedNilUse

	// WrongAssignCount occurs when the number of values on the right-hand side
	// of an assignment or initialization expression does not match the number
	// of variables on the left-hand side.
	//
	// Example:
	//  var x = 1, 2
	WrongAssignCount

	// UnassignableOperand occurs when the left-hand side of an assignment is
	// not assignable.
	//
	// Example:
	//  func f() {
	//  	const c = 1
	//  	c = 2
	//  }
	UnassignableOperand

	// NoNewVar occurs when a short variable declaration (':=') does not declare
	// new variables.
	//
	// Example:
	//  func f() {
	//  	x := 1
	//  	x := 2
	//  }
	NoNewVar

	// MultiValAssignOp occurs when an assignment operation (+=, *=, etc) does
	// not have single-valued left-hand or right-hand side.
	//
	// Per the spec:
	//  "In assignment operations, both the left- and right-hand expression lists
	//  must contain exactly one single-valued expression"
	//
	// Example:
	//  func f() int {
	//  	x, y := 1, 2
	//  	x, y += 1
	//  	return x + y
	//  }
	MultiValAssignOp

	// InvalidIfaceAssign occurs when a value of type T is used as an
	// interface, but T does not implement a method of the expected interface.
	//
	// Example:
	//  type I interface {
	//  	f()
	//  }
	//
	//  type T int
	//
	//  var x I = T(1)
	InvalidIfaceAssign

	// InvalidChanAssign occurs when a chan assignment is invalid.
	//
	// Per the spec, a value x is assignable to a channel type T if:
	//  "x is a bidirectional channel value, T is a channel type, x's type V and
	//  T have identical element types, and at least one of V or T is not a
	//  defined type."
	//
	// Example:
	//  type T1 chan int
	//  type T2 chan int
	//
	//  var x T1
	//  // Invalid assignment because both types are named
	//  var _ T2 = x
	InvalidChanAssign

	// IncompatibleAssign occurs when the type of the right-hand side expression
	// in an assignment cannot be assigned to the type of the variable being
	// assigned.
	//
	// Example:
	//  var x []int
	//  var _ int = x
	IncompatibleAssign

	// UnaddressableFieldAssign occurs when trying to assign to a struct field
	// in a map value.
	//
	// Example:
	//  func f() {
	//  	m := make(map[string]struct{i int})
	//  	m["foo"].i = 42
	//  }
	UnaddressableFieldAssign

	/* decls > type (+ other type expression codes) */

	// NotAType occurs when the identifier used as the underlying type in a type
	// declaration or the right-hand side of a type alias does not denote a type.
	//
	// Example:
	//  var S = 2
	//
	//  type T S
	NotAType

	// InvalidArrayLen occurs when an array length is not a constant value.
	//
	// Example:
	//  var n = 3
	//  var _ = [n]int{}
	InvalidArrayLen

	// BlankIfaceMethod occurs when a method name is '_'.
	//
	// Per the spec:
	//  "The name of each explicitly specified method must be unique and not
	//  blank."
	//
	// Example:
	//  type T interface {
	//  	_(int)
	//  }
	BlankIfaceMethod

	// IncomparableMapKey occurs when a map key type does not support the == and
	// != operators.
	//
	// Per the spec:
	//  "The comparison operators == and != must be fully defined for operands of
	//  the key type; thus the key type must not be a function, map, or slice."
	//
	// Example:
	//  var x map[T]int
	//
	//  type T []int
	IncomparableMapKey

	// InvalidIfaceEmbed occurs when a non-interface type is embedded in an
	// interface.
	//
	// Example:
	//  type T struct {}
	//
	//  func (T) m()
	//
	//  type I interface {
	//  	T
	//  }
	InvalidIfaceEmbed

	// InvalidPtrEmbed occurs when an embedded field is of the pointer form *T,
	// and T itself is itself a pointer, an unsafe.Pointer, or an interface.
	//
	// Per the spec:
	//  "An embedded field must be specified as a type name T or as a pointer to
	//  a non-interface type name *T, and T itself may not be a pointer type."
	//
	// Example:
	//  type T *int
	//
	//  type S struct {
	//  	*T
	//  }
	InvalidPtrEmbed

	/* decls > func and method */

	// BadRecv occurs when a method declaration does not have exactly one
	// receiver parameter.
	//
	// Example:
	//  func () _() {}
	BadRecv

	// InvalidRecv occurs when a receiver type expression is not of the form T
	// or *T, or T is a pointer type.
	//
	// Example:
	//  type T struct {}
	//
	//  func (**T) m() {}
	InvalidRecv

	// DuplicateFieldAndMethod occurs when an identifier appears as both a field
	// and method name.
	//
	// Example:
	//  type T struct {
	//  	m int
	//  }
	//
	//  func (T) m() {}
	DuplicateFieldAndMethod

	// DuplicateMethod occurs when two methods on the same receiver type have
	// the same name.
	//
	// Example:
	//  type T struct {}
	//  func (T) m() {}
	//  func (T) m(i int) int { return i }
	DuplicateMethod

	/* decls > special */

	// InvalidBlank occurs when a blank identifier is used as a value or type.
	//
	// Per the spec:
	//  "The blank identifier may appear as an operand only on the left-hand side
	//  of an assignment."
	//
	// Example:
	//  var x = _
	InvalidBlank

	// InvalidIota occurs when the predeclared identifier iota is used outside
	// of a constant declaration.
	//
	// Example:
	//  var x = iota
	InvalidIota

	// MissingInitBody occurs when an init function is missing its body.
	//
	// Example:
	//  func init()
	MissingInitBody

	// InvalidInitSig occurs when an init function declares parameters or
	// results.
	//
	// Example:
	//  func init() int { return 1 }
	InvalidInitSig

	// InvalidInitDecl occurs when init is declared as anything other than a
	// function.
	//
	// Example:
	//  var init = 1
	InvalidInitDecl

	// InvalidMainDecl occurs when main is declared as anything other than a
	// function, in a main package.
	InvalidMainDecl

	/* exprs */

	// TooManyValues occurs when a function returns too many values for the
	// expression context in which it is used.
	//
	// Example:
	//  func ReturnTwo() (int, int) {
	//  	return 1, 2
	//  }
	//
	//  var x = ReturnTwo()
	TooManyValues

	// NotAnExpr occurs when a type expression is used where a value expression
	// is expected.
	//
	// Example:
	//  type T struct {}
	//
	//  func f() {
	//  	T
	//  }
	NotAnExpr

	/* exprs > const */

	// TruncatedFloat occurs when a float constant is truncated to an integer
	// value.
	//
	// Example:
	//  var _ int = 98.6
	TruncatedFloat

	// NumericOverflow occurs when a numeric constant overflows its target type.
	//
	// Example:
	//  var x int8 = 1000
	NumericOverflow

	/* exprs > operation */

	// UndefinedOp occurs when an operator is not defined for the type(s) used
	// in an operation.
	//
	// Example:
	//  var c = "a" - "b"
	UndefinedOp

	// MismatchedTypes occurs when operand types are incompatible in a binary
	// operation.
	//
	// Example:
	//  var a = "hello"
	//  var b = 1
	//  var c = a - b
	MismatchedTypes

	// DivByZero occurs when a division operation is provable at compile
	// time to be a division by zero.
	//
	// Example:
	//  const divisor = 0
	//  var x int = 1/divisor
	DivByZero

	// NonNumericIncDec occurs when an increment or decrement operator is
	// applied to a non-numeric value.
	//
	// Example:
	//  func f() {
	//  	var c = "c"
	//  	c++
	//  }
	NonNumericIncDec

	/* exprs > ptr */

	// UnaddressableOperand occurs when the & operator is applied to an
	// unaddressable expression.
	//
	// Example:
	//  var x = &1
	UnaddressableOperand

	// InvalidIndirection occurs when a non-pointer value is indirected via the
	// '*' operator.
	//
	// Example:
	//  var x int
	//  var y = *x
	InvalidIndirection

	/* exprs > [] */

	// NonIndexableOperand occurs when an index operation is applied to a value
	// that cannot be indexed.
	//
	// Example:
	//  var x = 1
	//  var y = x[1]
	NonIndexableOperand

	// InvalidIndex occurs when an index argument is not of integer type,
	// negative, or out-of-bounds.
	//
	// Example:
	//  var s = [...]int{1,2,3}
	//  var x = s[5]
	//
	// Example:
	//  var s = []int{1,2,3}
	//  var _ = s[-1]
	//
	// Example:
	//  var s = []int{1,2,3}
	//  var i string
	//  var _ = s[i]
	InvalidIndex

	// SwappedSliceIndices occurs when constant indices in a slice expression
	// are decreasing in value.
	//
	// Example:
	//  var _ = []int{1,2,3}[2:1]
	SwappedSliceIndices

	/* operators > slice */

	// NonSliceableOperand occurs when a slice operation is applied to a value
	// whose type is not sliceable, or is unaddressable.
	//
	// Example:
	//  var x = [...]int{1, 2, 3}[:1]
	//
	// Example:
	//  var x = 1
	//  var y = 1[:1]
	NonSliceableOperand

	// InvalidSliceExpr occurs when a three-index slice expression (a[x:y:z]) is
	// applied to a string.
	//
	// Example:
	//  var s = "hello"
	//  var x = s[1:2:3]
	InvalidSliceExpr

	/* exprs > shift */

	// InvalidShiftCount occurs when the right-hand side of a shift operation is
	// either non-integer, negative, or too large.
	//
	// Example:
	//  var (
	//  	x string
	//  	y int = 1 << x
	//  )
	InvalidShiftCount

	// InvalidShiftOperand occurs when the shifted operand is not an integer.
	//
	// Example:
	//  var s = "hello"
	//  var x = s << 2
	InvalidShiftOperand

	/* exprs > chan */

	// InvalidReceive occurs when there is a channel receive from a value that
	// is either not a channel, or is a send-only channel.
	//
	// Example:
	//  func f() {
	//  	var x = 1
	//  	<-x
	//  }
	InvalidReceive

	// InvalidSend occurs when there is a channel send to a value that is not a
	// channel, or is a receive-only channel.
	//
	// Example:
	//  func f() {
	//  	var x = 1
	//  	x <- "hello!"
	//  }
	InvalidSend

	/* exprs > literal */

	// DuplicateLitKey occurs when an index is duplicated in a slice, array, or
	// map literal.
	//
	// Example:
	//  var _ = []int{0:1, 0:2}
	//
	// Example:
	//  var _ = map[string]int{"a": 1, "a": 2}
	DuplicateLitKey

	// MissingLitKey occurs when a map literal is missing a key expression.
	//
	// Example:
	//  var _ = map[string]int{1}
	MissingLitKey

	// InvalidLitIndex occurs when the key in a key-value element of a slice or
	// array literal is not an integer constant.
	//
	// Example:
	//  var i = 0
	//  var x = []string{i: "world"}
	InvalidLitIndex

	// OversizeArrayLit occurs when an array literal exceeds its length.
	//
	// Example:
	//  var _ = [2]int{1,2,3}
	OversizeArrayLit

	// MixedStructLit occurs when a struct literal contains a mix of positional
	// and named elements.
	//
	// Example:
	//  var _ = struct{i, j int}{i: 1, 2}
	MixedStructLit

	// InvalidStructLit occurs when a positional struct literal has an incorrect
	// number of values.
	//
	// Example:
	//  var _ = struct{i, j int}{1,2,3}
	InvalidStructLit

	// MissingLitField occurs when a struct literal refers to a field that does
	// not exist on the struct type.
	//
	// Example:
	//  var _ = struct{i int}{j: 2}
	MissingLitField

	// DuplicateLitField occurs when a struct literal contains duplicated
	// fields.
	//
	// Example:
	//  var _ = struct{i int}{i: 1, i: 2}
	DuplicateLitField

	// UnexportedLitField occurs when a positional struct literal implicitly
	// assigns an unexported field of an imported type.
	UnexportedLitField

	// InvalidLitField occurs when a field name is not a valid identifier.
	//
	// Example:
	//  var _ = struct{i int}{1: 1}
	InvalidLitField

	// UntypedLit occurs when a composite literal omits a required type
	// identifier.
	//
	// Example:
	//  type outer struct{
	//  	inner struct { i int }
	//  }
	//
	//  var _ = outer{inner: {1}}
	UntypedLit

	// InvalidLit occurs when a composite literal expression does not match its
	// type.
	//
	// Example:
	//  type P *struct{
	//  	x int
	//  }
	//  var _ = P {}
	InvalidLit

	/* exprs > selector */

	// AmbiguousSelector occurs when a selector is ambiguous.
	//
	// Example:
	//  type E1 struct { i int }
	//  type E2 struct { i int }
	//  type T struct { E1; E2 }
	//
	//  var x T
	//  var _ = x.i
	AmbiguousSelector

	// UndeclaredImportedName occurs when a package-qualified identifier is
	// undeclared by the imported package.
	//
	// Example:
	//  import "go/types"
	//
	//  var _ = types.NotAnActualIdentifier
	UndeclaredImportedName

	// UnexportedName occurs when a selector refers to an unexported identifier
	// of an imported package.
	//
	// Example:
	//  import "reflect"
	//
	//  type _ reflect.flag
	UnexportedName

	// UndeclaredName occurs when an identifier is not declared in the current
	// scope.
	//
	// Example:
	//  var x T
	UndeclaredName

	// MissingFieldOrMethod occurs when a selector references a field or method
	// that does not exist.
	//
	// Example:
	//  type T struct {}
	//
	//  var x = T{}.f
	MissingFieldOrMethod

	/* exprs > ... */

	// BadDotDotDotSyntax occurs when a "..." occurs in a context where it is
	// not valid.
	//
	// Example:
	//  var _ = map[int][...]int{0: {}}
	BadDotDotDotSyntax

	// NonVariadicDotDotDot occurs when a "..." is used on the final argument to
	// a non-variadic function.
	//
	// Example:
	//  func printArgs(s []string) {
	//  	for _, a := range s {
	//  		println(a)
	//  	}
	//  }
	//
	//  func f() {
	//  	s := []string{"a", "b", "c"}
	//  	printArgs(s...)
	//  }
	NonVariadicDotDotDot

	// MisplacedDotDotDot occurs when a "..." is used somewhere other than the
	// final argument to a function call.
	//
	// Example:
	//  func printArgs(args ...int) {
	//  	for _, a := range args {
	//  		println(a)
	//  	}
	//  }
	//
	//  func f() {
	//  	a := []int{1,2,3}
	//  	printArgs(0, a...)
	//  }
	MisplacedDotDotDot

	// InvalidDotDotDotOperand occurs when a "..." operator is applied to a
	// single-valued operand.
	//
	// Example:
	//  func printArgs(args ...int) {
	//  	for _, a := range args {
	//  		println(a)
	//  	}
	//  }
	//
	//  func f() {
	//  	a := 1
	//  	printArgs(a...)
	//  }
	//
	// Example:
	//  func args() (int, int) {
	//  	return 1, 2
	//  }
	//
	//  func printArgs(args ...int) {
	//  	for _, a := range args {
	//  		println(a)
	//  	}
	//  }
	//
	//  func g() {
	//  	printArgs(args()...)
	//  }
	InvalidDotDotDotOperand

	// InvalidDotDotDot occurs when a "..." is used in a non-variadic built-in
	// function.
	//
	// Example:
	//  var s = []int{1, 2, 3}
	//  var l = len(s...)
	InvalidDotDotDot

	/* exprs > built-in */

	// UncalledBuiltin occurs when a built-in function is used as a
	// function-valued expression, instead of being called.
	//
	// Per the spec:
	//  "The built-in functions do not have standard Go types, so they can only
	//  appear in call expressions; they cannot be used as function values."
	//
	// Example:
	//  var _ = copy
	UncalledBuiltin

	// InvalidAppend occurs when append is called with a first argument that is
	// not a slice.
	//
	// Example:
	//  var _ = append(1, 2)
	InvalidAppend

	// InvalidCap occurs when an argument to the cap built-in function is not of
	// supported type.
	//
	// See https://golang.org/ref/spec#Lengthand_capacity for information on
	// which underlying types are supported as arguments to cap and len.
	//
	// Example:
	//  var s = 2
	//  var x = cap(s)
	InvalidCap

	// InvalidClose occurs when close(...) is called with an argument that is
	// not of channel type, or that is a receive-only channel.
	//
	// Example:
	//  func f() {
	//  	var x int
	//  	close(x)
	//  }
	InvalidClose

	// InvalidCopy occurs when the arguments are not of slice type or do not
	// have compatible type.
	//
	// See https://golang.org/ref/spec#Appendingand_copying_slices for more
	// information on the type requirements for the copy built-in.
	//
	// Example:
	//  func f() {
	//  	var x []int
	//  	y := []int64{1,2,3}
	//  	copy(x, y)
	//  }
	InvalidCopy

	// InvalidComplex occurs when the complex built-in function is called with
	// arguments with incompatible types.
	//
	// Example:
	//  var _ = complex(float32(1), float64(2))
	InvalidComplex

	// InvalidDelete occurs when the delete built-in function is called with a
	// first argument that is not a map.
	//
	// Example:
	//  func f() {
	//  	m := "hello"
	//  	delete(m, "e")
	//  }
	InvalidDelete

	// InvalidImag occurs when the imag built-in function is called with an
	// argument that does not have complex type.
	//
	// Example:
	//  var _ = imag(int(1))
	InvalidImag

	// InvalidLen occurs when an argument to the len built-in function is not of
	// supported type.
	//
	// See https://golang.org/ref/spec#Lengthand_capacity for information on
	// which underlying types are supported as arguments to cap and len.
	//
	// Example:
	//  var s = 2
	//  var x = len(s)
	InvalidLen

	// SwappedMakeArgs occurs when make is called with three arguments, and its
	// length argument is larger than its capacity argument.
	//
	// Example:
	//  var x = make([]int, 3, 2)
	SwappedMakeArgs

	// InvalidMake occurs when make is called with an unsupported type argument.
	//
	// See https://golang.org/ref/spec#Makingslices_maps_and_channels for
	// information on the types that may be created using make.
	//
	// Example:
	//  var x = make(int)
	InvalidMake

	// InvalidReal occurs when the real built-in function is called with an
	// argument that does not have complex type.
	//
	// Example:
	//  var _ = real(int(1))
	InvalidReal

	/* exprs > assertion */

	// InvalidAssert occurs when a type assertion is applied to a
	// value that is not of interface type.
	//
	// Example:
	//  var x = 1
	//  var _ = x.(float64)
	InvalidAssert

	// ImpossibleAssert occurs for a type assertion x.(T) when the value x of
	// interface cannot have dynamic type T, due to a missing or mismatching
	// method on T.
	//
	// Example:
	//  type T int
	//
	//  func (t *T) m() int { return int(*t) }
	//
	//  type I interface { m() int }
	//
	//  var x I
	//  var _ = x.(T)
	ImpossibleAssert

	/* exprs > conversion */

	// InvalidConversion occurs when the argument type cannot be converted to the
	// target.
	//
	// See https://golang.org/ref/spec#Conversions for the rules of
	// convertibility.
	//
	// Example:
	//  var x float64
	//  var _ = string(x)
	InvalidConversion

	// InvalidUntypedConversion occurs when an there is no valid implicit
	// conversion from an untyped value satisfying the type constraints of the
	// context in which it is used.
	//
	// Example:
	//  var _ = 1 + ""
	InvalidUntypedConversion

	/* offsetof */

	// BadOffsetofSyntax occurs when unsafe.Offsetof is called with an argument
	// that is not a selector expression.
	//
	// Example:
	//  import "unsafe"
	//
	//  var x int
	//  var _ = unsafe.Offsetof(x)
	BadOffsetofSyntax

	// InvalidOffsetof occurs when unsafe.Offsetof is called with a method
	// selector, rather than a field selector, or when the field is embedded via
	// a pointer.
	//
	// Per the spec:
	//
	//  "If f is an embedded field, it must be reachable without pointer
	//  indirections through fields of the struct. "
	//
	// Example:
	//  import "unsafe"
	//
	//  type T struct { f int }
	//  type S struct { *T }
	//  var s S
	//  var _ = unsafe.Offsetof(s.f)
	//
	// Example:
	//  import "unsafe"
	//
	//  type S struct{}
	//
	//  func (S) m() {}
	//
	//  var s S
	//  var _ = unsafe.Offsetof(s.m)
	InvalidOffsetof

	/* control flow > scope */

	// UnusedExpr occurs when a side-effect free expression is used as a
	// statement. Such a statement has no effect.
	//
	// Example:
	//  func f(i int) {
	//  	i*i
	//  }
	UnusedExpr

	// UnusedVar occurs when a variable is declared but unused.
	//
	// Example:
	//  func f() {
	//  	x := 1
	//  }
	UnusedVar

	// MissingReturn occurs when a function with results is missing a return
	// statement.
	//
	// Example:
	//  func f() int {}
	MissingReturn

	// WrongResultCount occurs when a return statement returns an incorrect
	// number of values.
	//
	// Example:
	//  func ReturnOne() int {
	//  	return 1, 2
	//  }
	WrongResultCount

	// OutOfScopeResult occurs when the name of a value implicitly returned by
	// an empty return statement is shadowed in a nested scope.
	//
	// Example:
	//  func factor(n int) (i int) {
	//  	for i := 2; i < n; i++ {
	//  		if n%i == 0 {
	//  			return
	//  		}
	//  	}
	//  	return 0
	//  }
	OutOfScopeResult

	/* control flow > if */

	// InvalidCond occurs when an if condition is not a boolean expression.
	//
	// Example:
	//  func checkReturn(i int) {
	//  	if i {
	//  		panic("non-zero return")
	//  	}
	//  }
	InvalidCond

	/* control flow > for */

	// InvalidPostDecl occurs when there is a declaration in a for-loop post
	// statement.
	//
	// Example:
	//  func f() {
	//  	for i := 0; i < 10; j := 0 {}
	//  }
	InvalidPostDecl

	// InvalidChanRange occurs when a send-only channel used in a range
	// expression.
	//
	// Example:
	//  func sum(c chan<- int) {
	//  	s := 0
	//  	for i := range c {
	//  		s += i
	//  	}
	//  }
	InvalidChanRange

	// InvalidIterVar occurs when two iteration variables are used while ranging
	// over a channel.
	//
	// Example:
	//  func f(c chan int) {
	//  	for k, v := range c {
	//  		println(k, v)
	//  	}
	//  }
	InvalidIterVar

	// InvalidRangeExpr occurs when the type of a range expression is not array,
	// slice, string, map, or channel.
	//
	// Example:
	//  func f(i int) {
	//  	for j := range i {
	//  		println(j)
	//  	}
	//  }
	InvalidRangeExpr

	/* control flow > switch */

	// MisplacedBreak occurs when a break statement is not within a for, switch,
	// or select statement of the innermost function definition.
	//
	// Example:
	//  func f() {
	//  	break
	//  }
	MisplacedBreak

	// MisplacedContinue occurs when a continue statement is not within a for
	// loop of the innermost function definition.
	//
	// Example:
	//  func sumeven(n int) int {
	//  	proceed := func() {
	//  		continue
	//  	}
	//  	sum := 0
	//  	for i := 1; i <= n; i++ {
	//  		if i % 2 != 0 {
	//  			proceed()
	//  		}
	//  		sum += i
	//  	}
	//  	return sum
	//  }
	MisplacedContinue

	// MisplacedFallthrough occurs when a fallthrough statement is not within an
	// expression switch.
	//
	// Example:
	//  func typename(i interface{}) string {
	//  	switch i.(type) {
	//  	case int64:
	//  		fallthrough
	//  	case int:
	//  		return "int"
	//  	}
	//  	return "unsupported"
	//  }
	MisplacedFallthrough

	// DuplicateCase occurs when a type or expression switch has duplicate
	// cases.
	//
	// Example:
	//  func printInt(i int) {
	//  	switch i {
	//  	case 1:
	//  		println("one")
	//  	case 1:
	//  		println("One")
	//  	}
	//  }
	DuplicateCase

	// DuplicateDefault occurs when a type or expression switch has multiple
	// default clauses.
	//
	// Example:
	//  func printInt(i int) {
	//  	switch i {
	//  	case 1:
	//  		println("one")
	//  	default:
	//  		println("One")
	//  	default:
	//  		println("1")
	//  	}
	//  }
	DuplicateDefault

	// BadTypeKeyword occurs when a .(type) expression is used anywhere other
	// than a type switch.
	//
	// Example:
	//  type I interface {
	//  	m()
	//  }
	//  var t I
	//  var _ = t.(type)
	BadTypeKeyword

	// InvalidTypeSwitch occurs when .(type) is used on an expression that is
	// not of interface type.
	//
	// Example:
	//  func f(i int) {
	//  	switch x := i.(type) {}
	//  }
	InvalidTypeSwitch

	// InvalidExprSwitch occurs when a switch expression is not comparable.
	//
	// Example:
	//  func _() {
	//  	var a struct{ _ func() }
	//  	switch a /* ERROR cannot switch on a */ {
	//  	}
	//  }
	InvalidExprSwitch

	/* control flow > select */

	// InvalidSelectCase occurs when a select case is not a channel send or
	// receive.
	//
	// Example:
	//  func checkChan(c <-chan int) bool {
	//  	select {
	//  	case c:
	//  		return true
	//  	default:
	//  		return false
	//  	}
	//  }
	InvalidSelectCase

	/* control flow > labels and jumps */

	// UndeclaredLabel occurs when an undeclared label is jumped to.
	//
	// Example:
	//  func f() {
	//  	goto L
	//  }
	UndeclaredLabel

	// DuplicateLabel occurs when a label is declared more than once.
	//
	// Example:
	//  func f() int {
	//  L:
	//  L:
	//  	return 1
	//  }
	DuplicateLabel

	// MisplacedLabel occurs when a break or continue label is not on a for,
	// switch, or select statement.
	//
	// Example:
	//  func f() {
	//  L:
	//  	a := []int{1,2,3}
	//  	for _, e := range a {
	//  		if e > 10 {
	//  			break L
	//  		}
	//  		println(a)
	//  	}
	//  }
	MisplacedLabel

	// UnusedLabel occurs when a label is declared but not used.
	//
	// Example:
	//  func f() {
	//  L:
	//  }
	UnusedLabel

	// JumpOverDecl occurs when a label jumps over a variable declaration.
	//
	// Example:
	//  func f() int {
	//  	goto L
	//  	x := 2
	//  L:
	//  	x++
	//  	return x
	//  }
	JumpOverDecl

	// JumpIntoBlock occurs when a forward jump goes to a label inside a nested
	// block.
	//
	// Example:
	//  func f(x int) {
	//  	goto L
	//  	if x > 0 {
	//  	L:
	//  		print("inside block")
	//  	}
	// }
	JumpIntoBlock

	/* control flow > calls */

	// InvalidMethodExpr occurs when a pointer method is called but the argument
	// is not addressable.
	//
	// Example:
	//  type T struct {}
	//
	//  func (*T) m() int { return 1 }
	//
	//  var _ = T.m(T{})
	InvalidMethodExpr

	// WrongArgCount occurs when too few or too many arguments are passed by a
	// function call.
	//
	// Example:
	//  func f(i int) {}
	//  var x = f()
	WrongArgCount

	// InvalidCall occurs when an expression is called that is not of function
	// type.
	//
	// Example:
	//  var x = "x"
	//  var y = x()
	InvalidCall

	/* control flow > suspended */

	// UnusedResults occurs when a restricted expression-only built-in function
	// is suspended via go or defer. Such a suspension discards the results of
	// these side-effect free built-in functions, and therefore is ineffectual.
	//
	// Example:
	//  func f(a []int) int {
	//  	defer len(a)
	//  	return i
	//  }
	UnusedResults

	// InvalidDefer occurs when a deferred expression is not a function call,
	// for example if the expression is a type conversion.
	//
	// Example:
	//  func f(i int) int {
	//  	defer int32(i)
	//  	return i
	//  }
	InvalidDefer

	// InvalidGo occurs when a go expression is not a function call, for example
	// if the expression is a type conversion.
	//
	// Example:
	//  func f(i int) int {
	//  	go int32(i)
	//  	return i
	//  }
	InvalidGo

	// All codes below were added in Go 1.17.

	/* decl */

	// BadDecl occurs when a declaration has invalid syntax.
	BadDecl

	// RepeatedDecl occurs when an identifier occurs more than once on the left
	// hand side of a short variable declaration.
	//
	// Example:
	//  func _() {
	//  	x, y, y := 1, 2, 3
	//  }
	RepeatedDecl

	/* unsafe */

	// InvalidUnsafeAdd occurs when unsafe.Add is called with a
	// length argument that is not of integer type.
	//
	// Example:
	//  import "unsafe"
	//
	//  var p unsafe.Pointer
	//  var _ = unsafe.Add(p, float64(1))
	InvalidUnsafeAdd

	// InvalidUnsafeSlice occurs when unsafe.Slice is called with a
	// pointer argument that is not of pointer type or a length argument
	// that is not of integer type, negative, or out of bounds.
	//
	// Example:
	//  import "unsafe"
	//
	//  var x int
	//  var _ = unsafe.Slice(x, 1)
	//
	// Example:
	//  import "unsafe"
	//
	//  var x int
	//  var _ = unsafe.Slice(&x, float64(1))
	//
	// Example:
	//  import "unsafe"
	//
	//  var x int
	//  var _ = unsafe.Slice(&x, -1)
	//
	// Example:
	//  import "unsafe"
	//
	//  var x int
	//  var _ = unsafe.Slice(&x, uint64(1) << 63)
	InvalidUnsafeSlice

	// All codes below were added in Go 1.18.

	/* features */

	// UnsupportedFeature occurs when a language feature is used that is not
	// supported at this Go version.
	UnsupportedFeature

	/* type params */

	// NotAGenericType occurs when a non-generic type is used where a generic
	// type is expected: in type or function instantiation.
	//
	// Example:
	//  type T int
	//
	//  var _ T[int]
	NotAGenericType

	// WrongTypeArgCount occurs when a type or function is instantiated with an
	// incorrect number of type arguments, including when a generic type or
	// function is used without instantiation.
	//
	// Errors involving failed type inference are assigned other error codes.
	//
	// Example:
	//  type T[p any] int
	//
	//  var _ T[int, string]
	//
	// Example:
	//  func f[T any]() {}
	//
	//  var x = f
	WrongTypeArgCount

	// CannotInferTypeArgs occurs when type or function type argument inference
	// fails to infer all type arguments.
	//
	// Example:
	//  func f[T any]() {}
	//
	//  func _() {
	//  	f()
	//  }
	//
	// Example:
	//   type N[P, Q any] struct{}
	//
	//   var _ N[int]
	CannotInferTypeArgs

	// InvalidTypeArg occurs when a type argument does not satisfy its
	// corresponding type parameter constraints.
	//
	// Example:
	//  type T[P ~int] struct{}
	//
	//  var _ T[string]
	InvalidTypeArg // arguments? InferenceFailed

	// InvalidInstanceCycle occurs when an invalid cycle is detected
	// within the instantiation graph.
	//
	// Example:
	//  func f[T any]() { f[*T]() }
	InvalidInstanceCycle

	// InvalidUnion occurs when an embedded union or approximation element is
	// not valid.
	//
	// Example:
	//  type _ interface {
	//   	~int | interface{ m() }
	//  }
	InvalidUnion

	// MisplacedConstraintIface occurs when a constraint-type interface is used
	// outside of constraint position.
	//
	// Example:
	//   type I interface { ~int }
	//
	//   var _ I
	MisplacedConstraintIface

	// InvalidMethodTypeParams occurs when methods have type parameters.
	//
	// It cannot be encountered with an AST parsed using go/parser.
	InvalidMethodTypeParams

	// MisplacedTypeParam occurs when a type parameter is used in a place where
	// it is not permitted.
	//
	// Example:
	//  type T[P any] P
	//
	// Example:
	//  type T[P any] struct{ *P }
	MisplacedTypeParam

	// InvalidUnsafeSliceData occurs when unsafe.SliceData is called with
	// an argument that is not of slice type. It also occurs if it is used
	// in a package compiled for a language version before go1.20.
	//
	// Example:
	//  import "unsafe"
	//
	//  var x int
	//  var _ = unsafe.SliceData(x)
	InvalidUnsafeSliceData

	// InvalidUnsafeString occurs when unsafe.String is called with
	// a length argument that is not of integer type, negative, or
	// out of bounds. It also occurs if it is used in a package
	// compiled for a language version before go1.20.
	//
	// Example:
	//  import "unsafe"
	//
	//  var b [10]byte
	//  var _ = unsafe.String(&b[0], -1)
	InvalidUnsafeString

	// InvalidUnsafeStringData occurs if it is used in a package
	// compiled for a language version before go1.20.
	_ // not used anymore

)
