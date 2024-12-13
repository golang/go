// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"iter"
	"reflect"
	"strings"
	"sync"
	"testing"
)

var debug = flag.Bool("debug", false, "show the errors produced by the tests")

// T has lots of interesting pieces to use to test execution.
type T struct {
	// Basics
	True        bool
	I           int
	U16         uint16
	X, S        string
	FloatZero   float64
	ComplexZero complex128
	// Nested structs.
	U *U
	// Struct with String method.
	V0     V
	V1, V2 *V
	// Struct with Error method.
	W0     W
	W1, W2 *W
	// Slices
	SI      []int
	SICap   []int
	SIEmpty []int
	SB      []bool
	// Arrays
	AI [3]int
	// Maps
	MSI      map[string]int
	MSIone   map[string]int // one element, for deterministic output
	MSIEmpty map[string]int
	MXI      map[any]int
	MII      map[int]int
	MI32S    map[int32]string
	MI64S    map[int64]string
	MUI32S   map[uint32]string
	MUI64S   map[uint64]string
	MI8S     map[int8]string
	MUI8S    map[uint8]string
	SMSI     []map[string]int
	// Empty interfaces; used to see if we can dig inside one.
	Empty0 any // nil
	Empty1 any
	Empty2 any
	Empty3 any
	Empty4 any
	// Non-empty interfaces.
	NonEmptyInterface         I
	NonEmptyInterfacePtS      *I
	NonEmptyInterfaceNil      I
	NonEmptyInterfaceTypedNil I
	// Stringer.
	Str fmt.Stringer
	Err error
	// Pointers
	PI  *int
	PS  *string
	PSI *[]int
	NIL *int
	UPI unsafe.Pointer
	EmptyUPI unsafe.Pointer
	// Function (not method)
	BinaryFunc             func(string, string) string
	VariadicFunc           func(...string) string
	VariadicFuncInt        func(int, ...string) string
	NilOKFunc              func(*int) bool
	ErrFunc                func() (string, error)
	PanicFunc              func() string
	TooFewReturnCountFunc  func()
	TooManyReturnCountFunc func() (string, error, int)
	InvalidReturnTypeFunc  func() (string, bool)
	// Template to test evaluation of templates.
	Tmpl *Template
	// Unexported field; cannot be accessed by template.
	unexported int
}

type S []string

func (S) Method0() string {
	return "M0"
}

type U struct {
	V string
}

type V struct {
	j int
}

func (v *V) String() string {
	if v == nil {
		return "nilV"
	}
	return fmt.Sprintf("<%d>", v.j)
}

type W struct {
	k int
}

func (w *W) Error() string {
	if w == nil {
		return "nilW"
	}
	return fmt.Sprintf("[%d]", w.k)
}

var siVal = I(S{"a", "b"})

var tVal = &T{
	True:   true,
	I:      17,
	U16:    16,
	X:      "x",
	S:      "xyz",
	U:      &U{"v"},
	V0:     V{6666},
	V1:     &V{7777}, // leave V2 as nil
	W0:     W{888},
	W1:     &W{999}, // leave W2 as nil
	SI:     []int{3, 4, 5},
	SICap:  make([]int, 5, 10),
	AI:     [3]int{3, 4, 5},
	SB:     []bool{true, false},
	MSI:    map[string]int{"one": 1, "two": 2, "three": 3},
	MSIone: map[string]int{"one": 1},
	MXI:    map[any]int{"one": 1},
	MII:    map[int]int{1: 1},
	MI32S:  map[int32]string{1: "one", 2: "two"},
	MI64S:  map[int64]string{2: "i642", 3: "i643"},
	MUI32S: map[uint32]string{2: "u322", 3: "u323"},
	MUI64S: map[uint64]string{2: "ui642", 3: "ui643"},
	MI8S:   map[int8]string{2: "i82", 3: "i83"},
	MUI8S:  map[uint8]string{2: "u82", 3: "u83"},
	SMSI: []map[string]int{
		{"one": 1, "two": 2},
		{"eleven": 11, "twelve": 12},
	},
	Empty1:                    3,
	Empty2:                    "empty2",
	Empty3:                    []int{7, 8},
	Empty4:                    &U{"UinEmpty"},
	NonEmptyInterface:         &T{X: "x"},
	NonEmptyInterfacePtS:      &siVal,
	NonEmptyInterfaceTypedNil: (*T)(nil),
	Str:                       bytes.NewBuffer([]byte("foozle")),
	Err:                       errors.New("erroozle"),
	PI:                        newInt(23),
	PS:                        newString("a string"),
	PSI:                       newIntSlice(21, 22, 23),
	UPI:                       newUnsafePointer(23),
	BinaryFunc:                func(a, b string) string { return fmt.Sprintf("[%s=%s]", a, b) },
	VariadicFunc:              func(s ...string) string { return fmt.Sprint("<", strings.Join(s, "+"), ">") },
	VariadicFuncInt:           func(a int, s ...string) string { return fmt.Sprint(a, "=<", strings.Join(s, "+"), ">") },
	NilOKFunc:                 func(s *int) bool { return s == nil },
	ErrFunc:                   func() (string, error) { return "bla", nil },
	PanicFunc:                 func() string { panic("test panic") },
	TooFewReturnCountFunc:     func() {},
	TooManyReturnCountFunc:    func() (string, error, int) { return "", nil, 0 },
	InvalidReturnTypeFunc:     func() (string, bool) { return "", false },
	Tmpl:                      Must(New("x").Parse("test template")), // "x" is the value of .X
}

var tSliceOfNil = []*T{nil}

// A non-empty interface.
type I interface {
	Method0() string
}

var iVal I = tVal

// Helpers for creation.
func newInt(n int) *int {
	return &n
}

func newUnsafePointer(n int) unsafe.Pointer {
	return unsafe.Pointer(&n)
}

func newString(s string) *string {
	return &s
}

func newIntSlice(n ...int) *[]int {
	p := new([]int)
	*p = make([]int, len(n))
	copy(*p, n)
	return p
}

// Simple methods with and without arguments.
func (t *T) Method0() string {
	return "M0"
}

func (t *T) Method1(a int) int {
	return a
}

func (t *T) Method2(a uint16, b string) string {
	return fmt.Sprintf("Method2: %d %s", a, b)
}

func (t *T) Method3(v any) string {
	return fmt.Sprintf("Method3: %v", v)
}

func (t *T) Copy() *T {
	n := new(T)
	*n = *t
	return n
}

func (t *T) MAdd(a int, b []int) []int {
	v := make([]int, len(b))
	for i, x := range b {
		v[i] = x + a
	}
	return v
}

var myError = errors.New("my error")

// MyError returns a value and an error according to its argument.
func (t *T) MyError(error bool) (bool, error) {
	if error {
		return true, myError
	}
	return false, nil
}

// A few methods to test chaining.
func (t *T) GetU() *U {
	return t.U
}

func (u *U) TrueFalse(b bool) string {
	if b {
		return "true"
	}
	return ""
}

func typeOf(arg any) string {
	return fmt.Sprintf("%T", arg)
}

type execTest struct {
	name   string
	input  string
	output string
	data   any
	ok     bool
}

// bigInt and bigUint are hex string representing numbers either side
// of the max int boundary.
// We do it this way so the test doesn't depend on ints being 32 bits.
var (
	bigInt  = fmt.Sprintf("0x%x", int(1<<uint(reflect.TypeFor[int]().Bits()-1)-1))
	bigUint = fmt.Sprintf("0x%x", uint(1<<uint(reflect.TypeFor[int]().Bits()-1)))
)

var execTests = []execTest{
	// Trivial cases.
	{"empty", "", "", nil, true},
	{"text", "some text", "some text", nil, true},
	{"nil action", "{{nil}}", "", nil, false},

	// Ideal constants.
	{"ideal int", "{{typeOf 3}}", "int", 0, true},
	{"ideal float", "{{typeOf 1.0}}", "float64", 0, true},
	{"ideal exp float", "{{typeOf 1e1}}", "float64", 0, true},
	{"ideal complex", "{{typeOf 1i}}", "complex128", 0, true},
	{"ideal int", "{{typeOf " + bigInt + "}}", "int", 0, true},
	{"ideal too big", "{{typeOf " + bigUint + "}}", "", 0, false},
	{"ideal nil without type", "{{nil}}", "", 0, false},

	// Fields of structs.
	{".X", "-{{.X}}-", "-x-", tVal, true},
	{".U.V", "-{{.U.V}}-", "-v-", tVal, true},
	{".unexported", "{{.unexported}}", "", tVal, false},

	// Fields on maps.
	{"map .one", "{{.MSI.one}}", "1", tVal, true},
	{"map .two", "{{.MSI.two}}", "2", tVal, true},
	{"map .NO", "{{.MSI.NO}}", "<no value>", tVal, true},
	{"map .one interface", "{{.MXI.one}}", "1", tVal, true},
	{"map .WRONG args", "{{.MSI.one 1}}", "", tVal, false},
	{"map .WRONG type", "{{.MII.one}}", "", tVal, false},

	// Dots of all kinds to test basic evaluation.
	{"dot int", "<{{.}}>", "<13>", 13, true},
	{"dot uint", "<{{.}}>", "<14>", uint(14), true},
	{"dot float", "<{{.}}>", "<15.1>", 15.1, true},
	{"dot bool", "<{{.}}>", "<true>", true, true},
	{"dot complex", "<{{.}}>", "<(16.2-17i)>", 16.2 - 17i, true},
	{"dot string", "<{{.}}>", "<hello>", "hello", true},
	{"dot slice", "<{{.}}>", "<[-1 -2 -3]>", []int{-1, -2, -3}, true},
	{"dot map", "<{{.}}>", "<map[two:22]>", map[string]int{"two": 22}, true},
	{"dot struct", "<{{.}}>", "<{7 seven}>", struct {
		a int
		b string
	}{7, "seven"}, true},

	// Variables.
	{"$ int", "{{$}}", "123", 123, true},
	{"$.I", "{{$.I}}", "17", tVal, true},
	{"$.U.V", "{{$.U.V}}", "v", tVal, true},
	{"declare in action", "{{$x := $.U.V}}{{$x}}", "v", tVal, true},
	{"simple assignment", "{{$x := 2}}{{$x = 3}}{{$x}}", "3", tVal, true},
	{"nested assignment",
		"{{$x := 2}}{{if true}}{{$x = 3}}{{end}}{{$x}}",
		"3", tVal, true},
	{"nested assignment changes the last declaration",
		"{{$x := 1}}{{if true}}{{$x := 2}}{{if true}}{{$x = 3}}{{end}}{{end}}{{$x}}",
		"1", tVal, true},

	// Type with String method.
	{"V{6666}.String()", "-{{.V0}}-", "-<6666>-", tVal, true},
	{"&V{7777}.String()", "-{{.V1}}-", "-<7777>-", tVal, true},
	{"(*V)(nil).String()", "-{{.V2}}-", "-nilV-", tVal, true},

	// Type with Error method.
	{"W{888}.Error()", "-{{.W0}}-", "-[888]-", tVal, true},
	{"&W{999}.Error()", "-{{.W1}}-", "-[999]-", tVal, true},
	{"(*W)(nil).Error()", "-{{.W2}}-", "-nilW-", tVal, true},

	// Pointers.
	{"*int", "{{.PI}}", "23", tVal, true},
	{"*string", "{{.PS}}", "a string", tVal, true},
	{"*[]int", "{{.PSI}}", "[21 22 23]", tVal, true},
	{"*[]int[1]", "{{index .PSI 1}}", "22", tVal, true},
	{"NIL", "{{.NIL}}", "<nil>", tVal, true},

	// Empty interfaces holding values.
	{"empty nil", "{{.Empty0}}", "<no value>", tVal, true},
	{"empty with int", "{{.Empty1}}", "3", tVal, true},
	{"empty with string", "{{.Empty2}}", "empty2", tVal, true},
	{"empty with slice", "{{.Empty3}}", "[7 8]", tVal, true},
	{"empty with struct", "{{.Empty4}}", "{UinEmpty}", tVal, true},
	{"empty with struct, field", "{{.Empty4.V}}", "UinEmpty", tVal, true},

	// Edge cases with <no value> with an interface value
	{"field on interface", "{{.foo}}", "<no value>", nil, true},
	{"field on parenthesized interface", "{{(.).foo}}", "<no value>", nil, true},

	// Issue 31810: Parenthesized first element of pipeline with arguments.
	// See also TestIssue31810.
	{"unparenthesized non-function", "{{1 2}}", "", nil, false},
	{"parenthesized non-function", "{{(1) 2}}", "", nil, false},
	{"parenthesized non-function with no args", "{{(1)}}", "1", nil, true}, // This is fine.

	// Method calls.
	{".Method0", "-{{.Method0}}-", "-M0-", tVal, true},
	{".Method1(1234)", "-{{.Method1 1234}}-", "-1234-", tVal, true},
	{".Method1(.I)", "-{{.Method1 .I}}-", "-17-", tVal, true},
	{".Method2(3, .X)", "-{{.Method2 3 .X}}-", "-Method2: 3 x-", tVal, true},
	{".Method2(.U16, `str`)", "-{{.Method2 .U16 `str`}}-", "-Method2: 16 str-", tVal, true},
	{".Method2(.U16, $x)", "{{if $x := .X}}-{{.Method2 .U16 $x}}{{end}}-", "-Method2: 16 x-", tVal, true},
	{".Method3(nil constant)", "-{{.Method3 nil}}-", "-Method3: <nil>-", tVal, true},
	{".Method3(nil value)", "-{{.Method3 .MXI.unset}}-", "-Method3: <nil>-", tVal, true},
	{"method on var", "{{if $x := .}}-{{$x.Method2 .U16 $x.X}}{{end}}-", "-Method2: 16 x-", tVal, true},
	{"method on chained var",
		"{{range .MSIone}}{{if $.U.TrueFalse $.True}}{{$.U.TrueFalse $.True}}{{else}}WRONG{{end}}{{end}}",
		"true", tVal, true},
	{"chained method",
		"{{range .MSIone}}{{if $.GetU.TrueFalse $.True}}{{$.U.TrueFalse $.True}}{{else}}WRONG{{end}}{{end}}",
		"true", tVal, true},
	{"chained method on variable",
		"{{with $x := .}}{{with .SI}}{{$.GetU.TrueFalse $.True}}{{end}}{{end}}",
		"true", tVal, true},
	{".NilOKFunc not nil", "{{call .NilOKFunc .PI}}", "false", tVal, true},
	{".NilOKFunc nil", "{{call .NilOKFunc nil}}", "true", tVal, true},
	{"method on nil value from slice", "-{{range .}}{{.Method1 1234}}{{end}}-", "-1234-", tSliceOfNil, true},
	{"method on typed nil interface value", "{{.NonEmptyInterfaceTypedNil.Method0}}", "M0", tVal, true},

	// Function call builtin.
	{".BinaryFunc", "{{call .BinaryFunc `1` `2`}}", "[1=2]", tVal, true},
	{".VariadicFunc0", "{{call .VariadicFunc}}", "<>", tVal, true},
	{".VariadicFunc2", "{{call .VariadicFunc `he` `llo`}}", "<he+llo>", tVal, true},
	{".VariadicFuncInt", "{{call .VariadicFuncInt 33 `he` `llo`}}", "33=<he+llo>", tVal, true},
	{"if .BinaryFunc call", "{{ if .BinaryFunc}}{{call .BinaryFunc `1` `2`}}{{end}}", "[1=2]", tVal, true},
	{"if not .BinaryFunc call", "{{ if not .BinaryFunc}}{{call .BinaryFunc `1` `2`}}{{else}}No{{end}}", "No", tVal, true},
	{"Interface Call", `{{stringer .S}}`, "foozle", map[string]any{"S": bytes.NewBufferString("foozle")}, true},
	{".ErrFunc", "{{call .ErrFunc}}", "bla", tVal, true},
	{"call nil", "{{call nil}}", "", tVal, false},
	{"empty call", "{{call}}", "", tVal, false},
	{"empty call after pipe valid", "{{.ErrFunc | call}}", "bla", tVal, true},
	{"empty call after pipe invalid", "{{1 | call}}", "", tVal, false},

	// Erroneous function calls (check args).
	{".BinaryFuncTooFew", "{{call .BinaryFunc `1`}}", "", tVal, false},
	{".BinaryFuncTooMany", "{{call .BinaryFunc `1` `2` `3`}}", "", tVal, false},
	{".BinaryFuncBad0", "{{call .BinaryFunc 1 3}}", "", tVal, false},
	{".BinaryFuncBad1", "{{call .BinaryFunc `1` 3}}", "", tVal, false},
	{".VariadicFuncBad0", "{{call .VariadicFunc 3}}", "", tVal, false},
	{".VariadicFuncIntBad0", "{{call .VariadicFuncInt}}", "", tVal, false},
	{".VariadicFuncIntBad`", "{{call .VariadicFuncInt `x`}}", "", tVal, false},
	{".VariadicFuncNilBad", "{{call .VariadicFunc nil}}", "", tVal, false},

	// Pipelines.
	{"pipeline", "-{{.Method0 | .Method2 .U16}}-", "-Method2: 16 M0-", tVal, true},
	{"pipeline func", "-{{call .VariadicFunc `llo` | call .VariadicFunc `he` }}-", "-<he+<llo>>-", tVal, true},

	// Nil values aren't missing arguments.
	{"nil pipeline", "{{ .Empty0 | call .NilOKFunc }}", "true", tVal, true},
	{"nil call arg", "{{ call .NilOKFunc .Empty0 }}", "true", tVal, true},
	{"bad nil pipeline", "{{ .Empty0 | .VariadicFunc }}", "", tVal, false},

	// Parenthesized expressions
	{"parens in pipeline", "{{printf `%d %d %d` (1) (2 | add 3) (add 4 (add 5 6))}}", "1 5 15", tVal, true},

	// Parenthesized expressions with field accesses
	{"parens: $ in paren", "{{($).X}}", "x", tVal, true},
	{"parens: $.GetU in paren", "{{($.GetU).V}}", "v", tVal, true},
	{"parens: $ in paren in pipe", "{{($ | echo).X}}", "x", tVal, true},
	{"parens: spaces and args", `{{(makemap "up" "down" "left" "right").left}}`, "right", tVal, true},

	// If.
	{"if true", "{{if true}}TRUE{{end}}", "TRUE", tVal, true},
	{"if false", "{{if false}}TRUE{{else}}FALSE{{end}}", "FALSE", tVal, true},
	{"if nil", "{{if nil}}TRUE{{end}}", "", tVal, false},
	{"if on typed nil interface value", "{{if .NonEmptyInterfaceTypedNil}}TRUE{{ end }}", "", tVal, true},
	{"if 1", "{{if 1}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if 0", "{{if 0}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if 1.5", "{{if 1.5}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if 0.0", "{{if .FloatZero}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if 1.5i", "{{if 1.5i}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if 0.0i", "{{if .ComplexZero}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if nonNilPointer", "{{if .PI}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if nilPointer", "{{if .NIL}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if UPI", "{{if .UPI}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if EmptyUPI", "{{if .EmptyUPI}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if emptystring", "{{if ``}}NON-EMPTY{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"if string", "{{if `notempty`}}NON-EMPTY{{else}}EMPTY{{end}}", "NON-EMPTY", tVal, true},
	{"if emptyslice", "{{if .SIEmpty}}NON-EMPTY{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"if slice", "{{if .SI}}NON-EMPTY{{else}}EMPTY{{end}}", "NON-EMPTY", tVal, true},
	{"if emptymap", "{{if .MSIEmpty}}NON-EMPTY{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"if map", "{{if .MSI}}NON-EMPTY{{else}}EMPTY{{end}}", "NON-EMPTY", tVal, true},
	{"if map unset", "{{if .MXI.none}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if map not unset", "{{if not .MXI.none}}ZERO{{else}}NON-ZERO{{end}}", "ZERO", tVal, true},
	{"if $x with $y int", "{{if $x := true}}{{with $y := .I}}{{$x}},{{$y}}{{end}}{{end}}", "true,17", tVal, true},
	{"if $x with $x int", "{{if $x := true}}{{with $x := .I}}{{$x}},{{end}}{{$x}}{{end}}", "17,true", tVal, true},
	{"if else if", "{{if false}}FALSE{{else if true}}TRUE{{end}}", "TRUE", tVal, true},
	{"if else chain", "{{if eq 1 3}}1{{else if eq 2 3}}2{{else if eq 3 3}}3{{end}}", "3", tVal, true},

	// Print etc.
	{"print", `{{print "hello, print"}}`, "hello, print", tVal, true},
	{"print 123", `{{print 1 2 3}}`, "1 2 3", tVal, true},
	{"print nil", `{{print nil}}`, "<nil>", tVal, true},
	{"println", `{{println 1 2 3}}`, "1 2 3\n", tVal, true},
	{"printf int", `{{printf "%04x" 127}}`, "007f", tVal, true},
	{"printf float", `{{printf "%g" 3.5}}`, "3.5", tVal, true},
	{"printf complex", `{{printf "%g" 1+7i}}`, "(1+7i)", tVal, true},
	{"printf string", `{{printf "%s" "hello"}}`, "hello", tVal, true},
	{"printf function", `{{printf "%#q" zeroArgs}}`, "`zeroArgs`", tVal, true},
	{"printf field", `{{printf "%s" .U.V}}`, "v", tVal, true},
	{"printf method", `{{printf "%s" .Method0}}`, "M0", tVal, true},
	{"printf dot", `{{with .I}}{{printf "%d" .}}{{end}}`, "17", tVal, true},
	{"printf var", `{{with $x := .I}}{{printf "%d" $x}}{{end}}`, "17", tVal, true},
	{"printf lots", `{{printf "%d %s %g %s" 127 "hello" 7-3i .Method0}}`, "127 hello (7-3i) M0", tVal, true},

	// HTML.
	{"html", `{{html "<script>alert(\"XSS\");</script>"}}`,
		"&lt;script&gt;alert(&#34;XSS&#34;);&lt;/script&gt;", nil, true},
	{"html pipeline", `{{printf "<script>alert(\"XSS\");</script>" | html}}`,
		"&lt;script&gt;alert(&#34;XSS&#34;);&lt;/script&gt;", nil, true},
	{"html", `{{html .PS}}`, "a string", tVal, true},
	{"html typed nil", `{{html .NIL}}`, "&lt;nil&gt;", tVal, true},
	{"html untyped nil", `{{html .Empty0}}`, "&lt;no value&gt;", tVal, true},

	// JavaScript.
	{"js", `{{js .}}`, `It\'d be nice.`, `It'd be nice.`, true},

	// URL query.
	{"urlquery", `{{"http://www.example.org/"|urlquery}}`, "http%3A%2F%2Fwww.example.org%2F", nil, true},

	// Booleans
	{"not", "{{not true}} {{not false}}", "false true", nil, true},
	{"and", "{{and false 0}} {{and 1 0}} {{and 0 true}} {{and 1 1}}", "false 0 0 1", nil, true},
	{"or", "{{or 0 0}} {{or 1 0}} {{or 0 true}} {{or 1 1}}", "0 1 true 1", nil, true},
	{"or short-circuit", "{{or 0 1 (die)}}", "1", nil, true},
	{"and short-circuit", "{{and 1 0 (die)}}", "0", nil, true},
	{"or short-circuit2", "{{or 0 0 (die)}}", "", nil, false},
	{"and short-circuit2", "{{and 1 1 (die)}}", "", nil, false},
	{"and pipe-true", "{{1 | and 1}}", "1", nil, true},
	{"and pipe-false", "{{0 | and 1}}", "0", nil, true},
	{"or pipe-true", "{{1 | or 0}}", "1", nil, true},
	{"or pipe-false", "{{0 | or 0}}", "0", nil, true},
	{"and undef", "{{and 1 .Unknown}}", "<no value>", nil, true},
	{"or undef", "{{or 0 .Unknown}}", "<no value>", nil, true},
	{"boolean if", "{{if and true 1 `hi`}}TRUE{{else}}FALSE{{end}}", "TRUE", tVal, true},
	{"boolean if not", "{{if and true 1 `hi` | not}}TRUE{{else}}FALSE{{end}}", "FALSE", nil, true},
	{"boolean if pipe", "{{if true | not | and 1}}TRUE{{else}}FALSE{{end}}", "FALSE", nil, true},

	// Indexing.
	{"slice[0]", "{{index .SI 0}}", "3", tVal, true},
	{"slice[1]", "{{index .SI 1}}", "4", tVal, true},
	{"slice[HUGE]", "{{index .SI 10}}", "", tVal, false},
	{"slice[WRONG]", "{{index .SI `hello`}}", "", tVal, false},
	{"slice[nil]", "{{index .SI nil}}", "", tVal, false},
	{"map[one]", "{{index .MSI `one`}}", "1", tVal, true},
	{"map[two]", "{{index .MSI `two`}}", "2", tVal, true},
	{"map[NO]", "{{index .MSI `XXX`}}", "0", tVal, true},
	{"map[nil]", "{{index .MSI nil}}", "", tVal, false},
	{"map[``]", "{{index .MSI ``}}", "0", tVal, true},
	{"map[WRONG]", "{{index .MSI 10}}", "", tVal, false},
	{"double index", "{{index .SMSI 1 `eleven`}}", "11", tVal, true},
	{"nil[1]", "{{index nil 1}}", "", tVal, false},
	{"map MI64S", "{{index .MI64S 2}}", "i642", tVal, true},
	{"map MI32S", "{{index .MI32S 2}}", "two", tVal, true},
	{"map MUI64S", "{{index .MUI64S 3}}", "ui643", tVal, true},
	{"map MI8S", "{{index .MI8S 3}}", "i83", tVal, true},
	{"map MUI8S", "{{index .MUI8S 2}}", "u82", tVal, true},
	{"index of an interface field", "{{index .Empty3 0}}", "7", tVal, true},

	// Slicing.
	{"slice[:]", "{{slice .SI}}", "[3 4 5]", tVal, true},
	{"slice[1:]", "{{slice .SI 1}}", "[4 5]", tVal, true},
	{"slice[1:2]", "{{slice .SI 1 2}}", "[4]", tVal, true},
	{"slice[-1:]", "{{slice .SI -1}}", "", tVal, false},
	{"slice[1:-2]", "{{slice .SI 1 -2}}", "", tVal, false},
	{"slice[1:2:-1]", "{{slice .SI 1 2 -1}}", "", tVal, false},
	{"slice[2:1]", "{{slice .SI 2 1}}", "", tVal, false},
	{"slice[2:2:1]", "{{slice .SI 2 2 1}}", "", tVal, false},
	{"out of range", "{{slice .SI 4 5}}", "", tVal, false},
	{"out of range", "{{slice .SI 2 2 5}}", "", tVal, false},
	{"len(s) < indexes < cap(s)", "{{slice .SICap 6 10}}", "[0 0 0 0]", tVal, true},
	{"len(s) < indexes < cap(s)", "{{slice .SICap 6 10 10}}", "[0 0 0 0]", tVal, true},
	{"indexes > cap(s)", "{{slice .SICap 10 11}}", "", tVal, false},
	{"indexes > cap(s)", "{{slice .SICap 6 10 11}}", "", tVal, false},
	{"array[:]", "{{slice .AI}}", "[3 4 5]", tVal, true},
	{"array[1:]", "{{slice .AI 1}}", "[4 5]", tVal, true},
	{"array[1:2]", "{{slice .AI 1 2}}", "[4]", tVal, true},
	{"string[:]", "{{slice .S}}", "xyz", tVal, true},
	{"string[0:1]", "{{slice .S 0 1}}", "x", tVal, true},
	{"string[1:]", "{{slice .S 1}}", "yz", tVal, true},
	{"string[1:2]", "{{slice .S 1 2}}", "y", tVal, true},
	{"out of range", "{{slice .S 1 5}}", "", tVal, false},
	{"3-index slice of string", "{{slice .S 1 2 2}}", "", tVal, false},
	{"slice of an interface field", "{{slice .Empty3 0 1}}", "[7]", tVal, true},

	// Len.
	{"slice", "{{len .SI}}", "3", tVal, true},
	{"map", "{{len .MSI }}", "3", tVal, true},
	{"len of int", "{{len 3}}", "", tVal, false},
	{"len of nothing", "{{len .Empty0}}", "", tVal, false},
	{"len of an interface field", "{{len .Empty3}}", "2", tVal, true},

	// With.
	{"with true", "{{with true}}{{.}}{{end}}", "true", tVal, true},
	{"with false", "{{with false}}{{.}}{{else}}FALSE{{end}}", "FALSE", tVal, true},
	{"with 1", "{{with 1}}{{.}}{{else}}ZERO{{end}}", "1", tVal, true},
	{"with 0", "{{with 0}}{{.}}{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"with 1.5", "{{with 1.5}}{{.}}{{else}}ZERO{{end}}", "1.5", tVal, true},
	{"with 0.0", "{{with .FloatZero}}{{.}}{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"with 1.5i", "{{with 1.5i}}{{.}}{{else}}ZERO{{end}}", "(0+1.5i)", tVal, true},
	{"with 0.0i", "{{with .ComplexZero}}{{.}}{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"with emptystring", "{{with ``}}{{.}}{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"with string", "{{with `notempty`}}{{.}}{{else}}EMPTY{{end}}", "notempty", tVal, true},
	{"with emptyslice", "{{with .SIEmpty}}{{.}}{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"with slice", "{{with .SI}}{{.}}{{else}}EMPTY{{end}}", "[3 4 5]", tVal, true},
	{"with emptymap", "{{with .MSIEmpty}}{{.}}{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"with map", "{{with .MSIone}}{{.}}{{else}}EMPTY{{end}}", "map[one:1]", tVal, true},
	{"with empty interface, struct field", "{{with .Empty4}}{{.V}}{{end}}", "UinEmpty", tVal, true},
	{"with $x int", "{{with $x := .I}}{{$x}}{{end}}", "17", tVal, true},
	{"with $x struct.U.V", "{{with $x := $}}{{$x.U.V}}{{end}}", "v", tVal, true},
	{"with variable and action", "{{with $x := $}}{{$y := $.U.V}}{{$y}}{{end}}", "v", tVal, true},
	{"with on typed nil interface value", "{{with .NonEmptyInterfaceTypedNil}}TRUE{{ end }}", "", tVal, true},
	{"with else with", "{{with 0}}{{.}}{{else with true}}{{.}}{{end}}", "true", tVal, true},
	{"with else with chain", "{{with 0}}{{.}}{{else with false}}{{.}}{{else with `notempty`}}{{.}}{{end}}", "notempty", tVal, true},

	// Range.
	{"range []int", "{{range .SI}}-{{.}}-{{end}}", "-3--4--5-", tVal, true},
	{"range empty no else", "{{range .SIEmpty}}-{{.}}-{{end}}", "", tVal, true},
	{"range []int else", "{{range .SI}}-{{.}}-{{else}}EMPTY{{end}}", "-3--4--5-", tVal, true},
	{"range empty else", "{{range .SIEmpty}}-{{.}}-{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"range []int break else", "{{range .SI}}-{{.}}-{{break}}NOTREACHED{{else}}EMPTY{{end}}", "-3-", tVal, true},
	{"range []int continue else", "{{range .SI}}-{{.}}-{{continue}}NOTREACHED{{else}}EMPTY{{end}}", "-3--4--5-", tVal, true},
	{"range []bool", "{{range .SB}}-{{.}}-{{end}}", "-true--false-", tVal, true},
	{"range []int method", "{{range .SI | .MAdd .I}}-{{.}}-{{end}}", "-20--21--22-", tVal, true},
	{"range map", "{{range .MSI}}-{{.}}-{{end}}", "-1--3--2-", tVal, true},
	{"range empty map no else", "{{range .MSIEmpty}}-{{.}}-{{end}}", "", tVal, true},
	{"range map else", "{{range .MSI}}-{{.}}-{{else}}EMPTY{{end}}", "-1--3--2-", tVal, true},
	{"range empty map else", "{{range .MSIEmpty}}-{{.}}-{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"range empty interface", "{{range .Empty3}}-{{.}}-{{else}}EMPTY{{end}}", "-7--8-", tVal, true},
	{"range empty nil", "{{range .Empty0}}-{{.}}-{{end}}", "", tVal, true},
	{"range $x SI", "{{range $x := .SI}}<{{$x}}>{{end}}", "<3><4><5>", tVal, true},
	{"range $x $y SI", "{{range $x, $y := .SI}}<{{$x}}={{$y}}>{{end}}", "<0=3><1=4><2=5>", tVal, true},
	{"range $x MSIone", "{{range $x := .MSIone}}<{{$x}}>{{end}}", "<1>", tVal, true},
	{"range $x $y MSIone", "{{range $x, $y := .MSIone}}<{{$x}}={{$y}}>{{end}}", "<one=1>", tVal, true},
	{"range $x PSI", "{{range $x := .PSI}}<{{$x}}>{{end}}", "<21><22><23>", tVal, true},
	{"declare in range", "{{range $x := .PSI}}<{{$foo:=$x}}{{$x}}>{{end}}", "<21><22><23>", tVal, true},
	{"range count", `{{range $i, $x := count 5}}[{{$i}}]{{$x}}{{end}}`, "[0]a[1]b[2]c[3]d[4]e", tVal, true},
	{"range nil count", `{{range $i, $x := count 0}}{{else}}empty{{end}}`, "empty", tVal, true},
	{"range iter.Seq[int]", `{{range $i := .}}{{$i}}{{end}}`, "01", fVal1(2), true},
	{"i = range iter.Seq[int]", `{{$i := 0}}{{range $i = .}}{{$i}}{{end}}`, "01", fVal1(2), true},
	{"range iter.Seq[int] over two var", `{{range $i, $c := .}}{{$c}}{{end}}`, "", fVal1(2), false},
	{"i, c := range iter.Seq2[int,int]", `{{range $i, $c := .}}{{$i}}{{$c}}{{end}}`, "0112", fVal2(2), true},
	{"i, c = range iter.Seq2[int,int]", `{{$i := 0}}{{$c := 0}}{{range $i, $c = .}}{{$i}}{{$c}}{{end}}`, "0112", fVal2(2), true},
	{"i = range iter.Seq2[int,int]", `{{$i := 0}}{{range $i = .}}{{$i}}{{end}}`, "01", fVal2(2), true},
	{"i := range iter.Seq2[int,int]", `{{range $i := .}}{{$i}}{{end}}`, "01", fVal2(2), true},
	{"i,c,x range iter.Seq2[int,int]", `{{$i := 0}}{{$c := 0}}{{$x := 0}}{{range $i, $c = .}}{{$i}}{{$c}}{{end}}`, "0112", fVal2(2), true},
	{"i,x range iter.Seq[int]", `{{$i := 0}}{{$x := 0}}{{range $i = .}}{{$i}}{{end}}`, "01", fVal1(2), true},
	{"range iter.Seq[int] else", `{{range $i := .}}{{$i}}{{else}}empty{{end}}`, "empty", fVal1(0), true},
	{"range iter.Seq2[int,int] else", `{{range $i := .}}{{$i}}{{else}}empty{{end}}`, "empty", fVal2(0), true},
	{"range int8", rangeTestInt, rangeTestData[int8](), int8(5), true},
	{"range int16", rangeTestInt, rangeTestData[int16](), int16(5), true},
	{"range int32", rangeTestInt, rangeTestData[int32](), int32(5), true},
	{"range int64", rangeTestInt, rangeTestData[int64](), int64(5), true},
	{"range int", rangeTestInt, rangeTestData[int](), int(5), true},
	{"range uint8", rangeTestInt, rangeTestData[uint8](), uint8(5), true},
	{"range uint16", rangeTestInt, rangeTestData[uint16](), uint16(5), true},
	{"range uint32", rangeTestInt, rangeTestData[uint32](), uint32(5), true},
	{"range uint64", rangeTestInt, rangeTestData[uint64](), uint64(5), true},
	{"range uint", rangeTestInt, rangeTestData[uint](), uint(5), true},
	{"range uintptr", rangeTestInt, rangeTestData[uintptr](), uintptr(5), true},
	{"range uintptr(0)", `{{range $v := .}}{{print $v}}{{else}}empty{{end}}`, "empty", uintptr(0), true},
	{"range 5", `{{range $v := 5}}{{printf "%T%d" $v $v}}{{end}}`, rangeTestData[int](), nil, true},

	// Cute examples.
	{"or as if true", `{{or .SI "slice is empty"}}`, "[3 4 5]", tVal, true},
	{"or as if false", `{{or .SIEmpty "slice is empty"}}`, "slice is empty", tVal, true},

	// Error handling.
	{"error method, error", "{{.MyError true}}", "", tVal, false},
	{"error method, no error", "{{.MyError false}}", "false", tVal, true},

	// Numbers
	{"decimal", "{{print 1234}}", "1234", tVal, true},
	{"decimal _", "{{print 12_34}}", "1234", tVal, true},
	{"binary", "{{print 0b101}}", "5", tVal, true},
	{"binary _", "{{print 0b_1_0_1}}", "5", tVal, true},
	{"BINARY", "{{print 0B101}}", "5", tVal, true},
	{"octal0", "{{print 0377}}", "255", tVal, true},
	{"octal", "{{print 0o377}}", "255", tVal, true},
	{"octal _", "{{print 0o_3_7_7}}", "255", tVal, true},
	{"OCTAL", "{{print 0O377}}", "255", tVal, true},
	{"hex", "{{print 0x123}}", "291", tVal, true},
	{"hex _", "{{print 0x1_23}}", "291", tVal, true},
	{"HEX", "{{print 0X123ABC}}", "1194684", tVal, true},
	{"float", "{{print 123.4}}", "123.4", tVal, true},
	{"float _", "{{print 0_0_1_2_3.4}}", "123.4", tVal, true},
	{"hex float", "{{print +0x1.ep+2}}", "7.5", tVal, true},
	{"hex float _", "{{print +0x_1.e_0p+0_2}}", "7.5", tVal, true},
	{"HEX float", "{{print +0X1.EP+2}}", "7.5", tVal, true},
	{"print multi", "{{print 1_2_3_4 7.5_00_00_00}}", "1234 7.5", tVal, true},
	{"print multi2", "{{print 1234 0x0_1.e_0p+02}}", "1234 7.5", tVal, true},

	// Fixed bugs.
	// Must separate dot and receiver; otherwise args are evaluated with dot set to variable.
	{"bug0", "{{range .MSIone}}{{if $.Method1 .}}X{{end}}{{end}}", "X", tVal, true},
	// Do not loop endlessly in indirect for non-empty interfaces.
	// The bug appears with *interface only; looped forever.
	{"bug1", "{{.Method0}}", "M0", &iVal, true},
	// Was taking address of interface field, so method set was empty.
	{"bug2", "{{$.NonEmptyInterface.Method0}}", "M0", tVal, true},
	// Struct values were not legal in with - mere oversight.
	{"bug3", "{{with $}}{{.Method0}}{{end}}", "M0", tVal, true},
	// Nil interface values in if.
	{"bug4", "{{if .Empty0}}non-nil{{else}}nil{{end}}", "nil", tVal, true},
	// Stringer.
	{"bug5", "{{.Str}}", "foozle", tVal, true},
	{"bug5a", "{{.Err}}", "erroozle", tVal, true},
	// Args need to be indirected and dereferenced sometimes.
	{"bug6a", "{{vfunc .V0 .V1}}", "vfunc", tVal, true},
	{"bug6b", "{{vfunc .V0 .V0}}", "vfunc", tVal, true},
	{"bug6c", "{{vfunc .V1 .V0}}", "vfunc", tVal, true},
	{"bug6d", "{{vfunc .V1 .V1}}", "vfunc", tVal, true},
	// Legal parse but illegal execution: non-function should have no arguments.
	{"bug7a", "{{3 2}}", "", tVal, false},
	{"bug7b", "{{$x := 1}}{{$x 2}}", "", tVal, false},
	{"bug7c", "{{$x := 1}}{{3 | $x}}", "", tVal, false},
	// Pipelined arg was not being type-checked.
	{"bug8a", "{{3|oneArg}}", "", tVal, false},
	{"bug8b", "{{4|dddArg 3}}", "", tVal, false},
	// A bug was introduced that broke map lookups for lower-case names.
	{"bug9", "{{.cause}}", "neglect", map[string]string{"cause": "neglect"}, true},
	// Field chain starting with function did not work.
	{"bug10", "{{mapOfThree.three}}-{{(mapOfThree).three}}", "3-3", 0, true},
	// Dereferencing nil pointer while evaluating function arguments should not panic. Issue 7333.
	{"bug11", "{{valueString .PS}}", "", T{}, false},
	// 0xef gave constant type float64. Issue 8622.
	{"bug12xe", "{{printf `%T` 0xef}}", "int", T{}, true},
	{"bug12xE", "{{printf `%T` 0xEE}}", "int", T{}, true},
	{"bug12Xe", "{{printf `%T` 0Xef}}", "int", T{}, true},
	{"bug12XE", "{{printf `%T` 0XEE}}", "int", T{}, true},
	// Chained nodes did not work as arguments. Issue 8473.
	{"bug13", "{{print (.Copy).I}}", "17", tVal, true},
	// Didn't protect against nil or literal values in field chains.
	{"bug14a", "{{(nil).True}}", "", tVal, false},
	{"bug14b", "{{$x := nil}}{{$x.anything}}", "", tVal, false},
	{"bug14c", `{{$x := (1.0)}}{{$y := ("hello")}}{{$x.anything}}{{$y.true}}`, "", tVal, false},
	// Didn't call validateType on function results. Issue 10800.
	{"bug15", "{{valueString returnInt}}", "", tVal, false},
	// Variadic function corner cases. Issue 10946.
	{"bug16a", "{{true|printf}}", "", tVal, false},
	{"bug16b", "{{1|printf}}", "", tVal, false},
	{"bug16c", "{{1.1|printf}}", "", tVal, false},
	{"bug16d", "{{'x'|printf}}", "", tVal, false},
	{"bug16e", "{{0i|printf}}", "", tVal, false},
	{"bug16f", "{{true|twoArgs \"xxx\"}}", "", tVal, false},
	{"bug16g", "{{\"aaa\" |twoArgs \"bbb\"}}", "twoArgs=bbbaaa", tVal, true},
	{"bug16h", "{{1|oneArg}}", "", tVal, false},
	{"bug16i", "{{\"aaa\"|oneArg}}", "oneArg=aaa", tVal, true},
	{"bug16j", "{{1+2i|printf \"%v\"}}", "(1+2i)", tVal, true},
	{"bug16k", "{{\"aaa\"|printf }}", "aaa", tVal, true},
	{"bug17a", "{{.NonEmptyInterface.X}}", "x", tVal, true},
	{"bug17b", "-{{.NonEmptyInterface.Method1 1234}}-", "-1234-", tVal, true},
	{"bug17c", "{{len .NonEmptyInterfacePtS}}", "2", tVal, true},
	{"bug17d", "{{index .NonEmptyInterfacePtS 0}}", "a", tVal, true},
	{"bug17e", "{{range .NonEmptyInterfacePtS}}-{{.}}-{{end}}", "-a--b-", tVal, true},

	// More variadic function corner cases. Some runes would get evaluated
	// as constant floats instead of ints. Issue 34483.
	{"bug18a", "{{eq . '.'}}", "true", '.', true},
	{"bug18b", "{{eq . 'e'}}", "true", 'e', true},
	{"bug18c", "{{eq . 'P'}}", "true", 'P', true},

	{"issue56490", "{{$i := 0}}{{$x := 0}}{{range $i = .AI}}{{end}}{{$i}}", "5", tVal, true},
	{"issue60801", "{{$k := 0}}{{$v := 0}}{{range $k, $v = .AI}}{{$k}}={{$v}} {{end}}", "0=3 1=4 2=5 ", tVal, true},
}

func fVal1(i int) iter.Seq[int] {
	return func(yield func(int) bool) {
		for v := range i {
			if !yield(v) {
				break
			}
		}
	}
}

func fVal2(i int) iter.Seq2[int, int] {
	return func(yield func(int, int) bool) {
		for v := range i {
			if !yield(v, v+1) {
				break
			}
		}
	}
}

const rangeTestInt = `{{range $v := .}}{{printf "%T%d" $v $v}}{{end}}`

func rangeTestData[T int | int8 | int16 | int32 | int64 | uint | uint8 | uint16 | uint32 | uint64 | uintptr]() string {
	I := T(5)
	var buf strings.Builder
	for i := T(0); i < I; i++ {
		fmt.Fprintf(&buf, "%T%d", i, i)
	}
	return buf.String()
}

func zeroArgs() string {
	return "zeroArgs"
}

func oneArg(a string) string {
	return "oneArg=" + a
}

func twoArgs(a, b string) string {
	return "twoArgs=" + a + b
}

func dddArg(a int, b ...string) string {
	return fmt.Sprintln(a, b)
}

// count returns a channel that will deliver n sequential 1-letter strings starting at "a"
func count(n int) chan string {
	if n == 0 {
		return nil
	}
	c := make(chan string)
	go func() {
		for i := 0; i < n; i++ {
			c <- "abcdefghijklmnop"[i : i+1]
		}
		close(c)
	}()
	return c
}

// vfunc takes a *V and a V
func vfunc(V, *V) string {
	return "vfunc"
}

// valueString takes a string, not a pointer.
func valueString(v string) string {
	return "value is ignored"
}

// returnInt returns an int
func returnInt() int {
	return 7
}

func add(args ...int) int {
	sum := 0
	for _, x := range args {
		sum += x
	}
	return sum
}

func echo(arg any) any {
	return arg
}

func makemap(arg ...string) map[string]string {
	if len(arg)%2 != 0 {
		panic("bad makemap")
	}
	m := make(map[string]string)
	for i := 0; i < len(arg); i += 2 {
		m[arg[i]] = arg[i+1]
	}
	return m
}

func stringer(s fmt.Stringer) string {
	return s.String()
}

func mapOfThree() any {
	return map[string]int{"three": 3}
}

func testExecute(execTests []execTest, template *Template, t *testing.T) {
	b := new(strings.Builder)
	funcs := FuncMap{
		"add":         add,
		"count":       count,
		"dddArg":      dddArg,
		"die":         func() bool { panic("die") },
		"echo":        echo,
		"makemap":     makemap,
		"mapOfThree":  mapOfThree,
		"oneArg":      oneArg,
		"returnInt":   returnInt,
		"stringer":    stringer,
		"twoArgs":     twoArgs,
		"typeOf":      typeOf,
		"valueString": valueString,
		"vfunc":       vfunc,
		"zeroArgs":    zeroArgs,
	}
	for _, test := range execTests {
		var tmpl *Template
		var err error
		if template == nil {
			tmpl, err = New(test.name).Funcs(funcs).Parse(test.input)
		} else {
			tmpl, err = template.New(test.name).Funcs(funcs).Parse(test.input)
		}
		if err != nil {
			t.Errorf("%s: parse error: %s", test.name, err)
			continue
		}
		b.Reset()
		err = tmpl.Execute(b, test.data)
		switch {
		case !test.ok && err == nil:
			t.Errorf("%s: expected error; got none", test.name)
			continue
		case test.ok && err != nil:
			t.Errorf("%s: unexpected execute error: %s", test.name, err)
			continue
		case !test.ok && err != nil:
			// expected error, got one
			if *debug {
				fmt.Printf("%s: %s\n\t%s\n", test.name, test.input, err)
			}
		}
		result := b.String()
		if result != test.output {
			t.Errorf("%s: expected\n\t%q\ngot\n\t%q", test.name, test.output, result)
		}
	}
}

func TestExecute(t *testing.T) {
	testExecute(execTests, nil, t)
}

var delimPairs = []string{
	"", "", // default
	"{{", "}}", // same as default
	"<<", ">>", // distinct
	"|", "|", // same
	"(日)", "(本)", // peculiar
}

func TestDelims(t *testing.T) {
	const hello = "Hello, world"
	var value = struct{ Str string }{hello}
	for i := 0; i < len(delimPairs); i += 2 {
		text := ".Str"
		left := delimPairs[i+0]
		trueLeft := left
		right := delimPairs[i+1]
		trueRight := right
		if left == "" { // default case
			trueLeft = "{{"
		}
		if right == "" { // default case
			trueRight = "}}"
		}
		text = trueLeft + text + trueRight
		// Now add a comment
		text += trueLeft + "/*comment*/" + trueRight
		// Now add  an action containing a string.
		text += trueLeft + `"` + trueLeft + `"` + trueRight
		// At this point text looks like `{{.Str}}{{/*comment*/}}{{"{{"}}`.
		tmpl, err := New("delims").Delims(left, right).Parse(text)
		if err != nil {
			t.Fatalf("delim %q text %q parse err %s", left, text, err)
		}
		var b = new(strings.Builder)
		err = tmpl.Execute(b, value)
		if err != nil {
			t.Fatalf("delim %q exec err %s", left, err)
		}
		if b.String() != hello+trueLeft {
			t.Errorf("expected %q got %q", hello+trueLeft, b.String())
		}
	}
}

// Check that an error from a method flows back to the top.
func TestExecuteError(t *testing.T) {
	b := new(bytes.Buffer)
	tmpl := New("error")
	_, err := tmpl.Parse("{{.MyError true}}")
	if err != nil {
		t.Fatalf("parse error: %s", err)
	}
	err = tmpl.Execute(b, tVal)
	if err == nil {
		t.Errorf("expected error; got none")
	} else if !strings.Contains(err.Error(), myError.Error()) {
		if *debug {
			fmt.Printf("test execute error: %s\n", err)
		}
		t.Errorf("expected myError; got %s", err)
	}
}

const execErrorText = `line 1
line 2
line 3
{{template "one" .}}
{{define "one"}}{{template "two" .}}{{end}}
{{define "two"}}{{template "three" .}}{{end}}
{{define "three"}}{{index "hi" $}}{{end}}`

// Check that an error from a nested template contains all the relevant information.
func TestExecError(t *testing.T) {
	tmpl, err := New("top").Parse(execErrorText)
	if err != nil {
		t.Fatal("parse error:", err)
	}
	var b bytes.Buffer
	err = tmpl.Execute(&b, 5) // 5 is out of range indexing "hi"
	if err == nil {
		t.Fatal("expected error")
	}
	const want = `template: top:7:20: executing "three" at <index "hi" $>: error calling index: index out of range: 5`
	got := err.Error()
	if got != want {
		t.Errorf("expected\n%q\ngot\n%q", want, got)
	}
}

type CustomError struct{}

func (*CustomError) Error() string { return "heyo !" }

// Check that a custom error can be returned.
func TestExecError_CustomError(t *testing.T) {
	failingFunc := func() (string, error) {
		return "", &CustomError{}
	}
	tmpl := Must(New("top").Funcs(FuncMap{
		"err": failingFunc,
	}).Parse("{{ err }}"))

	var b bytes.Buffer
	err := tmpl.Execute(&b, nil)

	var e *CustomError
	if !errors.As(err, &e) {
		t.Fatalf("expected custom error; got %s", err)
	}
}

func TestJSEscaping(t *testing.T) {
	testCases := []struct {
		in, exp string
	}{
		{`a`, `a`},
		{`'foo`, `\'foo`},
		{`Go "jump" \`, `Go \"jump\" \\`},
		{`Yukihiro says "今日は世界"`, `Yukihiro says \"今日は世界\"`},
		{"unprintable \uFFFE", `unprintable \uFFFE`},
		{`<html>`, `\u003Chtml\u003E`},
		{`no = in attributes`, `no \u003D in attributes`},
		{`&#x27; does not become HTML entity`, `\u0026#x27; does not become HTML entity`},
	}
	for _, tc := range testCases {
		s := JSEscapeString(tc.in)
		if s != tc.exp {
			t.Errorf("JS escaping [%s] got [%s] want [%s]", tc.in, s, tc.exp)
		}
	}
}

// A nice example: walk a binary tree.

type Tree struct {
	Val         int
	Left, Right *Tree
}

// Use different delimiters to test Set.Delims.
// Also test the trimming of leading and trailing spaces.
const treeTemplate = `
	(- define "tree" -)
	[
		(- .Val -)
		(- with .Left -)
			(template "tree" . -)
		(- end -)
		(- with .Right -)
			(- template "tree" . -)
		(- end -)
	]
	(- end -)
`

func TestTree(t *testing.T) {
	var tree = &Tree{
		1,
		&Tree{
			2, &Tree{
				3,
				&Tree{
					4, nil, nil,
				},
				nil,
			},
			&Tree{
				5,
				&Tree{
					6, nil, nil,
				},
				nil,
			},
		},
		&Tree{
			7,
			&Tree{
				8,
				&Tree{
					9, nil, nil,
				},
				nil,
			},
			&Tree{
				10,
				&Tree{
					11, nil, nil,
				},
				nil,
			},
		},
	}
	tmpl, err := New("root").Delims("(", ")").Parse(treeTemplate)
	if err != nil {
		t.Fatal("parse error:", err)
	}
	var b strings.Builder
	const expect = "[1[2[3[4]][5[6]]][7[8[9]][10[11]]]]"
	// First by looking up the template.
	err = tmpl.Lookup("tree").Execute(&b, tree)
	if err != nil {
		t.Fatal("exec error:", err)
	}
	result := b.String()
	if result != expect {
		t.Errorf("expected %q got %q", expect, result)
	}
	// Then direct to execution.
	b.Reset()
	err = tmpl.ExecuteTemplate(&b, "tree", tree)
	if err != nil {
		t.Fatal("exec error:", err)
	}
	result = b.String()
	if result != expect {
		t.Errorf("expected %q got %q", expect, result)
	}
}

func TestExecuteOnNewTemplate(t *testing.T) {
	// This is issue 3872.
	New("Name").Templates()
	// This is issue 11379.
	new(Template).Templates()
	new(Template).Parse("")
	new(Template).New("abc").Parse("")
	new(Template).Execute(nil, nil)                // returns an error (but does not crash)
	new(Template).ExecuteTemplate(nil, "XXX", nil) // returns an error (but does not crash)
}

const testTemplates = `{{define "one"}}one{{end}}{{define "two"}}two{{end}}`

func TestMessageForExecuteEmpty(t *testing.T) {
	// Test a truly empty template.
	tmpl := New("empty")
	var b bytes.Buffer
	err := tmpl.Execute(&b, 0)
	if err == nil {
		t.Fatal("expected initial error")
	}
	got := err.Error()
	want := `template: empty: "empty" is an incomplete or empty template`
	if got != want {
		t.Errorf("expected error %s got %s", want, got)
	}
	// Add a non-empty template to check that the error is helpful.
	tests, err := New("").Parse(testTemplates)
	if err != nil {
		t.Fatal(err)
	}
	tmpl.AddParseTree("secondary", tests.Tree)
	err = tmpl.Execute(&b, 0)
	if err == nil {
		t.Fatal("expected second error")
	}
	got = err.Error()
	want = `template: empty: "empty" is an incomplete or empty template`
	if got != want {
		t.Errorf("expected error %s got %s", want, got)
	}
	// Make sure we can execute the secondary.
	err = tmpl.ExecuteTemplate(&b, "secondary", 0)
	if err != nil {
		t.Fatal(err)
	}
}

func TestFinalForPrintf(t *testing.T) {
	tmpl, err := New("").Parse(`{{"x" | printf}}`)
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	err = tmpl.Execute(&b, 0)
	if err != nil {
		t.Fatal(err)
	}
}

type cmpTest struct {
	expr  string
	truth string
	ok    bool
}

var cmpTests = []cmpTest{
	{"eq true true", "true", true},
	{"eq true false", "false", true},
	{"eq 1+2i 1+2i", "true", true},
	{"eq 1+2i 1+3i", "false", true},
	{"eq 1.5 1.5", "true", true},
	{"eq 1.5 2.5", "false", true},
	{"eq 1 1", "true", true},
	{"eq 1 2", "false", true},
	{"eq `xy` `xy`", "true", true},
	{"eq `xy` `xyz`", "false", true},
	{"eq .Uthree .Uthree", "true", true},
	{"eq .Uthree .Ufour", "false", true},
	{"eq 3 4 5 6 3", "true", true},
	{"eq 3 4 5 6 7", "false", true},
	{"ne true true", "false", true},
	{"ne true false", "true", true},
	{"ne 1+2i 1+2i", "false", true},
	{"ne 1+2i 1+3i", "true", true},
	{"ne 1.5 1.5", "false", true},
	{"ne 1.5 2.5", "true", true},
	{"ne 1 1", "false", true},
	{"ne 1 2", "true", true},
	{"ne `xy` `xy`", "false", true},
	{"ne `xy` `xyz`", "true", true},
	{"ne .Uthree .Uthree", "false", true},
	{"ne .Uthree .Ufour", "true", true},
	{"lt 1.5 1.5", "false", true},
	{"lt 1.5 2.5", "true", true},
	{"lt 1 1", "false", true},
	{"lt 1 2", "true", true},
	{"lt `xy` `xy`", "false", true},
	{"lt `xy` `xyz`", "true", true},
	{"lt .Uthree .Uthree", "false", true},
	{"lt .Uthree .Ufour", "true", true},
	{"le 1.5 1.5", "true", true},
	{"le 1.5 2.5", "true", true},
	{"le 2.5 1.5", "false", true},
	{"le 1 1", "true", true},
	{"le 1 2", "true", true},
	{"le 2 1", "false", true},
	{"le `xy` `xy`", "true", true},
	{"le `xy` `xyz`", "true", true},
	{"le `xyz` `xy`", "false", true},
	{"le .Uthree .Uthree", "true", true},
	{"le .Uthree .Ufour", "true", true},
	{"le .Ufour .Uthree", "false", true},
	{"gt 1.5 1.5", "false", true},
	{"gt 1.5 2.5", "false", true},
	{"gt 1 1", "false", true},
	{"gt 2 1", "true", true},
	{"gt 1 2", "false", true},
	{"gt `xy` `xy`", "false", true},
	{"gt `xy` `xyz`", "false", true},
	{"gt .Uthree .Uthree", "false", true},
	{"gt .Uthree .Ufour", "false", true},
	{"gt .Ufour .Uthree", "true", true},
	{"ge 1.5 1.5", "true", true},
	{"ge 1.5 2.5", "false", true},
	{"ge 2.5 1.5", "true", true},
	{"ge 1 1", "true", true},
	{"ge 1 2", "false", true},
	{"ge 2 1", "true", true},
	{"ge `xy` `xy`", "true", true},
	{"ge `xy` `xyz`", "false", true},
	{"ge `xyz` `xy`", "true", true},
	{"ge .Uthree .Uthree", "true", true},
	{"ge .Uthree .Ufour", "false", true},
	{"ge .Ufour .Uthree", "true", true},
	// Mixing signed and unsigned integers.
	{"eq .Uthree .Three", "true", true},
	{"eq .Three .Uthree", "true", true},
	{"le .Uthree .Three", "true", true},
	{"le .Three .Uthree", "true", true},
	{"ge .Uthree .Three", "true", true},
	{"ge .Three .Uthree", "true", true},
	{"lt .Uthree .Three", "false", true},
	{"lt .Three .Uthree", "false", true},
	{"gt .Uthree .Three", "false", true},
	{"gt .Three .Uthree", "false", true},
	{"eq .Ufour .Three", "false", true},
	{"lt .Ufour .Three", "false", true},
	{"gt .Ufour .Three", "true", true},
	{"eq .NegOne .Uthree", "false", true},
	{"eq .Uthree .NegOne", "false", true},
	{"ne .NegOne .Uthree", "true", true},
	{"ne .Uthree .NegOne", "true", true},
	{"lt .NegOne .Uthree", "true", true},
	{"lt .Uthree .NegOne", "false", true},
	{"le .NegOne .Uthree", "true", true},
	{"le .Uthree .NegOne", "false", true},
	{"gt .NegOne .Uthree", "false", true},
	{"gt .Uthree .NegOne", "true", true},
	{"ge .NegOne .Uthree", "false", true},
	{"ge .Uthree .NegOne", "true", true},
	{"eq (index `x` 0) 'x'", "true", true}, // The example that triggered this rule.
	{"eq (index `x` 0) 'y'", "false", true},
	{"eq .V1 .V2", "true", true},
	{"eq .Ptr .Ptr", "true", true},
	{"eq .Ptr .NilPtr", "false", true},
	{"eq .NilPtr .NilPtr", "true", true},
	{"eq .Iface1 .Iface1", "true", true},
	{"eq .Iface1 .NilIface", "false", true},
	{"eq .NilIface .NilIface", "true", true},
	{"eq .NilIface .Iface1", "false", true},
	{"eq .NilIface 0", "false", true},
	{"eq 0 .NilIface", "false", true},
	{"eq .Map .Map", "true", true},        // Uncomparable types but nil is OK.
	{"eq .Map nil", "true", true},         // Uncomparable types but nil is OK.
	{"eq nil .Map", "true", true},         // Uncomparable types but nil is OK.
	{"eq .Map .NonNilMap", "false", true}, // Uncomparable types but nil is OK.
	// Errors
	{"eq `xy` 1", "", false},                // Different types.
	{"eq 2 2.0", "", false},                 // Different types.
	{"lt true true", "", false},             // Unordered types.
	{"lt 1+0i 1+0i", "", false},             // Unordered types.
	{"eq .Ptr 1", "", false},                // Incompatible types.
	{"eq .Ptr .NegOne", "", false},          // Incompatible types.
	{"eq .Map .V1", "", false},              // Uncomparable types.
	{"eq .NonNilMap .NonNilMap", "", false}, // Uncomparable types.
}

func TestComparison(t *testing.T) {
	b := new(strings.Builder)
	var cmpStruct = struct {
		Uthree, Ufour    uint
		NegOne, Three    int
		Ptr, NilPtr      *int
		NonNilMap        map[int]int
		Map              map[int]int
		V1, V2           V
		Iface1, NilIface fmt.Stringer
	}{
		Uthree:    3,
		Ufour:     4,
		NegOne:    -1,
		Three:     3,
		Ptr:       new(int),
		NonNilMap: make(map[int]int),
		Iface1:    b,
	}
	for _, test := range cmpTests {
		text := fmt.Sprintf("{{if %s}}true{{else}}false{{end}}", test.expr)
		tmpl, err := New("empty").Parse(text)
		if err != nil {
			t.Fatalf("%q: %s", test.expr, err)
		}
		b.Reset()
		err = tmpl.Execute(b, &cmpStruct)
		if test.ok && err != nil {
			t.Errorf("%s errored incorrectly: %s", test.expr, err)
			continue
		}
		if !test.ok && err == nil {
			t.Errorf("%s did not error", test.expr)
			continue
		}
		if b.String() != test.truth {
			t.Errorf("%s: want %s; got %s", test.expr, test.truth, b.String())
		}
	}
}

func TestMissingMapKey(t *testing.T) {
	data := map[string]int{
		"x": 99,
	}
	tmpl, err := New("t1").Parse("{{.x}} {{.y}}")
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	// By default, just get "<no value>"
	err = tmpl.Execute(&b, data)
	if err != nil {
		t.Fatal(err)
	}
	want := "99 <no value>"
	got := b.String()
	if got != want {
		t.Errorf("got %q; expected %q", got, want)
	}
	// Same if we set the option explicitly to the default.
	tmpl.Option("missingkey=default")
	b.Reset()
	err = tmpl.Execute(&b, data)
	if err != nil {
		t.Fatal("default:", err)
	}
	want = "99 <no value>"
	got = b.String()
	if got != want {
		t.Errorf("got %q; expected %q", got, want)
	}
	// Next we ask for a zero value
	tmpl.Option("missingkey=zero")
	b.Reset()
	err = tmpl.Execute(&b, data)
	if err != nil {
		t.Fatal("zero:", err)
	}
	want = "99 0"
	got = b.String()
	if got != want {
		t.Errorf("got %q; expected %q", got, want)
	}
	// Now we ask for an error.
	tmpl.Option("missingkey=error")
	err = tmpl.Execute(&b, data)
	if err == nil {
		t.Errorf("expected error; got none")
	}
	// same Option, but now a nil interface: ask for an error
	err = tmpl.Execute(&b, nil)
	t.Log(err)
	if err == nil {
		t.Errorf("expected error for nil-interface; got none")
	}
}

// Test that the error message for multiline unterminated string
// refers to the line number of the opening quote.
func TestUnterminatedStringError(t *testing.T) {
	_, err := New("X").Parse("hello\n\n{{`unterminated\n\n\n\n}}\n some more\n\n")
	if err == nil {
		t.Fatal("expected error")
	}
	str := err.Error()
	if !strings.Contains(str, "X:3: unterminated raw quoted string") {
		t.Fatalf("unexpected error: %s", str)
	}
}

const alwaysErrorText = "always be failing"

var alwaysError = errors.New(alwaysErrorText)

type ErrorWriter int

func (e ErrorWriter) Write(p []byte) (int, error) {
	return 0, alwaysError
}

func TestExecuteGivesExecError(t *testing.T) {
	// First, a non-execution error shouldn't be an ExecError.
	tmpl, err := New("X").Parse("hello")
	if err != nil {
		t.Fatal(err)
	}
	err = tmpl.Execute(ErrorWriter(0), 0)
	if err == nil {
		t.Fatal("expected error; got none")
	}
	if err.Error() != alwaysErrorText {
		t.Errorf("expected %q error; got %q", alwaysErrorText, err)
	}
	// This one should be an ExecError.
	tmpl, err = New("X").Parse("hello, {{.X.Y}}")
	if err != nil {
		t.Fatal(err)
	}
	err = tmpl.Execute(io.Discard, 0)
	if err == nil {
		t.Fatal("expected error; got none")
	}
	eerr, ok := err.(ExecError)
	if !ok {
		t.Fatalf("did not expect ExecError %s", eerr)
	}
	expect := "field X in type int"
	if !strings.Contains(err.Error(), expect) {
		t.Errorf("expected %q; got %q", expect, err)
	}
}

func funcNameTestFunc() int {
	return 0
}

func TestGoodFuncNames(t *testing.T) {
	names := []string{
		"_",
		"a",
		"a1",
		"a1",
		"Ӵ",
	}
	for _, name := range names {
		tmpl := New("X").Funcs(
			FuncMap{
				name: funcNameTestFunc,
			},
		)
		if tmpl == nil {
			t.Fatalf("nil result for %q", name)
		}
	}
}

func TestBadFuncNames(t *testing.T) {
	names := []string{
		"",
		"2",
		"a-b",
	}
	for _, name := range names {
		testBadFuncName(name, t)
	}
}

func TestIsTrue(t *testing.T) {
	var nil_ptr *int
	var nil_chan chan int
	tests := []struct{
		v any
		want bool
	}{
		{1, true},
		{0, false},
		{uint8(1), true},
		{uint8(0), false},
		{float64(1.0), true},
		{float64(0.0), false},
		{complex64(1.0), true},
		{complex64(0.0), false},
		{true, true},
		{false, false},
		{[2]int{1,2}, true},
		{[0]int{}, false},
		{[]byte("abc"), true},
		{[]byte(""), false},
		{map[string] int {"a": 1, "b": 2}, true},
		{map[string] int {}, false},
		{make(chan int), true},
		{nil_chan, false},
		{new(int), true},
		{nil_ptr, false},
		{unsafe.Pointer(new(int)), true},
		{unsafe.Pointer(nil_ptr), false},
	}
	for _, test_case := range tests {
		got, _ := IsTrue(test_case.v)
		if got != test_case.want {
			t.Fatalf("expect result %v, got %v", test_case.want, got)
		}
	}
}

func testBadFuncName(name string, t *testing.T) {
	t.Helper()
	defer func() {
		recover()
	}()
	New("X").Funcs(
		FuncMap{
			name: funcNameTestFunc,
		},
	)
	// If we get here, the name did not cause a panic, which is how Funcs
	// reports an error.
	t.Errorf("%q succeeded incorrectly as function name", name)
}

func TestBlock(t *testing.T) {
	const (
		input   = `a({{block "inner" .}}bar({{.}})baz{{end}})b`
		want    = `a(bar(hello)baz)b`
		overlay = `{{define "inner"}}foo({{.}})bar{{end}}`
		want2   = `a(foo(goodbye)bar)b`
	)
	tmpl, err := New("outer").Parse(input)
	if err != nil {
		t.Fatal(err)
	}
	tmpl2, err := Must(tmpl.Clone()).Parse(overlay)
	if err != nil {
		t.Fatal(err)
	}

	var buf strings.Builder
	if err := tmpl.Execute(&buf, "hello"); err != nil {
		t.Fatal(err)
	}
	if got := buf.String(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	buf.Reset()
	if err := tmpl2.Execute(&buf, "goodbye"); err != nil {
		t.Fatal(err)
	}
	if got := buf.String(); got != want2 {
		t.Errorf("got %q, want %q", got, want2)
	}
}

func TestEvalFieldErrors(t *testing.T) {
	tests := []struct {
		name, src string
		value     any
		want      string
	}{
		{
			// Check that calling an invalid field on nil pointer
			// prints a field error instead of a distracting nil
			// pointer error. https://golang.org/issue/15125
			"MissingFieldOnNil",
			"{{.MissingField}}",
			(*T)(nil),
			"can't evaluate field MissingField in type *template.T",
		},
		{
			"MissingFieldOnNonNil",
			"{{.MissingField}}",
			&T{},
			"can't evaluate field MissingField in type *template.T",
		},
		{
			"ExistingFieldOnNil",
			"{{.X}}",
			(*T)(nil),
			"nil pointer evaluating *template.T.X",
		},
		{
			"MissingKeyOnNilMap",
			"{{.MissingKey}}",
			(*map[string]string)(nil),
			"nil pointer evaluating *map[string]string.MissingKey",
		},
		{
			"MissingKeyOnNilMapPtr",
			"{{.MissingKey}}",
			(*map[string]string)(nil),
			"nil pointer evaluating *map[string]string.MissingKey",
		},
		{
			"MissingKeyOnMapPtrToNil",
			"{{.MissingKey}}",
			&map[string]string{},
			"<nil>",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tmpl := Must(New("tmpl").Parse(tc.src))
			err := tmpl.Execute(io.Discard, tc.value)
			got := "<nil>"
			if err != nil {
				got = err.Error()
			}
			if !strings.HasSuffix(got, tc.want) {
				t.Fatalf("got error %q, want %q", got, tc.want)
			}
		})
	}
}

func TestMaxExecDepth(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	tmpl := Must(New("tmpl").Parse(`{{template "tmpl" .}}`))
	err := tmpl.Execute(io.Discard, nil)
	got := "<nil>"
	if err != nil {
		got = err.Error()
	}
	const want = "exceeded maximum template depth"
	if !strings.Contains(got, want) {
		t.Errorf("got error %q; want %q", got, want)
	}
}

func TestAddrOfIndex(t *testing.T) {
	// golang.org/issue/14916.
	// Before index worked on reflect.Values, the .String could not be
	// found on the (incorrectly unaddressable) V value,
	// in contrast to range, which worked fine.
	// Also testing that passing a reflect.Value to tmpl.Execute works.
	texts := []string{
		`{{range .}}{{.String}}{{end}}`,
		`{{with index . 0}}{{.String}}{{end}}`,
	}
	for _, text := range texts {
		tmpl := Must(New("tmpl").Parse(text))
		var buf strings.Builder
		err := tmpl.Execute(&buf, reflect.ValueOf([]V{{1}}))
		if err != nil {
			t.Fatalf("%s: Execute: %v", text, err)
		}
		if buf.String() != "<1>" {
			t.Fatalf("%s: template output = %q, want %q", text, &buf, "<1>")
		}
	}
}

func TestInterfaceValues(t *testing.T) {
	// golang.org/issue/17714.
	// Before index worked on reflect.Values, interface values
	// were always implicitly promoted to the underlying value,
	// except that nil interfaces were promoted to the zero reflect.Value.
	// Eliminating a round trip to interface{} and back to reflect.Value
	// eliminated this promotion, breaking these cases.
	tests := []struct {
		text string
		out  string
	}{
		{`{{index .Nil 1}}`, "ERROR: index of untyped nil"},
		{`{{index .Slice 2}}`, "2"},
		{`{{index .Slice .Two}}`, "2"},
		{`{{call .Nil 1}}`, "ERROR: call of nil"},
		{`{{call .PlusOne 1}}`, "2"},
		{`{{call .PlusOne .One}}`, "2"},
		{`{{and (index .Slice 0) true}}`, "0"},
		{`{{and .Zero true}}`, "0"},
		{`{{and (index .Slice 1) false}}`, "false"},
		{`{{and .One false}}`, "false"},
		{`{{or (index .Slice 0) false}}`, "false"},
		{`{{or .Zero false}}`, "false"},
		{`{{or (index .Slice 1) true}}`, "1"},
		{`{{or .One true}}`, "1"},
		{`{{not (index .Slice 0)}}`, "true"},
		{`{{not .Zero}}`, "true"},
		{`{{not (index .Slice 1)}}`, "false"},
		{`{{not .One}}`, "false"},
		{`{{eq (index .Slice 0) .Zero}}`, "true"},
		{`{{eq (index .Slice 1) .One}}`, "true"},
		{`{{ne (index .Slice 0) .Zero}}`, "false"},
		{`{{ne (index .Slice 1) .One}}`, "false"},
		{`{{ge (index .Slice 0) .One}}`, "false"},
		{`{{ge (index .Slice 1) .Zero}}`, "true"},
		{`{{gt (index .Slice 0) .One}}`, "false"},
		{`{{gt (index .Slice 1) .Zero}}`, "true"},
		{`{{le (index .Slice 0) .One}}`, "true"},
		{`{{le (index .Slice 1) .Zero}}`, "false"},
		{`{{lt (index .Slice 0) .One}}`, "true"},
		{`{{lt (index .Slice 1) .Zero}}`, "false"},
	}

	for _, tt := range tests {
		tmpl := Must(New("tmpl").Parse(tt.text))
		var buf strings.Builder
		err := tmpl.Execute(&buf, map[string]any{
			"PlusOne": func(n int) int {
				return n + 1
			},
			"Slice": []int{0, 1, 2, 3},
			"One":   1,
			"Two":   2,
			"Nil":   nil,
			"Zero":  0,
		})
		if strings.HasPrefix(tt.out, "ERROR:") {
			e := strings.TrimSpace(strings.TrimPrefix(tt.out, "ERROR:"))
			if err == nil || !strings.Contains(err.Error(), e) {
				t.Errorf("%s: Execute: %v, want error %q", tt.text, err, e)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: Execute: %v", tt.text, err)
			continue
		}
		if buf.String() != tt.out {
			t.Errorf("%s: template output = %q, want %q", tt.text, &buf, tt.out)
		}
	}
}

// Check that panics during calls are recovered and returned as errors.
func TestExecutePanicDuringCall(t *testing.T) {
	funcs := map[string]any{
		"doPanic": func() string {
			panic("custom panic string")
		},
	}
	tests := []struct {
		name    string
		input   string
		data    any
		wantErr string
	}{
		{
			"direct func call panics",
			"{{doPanic}}", (*T)(nil),
			`template: t:1:2: executing "t" at <doPanic>: error calling doPanic: custom panic string`,
		},
		{
			"indirect func call panics",
			"{{call doPanic}}", (*T)(nil),
			`template: t:1:7: executing "t" at <doPanic>: error calling doPanic: custom panic string`,
		},
		{
			"direct method call panics",
			"{{.GetU}}", (*T)(nil),
			`template: t:1:2: executing "t" at <.GetU>: error calling GetU: runtime error: invalid memory address or nil pointer dereference`,
		},
		{
			"indirect method call panics",
			"{{call .GetU}}", (*T)(nil),
			`template: t:1:7: executing "t" at <.GetU>: error calling GetU: runtime error: invalid memory address or nil pointer dereference`,
		},
		{
			"func field call panics",
			"{{call .PanicFunc}}", tVal,
			`template: t:1:2: executing "t" at <call .PanicFunc>: error calling call: test panic`,
		},
		{
			"method call on nil interface",
			"{{.NonEmptyInterfaceNil.Method0}}", tVal,
			`template: t:1:23: executing "t" at <.NonEmptyInterfaceNil.Method0>: nil pointer evaluating template.I.Method0`,
		},
	}
	for _, tc := range tests {
		b := new(bytes.Buffer)
		tmpl, err := New("t").Funcs(funcs).Parse(tc.input)
		if err != nil {
			t.Fatalf("parse error: %s", err)
		}
		err = tmpl.Execute(b, tc.data)
		if err == nil {
			t.Errorf("%s: expected error; got none", tc.name)
		} else if !strings.Contains(err.Error(), tc.wantErr) {
			if *debug {
				fmt.Printf("%s: test execute error: %s\n", tc.name, err)
			}
			t.Errorf("%s: expected error:\n%s\ngot:\n%s", tc.name, tc.wantErr, err)
		}
	}
}

func TestFunctionCheckDuringCall(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		data    any
		wantErr string
	}{{
		name:    "call nothing",
		input:   `{{call}}`,
		data:    tVal,
		wantErr: "wrong number of args for call: want at least 1 got 0",
	},
		{
			name:    "call non-function",
			input:   "{{call .True}}",
			data:    tVal,
			wantErr: "error calling call: non-function .True of type bool",
		},
		{
			name:    "call func with wrong argument",
			input:   "{{call .BinaryFunc 1}}",
			data:    tVal,
			wantErr: "error calling call: wrong number of args for .BinaryFunc: got 1 want 2",
		},
		{
			name:    "call variadic func with wrong argument",
			input:   `{{call .VariadicFuncInt}}`,
			data:    tVal,
			wantErr: "error calling call: wrong number of args for .VariadicFuncInt: got 0 want at least 1",
		},
		{
			name:    "call too few return number func",
			input:   `{{call .TooFewReturnCountFunc}}`,
			data:    tVal,
			wantErr: "error calling call: function .TooFewReturnCountFunc has 0 return values; should be 1 or 2",
		},
		{
			name:    "call too many return number func",
			input:   `{{call .TooManyReturnCountFunc}}`,
			data:    tVal,
			wantErr: "error calling call: function .TooManyReturnCountFunc has 3 return values; should be 1 or 2",
		},
		{
			name:    "call invalid return type func",
			input:   `{{call .InvalidReturnTypeFunc}}`,
			data:    tVal,
			wantErr: "error calling call: invalid function signature for .InvalidReturnTypeFunc: second return value should be error; is bool",
		},
		{
			name:    "call pipeline",
			input:   `{{call (len "test")}}`,
			data:    nil,
			wantErr: "error calling call: non-function len \"test\" of type int",
		},
	}

	for _, tc := range tests {
		b := new(bytes.Buffer)
		tmpl, err := New("t").Parse(tc.input)
		if err != nil {
			t.Fatalf("parse error: %s", err)
		}
		err = tmpl.Execute(b, tc.data)
		if err == nil {
			t.Errorf("%s: expected error; got none", tc.name)
		} else if tc.wantErr == "" || !strings.Contains(err.Error(), tc.wantErr) {
			if *debug {
				fmt.Printf("%s: test execute error: %s\n", tc.name, err)
			}
			t.Errorf("%s: expected error:\n%s\ngot:\n%s", tc.name, tc.wantErr, err)
		}
	}
}

// Issue 31810. Check that a parenthesized first argument behaves properly.
func TestIssue31810(t *testing.T) {
	// A simple value with no arguments is fine.
	var b strings.Builder
	const text = "{{ (.)  }}"
	tmpl, err := New("").Parse(text)
	if err != nil {
		t.Error(err)
	}
	err = tmpl.Execute(&b, "result")
	if err != nil {
		t.Error(err)
	}
	if b.String() != "result" {
		t.Errorf("%s got %q, expected %q", text, b.String(), "result")
	}

	// Even a plain function fails - need to use call.
	f := func() string { return "result" }
	b.Reset()
	err = tmpl.Execute(&b, f)
	if err == nil {
		t.Error("expected error with no call, got none")
	}

	// Works if the function is explicitly called.
	const textCall = "{{ (call .)  }}"
	tmpl, err = New("").Parse(textCall)
	b.Reset()
	err = tmpl.Execute(&b, f)
	if err != nil {
		t.Error(err)
	}
	if b.String() != "result" {
		t.Errorf("%s got %q, expected %q", textCall, b.String(), "result")
	}
}

// Issue 43065, range over send only channel
func TestIssue43065(t *testing.T) {
	var b bytes.Buffer
	tmp := Must(New("").Parse(`{{range .}}{{end}}`))
	ch := make(chan<- int)
	err := tmp.Execute(&b, ch)
	if err == nil {
		t.Error("expected err got nil")
	} else if !strings.Contains(err.Error(), "range over send-only channel") {
		t.Errorf("%s", err)
	}
}

// Issue 39807: data race in html/template & text/template
func TestIssue39807(t *testing.T) {
	var wg sync.WaitGroup

	tplFoo, err := New("foo").Parse(`{{ template "bar" . }}`)
	if err != nil {
		t.Error(err)
	}

	tplBar, err := New("bar").Parse("bar")
	if err != nil {
		t.Error(err)
	}

	gofuncs := 10
	numTemplates := 10

	for i := 1; i <= gofuncs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < numTemplates; j++ {
				_, err := tplFoo.AddParseTree(tplBar.Name(), tplBar.Tree)
				if err != nil {
					t.Error(err)
				}
				err = tplFoo.Execute(io.Discard, nil)
				if err != nil {
					t.Error(err)
				}
			}
		}()
	}

	wg.Wait()
}

// Issue 48215: embedded nil pointer causes panic.
// Fixed by adding FieldByIndexErr to the reflect package.
func TestIssue48215(t *testing.T) {
	type A struct {
		S string
	}
	type B struct {
		*A
	}
	tmpl, err := New("").Parse(`{{ .S }}`)
	if err != nil {
		t.Fatal(err)
	}
	err = tmpl.Execute(io.Discard, B{})
	// We expect an error, not a panic.
	if err == nil {
		t.Fatal("did not get error for nil embedded struct")
	}
	if !strings.Contains(err.Error(), "reflect: indirection through nil pointer to embedded struct field A") {
		t.Fatal(err)
	}
}
