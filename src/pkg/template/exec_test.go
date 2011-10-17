// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"
)

var debug = flag.Bool("debug", false, "show the errors produced by the tests")

// T has lots of interesting pieces to use to test execution.
type T struct {
	// Basics
	True        bool
	I           int
	U16         uint16
	X           string
	FloatZero   float64
	ComplexZero float64
	// Nested structs.
	U *U
	// Struct with String method.
	V0     V
	V1, V2 *V
	// Slices
	SI      []int
	SIEmpty []int
	SB      []bool
	// Maps
	MSI      map[string]int
	MSIone   map[string]int // one element, for deterministic output
	MSIEmpty map[string]int
	MXI      map[interface{}]int
	MII      map[int]int
	SMSI     []map[string]int
	// Empty interfaces; used to see if we can dig inside one.
	Empty0 interface{} // nil
	Empty1 interface{}
	Empty2 interface{}
	Empty3 interface{}
	Empty4 interface{}
	// Non-empty interface.
	NonEmptyInterface I
	// Stringer.
	Str fmt.Stringer
	// Pointers
	PI  *int
	PSI *[]int
	NIL *int
	// Template to test evaluation of templates.
	Tmpl *Template
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

var tVal = &T{
	True:   true,
	I:      17,
	U16:    16,
	X:      "x",
	U:      &U{"v"},
	V0:     V{6666},
	V1:     &V{7777}, // leave V2 as nil
	SI:     []int{3, 4, 5},
	SB:     []bool{true, false},
	MSI:    map[string]int{"one": 1, "two": 2, "three": 3},
	MSIone: map[string]int{"one": 1},
	MXI:    map[interface{}]int{"one": 1},
	MII:    map[int]int{1: 1},
	SMSI: []map[string]int{
		{"one": 1, "two": 2},
		{"eleven": 11, "twelve": 12},
	},
	Empty1:            3,
	Empty2:            "empty2",
	Empty3:            []int{7, 8},
	Empty4:            &U{"UinEmpty"},
	NonEmptyInterface: new(T),
	Str:               os.NewError("foozle"),
	PI:                newInt(23),
	PSI:               newIntSlice(21, 22, 23),
	Tmpl:              Must(New("x").Parse("test template")), // "x" is the value of .X
}

// A non-empty interface.
type I interface {
	Method0() string
}

var iVal I = tVal

// Helpers for creation.
func newInt(n int) *int {
	p := new(int)
	*p = n
	return p
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

func (t *T) MAdd(a int, b []int) []int {
	v := make([]int, len(b))
	for i, x := range b {
		v[i] = x + a
	}
	return v
}

// MSort is used to sort map keys for stable output. (Nice trick!)
func (t *T) MSort(m map[string]int) []string {
	keys := make([]string, len(m))
	i := 0
	for k := range m {
		keys[i] = k
		i++
	}
	sort.Strings(keys)
	return keys
}

// EPERM returns a value and an os.Error according to its argument.
func (t *T) EPERM(error bool) (bool, os.Error) {
	if error {
		return true, os.EPERM
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

func typeOf(arg interface{}) string {
	return fmt.Sprintf("%T", arg)
}

type execTest struct {
	name   string
	input  string
	output string
	data   interface{}
	ok     bool
}

// bigInt and bigUint are hex string representing numbers either side
// of the max int boundary.
// We do it this way so the test doesn't depend on ints being 32 bits.
var (
	bigInt  = fmt.Sprintf("0x%x", int(1<<uint(reflect.TypeOf(0).Bits()-1)-1))
	bigUint = fmt.Sprintf("0x%x", uint(1<<uint(reflect.TypeOf(0).Bits()-1)))
)

var execTests = []execTest{
	// Trivial cases.
	{"empty", "", "", nil, true},
	{"text", "some text", "some text", nil, true},

	// Ideal constants.
	{"ideal int", "{{typeOf 3}}", "int", 0, true},
	{"ideal float", "{{typeOf 1.0}}", "float64", 0, true},
	{"ideal exp float", "{{typeOf 1e1}}", "float64", 0, true},
	{"ideal complex", "{{typeOf 1i}}", "complex128", 0, true},
	{"ideal int", "{{typeOf " + bigInt + "}}", "int", 0, true},
	{"ideal too big", "{{typeOf " + bigUint + "}}", "", 0, false},

	// Fields of structs.
	{".X", "-{{.X}}-", "-x-", tVal, true},
	{".U.V", "-{{.U.V}}-", "-v-", tVal, true},

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

	// Type with String method.
	{"V{6666}.String()", "-{{.V0}}-", "-<6666>-", tVal, true},
	{"&V{7777}.String()", "-{{.V1}}-", "-<7777>-", tVal, true},
	{"(*V)(nil).String()", "-{{.V2}}-", "-nilV-", tVal, true},

	// Pointers.
	{"*int", "{{.PI}}", "23", tVal, true},
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

	// Method calls.
	{".Method0", "-{{.Method0}}-", "-M0-", tVal, true},
	{".Method1(1234)", "-{{.Method1 1234}}-", "-1234-", tVal, true},
	{".Method1(.I)", "-{{.Method1 .I}}-", "-17-", tVal, true},
	{".Method2(3, .X)", "-{{.Method2 3 .X}}-", "-Method2: 3 x-", tVal, true},
	{".Method2(.U16, `str`)", "-{{.Method2 .U16 `str`}}-", "-Method2: 16 str-", tVal, true},
	{".Method2(.U16, $x)", "{{if $x := .X}}-{{.Method2 .U16 $x}}{{end}}-", "-Method2: 16 x-", tVal, true},
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

	// Pipelines.
	{"pipeline", "-{{.Method0 | .Method2 .U16}}-", "-Method2: 16 M0-", tVal, true},

	// If.
	{"if true", "{{if true}}TRUE{{end}}", "TRUE", tVal, true},
	{"if false", "{{if false}}TRUE{{else}}FALSE{{end}}", "FALSE", tVal, true},
	{"if 1", "{{if 1}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if 0", "{{if 0}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if 1.5", "{{if 1.5}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if 0.0", "{{if .FloatZero}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if 1.5i", "{{if 1.5i}}NON-ZERO{{else}}ZERO{{end}}", "NON-ZERO", tVal, true},
	{"if 0.0i", "{{if .ComplexZero}}NON-ZERO{{else}}ZERO{{end}}", "ZERO", tVal, true},
	{"if emptystring", "{{if ``}}NON-EMPTY{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"if string", "{{if `notempty`}}NON-EMPTY{{else}}EMPTY{{end}}", "NON-EMPTY", tVal, true},
	{"if emptyslice", "{{if .SIEmpty}}NON-EMPTY{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"if slice", "{{if .SI}}NON-EMPTY{{else}}EMPTY{{end}}", "NON-EMPTY", tVal, true},
	{"if emptymap", "{{if .MSIEmpty}}NON-EMPTY{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"if map", "{{if .MSI}}NON-EMPTY{{else}}EMPTY{{end}}", "NON-EMPTY", tVal, true},
	{"if $x with $y int", "{{if $x := true}}{{with $y := .I}}{{$x}},{{$y}}{{end}}{{end}}", "true,17", tVal, true},
	{"if $x with $x int", "{{if $x := true}}{{with $x := .I}}{{$x}},{{end}}{{$x}}{{end}}", "17,true", tVal, true},

	// Print etc.
	{"print", `{{print "hello, print"}}`, "hello, print", tVal, true},
	{"print", `{{print 1 2 3}}`, "1 2 3", tVal, true},
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

	// JavaScript.
	{"js", `{{js .}}`, `It\'d be nice.`, `It'd be nice.`, true},

	// URL query.
	{"urlquery", `{{"http://www.example.org/"|urlquery}}`, "http%3A%2F%2Fwww.example.org%2F", nil, true},

	// Booleans
	{"not", "{{not true}} {{not false}}", "false true", nil, true},
	{"and", "{{and false 0}} {{and 1 0}} {{and 0 true}} {{and 1 1}}", "false 0 0 1", nil, true},
	{"or", "{{or 0 0}} {{or 1 0}} {{or 0 true}} {{or 1 1}}", "0 1 true 1", nil, true},
	{"boolean if", "{{if and true 1 `hi`}}TRUE{{else}}FALSE{{end}}", "TRUE", tVal, true},
	{"boolean if not", "{{if and true 1 `hi` | not}}TRUE{{else}}FALSE{{end}}", "FALSE", nil, true},

	// Indexing.
	{"slice[0]", "{{index .SI 0}}", "3", tVal, true},
	{"slice[1]", "{{index .SI 1}}", "4", tVal, true},
	{"slice[HUGE]", "{{index .SI 10}}", "", tVal, false},
	{"slice[WRONG]", "{{index .SI `hello`}}", "", tVal, false},
	{"map[one]", "{{index .MSI `one`}}", "1", tVal, true},
	{"map[two]", "{{index .MSI `two`}}", "2", tVal, true},
	{"map[NO]", "{{index .MSI `XXX`}}", "", tVal, true},
	{"map[WRONG]", "{{index .MSI 10}}", "", tVal, false},
	{"double index", "{{index .SMSI 1 `eleven`}}", "11", tVal, true},

	// Len.
	{"slice", "{{len .SI}}", "3", tVal, true},
	{"map", "{{len .MSI }}", "3", tVal, true},
	{"len of int", "{{len 3}}", "", tVal, false},
	{"len of nothing", "{{len .Empty0}}", "", tVal, false},

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

	// Range.
	{"range []int", "{{range .SI}}-{{.}}-{{end}}", "-3--4--5-", tVal, true},
	{"range empty no else", "{{range .SIEmpty}}-{{.}}-{{end}}", "", tVal, true},
	{"range []int else", "{{range .SI}}-{{.}}-{{else}}EMPTY{{end}}", "-3--4--5-", tVal, true},
	{"range empty else", "{{range .SIEmpty}}-{{.}}-{{else}}EMPTY{{end}}", "EMPTY", tVal, true},
	{"range []bool", "{{range .SB}}-{{.}}-{{end}}", "-true--false-", tVal, true},
	{"range []int method", "{{range .SI | .MAdd .I}}-{{.}}-{{end}}", "-20--21--22-", tVal, true},
	{"range map", "{{range .MSI | .MSort}}-{{.}}-{{end}}", "-one--three--two-", tVal, true},
	{"range empty map no else", "{{range .MSIEmpty}}-{{.}}-{{end}}", "", tVal, true},
	{"range map else", "{{range .MSI | .MSort}}-{{.}}-{{else}}EMPTY{{end}}", "-one--three--two-", tVal, true},
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

	// Cute examples.
	{"or as if true", `{{or .SI "slice is empty"}}`, "[3 4 5]", tVal, true},
	{"or as if false", `{{or .SIEmpty "slice is empty"}}`, "slice is empty", tVal, true},

	// Error handling.
	{"error method, error", "{{.EPERM true}}", "", tVal, false},
	{"error method, no error", "{{.EPERM false}}", "false", tVal, true},

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
	// Args need to be indirected and dereferenced sometimes.
	{"bug6a", "{{vfunc .V0 .V1}}", "vfunc", tVal, true},
	{"bug6b", "{{vfunc .V0 .V0}}", "vfunc", tVal, true},
	{"bug6c", "{{vfunc .V1 .V0}}", "vfunc", tVal, true},
	{"bug6d", "{{vfunc .V1 .V1}}", "vfunc", tVal, true},
}

func zeroArgs() string {
	return "zeroArgs"
}

func oneArg(a string) string {
	return "oneArg=" + a
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

func testExecute(execTests []execTest, set *Set, t *testing.T) {
	b := new(bytes.Buffer)
	funcs := FuncMap{
		"count":    count,
		"oneArg":   oneArg,
		"typeOf":   typeOf,
		"vfunc":    vfunc,
		"zeroArgs": zeroArgs,
	}
	for _, test := range execTests {
		tmpl := New(test.name).Funcs(funcs)
		_, err := tmpl.ParseInSet(test.input, set)
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
		var b = new(bytes.Buffer)
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
	_, err := tmpl.Parse("{{.EPERM true}}")
	if err != nil {
		t.Fatalf("parse error: %s", err)
	}
	err = tmpl.Execute(b, tVal)
	if err == nil {
		t.Errorf("expected error; got none")
	} else if !strings.Contains(err.String(), os.EPERM.String()) {
		if *debug {
			fmt.Printf("test execute error: %s\n", err)
		}
		t.Errorf("expected os.EPERM; got %s", err)
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
		{"unprintable \uFDFF", `unprintable \uFDFF`},
		{`<html>`, `\x3Chtml\x3E`},
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
const treeTemplate = `
	(define "tree")
	[
		(.Val)
		(with .Left)
			(template "tree" .)
		(end)
		(with .Right)
			(template "tree" .)
		(end)
	]
	(end)
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
	set := new(Set)
	_, err := set.Delims("(", ")").Parse(treeTemplate)
	if err != nil {
		t.Fatal("parse error:", err)
	}
	var b bytes.Buffer
	err = set.Execute(&b, "tree", tree)
	if err != nil {
		t.Fatal("exec error:", err)
	}
	stripSpace := func(r int) int {
		if r == '\t' || r == '\n' {
			return -1
		}
		return r
	}
	result := strings.Map(stripSpace, b.String())
	const expect = "[1[2[3[4]][5[6]]][7[8[9]][10[11]]]]"
	if result != expect {
		t.Errorf("expected %q got %q", expect, result)
	}
}
