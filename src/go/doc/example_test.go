// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc_test

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/doc"
	"go/format"
	"go/parser"
	"go/token"
	"reflect"
	"strings"
	"testing"
)

const exampleTestFile = `
package foo_test

import (
	"flag"
	"fmt"
	"log"
	"sort"
	"os/exec"
)

func ExampleHello() {
	fmt.Println("Hello, world!")
	// Output: Hello, world!
}

func ExampleImport() {
	out, err := exec.Command("date").Output()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("The date is %s\n", out)
}

func ExampleKeyValue() {
	v := struct {
		a string
		b int
	}{
		a: "A",
		b: 1,
	}
	fmt.Print(v)
	// Output: a: "A", b: 1
}

func ExampleKeyValueImport() {
	f := flag.Flag{
		Name: "play",
	}
	fmt.Print(f)
	// Output: Name: "play"
}

var keyValueTopDecl = struct {
	a string
	b int
}{
	a: "B",
	b: 2,
}

func ExampleKeyValueTopDecl() {
	fmt.Print(keyValueTopDecl)
	// Output: a: "B", b: 2
}

// Person represents a person by name and age.
type Person struct {
    Name string
    Age  int
}

// String returns a string representation of the Person.
func (p Person) String() string {
    return fmt.Sprintf("%s: %d", p.Name, p.Age)
}

// ByAge implements sort.Interface for []Person based on
// the Age field.
type ByAge []Person

// Len returns the number of elements in ByAge.
func (a (ByAge)) Len() int { return len(a) }

// Swap swaps the elements in ByAge.
func (a ByAge) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByAge) Less(i, j int) bool { return a[i].Age < a[j].Age }

// people is the array of Person
var people = []Person{
	{"Bob", 31},
	{"John", 42},
	{"Michael", 17},
	{"Jenny", 26},
}

func ExampleSort() {
    fmt.Println(people)
    sort.Sort(ByAge(people))
    fmt.Println(people)
    // Output:
    // [Bob: 31 John: 42 Michael: 17 Jenny: 26]
    // [Michael: 17 Jenny: 26 Bob: 31 John: 42]
}
`

var exampleTestCases = []struct {
	Name, Play, Output string
}{
	{
		Name:   "Hello",
		Play:   exampleHelloPlay,
		Output: "Hello, world!\n",
	},
	{
		Name: "Import",
		Play: exampleImportPlay,
	},
	{
		Name:   "KeyValue",
		Play:   exampleKeyValuePlay,
		Output: "a: \"A\", b: 1\n",
	},
	{
		Name:   "KeyValueImport",
		Play:   exampleKeyValueImportPlay,
		Output: "Name: \"play\"\n",
	},
	{
		Name:   "KeyValueTopDecl",
		Play:   exampleKeyValueTopDeclPlay,
		Output: "a: \"B\", b: 2\n",
	},
	{
		Name:   "Sort",
		Play:   exampleSortPlay,
		Output: "[Bob: 31 John: 42 Michael: 17 Jenny: 26]\n[Michael: 17 Jenny: 26 Bob: 31 John: 42]\n",
	},
}

const exampleHelloPlay = `package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello, world!")
}
`
const exampleImportPlay = `package main

import (
	"fmt"
	"log"
	"os/exec"
)

func main() {
	out, err := exec.Command("date").Output()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("The date is %s\n", out)
}
`

const exampleKeyValuePlay = `package main

import (
	"fmt"
)

func main() {
	v := struct {
		a string
		b int
	}{
		a: "A",
		b: 1,
	}
	fmt.Print(v)
}
`

const exampleKeyValueImportPlay = `package main

import (
	"flag"
	"fmt"
)

func main() {
	f := flag.Flag{
		Name: "play",
	}
	fmt.Print(f)
}
`

const exampleKeyValueTopDeclPlay = `package main

import (
	"fmt"
)

var keyValueTopDecl = struct {
	a string
	b int
}{
	a: "B",
	b: 2,
}

func main() {
	fmt.Print(keyValueTopDecl)
}
`

const exampleSortPlay = `package main

import (
	"fmt"
	"sort"
)

// Person represents a person by name and age.
type Person struct {
	Name string
	Age  int
}

// String returns a string representation of the Person.
func (p Person) String() string {
	return fmt.Sprintf("%s: %d", p.Name, p.Age)
}

// ByAge implements sort.Interface for []Person based on
// the Age field.
type ByAge []Person

// Len returns the number of elements in ByAge.
func (a ByAge) Len() int { return len(a) }

// Swap swaps the elements in ByAge.
func (a ByAge) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByAge) Less(i, j int) bool { return a[i].Age < a[j].Age }

// people is the array of Person
var people = []Person{
	{"Bob", 31},
	{"John", 42},
	{"Michael", 17},
	{"Jenny", 26},
}

func main() {
	fmt.Println(people)
	sort.Sort(ByAge(people))
	fmt.Println(people)
}
`

func TestExamples(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "test.go", strings.NewReader(exampleTestFile), parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	for i, e := range doc.Examples(file) {
		c := exampleTestCases[i]
		if e.Name != c.Name {
			t.Errorf("got Name == %q, want %q", e.Name, c.Name)
		}
		if w := c.Play; w != "" {
			g := formatFile(t, fset, e.Play)
			if g != w {
				t.Errorf("%s: got Play == %q, want %q", c.Name, g, w)
			}
		}
		if g, w := e.Output, c.Output; g != w {
			t.Errorf("%s: got Output == %q, want %q", c.Name, g, w)
		}
	}
}

const exampleWholeFile = `package foo_test

type X int

func (X) Foo() {
}

func (X) TestBlah() {
}

func (X) BenchmarkFoo() {
}

func Example() {
	fmt.Println("Hello, world!")
	// Output: Hello, world!
}
`

const exampleWholeFileOutput = `package main

type X int

func (X) Foo() {
}

func (X) TestBlah() {
}

func (X) BenchmarkFoo() {
}

func main() {
	fmt.Println("Hello, world!")
}
`

const exampleWholeFileFunction = `package foo_test

func Foo(x int) {
}

func Example() {
	fmt.Println("Hello, world!")
	// Output: Hello, world!
}
`

const exampleWholeFileFunctionOutput = `package main

func Foo(x int) {
}

func main() {
	fmt.Println("Hello, world!")
}
`

const exampleWholeFileExternalFunction = `package foo_test

func foo(int)

func Example() {
	foo(42)
	// Output:
}
`

const exampleWholeFileExternalFunctionOutput = `package main

func foo(int)

func main() {
	foo(42)
}
`

var exampleWholeFileTestCases = []struct {
	Title, Source, Play, Output string
}{
	{
		"Methods",
		exampleWholeFile,
		exampleWholeFileOutput,
		"Hello, world!\n",
	},
	{
		"Function",
		exampleWholeFileFunction,
		exampleWholeFileFunctionOutput,
		"Hello, world!\n",
	},
	{
		"ExternalFunction",
		exampleWholeFileExternalFunction,
		exampleWholeFileExternalFunctionOutput,
		"",
	},
}

func TestExamplesWholeFile(t *testing.T) {
	for _, c := range exampleWholeFileTestCases {
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, "test.go", strings.NewReader(c.Source), parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		es := doc.Examples(file)
		if len(es) != 1 {
			t.Fatalf("%s: wrong number of examples; got %d want 1", c.Title, len(es))
		}
		e := es[0]
		if e.Name != "" {
			t.Errorf("%s: got Name == %q, want %q", c.Title, e.Name, "")
		}
		if g, w := formatFile(t, fset, e.Play), c.Play; g != w {
			t.Errorf("%s: got Play == %q, want %q", c.Title, g, w)
		}
		if g, w := e.Output, c.Output; g != w {
			t.Errorf("%s: got Output == %q, want %q", c.Title, g, w)
		}
	}
}

const exampleInspectSignature = `package foo_test

import (
	"bytes"
	"io"
)

func getReader() io.Reader { return nil }

func do(b bytes.Reader) {}

func Example() {
	getReader()
	do()
	// Output:
}

func ExampleIgnored() {
}
`

const exampleInspectSignatureOutput = `package main

import (
	"bytes"
	"io"
)

func getReader() io.Reader { return nil }

func do(b bytes.Reader) {}

func main() {
	getReader()
	do()
}
`

func TestExampleInspectSignature(t *testing.T) {
	// Verify that "bytes" and "io" are imported. See issue #28492.
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "test.go", strings.NewReader(exampleInspectSignature), parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	es := doc.Examples(file)
	if len(es) != 2 {
		t.Fatalf("wrong number of examples; got %d want 2", len(es))
	}
	// We are interested in the first example only.
	e := es[0]
	if e.Name != "" {
		t.Errorf("got Name == %q, want %q", e.Name, "")
	}
	if g, w := formatFile(t, fset, e.Play), exampleInspectSignatureOutput; g != w {
		t.Errorf("got Play == %q, want %q", g, w)
	}
	if g, w := e.Output, ""; g != w {
		t.Errorf("got Output == %q, want %q", g, w)
	}
}

const exampleEmpty = `
package p
func Example() {}
func Example_a()
`

const exampleEmptyOutput = `package main

func main() {}
func main()
`

func TestExampleEmpty(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "test.go", strings.NewReader(exampleEmpty), parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	es := doc.Examples(file)
	if len(es) != 1 {
		t.Fatalf("wrong number of examples; got %d want 1", len(es))
	}
	e := es[0]
	if e.Name != "" {
		t.Errorf("got Name == %q, want %q", e.Name, "")
	}
	if g, w := formatFile(t, fset, e.Play), exampleEmptyOutput; g != w {
		t.Errorf("got Play == %q, want %q", g, w)
	}
	if g, w := e.Output, ""; g != w {
		t.Errorf("got Output == %q, want %q", g, w)
	}
}

func formatFile(t *testing.T, fset *token.FileSet, n *ast.File) string {
	if n == nil {
		return "<nil>"
	}
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, n); err != nil {
		t.Fatal(err)
	}
	return buf.String()
}

// This example illustrates how to use NewFromFiles
// to compute package documentation with examples.
func ExampleNewFromFiles() {
	// src and test are two source files that make up
	// a package whose documentation will be computed.
	const src = `
// This is the package comment.
package p

import "fmt"

// This comment is associated with the Greet function.
func Greet(who string) {
	fmt.Printf("Hello, %s!\n", who)
}
`
	const test = `
package p_test

// This comment is associated with the ExampleGreet_world example.
func ExampleGreet_world() {
	Greet("world")
}
`

	// Create the AST by parsing src and test.
	fset := token.NewFileSet()
	files := []*ast.File{
		mustParse(fset, "src.go", src),
		mustParse(fset, "src_test.go", test),
	}

	// Compute package documentation with examples.
	p, err := doc.NewFromFiles(fset, files, "example.com/p")
	if err != nil {
		panic(err)
	}

	fmt.Printf("package %s - %s", p.Name, p.Doc)
	fmt.Printf("func %s - %s", p.Funcs[0].Name, p.Funcs[0].Doc)
	fmt.Printf(" ⤷ example with suffix %q - %s", p.Funcs[0].Examples[0].Suffix, p.Funcs[0].Examples[0].Doc)

	// Output:
	// package p - This is the package comment.
	// func Greet - This comment is associated with the Greet function.
	//  ⤷ example with suffix "world" - This comment is associated with the ExampleGreet_world example.
}

func TestClassifyExamples(t *testing.T) {
	const src = `
package p

const Const1 = 0
var   Var1   = 0

type (
	Type1     int
	Type1_Foo int
	Type1_foo int
	type2     int

	Embed struct { Type1 }
	Uembed struct { type2 }
)

func Func1()     {}
func Func1_Foo() {}
func Func1_foo() {}
func func2()     {}

func (Type1) Func1() {}
func (Type1) Func1_Foo() {}
func (Type1) Func1_foo() {}
func (Type1) func2() {}

func (type2) Func1() {}

type (
	Conflict          int
	Conflict_Conflict int
	Conflict_conflict int
)

func (Conflict) Conflict() {}
`
	const test = `
package p_test

func ExampleConst1() {} // invalid - no support for consts and vars
func ExampleVar1()   {} // invalid - no support for consts and vars

func Example()               {}
func Example_()              {} // invalid - suffix must start with a lower-case letter
func Example_suffix()        {}
func Example_suffix_xX_X_x() {}
func Example_世界()           {} // invalid - suffix must start with a lower-case letter
func Example_123()           {} // invalid - suffix must start with a lower-case letter
func Example_BadSuffix()     {} // invalid - suffix must start with a lower-case letter

func ExampleType1()               {}
func ExampleType1_()              {} // invalid - suffix must start with a lower-case letter
func ExampleType1_suffix()        {}
func ExampleType1_BadSuffix()     {} // invalid - suffix must start with a lower-case letter
func ExampleType1_Foo()           {}
func ExampleType1_Foo_suffix()    {}
func ExampleType1_Foo_BadSuffix() {} // invalid - suffix must start with a lower-case letter
func ExampleType1_foo()           {}
func ExampleType1_foo_suffix()    {}
func ExampleType1_foo_Suffix()    {} // matches Type1, instead of Type1_foo
func Exampletype2()               {} // invalid - cannot match unexported

func ExampleFunc1()               {}
func ExampleFunc1_()              {} // invalid - suffix must start with a lower-case letter
func ExampleFunc1_suffix()        {}
func ExampleFunc1_BadSuffix()     {} // invalid - suffix must start with a lower-case letter
func ExampleFunc1_Foo()           {}
func ExampleFunc1_Foo_suffix()    {}
func ExampleFunc1_Foo_BadSuffix() {} // invalid - suffix must start with a lower-case letter
func ExampleFunc1_foo()           {}
func ExampleFunc1_foo_suffix()    {}
func ExampleFunc1_foo_Suffix()    {} // matches Func1, instead of Func1_foo
func Examplefunc1()               {} // invalid - cannot match unexported

func ExampleType1_Func1()               {}
func ExampleType1_Func1_()              {} // invalid - suffix must start with a lower-case letter
func ExampleType1_Func1_suffix()        {}
func ExampleType1_Func1_BadSuffix()     {} // invalid - suffix must start with a lower-case letter
func ExampleType1_Func1_Foo()           {}
func ExampleType1_Func1_Foo_suffix()    {}
func ExampleType1_Func1_Foo_BadSuffix() {} // invalid - suffix must start with a lower-case letter
func ExampleType1_Func1_foo()           {}
func ExampleType1_Func1_foo_suffix()    {}
func ExampleType1_Func1_foo_Suffix()    {} // matches Type1.Func1, instead of Type1.Func1_foo
func ExampleType1_func2()               {} // matches Type1, instead of Type1.func2

func ExampleEmbed_Func1()         {} // invalid - no support for forwarded methods from embedding exported type
func ExampleUembed_Func1()        {} // methods from embedding unexported types are OK
func ExampleUembed_Func1_suffix() {}

func ExampleConflict_Conflict()        {} // ambiguous with either Conflict or Conflict_Conflict type
func ExampleConflict_conflict()        {} // ambiguous with either Conflict or Conflict_conflict type
func ExampleConflict_Conflict_suffix() {} // ambiguous with either Conflict or Conflict_Conflict type
func ExampleConflict_conflict_suffix() {} // ambiguous with either Conflict or Conflict_conflict type
`

	// Parse literal source code as a *doc.Package.
	fset := token.NewFileSet()
	files := []*ast.File{
		mustParse(fset, "src.go", src),
		mustParse(fset, "src_test.go", test),
	}
	p, err := doc.NewFromFiles(fset, files, "example.com/p")
	if err != nil {
		t.Fatalf("doc.NewFromFiles: %v", err)
	}

	// Collect the association of examples to top-level identifiers.
	got := map[string][]string{}
	got[""] = exampleNames(p.Examples)
	for _, f := range p.Funcs {
		got[f.Name] = exampleNames(f.Examples)
	}
	for _, t := range p.Types {
		got[t.Name] = exampleNames(t.Examples)
		for _, f := range t.Funcs {
			got[f.Name] = exampleNames(f.Examples)
		}
		for _, m := range t.Methods {
			got[t.Name+"."+m.Name] = exampleNames(m.Examples)
		}
	}

	want := map[string][]string{
		"": {"", "suffix", "suffix_xX_X_x"}, // Package-level examples.

		"Type1":     {"", "foo_Suffix", "func2", "suffix"},
		"Type1_Foo": {"", "suffix"},
		"Type1_foo": {"", "suffix"},

		"Func1":     {"", "foo_Suffix", "suffix"},
		"Func1_Foo": {"", "suffix"},
		"Func1_foo": {"", "suffix"},

		"Type1.Func1":     {"", "foo_Suffix", "suffix"},
		"Type1.Func1_Foo": {"", "suffix"},
		"Type1.Func1_foo": {"", "suffix"},

		"Uembed.Func1": {"", "suffix"},

		// These are implementation dependent due to the ambiguous parsing.
		"Conflict_Conflict": {"", "suffix"},
		"Conflict_conflict": {"", "suffix"},
	}

	for id := range got {
		if !reflect.DeepEqual(got[id], want[id]) {
			t.Errorf("classification mismatch for %q:\ngot  %q\nwant %q", id, got[id], want[id])
		}
	}
}

func exampleNames(exs []*doc.Example) (out []string) {
	for _, ex := range exs {
		out = append(out, ex.Suffix)
	}
	return out
}

func mustParse(fset *token.FileSet, filename, src string) *ast.File {
	f, err := parser.ParseFile(fset, filename, src, parser.ParseComments)
	if err != nil {
		panic(err)
	}
	return f
}
