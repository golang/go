// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc_test

import (
	"bytes"
	"go/ast"
	"go/doc"
	"go/format"
	"go/parser"
	"go/token"
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

func TestExamplesWholeFile(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "test.go", strings.NewReader(exampleWholeFile), parser.ParseComments)
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
	if g, w := formatFile(t, fset, e.Play), exampleWholeFileOutput; g != w {
		t.Errorf("got Play == %q, want %q", g, w)
	}
	if g, w := e.Output, "Hello, world!\n"; g != w {
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
