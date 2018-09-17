package analysistest_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/findcall"
)

// TestTheTest tests the analysistest testing infrastructure.
func TestTheTest(t *testing.T) {
	// We'll simulate a partly failing test of the findcall analysis,
	// which (by default) reports calls to functions named 'println'.
	filemap := map[string]string{"a/b.go": `package main

func main() {
	println("hello, world") // want "call of println"
	println("hello, world") // want "wrong expectation text"
	println() // trigger an unexpected diagnostic
	print()	// want "unsatisfied expectation"
	print() // want: "ill-formed 'want' comment"
}
`}
	dir, cleanup, err := analysistest.WriteFiles(filemap)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	var got []string
	t2 := errorfunc(func(s string) { got = append(got, s) }) // a fake *testing.T
	analysistest.Run(t2, dir, findcall.Analyzer, "a")

	want := []string{
		`a/b.go:8:10: in 'want' comment: invalid syntax`,
		`a/b.go:5:9: diagnostic "call of println(...)" does not match pattern "wrong expectation text"`,
		`a/b.go:6:9: unexpected diagnostic: call of println(...)`,
		`a/b.go:7: expected diagnostic matching "unsatisfied expectation"`,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got:\n%s\nwant:\n%s",
			strings.Join(got, "\n"),
			strings.Join(want, "\n"))
	}
}

type errorfunc func(string)

func (f errorfunc) Errorf(format string, args ...interface{}) {
	f(fmt.Sprintf(format, args...))
}
