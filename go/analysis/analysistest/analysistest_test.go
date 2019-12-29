package analysistest_test

import (
	"fmt"
	"log"
	"os"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/findcall"
	"golang.org/x/tools/internal/testenv"
)

func init() {
	// This test currently requires GOPATH mode.
	// Explicitly disabling module mode should suffice, but
	// we'll also turn off GOPROXY just for good measure.
	if err := os.Setenv("GO111MODULE", "off"); err != nil {
		log.Fatal(err)
	}
	if err := os.Setenv("GOPROXY", "off"); err != nil {
		log.Fatal(err)
	}
}

// TestTheTest tests the analysistest testing infrastructure.
func TestTheTest(t *testing.T) {
	testenv.NeedsTool(t, "go")

	// We'll simulate a partly failing test of the findcall analysis,
	// which (by default) reports calls to functions named 'println'.
	findcall.Analyzer.Flags.Set("name", "println")

	filemap := map[string]string{"a/b.go": `package main // want package:"found"

func main() {
	// The expectation is ill-formed:
	print() // want: "diagnostic"
	print() // want foo"fact"
	print() // want foo:
	print() // want "\xZZ scan error"

	// A diagnostic is reported at this line, but the expectation doesn't match:
	println("hello, world") // want "wrong expectation text"

	// An unexpected diagnostic is reported at this line:
	println() // trigger an unexpected diagnostic

	// No diagnostic is reported at this line:
	print()	// want "unsatisfied expectation"

	// OK
	println("hello, world") // want "call of println"

	// OK /* */-form.
	println("안녕, 세계") /* want "call of println" */

	// OK  (nested comment)
	println("Γειά σου, Κόσμε") // some comment // want "call of println"

	// OK (nested comment in /**/)
	println("你好，世界") /* some comment // want "call of println" */

	// OK (multiple expectations on same line)
	println(); println() // want "call of println(...)" "call of println(...)"
}

// OK (facts and diagnostics on same line)
func println(...interface{}) { println() } // want println:"found" "call of println(...)"

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
		`a/b.go:5: in 'want' comment: unexpected ":"`,
		`a/b.go:6: in 'want' comment: got String after foo, want ':'`,
		`a/b.go:7: in 'want' comment: got EOF, want regular expression`,
		`a/b.go:8: in 'want' comment: invalid char escape`,
		`a/b.go:11:9: diagnostic "call of println(...)" does not match pattern "wrong expectation text"`,
		`a/b.go:14:9: unexpected diagnostic: call of println(...)`,
		`a/b.go:11: no diagnostic was reported matching "wrong expectation text"`,
		`a/b.go:17: no diagnostic was reported matching "unsatisfied expectation"`,
	}
	// Go 1.13's scanner error messages uses the word invalid where Go 1.12 used illegal. Convert them
	// to keep tests compatible with both.
	// TODO(matloob): Remove this once Go 1.13 is released.
	for i := range got {
		got[i] = strings.Replace(got[i], "illegal", "invalid", -1)
	} //
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
