// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generate

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

type splitTest struct {
	in  string
	out []string
}

// Same as above, except including source line number to set
type splitTestWithLine struct {
	in         string
	out        []string
	lineNumber int
}

const anyLineNo = 0

var splitTests = []splitTest{
	{"", nil},
	{"x", []string{"x"}},
	{" a b\tc ", []string{"a", "b", "c"}},
	{` " a " `, []string{" a "}},
	{"$GOARCH", []string{runtime.GOARCH}},
	{"$GOOS", []string{runtime.GOOS}},
	{"$GOFILE", []string{"proc.go"}},
	{"$GOPACKAGE", []string{"sys"}},
	{"a $XXNOTDEFINEDXX b", []string{"a", "", "b"}},
	{"/$XXNOTDEFINED/", []string{"//"}},
	{"/$DOLLAR/", []string{"/$/"}},
	{"yacc -o $GOARCH/yacc_$GOFILE", []string{"go", "tool", "yacc", "-o", runtime.GOARCH + "/yacc_proc.go"}},
}

func TestGenerateCommandParse(t *testing.T) {
	dir := filepath.Join(testenv.GOROOT(t), "src", "sys")
	g := &Generator{
		r:        nil, // Unused here.
		path:     filepath.Join(dir, "proc.go"),
		dir:      dir,
		file:     "proc.go",
		pkg:      "sys",
		commands: make(map[string][]string),
	}
	g.setEnv()
	g.setShorthand([]string{"-command", "yacc", "go", "tool", "yacc"})
	for _, test := range splitTests {
		// First with newlines.
		got := g.split("//go:generate " + test.in + "\n")
		if !reflect.DeepEqual(got, test.out) {
			t.Errorf("split(%q): got %q expected %q", test.in, got, test.out)
		}
		// Then with CRLFs, thank you Windows.
		got = g.split("//go:generate " + test.in + "\r\n")
		if !reflect.DeepEqual(got, test.out) {
			t.Errorf("split(%q): got %q expected %q", test.in, got, test.out)
		}
	}
}

// These environment variables will be undefined before the splitTestWithLine tests
var undefEnvList = []string{
	"_XYZZY_",
}

// These environment variables will be defined before the splitTestWithLine tests
var defEnvMap = map[string]string{
	"_PLUGH_": "SomeVal",
	"_X":      "Y",
}

// TestGenerateCommandShortHand - similar to TestGenerateCommandParse,
// except:
//  1. if the result starts with -command, record that shorthand
//     before moving on to the next test.
//  2. If a source line number is specified, set that in the parser
//     before executing the test.  i.e., execute the split as if it
//     processing that source line.
func TestGenerateCommandShorthand(t *testing.T) {
	dir := filepath.Join(testenv.GOROOT(t), "src", "sys")
	g := &Generator{
		r:        nil, // Unused here.
		path:     filepath.Join(dir, "proc.go"),
		dir:      dir,
		file:     "proc.go",
		pkg:      "sys",
		commands: make(map[string][]string),
	}

	var inLine string
	var expected, got []string

	g.setEnv()

	// Set up the system environment variables
	for i := range undefEnvList {
		os.Unsetenv(undefEnvList[i])
	}
	for k := range defEnvMap {
		os.Setenv(k, defEnvMap[k])
	}

	// simple command from environment variable
	inLine = "//go:generate -command CMD0 \"ab${_X}cd\""
	expected = []string{"-command", "CMD0", "abYcd"}
	got = g.split(inLine + "\n")

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("split(%q): got %q expected %q", inLine, got, expected)
	}

	// try again, with an extra level of indirection (should leave variable in command)
	inLine = "//go:generate -command CMD0 \"ab${DOLLAR}{_X}cd\""
	expected = []string{"-command", "CMD0", "ab${_X}cd"}
	got = g.split(inLine + "\n")

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("split(%q): got %q expected %q", inLine, got, expected)
	}

	// Now the interesting part, record that output as a command
	g.setShorthand(got)

	// see that the command still substitutes correctly from env. variable
	inLine = "//go:generate CMD0"
	expected = []string{"abYcd"}
	got = g.split(inLine + "\n")

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("split(%q): got %q expected %q", inLine, got, expected)
	}

	// Now change the value of $X and see if the recorded definition is
	// still intact (vs. having the $_X already substituted out)

	os.Setenv("_X", "Z")
	inLine = "//go:generate CMD0"
	expected = []string{"abZcd"}
	got = g.split(inLine + "\n")

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("split(%q): got %q expected %q", inLine, got, expected)
	}

	// What if the variable is now undefined?  Should be empty substitution.

	os.Unsetenv("_X")
	inLine = "//go:generate CMD0"
	expected = []string{"abcd"}
	got = g.split(inLine + "\n")

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("split(%q): got %q expected %q", inLine, got, expected)
	}

	// Try another undefined variable as an extra check
	os.Unsetenv("_Z")
	inLine = "//go:generate -command CMD1 \"ab${_Z}cd\""
	expected = []string{"-command", "CMD1", "abcd"}
	got = g.split(inLine + "\n")

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("split(%q): got %q expected %q", inLine, got, expected)
	}

	g.setShorthand(got)

	inLine = "//go:generate CMD1"
	expected = []string{"abcd"}
	got = g.split(inLine + "\n")

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("split(%q): got %q expected %q", inLine, got, expected)
	}

	const val = "someNewValue"
	os.Setenv("_Z", val)

	// try again with the properly-escaped variable.

	inLine = "//go:generate -command CMD2 \"ab${DOLLAR}{_Z}cd\""
	expected = []string{"-command", "CMD2", "ab${_Z}cd"}
	got = g.split(inLine + "\n")

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("split(%q): got %q expected %q", inLine, got, expected)
	}

	g.setShorthand(got)

	inLine = "//go:generate CMD2"
	expected = []string{"ab" + val + "cd"}
	got = g.split(inLine + "\n")

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("split(%q): got %q expected %q", inLine, got, expected)
	}
}

// Command-related tests for TestGenerateCommandShortHand2
// -- Note line numbers included to check substitutions from "built-in" variable - $GOLINE
var splitTestsLines = []splitTestWithLine{
	{"-command TEST1 $GOLINE", []string{"-command", "TEST1", "22"}, 22},
	{"-command TEST2 ${DOLLAR}GOLINE", []string{"-command", "TEST2", "$GOLINE"}, 26},
	{"TEST1", []string{"22"}, 33},
	{"TEST2", []string{"66"}, 66},
	{"TEST1 ''", []string{"22", "''"}, 99},
	{"TEST2 ''", []string{"44", "''"}, 44},
}

// TestGenerateCommandShortHand2 - similar to TestGenerateCommandParse,
// except:
//  1. if the result starts with -command, record that shorthand
//     before moving on to the next test.
//  2. If a source line number is specified, set that in the parser
//     before executing the test.  i.e., execute the split as if it
//     processing that source line.
func TestGenerateCommandShortHand2(t *testing.T) {
	dir := filepath.Join(testenv.GOROOT(t), "src", "sys")
	g := &Generator{
		r:        nil, // Unused here.
		path:     filepath.Join(dir, "proc.go"),
		dir:      dir,
		file:     "proc.go",
		pkg:      "sys",
		commands: make(map[string][]string),
	}
	g.setEnv()
	for _, test := range splitTestsLines {
		// if the test specified a line number, reflect that
		if test.lineNumber != anyLineNo {
			g.lineNum = test.lineNumber
			g.setEnv()
		}
		// First with newlines.
		got := g.split("//go:generate " + test.in + "\n")
		if !reflect.DeepEqual(got, test.out) {
			t.Errorf("split(%q): got %q expected %q", test.in, got, test.out)
		}
		// Then with CRLFs, thank you Windows.
		got = g.split("//go:generate " + test.in + "\r\n")
		if !reflect.DeepEqual(got, test.out) {
			t.Errorf("split(%q): got %q expected %q", test.in, got, test.out)
		}
		if got[0] == "-command" { // record commands
			g.setShorthand(got)
		}
	}
}
