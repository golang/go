// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc_test

import (
	"cmd/internal/objfile"
	"debug/dwarf"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

type testline struct {
	// line is one line of go source
	line string

	// scopes is a list of scope IDs of all the lexical scopes that this line
	// of code belongs to.
	// Scope IDs are assigned by traversing the tree of lexical blocks of a
	// function in pre-order
	// Scope IDs are function specific, i.e. scope 0 is always the root scope
	// of the function that this line belongs to. Empty scopes are not assigned
	// an ID (because they are not saved in debug_info).
	// Scope 0 is always omitted from this list since all lines always belong
	// to it.
	scopes []int

	// vars is the list of variables that belong in scopes[len(scopes)-1].
	// Local variables are prefixed with "var ", formal parameters with "arg ".
	// Must be ordered alphabetically.
	// Set to nil to skip the check.
	vars []string
}

var testfile = []testline{
	{line: "package main"},
	{line: "func f1(x int) { }"},
	{line: "func f2(x int) { }"},
	{line: "func f3(x int) { }"},
	{line: "func f4(x int) { }"},
	{line: "func f5(x int) { }"},
	{line: "func f6(x int) { }"},
	{line: "func fi(x interface{}) { if a, ok := x.(error); ok { a.Error() } }"},
	{line: "func gret1() int { return 2 }"},
	{line: "func gretbool() bool { return true }"},
	{line: "func gret3() (int, int, int) { return 0, 1, 2 }"},
	{line: "var v = []int{ 0, 1, 2 }"},
	{line: "var ch = make(chan int)"},
	{line: "var floatch = make(chan float64)"},
	{line: "var iface interface{}"},
	{line: "func TestNestedFor() {", vars: []string{"var a int"}},
	{line: "	a := 0"},
	{line: "	f1(a)"},
	{line: "	for i := 0; i < 5; i++ {", scopes: []int{1}, vars: []string{"var i int"}},
	{line: "		f2(i)", scopes: []int{1}},
	{line: "		for i := 0; i < 5; i++ {", scopes: []int{1, 2}, vars: []string{"var i int"}},
	{line: "			f3(i)", scopes: []int{1, 2}},
	{line: "		}"},
	{line: "		f4(i)", scopes: []int{1}},
	{line: "	}"},
	{line: "	f5(a)"},
	{line: "}"},
	{line: "func TestOas2() {", vars: []string{}},
	{line: "	if a, b, c := gret3(); a != 1 {", scopes: []int{1}, vars: []string{"var a int", "var b int", "var c int"}},
	{line: "		f1(a)", scopes: []int{1}},
	{line: "		f1(b)", scopes: []int{1}},
	{line: "		f1(c)", scopes: []int{1}},
	{line: "	}"},
	{line: "	for i, x := range v {", scopes: []int{2}, vars: []string{"var i int", "var x int"}},
	{line: "		f1(i)", scopes: []int{2}},
	{line: "		f1(x)", scopes: []int{2}},
	{line: "	}"},
	{line: "	if a, ok := <- ch; ok {", scopes: []int{3}, vars: []string{"var a int", "var ok bool"}},
	{line: "		f1(a)", scopes: []int{3}},
	{line: "	}"},
	{line: "	if a, ok := iface.(int); ok {", scopes: []int{4}, vars: []string{"var a int", "var ok bool"}},
	{line: "		f1(a)", scopes: []int{4}},
	{line: "	}"},
	{line: "}"},
	{line: "func TestIfElse() {"},
	{line: "	if x := gret1(); x != 0 {", scopes: []int{1}, vars: []string{"var x int"}},
	{line: "		a := 0", scopes: []int{1, 2}, vars: []string{"var a int"}},
	{line: "		f1(a); f1(x)", scopes: []int{1, 2}},
	{line: "	} else {"},
	{line: "		b := 1", scopes: []int{1, 3}, vars: []string{"var b int"}},
	{line: "		f1(b); f1(x+1)", scopes: []int{1, 3}},
	{line: "	}"},
	{line: "}"},
	{line: "func TestSwitch() {", vars: []string{}},
	{line: "	switch x := gret1(); x {", scopes: []int{1}, vars: []string{"var x int"}},
	{line: "	case 0:", scopes: []int{1, 2}},
	{line: "		i := x + 5", scopes: []int{1, 2}, vars: []string{"var i int"}},
	{line: "		f1(x); f1(i)", scopes: []int{1, 2}},
	{line: "	case 1:", scopes: []int{1, 3}},
	{line: "		j := x + 10", scopes: []int{1, 3}, vars: []string{"var j int"}},
	{line: "		f1(x); f1(j)", scopes: []int{1, 3}},
	{line: "	case 2:", scopes: []int{1, 4}},
	{line: "		k := x + 2", scopes: []int{1, 4}, vars: []string{"var k int"}},
	{line: "		f1(x); f1(k)", scopes: []int{1, 4}},
	{line: "	}"},
	{line: "}"},
	{line: "func TestTypeSwitch() {", vars: []string{}},
	{line: "	switch x := iface.(type) {"},
	{line: "	case int:", scopes: []int{1}},
	{line: "		f1(x)", scopes: []int{1}, vars: []string{"var x int"}},
	{line: "	case uint8:", scopes: []int{2}},
	{line: "		f1(int(x))", scopes: []int{2}, vars: []string{"var x uint8"}},
	{line: "	case float64:", scopes: []int{3}},
	{line: "		f1(int(x)+1)", scopes: []int{3}, vars: []string{"var x float64"}},
	{line: "	}"},
	{line: "}"},
	{line: "func TestSelectScope() {"},
	{line: "	select {"},
	{line: "	case i := <- ch:", scopes: []int{1}},
	{line: "		f1(i)", scopes: []int{1}, vars: []string{"var i int"}},
	{line: "	case f := <- floatch:", scopes: []int{2}},
	{line: "		f1(int(f))", scopes: []int{2}, vars: []string{"var f float64"}},
	{line: "	}"},
	{line: "}"},
	{line: "func TestBlock() {", vars: []string{"var a int"}},
	{line: "	a := 1"},
	{line: "	{"},
	{line: "		b := 2", scopes: []int{1}, vars: []string{"var b int"}},
	{line: "		f1(b)", scopes: []int{1}},
	{line: "		f1(a)", scopes: []int{1}},
	{line: "	}"},
	{line: "}"},
	{line: "func TestDiscontiguousRanges() {", vars: []string{"var a int"}},
	{line: "	a := 0"},
	{line: "	f1(a)"},
	{line: "	{"},
	{line: "		b := 0", scopes: []int{1}, vars: []string{"var b int"}},
	{line: "		f2(b)", scopes: []int{1}},
	{line: "		if gretbool() {", scopes: []int{1}},
	{line: "			c := 0", scopes: []int{1, 2}, vars: []string{"var c int"}},
	{line: "			f3(c)", scopes: []int{1, 2}},
	{line: "		} else {"},
	{line: "			c := 1.1", scopes: []int{1, 3}, vars: []string{"var c float64"}},
	{line: "			f4(int(c))", scopes: []int{1, 3}},
	{line: "		}"},
	{line: "		f5(b)", scopes: []int{1}},
	{line: "	}"},
	{line: "	f6(a)"},
	{line: "}"},
	{line: "func TestClosureScope() {", vars: []string{"var a int", "var b int", "var f func(int)"}},
	{line: "	a := 1; b := 1"},
	{line: "	f := func(c int) {", scopes: []int{0}, vars: []string{"arg c int", "var &b *int", "var a int", "var d int"}},
	{line: "		d := 3"},
	{line: "		f1(c); f1(d)"},
	{line: "		if e := 3; e != 0 {", scopes: []int{1}, vars: []string{"var e int"}},
	{line: "			f1(e)", scopes: []int{1}},
	{line: "			f1(a)", scopes: []int{1}},
	{line: "			b = 2", scopes: []int{1}},
	{line: "		}"},
	{line: "	}"},
	{line: "	f(3); f1(b)"},
	{line: "}"},
	{line: "func TestEscape() {"},
	{line: "	a := 1", vars: []string{"var a int"}},
	{line: "	{"},
	{line: "		b := 2", scopes: []int{1}, vars: []string{"var &b *int", "var p *int"}},
	{line: "		p := &b", scopes: []int{1}},
	{line: "		f1(a)", scopes: []int{1}},
	{line: "		fi(p)", scopes: []int{1}},
	{line: "	}"},
	{line: "}"},
	{line: "func TestCaptureVar(flag bool) func() int {"},
	{line: "	a := 1", vars: []string{"arg flag bool", "arg ~r1 func() int", "var a int"}},
	{line: "	if flag {"},
	{line: "		b := 2", scopes: []int{1}, vars: []string{"var b int", "var f func() int"}},
	{line: "		f := func() int {", scopes: []int{1, 0}},
	{line: "			return b + 1"},
	{line: "		}"},
	{line: "		return f", scopes: []int{1}},
	{line: "	}"},
	{line: "	f1(a)"},
	{line: "	return nil"},
	{line: "}"},
	{line: "func main() {"},
	{line: "	TestNestedFor()"},
	{line: "	TestOas2()"},
	{line: "	TestIfElse()"},
	{line: "	TestSwitch()"},
	{line: "	TestTypeSwitch()"},
	{line: "	TestSelectScope()"},
	{line: "	TestBlock()"},
	{line: "	TestDiscontiguousRanges()"},
	{line: "	TestClosureScope()"},
	{line: "	TestEscape()"},
	{line: "	TestCaptureVar(true)"},
	{line: "}"},
}

const detailOutput = false

// Compiles testfile checks that the description of lexical blocks emitted
// by the linker in debug_info, for each function in the main package,
// corresponds to what we expect it to be.
func TestScopeRanges(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	dir, err := ioutil.TempDir("", "TestScopeRanges")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	src, f := gobuild(t, dir, false, testfile)
	defer f.Close()

	// the compiler uses forward slashes for paths even on windows
	src = strings.Replace(src, "\\", "/", -1)

	pcln, err := f.PCLineTable()
	if err != nil {
		t.Fatal(err)
	}
	dwarfData, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}
	dwarfReader := dwarfData.Reader()

	lines := make(map[line][]*lexblock)

	for {
		entry, err := dwarfReader.Next()
		if err != nil {
			t.Fatal(err)
		}
		if entry == nil {
			break
		}

		if entry.Tag != dwarf.TagSubprogram {
			continue
		}

		name, ok := entry.Val(dwarf.AttrName).(string)
		if !ok || !strings.HasPrefix(name, "main.Test") {
			continue
		}

		var scope lexblock
		ctxt := scopexplainContext{
			dwarfData:   dwarfData,
			dwarfReader: dwarfReader,
			scopegen:    1,
		}

		readScope(&ctxt, &scope, entry)

		scope.markLines(pcln, lines)
	}

	anyerror := false
	for i := range testfile {
		tgt := testfile[i].scopes
		out := lines[line{src, i + 1}]

		if detailOutput {
			t.Logf("%s // %v", testfile[i].line, out)
		}

		scopesok := checkScopes(tgt, out)
		if !scopesok {
			t.Logf("mismatch at line %d %q: expected: %v got: %v\n", i, testfile[i].line, tgt, scopesToString(out))
		}

		varsok := true
		if testfile[i].vars != nil {
			if len(out) > 0 {
				varsok = checkVars(testfile[i].vars, out[len(out)-1].vars)
				if !varsok {
					t.Logf("variable mismatch at line %d %q for scope %d: expected: %v got: %v\n", i, testfile[i].line, out[len(out)-1].id, testfile[i].vars, out[len(out)-1].vars)
				}
			}
		}

		anyerror = anyerror || !scopesok || !varsok
	}

	if anyerror {
		t.Fatalf("mismatched output")
	}
}

func scopesToString(v []*lexblock) string {
	r := make([]string, len(v))
	for i, s := range v {
		r[i] = strconv.Itoa(s.id)
	}
	return "[ " + strings.Join(r, ", ") + " ]"
}

func checkScopes(tgt []int, out []*lexblock) bool {
	if len(out) > 0 {
		// omit scope 0
		out = out[1:]
	}
	if len(tgt) != len(out) {
		return false
	}
	for i := range tgt {
		if tgt[i] != out[i].id {
			return false
		}
	}
	return true
}

func checkVars(tgt, out []string) bool {
	if len(tgt) != len(out) {
		return false
	}
	for i := range tgt {
		if tgt[i] != out[i] {
			return false
		}
	}
	return true
}

type lexblock struct {
	id     int
	ranges [][2]uint64
	vars   []string
	scopes []lexblock
}

type line struct {
	file   string
	lineno int
}

type scopexplainContext struct {
	dwarfData   *dwarf.Data
	dwarfReader *dwarf.Reader
	scopegen    int
	lines       map[line][]int
}

// readScope reads the DW_TAG_lexical_block or the DW_TAG_subprogram in
// entry and writes a description in scope.
// Nested DW_TAG_lexical_block entries are read recursively.
func readScope(ctxt *scopexplainContext, scope *lexblock, entry *dwarf.Entry) {
	var err error
	scope.ranges, err = ctxt.dwarfData.Ranges(entry)
	if err != nil {
		panic(err)
	}
	for {
		e, err := ctxt.dwarfReader.Next()
		if err != nil {
			panic(err)
		}
		switch e.Tag {
		case 0:
			sort.Strings(scope.vars)
			return
		case dwarf.TagFormalParameter:
			typ, err := ctxt.dwarfData.Type(e.Val(dwarf.AttrType).(dwarf.Offset))
			if err != nil {
				panic(err)
			}
			scope.vars = append(scope.vars, "arg "+e.Val(dwarf.AttrName).(string)+" "+typ.String())
		case dwarf.TagVariable:
			typ, err := ctxt.dwarfData.Type(e.Val(dwarf.AttrType).(dwarf.Offset))
			if err != nil {
				panic(err)
			}
			scope.vars = append(scope.vars, "var "+e.Val(dwarf.AttrName).(string)+" "+typ.String())
		case dwarf.TagLexDwarfBlock:
			scope.scopes = append(scope.scopes, lexblock{id: ctxt.scopegen})
			ctxt.scopegen++
			readScope(ctxt, &scope.scopes[len(scope.scopes)-1], e)
		}
	}
}

// markLines marks all lines that belong to this scope with this scope
// Recursively calls markLines for all children scopes.
func (scope *lexblock) markLines(pcln objfile.Liner, lines map[line][]*lexblock) {
	for _, r := range scope.ranges {
		for pc := r[0]; pc < r[1]; pc++ {
			file, lineno, _ := pcln.PCToLine(pc)
			l := line{file, lineno}
			if len(lines[l]) == 0 || lines[l][len(lines[l])-1] != scope {
				lines[l] = append(lines[l], scope)
			}
		}
	}

	for i := range scope.scopes {
		scope.scopes[i].markLines(pcln, lines)
	}
}

func gobuild(t *testing.T, dir string, optimized bool, testfile []testline) (string, *objfile.File) {
	src := filepath.Join(dir, "test.go")
	dst := filepath.Join(dir, "out.o")

	f, err := os.Create(src)
	if err != nil {
		t.Fatal(err)
	}
	for i := range testfile {
		f.Write([]byte(testfile[i].line))
		f.Write([]byte{'\n'})
	}
	f.Close()

	args := []string{"build"}
	if !optimized {
		args = append(args, "-gcflags=-N -l")
	}
	args = append(args, "-o", dst, src)

	cmd := exec.Command(testenv.GoToolPath(t), args...)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Logf("build: %s\n", string(b))
		t.Fatal(err)
	}

	pkg, err := objfile.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	return src, pkg
}

// TestEmptyDwarfRanges tests that no list entry in debug_ranges has start == end.
// See issue #23928.
func TestEmptyDwarfRanges(t *testing.T) {
	testenv.MustHaveGoRun(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	dir, err := ioutil.TempDir("", "TestEmptyDwarfRanges")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	_, f := gobuild(t, dir, true, []testline{{line: "package main"}, {line: "func main(){ println(\"hello\") }"}})
	defer f.Close()

	dwarfData, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}
	dwarfReader := dwarfData.Reader()

	for {
		entry, err := dwarfReader.Next()
		if err != nil {
			t.Fatal(err)
		}
		if entry == nil {
			break
		}

		ranges, err := dwarfData.Ranges(entry)
		if err != nil {
			t.Fatal(err)
		}
		if ranges == nil {
			continue
		}

		for _, rng := range ranges {
			if rng[0] == rng[1] {
				t.Errorf("range entry with start == end: %v", rng)
			}
		}
	}
}
