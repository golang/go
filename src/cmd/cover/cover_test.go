// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
)

const (
	// Data directory, also the package directory for the test.
	testdata = "testdata"
)

var (
	// Input files.
	testMain       = filepath.Join(testdata, "main.go")
	testTest       = filepath.Join(testdata, "test.go")
	coverProfile   = filepath.Join(testdata, "profile.cov")
	toolexecSource = filepath.Join(testdata, "toolexec.go")

	// The HTML test files are in a separate directory
	// so they are a complete package.
	htmlGolden = filepath.Join(testdata, "html", "html.golden")

	// Temporary files.
	tmpTestMain    string
	coverInput     string
	coverOutput    string
	htmlProfile    string
	htmlHTML       string
	htmlUDir       string
	htmlU          string
	htmlUTest      string
	htmlUProfile   string
	htmlUHTML      string
	lineDupDir     string
	lineDupGo      string
	lineDupTestGo  string
	lineDupProfile string
)

var (
	// testTempDir is a temporary directory created in TestMain.
	testTempDir string

	// testcover is a newly built version of the cover program.
	testcover string

	// toolexec is a program to use as the go tool's -toolexec argument.
	toolexec string

	// testcoverErr records an error building testcover or toolexec.
	testcoverErr error

	// testcoverOnce is used to build testcover once.
	testcoverOnce sync.Once

	// toolexecArg is the argument to pass to the go tool.
	toolexecArg string
)

var debug = flag.Bool("debug", false, "keep rewritten files for debugging")

// We use TestMain to set up a temporary directory and remove it when
// the tests are done.
func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "go-testcover")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Setenv("GOPATH", filepath.Join(dir, "_gopath"))

	testTempDir = dir

	tmpTestMain = filepath.Join(dir, "main.go")
	coverInput = filepath.Join(dir, "test_line.go")
	coverOutput = filepath.Join(dir, "test_cover.go")
	htmlProfile = filepath.Join(dir, "html.cov")
	htmlHTML = filepath.Join(dir, "html.html")
	htmlUDir = filepath.Join(dir, "htmlunformatted")
	htmlU = filepath.Join(htmlUDir, "htmlunformatted.go")
	htmlUTest = filepath.Join(htmlUDir, "htmlunformatted_test.go")
	htmlUProfile = filepath.Join(htmlUDir, "htmlunformatted.cov")
	htmlUHTML = filepath.Join(htmlUDir, "htmlunformatted.html")
	lineDupDir = filepath.Join(dir, "linedup")
	lineDupGo = filepath.Join(lineDupDir, "linedup.go")
	lineDupTestGo = filepath.Join(lineDupDir, "linedup_test.go")
	lineDupProfile = filepath.Join(lineDupDir, "linedup.out")

	status := m.Run()

	if !*debug {
		os.RemoveAll(dir)
	}

	os.Exit(status)
}

// buildCover builds a version of the cover program for testing.
// This ensures that "go test cmd/cover" tests the current cmd/cover.
func buildCover(t *testing.T) {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	testcoverOnce.Do(func() {
		var wg sync.WaitGroup
		wg.Add(2)

		var err1, err2 error
		go func() {
			defer wg.Done()
			testcover = filepath.Join(testTempDir, "cover.exe")
			t.Logf("running [go build -o %s]", testcover)
			out, err := exec.Command(testenv.GoToolPath(t), "build", "-o", testcover).CombinedOutput()
			if len(out) > 0 {
				t.Logf("%s", out)
			}
			err1 = err
		}()

		go func() {
			defer wg.Done()
			toolexec = filepath.Join(testTempDir, "toolexec.exe")
			t.Logf("running [go -build -o %s %s]", toolexec, toolexecSource)
			out, err := exec.Command(testenv.GoToolPath(t), "build", "-o", toolexec, toolexecSource).CombinedOutput()
			if len(out) > 0 {
				t.Logf("%s", out)
			}
			err2 = err
		}()

		wg.Wait()

		testcoverErr = err1
		if err2 != nil && err1 == nil {
			testcoverErr = err2
		}

		toolexecArg = "-toolexec=" + toolexec + " " + testcover
	})
	if testcoverErr != nil {
		t.Fatal("failed to build testcover or toolexec program:", testcoverErr)
	}
}

// Run this shell script, but do it in Go so it can be run by "go test".
//
//	replace the word LINE with the line number < testdata/test.go > testdata/test_line.go
// 	go build -o testcover
// 	testcover -mode=count -var=CoverTest -o ./testdata/test_cover.go testdata/test_line.go
//	go run ./testdata/main.go ./testdata/test.go
//
func TestCover(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoRun(t)
	buildCover(t)

	// Read in the test file (testTest) and write it, with LINEs specified, to coverInput.
	file, err := os.ReadFile(testTest)
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(file, []byte("\n"))
	for i, line := range lines {
		lines[i] = bytes.ReplaceAll(line, []byte("LINE"), []byte(fmt.Sprint(i+1)))
	}

	// Add a function that is not gofmt'ed. This used to cause a crash.
	// We don't put it in test.go because then we would have to gofmt it.
	// Issue 23927.
	lines = append(lines, []byte("func unFormatted() {"),
		[]byte("\tif true {"),
		[]byte("\t}else{"),
		[]byte("\t}"),
		[]byte("}"))
	lines = append(lines, []byte("func unFormatted2(b bool) {if b{}else{}}"))

	if err := os.WriteFile(coverInput, bytes.Join(lines, []byte("\n")), 0666); err != nil {
		t.Fatal(err)
	}

	// testcover -mode=count -var=thisNameMustBeVeryLongToCauseOverflowOfCounterIncrementStatementOntoNextLineForTest -o ./testdata/test_cover.go testdata/test_line.go
	cmd := exec.Command(testcover, "-mode=count", "-var=thisNameMustBeVeryLongToCauseOverflowOfCounterIncrementStatementOntoNextLineForTest", "-o", coverOutput, coverInput)
	run(cmd, t)

	cmd = exec.Command(testcover, "-mode=set", "-var=Not_an-identifier", "-o", coverOutput, coverInput)
	err = cmd.Run()
	if err == nil {
		t.Error("Expected cover to fail with an error")
	}

	// Copy testmain to testTempDir, so that it is in the same directory
	// as coverOutput.
	b, err := os.ReadFile(testMain)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(tmpTestMain, b, 0444); err != nil {
		t.Fatal(err)
	}

	// go run ./testdata/main.go ./testdata/test.go
	cmd = exec.Command(testenv.GoToolPath(t), "run", tmpTestMain, coverOutput)
	run(cmd, t)

	file, err = os.ReadFile(coverOutput)
	if err != nil {
		t.Fatal(err)
	}
	// compiler directive must appear right next to function declaration.
	if got, err := regexp.MatchString(".*\n//go:nosplit\nfunc someFunction().*", string(file)); err != nil || !got {
		t.Error("misplaced compiler directive")
	}
	// "go:linkname" compiler directive should be present.
	if got, err := regexp.MatchString(`.*go\:linkname some\_name some\_name.*`, string(file)); err != nil || !got {
		t.Error("'go:linkname' compiler directive not found")
	}

	// Other comments should be preserved too.
	c := ".*// This comment didn't appear in generated go code.*"
	if got, err := regexp.MatchString(c, string(file)); err != nil || !got {
		t.Errorf("non compiler directive comment %q not found", c)
	}
}

// TestDirectives checks that compiler directives are preserved and positioned
// correctly. Directives that occur before top-level declarations should remain
// above those declarations, even if they are not part of the block of
// documentation comments.
func TestDirectives(t *testing.T) {
	t.Parallel()
	buildCover(t)

	// Read the source file and find all the directives. We'll keep
	// track of whether each one has been seen in the output.
	testDirectives := filepath.Join(testdata, "directives.go")
	source, err := os.ReadFile(testDirectives)
	if err != nil {
		t.Fatal(err)
	}
	sourceDirectives := findDirectives(source)

	// testcover -mode=atomic ./testdata/directives.go
	cmd := exec.Command(testcover, "-mode=atomic", testDirectives)
	cmd.Stderr = os.Stderr
	output, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}

	// Check that all directives are present in the output.
	outputDirectives := findDirectives(output)
	foundDirective := make(map[string]bool)
	for _, p := range sourceDirectives {
		foundDirective[p.name] = false
	}
	for _, p := range outputDirectives {
		if found, ok := foundDirective[p.name]; !ok {
			t.Errorf("unexpected directive in output: %s", p.text)
		} else if found {
			t.Errorf("directive found multiple times in output: %s", p.text)
		}
		foundDirective[p.name] = true
	}
	for name, found := range foundDirective {
		if !found {
			t.Errorf("missing directive: %s", name)
		}
	}

	// Check that directives that start with the name of top-level declarations
	// come before the beginning of the named declaration and after the end
	// of the previous declaration.
	fset := token.NewFileSet()
	astFile, err := parser.ParseFile(fset, testDirectives, output, 0)
	if err != nil {
		t.Fatal(err)
	}

	prevEnd := 0
	for _, decl := range astFile.Decls {
		var name string
		switch d := decl.(type) {
		case *ast.FuncDecl:
			name = d.Name.Name
		case *ast.GenDecl:
			if len(d.Specs) == 0 {
				// An empty group declaration. We still want to check that
				// directives can be associated with it, so we make up a name
				// to match directives in the test data.
				name = "_empty"
			} else if spec, ok := d.Specs[0].(*ast.TypeSpec); ok {
				name = spec.Name.Name
			}
		}
		pos := fset.Position(decl.Pos()).Offset
		end := fset.Position(decl.End()).Offset
		if name == "" {
			prevEnd = end
			continue
		}
		for _, p := range outputDirectives {
			if !strings.HasPrefix(p.name, name) {
				continue
			}
			if p.offset < prevEnd || pos < p.offset {
				t.Errorf("directive %s does not appear before definition %s", p.text, name)
			}
		}
		prevEnd = end
	}
}

type directiveInfo struct {
	text   string // full text of the comment, not including newline
	name   string // text after //go:
	offset int    // byte offset of first slash in comment
}

func findDirectives(source []byte) []directiveInfo {
	var directives []directiveInfo
	directivePrefix := []byte("\n//go:")
	offset := 0
	for {
		i := bytes.Index(source[offset:], directivePrefix)
		if i < 0 {
			break
		}
		i++ // skip newline
		p := source[offset+i:]
		j := bytes.IndexByte(p, '\n')
		if j < 0 {
			// reached EOF
			j = len(p)
		}
		directive := directiveInfo{
			text:   string(p[:j]),
			name:   string(p[len(directivePrefix)-1 : j]),
			offset: offset + i,
		}
		directives = append(directives, directive)
		offset += i + j
	}
	return directives
}

// Makes sure that `cover -func=profile.cov` reports accurate coverage.
// Issue #20515.
func TestCoverFunc(t *testing.T) {
	t.Parallel()
	buildCover(t)
	// testcover -func ./testdata/profile.cov
	cmd := exec.Command(testcover, "-func", coverProfile)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			t.Logf("%s", ee.Stderr)
		}
		t.Fatal(err)
	}

	if got, err := regexp.Match(".*total:.*100.0.*", out); err != nil || !got {
		t.Logf("%s", out)
		t.Errorf("invalid coverage counts. got=(%v, %v); want=(true; nil)", got, err)
	}
}

// Check that cover produces correct HTML.
// Issue #25767.
func TestCoverHTML(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoRun(t)
	buildCover(t)

	// go test -coverprofile testdata/html/html.cov cmd/cover/testdata/html
	cmd := exec.Command(testenv.GoToolPath(t), "test", toolexecArg, "-coverprofile", htmlProfile, "cmd/cover/testdata/html")
	run(cmd, t)
	// testcover -html testdata/html/html.cov -o testdata/html/html.html
	cmd = exec.Command(testcover, "-html", htmlProfile, "-o", htmlHTML)
	run(cmd, t)

	// Extract the parts of the HTML with comment markers,
	// and compare against a golden file.
	entireHTML, err := os.ReadFile(htmlHTML)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	scan := bufio.NewScanner(bytes.NewReader(entireHTML))
	in := false
	for scan.Scan() {
		line := scan.Text()
		if strings.Contains(line, "// START") {
			in = true
		}
		if in {
			fmt.Fprintln(&out, line)
		}
		if strings.Contains(line, "// END") {
			in = false
		}
	}
	if scan.Err() != nil {
		t.Error(scan.Err())
	}
	golden, err := os.ReadFile(htmlGolden)
	if err != nil {
		t.Fatalf("reading golden file: %v", err)
	}
	// Ignore white space differences.
	// Break into lines, then compare by breaking into words.
	goldenLines := strings.Split(string(golden), "\n")
	outLines := strings.Split(out.String(), "\n")
	// Compare at the line level, stopping at first different line so
	// we don't generate tons of output if there's an inserted or deleted line.
	for i, goldenLine := range goldenLines {
		if i >= len(outLines) {
			t.Fatalf("output shorter than golden; stops before line %d: %s\n", i+1, goldenLine)
		}
		// Convert all white space to simple spaces, for easy comparison.
		goldenLine = strings.Join(strings.Fields(goldenLine), " ")
		outLine := strings.Join(strings.Fields(outLines[i]), " ")
		if outLine != goldenLine {
			t.Fatalf("line %d differs: got:\n\t%s\nwant:\n\t%s", i+1, outLine, goldenLine)
		}
	}
	if len(goldenLines) != len(outLines) {
		t.Fatalf("output longer than golden; first extra output line %d: %q\n", len(goldenLines)+1, outLines[len(goldenLines)])
	}
}

// Test HTML processing with a source file not run through gofmt.
// Issue #27350.
func TestHtmlUnformatted(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoRun(t)
	buildCover(t)

	if err := os.Mkdir(htmlUDir, 0777); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(htmlUDir, "go.mod"), []byte("module htmlunformatted\n"), 0666); err != nil {
		t.Fatal(err)
	}

	const htmlUContents = `
package htmlunformatted

var g int

func F() {
//line x.go:1
	{ { F(); goto lab } }
lab:
}`

	const htmlUTestContents = `package htmlunformatted`

	if err := os.WriteFile(htmlU, []byte(htmlUContents), 0444); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(htmlUTest, []byte(htmlUTestContents), 0444); err != nil {
		t.Fatal(err)
	}

	// go test -covermode=count -coverprofile TMPDIR/htmlunformatted.cov
	cmd := exec.Command(testenv.GoToolPath(t), "test", toolexecArg, "-covermode=count", "-coverprofile", htmlUProfile)
	cmd.Dir = htmlUDir
	run(cmd, t)

	// testcover -html TMPDIR/htmlunformatted.cov -o unformatted.html
	cmd = exec.Command(testcover, "-html", htmlUProfile, "-o", htmlUHTML)
	cmd.Dir = htmlUDir
	run(cmd, t)
}

// lineDupContents becomes linedup.go in TestFuncWithDuplicateLines.
const lineDupContents = `
package linedup

var G int

func LineDup(c int) {
	for i := 0; i < c; i++ {
//line ld.go:100
		if i % 2 == 0 {
			G++
		}
		if i % 3 == 0 {
			G++; G++
		}
//line ld.go:100
		if i % 4 == 0 {
			G++; G++; G++
		}
		if i % 5 == 0 {
			G++; G++; G++; G++
		}
	}
}
`

// lineDupTestContents becomes linedup_test.go in TestFuncWithDuplicateLines.
const lineDupTestContents = `
package linedup

import "testing"

func TestLineDup(t *testing.T) {
	LineDup(100)
}
`

// Test -func with duplicate //line directives with different numbers
// of statements.
func TestFuncWithDuplicateLines(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoRun(t)
	buildCover(t)

	if err := os.Mkdir(lineDupDir, 0777); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(lineDupDir, "go.mod"), []byte("module linedup\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(lineDupGo, []byte(lineDupContents), 0444); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(lineDupTestGo, []byte(lineDupTestContents), 0444); err != nil {
		t.Fatal(err)
	}

	// go test -cover -covermode count -coverprofile TMPDIR/linedup.out
	cmd := exec.Command(testenv.GoToolPath(t), "test", toolexecArg, "-cover", "-covermode", "count", "-coverprofile", lineDupProfile)
	cmd.Dir = lineDupDir
	run(cmd, t)

	// testcover -func=TMPDIR/linedup.out
	cmd = exec.Command(testcover, "-func", lineDupProfile)
	cmd.Dir = lineDupDir
	run(cmd, t)
}

func run(c *exec.Cmd, t *testing.T) {
	t.Helper()
	t.Log("running", c.Args)
	out, err := c.CombinedOutput()
	if len(out) > 0 {
		t.Logf("%s", out)
	}
	if err != nil {
		t.Fatal(err)
	}
}
