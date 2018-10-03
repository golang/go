// Package analysistest provides utilities for testing analyzers.
package analysistest

import (
	"fmt"
	"go/token"
	"go/types"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/scanner"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/checker"
	"golang.org/x/tools/go/packages"
)

// WriteFiles is a helper function that creates a temporary directory
// and populates it with a GOPATH-style project using filemap (which
// maps file names to contents). On success it returns the name of the
// directory and a cleanup function to delete it.
func WriteFiles(filemap map[string]string) (dir string, cleanup func(), err error) {
	gopath, err := ioutil.TempDir("", "analysistest")
	if err != nil {
		return "", nil, err
	}
	cleanup = func() { os.RemoveAll(gopath) }

	for name, content := range filemap {
		filename := filepath.Join(gopath, "src", name)
		os.MkdirAll(filepath.Dir(filename), 0777) // ignore error
		if err := ioutil.WriteFile(filename, []byte(content), 0666); err != nil {
			cleanup()
			return "", nil, err
		}
	}
	return gopath, cleanup, nil
}

// TestData returns the effective filename of
// the program's "testdata" directory.
// This function may be overridden by projects using
// an alternative build system (such as Blaze) that
// does not run a test in its package directory.
var TestData = func() string {
	testdata, err := filepath.Abs("testdata")
	if err != nil {
		log.Fatal(err)
	}
	return testdata
}

// Testing is an abstraction of a *testing.T.
type Testing interface {
	Errorf(format string, args ...interface{})
}

// Run applies an analysis to each named package.
// It loads each package from the specified GOPATH-style project
// directory using golang.org/x/tools/go/packages, runs the analysis on
// it, and checks that each the analysis emits the expected diagnostics
// and facts specified by the contents of '// want ...' comments in the
// package's source files.
//
// An expectation of a Diagnostic is specified by a string literal
// containing a regular expression that must match the diagnostic
// message. For example:
//
//	fmt.Printf("%s", 1) // want `cannot provide int 1 to %s`
//
// An expectation of a Fact associated with an object is specified by
// 'name:"pattern"', where name is the name of the object, which must be
// declared on the same line as the comment, and pattern is a regular
// expression that must match the string representation of the fact,
// fmt.Sprint(fact). For example:
//
//	func panicf(format string, args interface{}) { // want panicf:"printfWrapper"
//
// Package facts are specified by the name "package".
//
// A single 'want' comment may contain a mixture of diagnostic and fact
// expectations, including multiple facts about the same object:
//
//	// want "diag" "diag2" x:"fact1" x:"fact2" y:"fact3"
//
// Unexpected diagnostics and facts, and unmatched expectations, are
// reported as errors to the Testing.
//
// You may wish to call this function from within a (*testing.T).Run
// subtest to ensure that errors have adequate contextual description.
func Run(t Testing, dir string, a *analysis.Analyzer, pkgnames ...string) {
	if pkgnames == nil {
		t.Errorf("Run: no packages")
	}
	for _, pkgname := range pkgnames {
		pkg, err := loadPackage(dir, pkgname)
		if err != nil {
			t.Errorf("loading %s: %v", pkgname, err)
			continue
		}

		pass, diagnostics, facts, err := checker.Analyze(pkg, a)
		if err != nil {
			t.Errorf("analyzing %s: %v", pkgname, err)
			continue
		}

		check(t, dir, pass, diagnostics, facts)
	}
}

// loadPackage loads the specified package (from source, with
// dependencies) from dir, which is the root of a GOPATH-style project tree.
func loadPackage(dir, pkgpath string) (*packages.Package, error) {
	// packages.Load loads the real standard library, not a minimal
	// fake version, which would be more efficient, especially if we
	// have many small tests that import, say, net/http.
	// However there is no easy way to make go/packages to consume
	// a list of packages we generate and then do the parsing and
	// typechecking, though this feature seems to be a recurring need.

	cfg := &packages.Config{
		Mode:  packages.LoadAllSyntax,
		Dir:   dir,
		Tests: true,
		Env:   append(os.Environ(), "GOPATH="+dir, "GO111MODULE=off", "GOPROXY=off"),
	}
	pkgs, err := packages.Load(cfg, pkgpath)
	if err != nil {
		return nil, err
	}
	if packages.PrintErrors(pkgs) > 0 {
		return nil, fmt.Errorf("loading %s failed", pkgpath)
	}
	if len(pkgs) != 1 {
		return nil, fmt.Errorf("pattern %q expanded to %d packages, want 1",
			pkgpath, len(pkgs))
	}

	return pkgs[0], nil
}

// check inspects an analysis pass on which the analysis has already
// been run, and verifies that all reported diagnostics and facts match
// specified by the contents of "// want ..." comments in the package's
// source files, which must have been parsed with comments enabled.
func check(t Testing, gopath string, pass *analysis.Pass, diagnostics []analysis.Diagnostic, facts map[types.Object][]analysis.Fact) {

	// Read expectations out of comments.
	type key struct {
		file string
		line int
	}
	want := make(map[key][]expectation)
	for _, f := range pass.Files {
		for _, c := range f.Comments {
			posn := pass.Fset.Position(c.Pos())
			sanitize(gopath, &posn)
			text := strings.TrimSpace(c.Text())

			// Any comment starting with "want" is treated
			// as an expectation, even without following whitespace.
			if rest := strings.TrimPrefix(text, "want"); rest != text {
				expects, err := parseExpectations(rest)
				if err != nil {
					t.Errorf("%s: in 'want' comment: %s", posn, err)
					continue
				}
				if false {
					log.Printf("%s: %v", posn, expects)
				}
				want[key{posn.Filename, posn.Line}] = expects
			}
		}
	}

	checkMessage := func(posn token.Position, kind, name, message string) {
		sanitize(gopath, &posn)
		k := key{posn.Filename, posn.Line}
		expects := want[k]
		var unmatched []string
		for i, exp := range expects {
			if exp.kind == kind && exp.name == name {
				if exp.rx.MatchString(message) {
					// matched: remove the expectation.
					expects[i] = expects[len(expects)-1]
					expects = expects[:len(expects)-1]
					want[k] = expects
					return
				}
				unmatched = append(unmatched, fmt.Sprintf("%q", exp.rx))
			}
		}
		if unmatched == nil {
			t.Errorf("%v: unexpected %s: %v", posn, kind, message)
		} else {
			t.Errorf("%v: %s %q does not match pattern %s",
				posn, kind, message, strings.Join(unmatched, " or "))
		}
	}

	// Check the diagnostics match expectations.
	for _, f := range diagnostics {
		posn := pass.Fset.Position(f.Pos)
		checkMessage(posn, "diagnostic", "", f.Message)
	}

	// Check the facts match expectations.
	// Report errors in lexical order for determinism.
	var objects []types.Object
	for obj := range facts {
		objects = append(objects, obj)
	}
	sort.Slice(objects, func(i, j int) bool {
		return objects[i].Pos() < objects[j].Pos()
	})
	for _, obj := range objects {
		var posn token.Position
		var name string
		if obj != nil {
			// Object facts are reported on the declaring line.
			name = obj.Name()
			posn = pass.Fset.Position(obj.Pos())
		} else {
			// Package facts are reported at the start of the file.
			name = "package"
			posn = pass.Fset.Position(pass.Files[0].Pos())
			posn.Line = 1
		}

		for _, fact := range facts[obj] {
			checkMessage(posn, "fact", name, fmt.Sprint(fact))
		}
	}

	// Reject surplus expectations.
	var surplus []string
	for key, expects := range want {
		for _, exp := range expects {
			err := fmt.Sprintf("%s:%d: no %s was reported matching %q", key.file, key.line, exp.kind, exp.rx)
			surplus = append(surplus, err)
		}
	}
	sort.Strings(surplus)
	for _, err := range surplus {
		t.Errorf("%s", err)
	}
}

type expectation struct {
	kind string // either "fact" or "diagnostic"
	name string // name of object to which fact belongs, or "package" ("fact" only)
	rx   *regexp.Regexp
}

func (ex expectation) String() string {
	return fmt.Sprintf("%s %s:%q", ex.kind, ex.name, ex.rx) // for debugging
}

// parseExpectations parses the content of a "// want ..." comment
// and returns the expections, a mixture of diagnostics ("rx") and
// facts (name:"rx").
func parseExpectations(text string) ([]expectation, error) {
	var scanErr string
	sc := new(scanner.Scanner).Init(strings.NewReader(text))
	sc.Error = func(s *scanner.Scanner, msg string) {
		scanErr = msg // e.g. bad string escape
	}
	sc.Mode = scanner.ScanIdents | scanner.ScanStrings | scanner.ScanRawStrings

	scanRegexp := func(tok rune) (*regexp.Regexp, error) {
		if tok != scanner.String && tok != scanner.RawString {
			return nil, fmt.Errorf("got %s, want regular expression",
				scanner.TokenString(tok))
		}
		pattern, _ := strconv.Unquote(sc.TokenText()) // can't fail
		return regexp.Compile(pattern)
	}

	var expects []expectation
	for {
		tok := sc.Scan()
		switch tok {
		case scanner.String, scanner.RawString:
			rx, err := scanRegexp(tok)
			if err != nil {
				return nil, err
			}
			expects = append(expects, expectation{"diagnostic", "", rx})

		case scanner.Ident:
			name := sc.TokenText()
			tok = sc.Scan()
			if tok != ':' {
				return nil, fmt.Errorf("got %s after %s, want ':'",
					scanner.TokenString(tok), name)
			}
			tok = sc.Scan()
			rx, err := scanRegexp(tok)
			if err != nil {
				return nil, err
			}
			expects = append(expects, expectation{"fact", name, rx})

		case scanner.EOF:
			if scanErr != "" {
				return nil, fmt.Errorf("%s", scanErr)
			}
			return expects, nil

		default:
			return nil, fmt.Errorf("unexpected %s", scanner.TokenString(tok))
		}
	}
}

// sanitize removes the GOPATH portion of the filename,
// typically a gnarly /tmp directory.
func sanitize(gopath string, posn *token.Position) {
	prefix := gopath + string(os.PathSeparator) + "src" + string(os.PathSeparator)
	posn.Filename = filepath.ToSlash(strings.TrimPrefix(posn.Filename, prefix))
}
