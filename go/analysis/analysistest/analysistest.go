// Package analysistest provides utilities for testing analyzers.
package analysistest

import (
	"fmt"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

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
// it, and checks that each the analysis generates the diagnostics
// specified by 'want "..."' comments in the package's source files.
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

		pass, diagnostics, err := checker.Analyze(pkg, a)
		if err != nil {
			t.Errorf("analyzing %s: %v", pkgname, err)
			continue
		}

		checkDiagnostics(t, dir, pass, diagnostics)
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
	//
	// It is possible to write a custom driver, but it's fairly
	// involved and requires setting a global (environment) variable.
	//
	// Also, using the "go list" driver will probably not work in google3.
	//
	// TODO(adonovan): extend go/packages to allow bypassing the driver.

	cfg := &packages.Config{
		Mode:  packages.LoadAllSyntax,
		Dir:   dir,
		Tests: true,
		Env:   append(os.Environ(), "GOPATH="+dir),
	}
	pkgs, err := packages.Load(cfg, pkgpath)
	if err != nil {
		return nil, err
	}
	if len(pkgs) != 1 {
		return nil, fmt.Errorf("pattern %q expanded to %d packages, want 1",
			pkgpath, len(pkgs))
	}

	return pkgs[0], nil
}

// checkDiagnostics inspects an analysis pass on which the analysis has
// already been run, and verifies that all reported diagnostics match those
// specified by 'want "..."' comments in the package's source files,
// which must have been parsed with comments enabled. Surplus diagnostics
// and unmatched expectations are reported as errors to the Testing.
func checkDiagnostics(t Testing, gopath string, pass *analysis.Pass, diagnostics []analysis.Diagnostic) {
	// Read expectations out of comments.
	type key struct {
		file string
		line int
	}
	wantErrs := make(map[key]*regexp.Regexp)
	for _, f := range pass.Files {
		for _, c := range f.Comments {
			posn := pass.Fset.Position(c.Pos())
			sanitize(gopath, &posn)
			text := strings.TrimSpace(c.Text())
			if !strings.HasPrefix(text, "want") {
				continue
			}
			text = strings.TrimSpace(text[len("want"):])
			pattern, err := strconv.Unquote(text)
			if err != nil {
				t.Errorf("%s: in 'want' comment: %v", posn, err)
				continue
			}
			rx, err := regexp.Compile(pattern)
			if err != nil {
				t.Errorf("%s: %v", posn, err)
				continue
			}
			wantErrs[key{posn.Filename, posn.Line}] = rx
		}
	}

	// Check the diagnostics match expectations.
	for _, f := range diagnostics {
		posn := pass.Fset.Position(f.Pos)
		sanitize(gopath, &posn)
		rx, ok := wantErrs[key{posn.Filename, posn.Line}]
		if !ok {
			t.Errorf("%v: unexpected diagnostic: %v", posn, f.Message)
			continue
		}
		delete(wantErrs, key{posn.Filename, posn.Line})
		if !rx.MatchString(f.Message) {
			t.Errorf("%v: diagnostic %q does not match pattern %q", posn, f.Message, rx)
		}
	}
	for key, rx := range wantErrs {
		t.Errorf("%s:%d: expected diagnostic matching %q", key.file, key.line, rx)
	}
}

// sanitize removes the GOPATH portion of the filename,
// typically a gnarly /tmp directory.
func sanitize(gopath string, posn *token.Position) {
	prefix := gopath + string(os.PathSeparator) + "src" + string(os.PathSeparator)
	posn.Filename = filepath.ToSlash(strings.TrimPrefix(posn.Filename, prefix))
}
