package packages

// This file defines the "go list" implementation of the Packages metadata query.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// golistPackages uses the "go list" command to expand the
// pattern words and return metadata for the specified packages.
func golistPackages(ctx context.Context, gopath string, cgo, export bool, words []string) ([]*Package, error) {
	// Fields must match go list;
	// see $GOROOT/src/cmd/go/internal/load/pkg.go.
	type jsonPackage struct {
		ImportPath   string
		Dir          string
		Name         string
		Export       string
		GoFiles      []string
		CFiles       []string
		CgoFiles     []string
		SFiles       []string
		Imports      []string
		ImportMap    map[string]string
		Deps         []string
		TestGoFiles  []string
		TestImports  []string
		XTestGoFiles []string
		XTestImports []string
		ForTest      string // q in a "p [q.test]" package, else ""
		DepOnly      bool
	}

	// go list uses the following identifiers in ImportPath and Imports:
	//
	// 	"p"			-- importable package or main (command)
	//      "q.test"		-- q's test executable
	// 	"p [q.test]"		-- variant of p as built for q's test executable
	//	"q_test [q.test]"	-- q's external test package
	//
	// The packages p that are built differently for a test q.test
	// are q itself, plus any helpers used by the external test q_test,
	// typically including "testing" and all its dependencies.

	// Run "go list" for complete
	// information on the specified packages.

	buf, err := golist(ctx, gopath, cgo, export, words)
	if err != nil {
		return nil, err
	}
	// Decode the JSON and convert it to Package form.
	var result []*Package
	for dec := json.NewDecoder(buf); dec.More(); {
		p := new(jsonPackage)
		if err := dec.Decode(p); err != nil {
			return nil, fmt.Errorf("JSON decoding failed: %v", err)
		}

		// Bad package?
		if p.Name == "" {
			// This could be due to:
			// - no such package
			// - package directory contains no Go source files
			// - all package declarations are mangled
			// - and possibly other things.
			//
			// For now, we throw it away and let later
			// stages rediscover the problem, but this
			// discards the error message computed by go list
			// and computes a new one---by different logic:
			// if only one of the package declarations is
			// bad, for example, should we report an error
			// in Metadata mode?
			// Unless we parse and typecheck, we might not
			// notice there's a problem.
			//
			// Perhaps we should save a map of PackageID to
			// errors for such cases.
			continue
		}

		id := p.ImportPath

		// Extract the PkgPath from the package's ID.
		pkgpath := id
		if i := strings.IndexByte(id, ' '); i >= 0 {
			pkgpath = id[:i]
		}

		// Is this a test?
		// ("foo [foo.test]" package or "foo.test" command)
		isTest := p.ForTest != "" || strings.HasSuffix(pkgpath, ".test")

		if pkgpath == "unsafe" {
			p.GoFiles = nil // ignore fake unsafe.go file
		}

		export := p.Export
		if export != "" && !filepath.IsAbs(export) {
			export = filepath.Join(p.Dir, export)
		}

		// imports
		//
		// Imports contains the IDs of all imported packages.
		// ImportsMap records (path, ID) only where they differ.
		ids := make(map[string]bool)
		for _, id := range p.Imports {
			ids[id] = true
		}
		imports := make(map[string]string)
		for path, id := range p.ImportMap {
			imports[path] = id // non-identity import
			delete(ids, id)
		}
		for id := range ids {
			// Go issue 26136: go list omits imports in cgo-generated files.
			if id == "C" && cgo {
				imports["unsafe"] = "unsafe"
				imports["syscall"] = "syscall"
				if pkgpath != "runtime/cgo" {
					imports["runtime/cgo"] = "runtime/cgo"
				}
				continue
			}

			imports[id] = id // identity import
		}

		pkg := &Package{
			ID:        id,
			Name:      p.Name,
			PkgPath:   pkgpath,
			IsTest:    isTest,
			Srcs:      absJoin(p.Dir, p.GoFiles, p.CgoFiles),
			OtherSrcs: absJoin(p.Dir, p.SFiles, p.CFiles),
			imports:   imports,
			export:    export,
			indirect:  p.DepOnly,
		}
		result = append(result, pkg)
	}

	return result, nil
}

// absJoin absolutizes and flattens the lists of files.
func absJoin(dir string, fileses ...[]string) (res []string) {
	for _, files := range fileses {
		for _, file := range files {
			if !filepath.IsAbs(file) {
				file = filepath.Join(dir, file)
			}
			res = append(res, file)
		}
	}
	return res
}

// golist returns the JSON-encoded result of a "go list args..." query.
func golist(ctx context.Context, gopath string, cgo, export bool, args []string) (*bytes.Buffer, error) {
	out := new(bytes.Buffer)
	if len(args) == 0 {
		return out, nil
	}

	const test = true // TODO(adonovan): expose a flag for this.

	cmd := exec.CommandContext(ctx, "go", append([]string{
		"list",
		"-e",
		fmt.Sprintf("-cgo=%t", cgo),
		fmt.Sprintf("-test=%t", test),
		fmt.Sprintf("-export=%t", export),
		"-deps",
		"-json",
		"--",
	}, args...)...)
	cmd.Env = append(append([]string(nil), os.Environ()...), "GOPATH="+gopath)
	if !cgo {
		cmd.Env = append(cmd.Env, "CGO_ENABLED=0")
	}
	cmd.Stdout = out
	cmd.Stderr = new(bytes.Buffer)
	if err := cmd.Run(); err != nil {
		exitErr, ok := err.(*exec.ExitError)
		if !ok {
			// Catastrophic error:
			// - executable not found
			// - context cancellation
			return nil, fmt.Errorf("couldn't exec 'go list': %s %T", err, err)
		}

		// Old go list?
		if strings.Contains(fmt.Sprint(cmd.Stderr), "flag provided but not defined") {
			return nil, fmt.Errorf("unsupported version of go list: %s: %s", exitErr, cmd.Stderr)
		}

		// Export mode entails a build.
		// If that build fails, errors appear on stderr
		// (despite the -e flag) and the Export field is blank.
		// Do not fail in that case.
		if !export {
			return nil, fmt.Errorf("go list: %s: %s", exitErr, cmd.Stderr)
		}
	}

	// Print standard error output from "go list".
	// Due to the -e flag, this should be empty.
	// However, in -export mode it contains build errors.
	// Should go list save build errors in the Package.Error JSON field?
	// See https://github.com/golang/go/issues/26319.
	// If so, then we should continue to print stderr as go list
	// will be silent unless something unexpected happened.
	// If not, perhaps we should suppress it to reduce noise.
	if stderr := fmt.Sprint(cmd.Stderr); stderr != "" {
		fmt.Fprintf(os.Stderr, "go list stderr <<%s>>\n", stderr)
	}

	// debugging
	if false {
		fmt.Fprintln(os.Stderr, out)
	}

	return out, nil
}
