// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The fiximports command fixes import declarations to use the canonical
// import path for packages that have an "import comment" as defined by
// https://golang.org/s/go14customimport.
//
//
// Background
//
// The Go 1 custom import path mechanism lets the maintainer of a
// package give it a stable name by which clients may import and "go
// get" it, independent of the underlying version control system (such
// as Git) or server (such as github.com) that hosts it.  Requests for
// the custom name are redirected to the underlying name.  This allows
// packages to be migrated from one underlying server or system to
// another without breaking existing clients.
//
// Because this redirect mechanism creates aliases for existing
// packages, it's possible for a single program to import the same
// package by its canonical name and by an alias.  The resulting
// executable will contain two copies of the package, which is wasteful
// at best and incorrect at worst.
//
// To avoid this, "go build" reports an error if it encounters a special
// comment like the one below, and if the import path in the comment
// does not match the path of the enclosing package relative to
// GOPATH/src:
//
//      $ grep ^package $GOPATH/src/github.com/bob/vanity/foo/foo.go
// 	package foo // import "vanity.com/foo"
//
// The error from "go build" indicates that the package canonically
// known as "vanity.com/foo" is locally installed under the
// non-canonical name "github.com/bob/vanity/foo".
//
//
// Usage
//
// When a package that you depend on introduces a custom import comment,
// and your workspace imports it by the non-canonical name, your build
// will stop working as soon as you update your copy of that package
// using "go get -u".
//
// The purpose of the fiximports tool is to fix up all imports of the
// non-canonical path within a Go workspace, replacing them with imports
// of the canonical path.  Following a run of fiximports, the workspace
// will no longer depend on the non-canonical copy of the package, so it
// should be safe to delete.  It may be necessary to run "go get -u"
// again to ensure that the package is locally installed under its
// canonical path, if it was not already.
//
// The fiximports tool operates locally; it does not make HTTP requests
// and does not discover new custom import comments.  It only operates
// on non-canonical packages present in your workspace.
//
// The -baddomains flag is a list of domain names that should always be
// considered non-canonical.  You can use this if you wish to make sure
// that you no longer have any dependencies on packages from that
// domain, even those that do not yet provide a canonical import path
// comment.  For example, the default value of -baddomains includes the
// moribund code hosting site code.google.com, so fiximports will report
// an error for each import of a package from this domain remaining
// after canonicalization.
//
// To see the changes fiximports would make without applying them, use
// the -n flag.
//
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// flags
var (
	dryrun     = flag.Bool("n", false, "dry run: show changes, but don't apply them")
	badDomains = flag.String("baddomains", "code.google.com",
		"a comma-separated list of domains from which packages should not be imported")
	replaceFlag = flag.String("replace", "",
		"a comma-separated list of noncanonical=canonical pairs of package paths.  If both items in a pair end with '...', they are treated as path prefixes.")
)

// seams for testing
var (
	stderr    io.Writer = os.Stderr
	writeFile           = ioutil.WriteFile
)

const usage = `fiximports: rewrite import paths to use canonical package names.

Usage: fiximports [-n] package...

The package... arguments specify a list of packages
in the style of the go tool; see "go help packages".
Hint: use "all" or "..." to match the entire workspace.

For details, see http://godoc.org/golang.org/x/tools/cmd/fiximports.

Flags:
  -n:	       dry run: show changes, but don't apply them
  -baddomains  a comma-separated list of domains from which packages
               should not be imported
`

func main() {
	flag.Parse()

	if len(flag.Args()) == 0 {
		fmt.Fprintf(stderr, usage)
		os.Exit(1)
	}
	if !fiximports(flag.Args()...) {
		os.Exit(1)
	}
}

type canonicalName struct{ path, name string }

// fiximports fixes imports in the specified packages.
// Invariant: a false result implies an error was already printed.
func fiximports(packages ...string) bool {
	// importedBy is the transpose of the package import graph.
	importedBy := make(map[string]map[*build.Package]bool)

	// addEdge adds an edge to the import graph.
	addEdge := func(from *build.Package, to string) {
		if to == "C" || to == "unsafe" {
			return // fake
		}
		pkgs := importedBy[to]
		if pkgs == nil {
			pkgs = make(map[*build.Package]bool)
			importedBy[to] = pkgs
		}
		pkgs[from] = true
	}

	// List metadata for all packages in the workspace.
	pkgs, err := list("...")
	if err != nil {
		fmt.Fprintf(stderr, "importfix: %v\n", err)
		return false
	}

	// packageName maps each package's path to its name.
	packageName := make(map[string]string)
	for _, p := range pkgs {
		packageName[p.ImportPath] = p.Package.Name
	}

	// canonical maps each non-canonical package path to
	// its canonical path and name.
	// A present nil value indicates that the canonical package
	// is unknown: hosted on a bad domain with no redirect.
	canonical := make(map[string]canonicalName)
	domains := strings.Split(*badDomains, ",")

	type replaceItem struct {
		old, new    string
		matchPrefix bool
	}
	var replace []replaceItem
	for _, pair := range strings.Split(*replaceFlag, ",") {
		if pair == "" {
			continue
		}
		words := strings.Split(pair, "=")
		if len(words) != 2 {
			fmt.Fprintf(stderr, "importfix: -replace: %q is not of the form \"canonical=noncanonical\".\n", pair)
			return false
		}
		replace = append(replace, replaceItem{
			old: strings.TrimSuffix(words[0], "..."),
			new: strings.TrimSuffix(words[1], "..."),
			matchPrefix: strings.HasSuffix(words[0], "...") &&
				strings.HasSuffix(words[1], "..."),
		})
	}

	// Find non-canonical packages and populate importedBy graph.
	for _, p := range pkgs {
		if p.Error != nil {
			msg := p.Error.Err
			if strings.Contains(msg, "code in directory") &&
				strings.Contains(msg, "expects import") {
				// don't show the very errors we're trying to fix
			} else {
				fmt.Fprintln(stderr, p.Error)
			}
		}

		for _, imp := range p.Imports {
			addEdge(&p.Package, imp)
		}
		for _, imp := range p.TestImports {
			addEdge(&p.Package, imp)
		}
		for _, imp := range p.XTestImports {
			addEdge(&p.Package, imp)
		}

		// Does package have an explicit import comment?
		if p.ImportComment != "" {
			if p.ImportComment != p.ImportPath {
				canonical[p.ImportPath] = canonicalName{
					path: p.Package.ImportComment,
					name: p.Package.Name,
				}
			}
		} else {
			// Is package matched by a -replace item?
			var newPath string
			for _, item := range replace {
				if item.matchPrefix {
					if strings.HasPrefix(p.ImportPath, item.old) {
						newPath = item.new + p.ImportPath[len(item.old):]
						break
					}
				} else if p.ImportPath == item.old {
					newPath = item.new
					break
				}
			}
			if newPath != "" {
				newName := packageName[newPath]
				if newName == "" {
					newName = filepath.Base(newPath) // a guess
				}
				canonical[p.ImportPath] = canonicalName{
					path: newPath,
					name: newName,
				}
				continue
			}

			// Is package matched by a -baddomains item?
			for _, domain := range domains {
				slash := strings.Index(p.ImportPath, "/")
				if slash < 0 {
					continue // no slash: standard package
				}
				if p.ImportPath[:slash] == domain {
					// Package comes from bad domain and has no import comment.
					// Report an error each time this package is imported.
					canonical[p.ImportPath] = canonicalName{}

					// TODO(adonovan): should we make an HTTP request to
					// see if there's an HTTP redirect, a "go-import" meta tag,
					// or an import comment in the the latest revision?
					// It would duplicate a lot of logic from "go get".
				}
				break
			}
		}
	}

	// Find all clients (direct importers) of canonical packages.
	// These are the packages that need fixing up.
	clients := make(map[*build.Package]bool)
	for path := range canonical {
		for client := range importedBy[path] {
			clients[client] = true
		}
	}

	// Restrict rewrites to the set of packages specified by the user.
	if len(packages) == 1 && (packages[0] == "all" || packages[0] == "...") {
		// no restriction
	} else {
		pkgs, err := list(packages...)
		if err != nil {
			fmt.Fprintf(stderr, "importfix: %v\n", err)
			return false
		}
		seen := make(map[string]bool)
		for _, p := range pkgs {
			seen[p.ImportPath] = true
		}
		for client := range clients {
			if !seen[client.ImportPath] {
				delete(clients, client)
			}
		}
	}

	// Rewrite selected client packages.
	ok := true
	for client := range clients {
		if !rewritePackage(client, canonical) {
			ok = false

			// There were errors.
			// Show direct and indirect imports of client.
			seen := make(map[string]bool)
			var direct, indirect []string
			for p := range importedBy[client.ImportPath] {
				direct = append(direct, p.ImportPath)
				seen[p.ImportPath] = true
			}

			var visit func(path string)
			visit = func(path string) {
				for q := range importedBy[path] {
					qpath := q.ImportPath
					if !seen[qpath] {
						seen[qpath] = true
						indirect = append(indirect, qpath)
						visit(qpath)
					}
				}
			}

			if direct != nil {
				fmt.Fprintf(stderr, "\timported directly by:\n")
				sort.Strings(direct)
				for _, path := range direct {
					fmt.Fprintf(stderr, "\t\t%s\n", path)
					visit(path)
				}

				if indirect != nil {
					fmt.Fprintf(stderr, "\timported indirectly by:\n")
					sort.Strings(indirect)
					for _, path := range indirect {
						fmt.Fprintf(stderr, "\t\t%s\n", path)
					}
				}
			}
		}
	}

	return ok
}

// Invariant: false result => error already printed.
func rewritePackage(client *build.Package, canonical map[string]canonicalName) bool {
	ok := true

	used := make(map[string]bool)
	var filenames []string
	filenames = append(filenames, client.GoFiles...)
	filenames = append(filenames, client.TestGoFiles...)
	filenames = append(filenames, client.XTestGoFiles...)
	var first bool
	for _, filename := range filenames {
		if !first {
			first = true
			fmt.Fprintf(stderr, "%s\n", client.ImportPath)
		}
		err := rewriteFile(filepath.Join(client.Dir, filename), canonical, used)
		if err != nil {
			fmt.Fprintf(stderr, "\tERROR: %v\n", err)
			ok = false
		}
	}

	// Show which imports were renamed in this package.
	var keys []string
	for key := range used {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if p := canonical[key]; p.path != "" {
			fmt.Fprintf(stderr, "\tfixed: %s -> %s\n", key, p.path)
		} else {
			fmt.Fprintf(stderr, "\tERROR: %s has no import comment\n", key)
			ok = false
		}
	}

	return ok
}

// rewrite reads, modifies, and writes filename, replacing all imports
// of packages P in canonical by canonical[P].
// It records in used which canonical packages were imported.
// used[P]=="" indicates that P was imported but its canonical path is unknown.
func rewriteFile(filename string, canonical map[string]canonicalName, used map[string]bool) error {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
	if err != nil {
		return err
	}
	var changed bool
	for _, imp := range f.Imports {
		impPath, err := strconv.Unquote(imp.Path.Value)
		if err != nil {
			log.Printf("%s: bad import spec %q: %v",
				fset.Position(imp.Pos()), imp.Path.Value, err)
			continue
		}
		canon, ok := canonical[impPath]
		if !ok {
			continue // import path is canonical
		}

		used[impPath] = true

		if canon.path == "" {
			// The canonical path is unknown (a -baddomain).
			// Show the offending import.
			// TODO(adonovan): should we show the actual source text?
			fmt.Fprintf(stderr, "\t%s:%d: import %q\n",
				shortPath(filename),
				fset.Position(imp.Pos()).Line, impPath)
			continue
		}

		changed = true

		imp.Path.Value = strconv.Quote(canon.path)

		// Add a renaming import if necessary.
		//
		// This is a guess at best.  We can't see whether a 'go
		// get' of the canonical import path would have the same
		// name or not.  Assume it's the last segment.
		newBase := path.Base(canon.path)
		if imp.Name == nil && newBase != canon.name {
			imp.Name = &ast.Ident{Name: canon.name}
		}
	}

	if changed && !*dryrun {
		var buf bytes.Buffer
		if err := format.Node(&buf, fset, f); err != nil {
			return fmt.Errorf("%s: couldn't format file: %v", filename, err)
		}
		return writeFile(filename, buf.Bytes(), 0644)
	}

	return nil
}

// listPackage is a copy of cmd/go/list.Package.
// It has more fields than build.Package and we need some of them.
type listPackage struct {
	build.Package
	Error *packageError // error loading package
}

// A packageError describes an error loading information about a package.
type packageError struct {
	ImportStack []string // shortest path from package named on command line to this one
	Pos         string   // position of error
	Err         string   // the error itself
}

func (e packageError) Error() string {
	if e.Pos != "" {
		return e.Pos + ": " + e.Err
	}
	return e.Err
}

// list runs 'go list' with the specified arguments and returns the
// metadata for matching packages.
func list(args ...string) ([]*listPackage, error) {
	cmd := exec.Command("go", append([]string{"list", "-e", "-json"}, args...)...)
	cmd.Stdout = new(bytes.Buffer)
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		return nil, err
	}

	dec := json.NewDecoder(cmd.Stdout.(io.Reader))
	var pkgs []*listPackage
	for {
		var p listPackage
		if err := dec.Decode(&p); err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		pkgs = append(pkgs, &p)
	}
	return pkgs, nil
}

// cwd contains the current working directory of the tool.
//
// It is initialized directly so that its value will be set for any other
// package variables or init functions that depend on it, such as the gopath
// variable in main_test.go.
var cwd string = func() string {
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatalf("os.Getwd: %v", err)
	}
	return cwd
}()

// shortPath returns an absolute or relative name for path, whatever is shorter.
// Plundered from $GOROOT/src/cmd/go/build.go.
func shortPath(path string) string {
	if rel, err := filepath.Rel(cwd, path); err == nil && len(rel) < len(path) {
		return rel
	}
	return path
}
