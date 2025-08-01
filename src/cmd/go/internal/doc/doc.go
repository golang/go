// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package doc implements the “go doc” command.
package doc

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"go/build"
	"go/token"
	"io"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/internal/telemetry/counter"
)

var CmdDoc = &base.Command{
	Run:         runDoc,
	UsageLine:   "go doc [doc flags] [package|[package.]symbol[.methodOrField]]",
	CustomFlags: true,
	Short:       "show documentation for package or symbol",
	Long: `
Doc prints the documentation comments associated with the item identified by its
arguments (a package, const, func, type, var, method, or struct field)
followed by a one-line summary of each of the first-level items "under"
that item (package-level declarations for a package, methods for a type,
etc.).

Doc accepts zero, one, or two arguments.

Given no arguments, that is, when run as

	go doc

it prints the package documentation for the package in the current directory.
If the package is a command (package main), the exported symbols of the package
are elided from the presentation unless the -cmd flag is provided.

When run with one argument, the argument is treated as a Go-syntax-like
representation of the item to be documented. What the argument selects depends
on what is installed in GOROOT and GOPATH, as well as the form of the argument,
which is schematically one of these:

	go doc <pkg>
	go doc <sym>[.<methodOrField>]
	go doc [<pkg>.]<sym>[.<methodOrField>]
	go doc [<pkg>.][<sym>.]<methodOrField>

The first item in this list matched by the argument is the one whose documentation
is printed. (See the examples below.) However, if the argument starts with a capital
letter it is assumed to identify a symbol or method in the current directory.

For packages, the order of scanning is determined lexically in breadth-first order.
That is, the package presented is the one that matches the search and is nearest
the root and lexically first at its level of the hierarchy. The GOROOT tree is
always scanned in its entirety before GOPATH.

If there is no package specified or matched, the package in the current
directory is selected, so "go doc Foo" shows the documentation for symbol Foo in
the current package.

The package path must be either a qualified path or a proper suffix of a
path. The go tool's usual package mechanism does not apply: package path
elements like . and ... are not implemented by go doc.

When run with two arguments, the first is a package path (full path or suffix),
and the second is a symbol, or symbol with method or struct field:

	go doc <pkg> <sym>[.<methodOrField>]

In all forms, when matching symbols, lower-case letters in the argument match
either case but upper-case letters match exactly. This means that there may be
multiple matches of a lower-case argument in a package if different symbols have
different cases. If this occurs, documentation for all matches is printed.

Examples:
	go doc
		Show documentation for current package.
	go doc -http
		Serve HTML documentation over HTTP for the current package.
	go doc Foo
		Show documentation for Foo in the current package.
		(Foo starts with a capital letter so it cannot match
		a package path.)
	go doc encoding/json
		Show documentation for the encoding/json package.
	go doc json
		Shorthand for encoding/json.
	go doc json.Number (or go doc json.number)
		Show documentation and method summary for json.Number.
	go doc json.Number.Int64 (or go doc json.number.int64)
		Show documentation for json.Number's Int64 method.
	go doc cmd/doc
		Show package docs for the doc command.
	go doc -cmd cmd/doc
		Show package docs and exported symbols within the doc command.
	go doc template.new
		Show documentation for html/template's New function.
		(html/template is lexically before text/template)
	go doc text/template.new # One argument
		Show documentation for text/template's New function.
	go doc text/template new # Two arguments
		Show documentation for text/template's New function.

	At least in the current tree, these invocations all print the
	documentation for json.Decoder's Decode method:

	go doc json.Decoder.Decode
	go doc json.decoder.decode
	go doc json.decode
	cd go/src/encoding/json; go doc decode

Flags:
	-all
		Show all the documentation for the package.
	-c
		Respect case when matching symbols.
	-cmd
		Treat a command (package main) like a regular package.
		Otherwise package main's exported symbols are hidden
		when showing the package's top-level documentation.
  	-http
		Serve HTML docs over HTTP.
	-short
		One-line representation for each symbol.
	-src
		Show the full source code for the symbol. This will
		display the full Go source of its declaration and
		definition, such as a function definition (including
		the body), type declaration or enclosing const
		block. The output may therefore include unexported
		details.
	-u
		Show documentation for unexported as well as exported
		symbols, methods, and fields.
`,
}

func runDoc(ctx context.Context, cmd *base.Command, args []string) {
	log.SetFlags(0)
	log.SetPrefix("doc: ")
	dirsInit()
	var flagSet flag.FlagSet
	err := do(os.Stdout, &flagSet, args)
	if err != nil {
		log.Fatal(err)
	}
}

var (
	unexported bool   // -u flag
	matchCase  bool   // -c flag
	chdir      string // -C flag
	showAll    bool   // -all flag
	showCmd    bool   // -cmd flag
	showSrc    bool   // -src flag
	short      bool   // -short flag
	serveHTTP  bool   // -http flag
)

// usage is a replacement usage function for the flags package.
func usage(flagSet *flag.FlagSet) {
	fmt.Fprintf(os.Stderr, "Usage of [go] doc:\n")
	fmt.Fprintf(os.Stderr, "\tgo doc\n")
	fmt.Fprintf(os.Stderr, "\tgo doc <pkg>\n")
	fmt.Fprintf(os.Stderr, "\tgo doc <sym>[.<methodOrField>]\n")
	fmt.Fprintf(os.Stderr, "\tgo doc [<pkg>.]<sym>[.<methodOrField>]\n")
	fmt.Fprintf(os.Stderr, "\tgo doc [<pkg>.][<sym>.]<methodOrField>\n")
	fmt.Fprintf(os.Stderr, "\tgo doc <pkg> <sym>[.<methodOrField>]\n")
	fmt.Fprintf(os.Stderr, "For more information run\n")
	fmt.Fprintf(os.Stderr, "\tgo help doc\n\n")
	fmt.Fprintf(os.Stderr, "Flags:\n")
	flagSet.PrintDefaults()
	os.Exit(2)
}

// do is the workhorse, broken out of runDoc to make testing easier.
func do(writer io.Writer, flagSet *flag.FlagSet, args []string) (err error) {
	flagSet.Usage = func() { usage(flagSet) }
	unexported = false
	matchCase = false
	flagSet.StringVar(&chdir, "C", "", "change to `dir` before running command")
	flagSet.BoolVar(&unexported, "u", false, "show unexported symbols as well as exported")
	flagSet.BoolVar(&matchCase, "c", false, "symbol matching honors case (paths not affected)")
	flagSet.BoolVar(&showAll, "all", false, "show all documentation for package")
	flagSet.BoolVar(&showCmd, "cmd", false, "show symbols with package docs even if package is a command")
	flagSet.BoolVar(&showSrc, "src", false, "show source code for symbol")
	flagSet.BoolVar(&short, "short", false, "one-line representation for each symbol")
	flagSet.BoolVar(&serveHTTP, "http", false, "serve HTML docs over HTTP")
	flagSet.Parse(args)
	counter.CountFlags("doc/flag:", *flag.CommandLine)
	if chdir != "" {
		if err := os.Chdir(chdir); err != nil {
			return err
		}
	}
	if serveHTTP {
		// Special case: if there are no arguments, try to go to an appropriate page
		// depending on whether we're in a module or workspace. The pkgsite homepage
		// is often not the most useful page.
		if len(flagSet.Args()) == 0 {
			mod, err := runCmd(append(os.Environ(), "GOWORK=off"), "go", "list", "-m")
			if err == nil && mod != "" && mod != "command-line-arguments" {
				// If there's a module, go to the module's doc page.
				return doPkgsite(mod)
			}
			gowork, err := runCmd(nil, "go", "env", "GOWORK")
			if err == nil && gowork != "" {
				// Outside a module, but in a workspace, go to the home page
				// with links to each of the modules' pages.
				return doPkgsite("")
			}
			// Outside a module or workspace, go to the documentation for the standard library.
			return doPkgsite("std")
		}

		// If args are provided, we need to figure out which page to open on the pkgsite
		// instance. Run the logic below to determine a match for a symbol, method,
		// or field, but don't actually print the documentation to the output.
		writer = io.Discard
	}
	var paths []string
	var symbol, method string
	// Loop until something is printed.
	dirs.Reset()
	for i := 0; ; i++ {
		buildPackage, userPath, sym, more := parseArgs(flagSet, flagSet.Args())
		if i > 0 && !more { // Ignore the "more" bit on the first iteration.
			return failMessage(paths, symbol, method)
		}
		if buildPackage == nil {
			return fmt.Errorf("no such package: %s", userPath)
		}

		// The builtin package needs special treatment: its symbols are lower
		// case but we want to see them, always.
		if buildPackage.ImportPath == "builtin" {
			unexported = true
		}

		symbol, method = parseSymbol(flagSet, sym)
		pkg := parsePackage(writer, buildPackage, userPath)
		paths = append(paths, pkg.prettyPath())

		defer func() {
			pkg.flush()
			e := recover()
			if e == nil {
				return
			}
			pkgError, ok := e.(PackageError)
			if ok {
				err = pkgError
				return
			}
			panic(e)
		}()

		var found bool
		switch {
		case symbol == "":
			pkg.packageDoc() // The package exists, so we got some output.
			found = true
		case method == "":
			if pkg.symbolDoc(symbol) {
				found = true
			}
		case pkg.printMethodDoc(symbol, method):
			found = true
		case pkg.printFieldDoc(symbol, method):
			found = true
		}
		if found {
			if serveHTTP {
				path, err := objectPath(userPath, pkg, symbol, method)
				if err != nil {
					return err
				}
				return doPkgsite(path)
			}
			return nil
		}
	}
}

func runCmd(env []string, cmdline ...string) (string, error) {
	var stdout, stderr strings.Builder
	cmd := exec.Command(cmdline[0], cmdline[1:]...)
	cmd.Env = env
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("go doc: %s: %v\n%s\n", strings.Join(cmdline, " "), err, stderr.String())
	}
	return strings.TrimSpace(stdout.String()), nil
}

func objectPath(userPath string, pkg *Package, symbol, method string) (string, error) {
	var err error
	path := pkg.build.ImportPath
	if path == "." {
		// go/build couldn't determine the import path, probably
		// because this was a relative path into a module. Use
		// go list to get the import path.
		path, err = runCmd(nil, "go", "list", userPath)
		if err != nil {
			return "", err
		}
	}

	object := symbol
	if symbol != "" && method != "" {
		object = symbol + "." + method
	}
	if object != "" {
		path = path + "#" + object
	}
	return path, nil
}

// failMessage creates a nicely formatted error message when there is no result to show.
func failMessage(paths []string, symbol, method string) error {
	var b bytes.Buffer
	if len(paths) > 1 {
		b.WriteString("s")
	}
	b.WriteString(" ")
	for i, path := range paths {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(path)
	}
	if method == "" {
		return fmt.Errorf("no symbol %s in package%s", symbol, &b)
	}
	return fmt.Errorf("no method or field %s.%s in package%s", symbol, method, &b)
}

// parseArgs analyzes the arguments (if any) and returns the package
// it represents, the part of the argument the user used to identify
// the path (or "" if it's the current package) and the symbol
// (possibly with a .method) within that package.
// parseSymbol is used to analyze the symbol itself.
// The boolean final argument reports whether it is possible that
// there may be more directories worth looking at. It will only
// be true if the package path is a partial match for some directory
// and there may be more matches. For example, if the argument
// is rand.Float64, we must scan both crypto/rand and math/rand
// to find the symbol, and the first call will return crypto/rand, true.
func parseArgs(flagSet *flag.FlagSet, args []string) (pkg *build.Package, path, symbol string, more bool) {
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	if len(args) == 0 {
		// Easy: current directory.
		return importDir(wd), "", "", false
	}
	arg := args[0]
	// We have an argument. If it is a directory name beginning with . or ..,
	// use the absolute path name. This discriminates "./errors" from "errors"
	// if the current directory contains a non-standard errors package.
	if isDotSlash(arg) {
		arg = filepath.Join(wd, arg)
	}
	switch len(args) {
	default:
		usage(flagSet)
	case 1:
		// Done below.
	case 2:
		// Package must be findable and importable.
		pkg, err := build.Import(args[0], wd, build.ImportComment)
		if err == nil {
			return pkg, args[0], args[1], false
		}
		for {
			packagePath, ok := findNextPackage(arg)
			if !ok {
				break
			}
			if pkg, err := build.ImportDir(packagePath, build.ImportComment); err == nil {
				return pkg, arg, args[1], true
			}
		}
		return nil, args[0], args[1], false
	}
	// Usual case: one argument.
	// If it contains slashes, it begins with either a package path
	// or an absolute directory.
	// First, is it a complete package path as it is? If so, we are done.
	// This avoids confusion over package paths that have other
	// package paths as their prefix.
	var importErr error
	if filepath.IsAbs(arg) {
		pkg, importErr = build.ImportDir(arg, build.ImportComment)
		if importErr == nil {
			return pkg, arg, "", false
		}
	} else {
		pkg, importErr = build.Import(arg, wd, build.ImportComment)
		if importErr == nil {
			return pkg, arg, "", false
		}
	}
	// Another disambiguator: If the argument starts with an upper
	// case letter, it can only be a symbol in the current directory.
	// Kills the problem caused by case-insensitive file systems
	// matching an upper case name as a package name.
	if !strings.ContainsAny(arg, `/\`) && token.IsExported(arg) {
		pkg, err := build.ImportDir(".", build.ImportComment)
		if err == nil {
			return pkg, "", arg, false
		}
	}
	// If it has a slash, it must be a package path but there is a symbol.
	// It's the last package path we care about.
	slash := strings.LastIndex(arg, "/")
	// There may be periods in the package path before or after the slash
	// and between a symbol and method.
	// Split the string at various periods to see what we find.
	// In general there may be ambiguities but this should almost always
	// work.
	var period int
	// slash+1: if there's no slash, the value is -1 and start is 0; otherwise
	// start is the byte after the slash.
	for start := slash + 1; start < len(arg); start = period + 1 {
		period = strings.Index(arg[start:], ".")
		symbol := ""
		if period < 0 {
			period = len(arg)
		} else {
			period += start
			symbol = arg[period+1:]
		}
		// Have we identified a package already?
		pkg, err := build.Import(arg[0:period], wd, build.ImportComment)
		if err == nil {
			return pkg, arg[0:period], symbol, false
		}
		// See if we have the basename or tail of a package, as in json for encoding/json
		// or ivy/value for robpike.io/ivy/value.
		pkgName := arg[:period]
		for {
			path, ok := findNextPackage(pkgName)
			if !ok {
				break
			}
			if pkg, err = build.ImportDir(path, build.ImportComment); err == nil {
				return pkg, arg[0:period], symbol, true
			}
		}
		dirs.Reset() // Next iteration of for loop must scan all the directories again.
	}
	// If it has a slash, we've failed.
	if slash >= 0 {
		// build.Import should always include the path in its error message,
		// and we should avoid repeating it. Unfortunately, build.Import doesn't
		// return a structured error. That can't easily be fixed, since it
		// invokes 'go list' and returns the error text from the loaded package.
		// TODO(golang.org/issue/34750): load using golang.org/x/tools/go/packages
		// instead of go/build.
		importErrStr := importErr.Error()
		if strings.Contains(importErrStr, arg[:period]) {
			log.Fatal(importErrStr)
		} else {
			log.Fatalf("no such package %s: %s", arg[:period], importErrStr)
		}
	}
	// Guess it's a symbol in the current directory.
	return importDir(wd), "", arg, false
}

// dotPaths lists all the dotted paths legal on Unix-like and
// Windows-like file systems. We check them all, as the chance
// of error is minute and even on Windows people will use ./
// sometimes.
var dotPaths = []string{
	`./`,
	`../`,
	`.\`,
	`..\`,
}

// isDotSlash reports whether the path begins with a reference
// to the local . or .. directory.
func isDotSlash(arg string) bool {
	if arg == "." || arg == ".." {
		return true
	}
	for _, dotPath := range dotPaths {
		if strings.HasPrefix(arg, dotPath) {
			return true
		}
	}
	return false
}

// importDir is just an error-catching wrapper for build.ImportDir.
func importDir(dir string) *build.Package {
	pkg, err := build.ImportDir(dir, build.ImportComment)
	if err != nil {
		log.Fatal(err)
	}
	return pkg
}

// parseSymbol breaks str apart into a symbol and method.
// Both may be missing or the method may be missing.
// If present, each must be a valid Go identifier.
func parseSymbol(flagSet *flag.FlagSet, str string) (symbol, method string) {
	if str == "" {
		return
	}
	elem := strings.Split(str, ".")
	switch len(elem) {
	case 1:
	case 2:
		method = elem[1]
	default:
		log.Printf("too many periods in symbol specification")
		usage(flagSet)
	}
	symbol = elem[0]
	return
}

// isExported reports whether the name is an exported identifier.
// If the unexported flag (-u) is true, isExported returns true because
// it means that we treat the name as if it is exported.
func isExported(name string) bool {
	return unexported || token.IsExported(name)
}

// findNextPackage returns the next full file name path that matches the
// (perhaps partial) package path pkg. The boolean reports if any match was found.
func findNextPackage(pkg string) (string, bool) {
	if filepath.IsAbs(pkg) {
		if dirs.offset == 0 {
			dirs.offset = -1
			return pkg, true
		}
		return "", false
	}
	if pkg == "" || token.IsExported(pkg) { // Upper case symbol cannot be a package name.
		return "", false
	}
	pkg = path.Clean(pkg)
	pkgSuffix := "/" + pkg
	for {
		d, ok := dirs.Next()
		if !ok {
			return "", false
		}
		if d.importPath == pkg || strings.HasSuffix(d.importPath, pkgSuffix) {
			return d.dir, true
		}
	}
}

var buildCtx = build.Default

// splitGopath splits $GOPATH into a list of roots.
func splitGopath() []string {
	return filepath.SplitList(buildCtx.GOPATH)
}
