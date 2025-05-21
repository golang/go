// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Doc (usually run as go doc) accepts zero, one or two arguments.
//
// Zero arguments:
//
//	go doc
//
// Show the documentation for the package in the current directory.
//
// One argument:
//
//	go doc <pkg>
//	go doc <sym>[.<methodOrField>]
//	go doc [<pkg>.]<sym>[.<methodOrField>]
//	go doc [<pkg>.][<sym>.]<methodOrField>
//
// The first item in this list that succeeds is the one whose documentation
// is printed. If there is a symbol but no package, the package in the current
// directory is chosen. However, if the argument begins with a capital
// letter it is always assumed to be a symbol in the current directory.
//
// Two arguments:
//
//	go doc <pkg> <sym>[.<methodOrField>]
//
// Show the documentation for the package, symbol, and method or field. The
// first argument must be a full package path. This is similar to the
// command-line usage for the godoc command.
//
// For commands, unless the -cmd flag is present "go doc command"
// shows only the package-level docs for the package.
//
// The -src flag causes doc to print the full source code for the symbol, such
// as the body of a struct, function or method.
//
// The -all flag causes doc to print all documentation for the package and
// all its visible symbols. The argument must identify a package.
//
// For complete documentation, run "go help doc".
package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"go/token"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"path/filepath"
	"strings"

	"cmd/internal/telemetry/counter"
)

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
func usage() {
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
	flag.PrintDefaults()
	os.Exit(2)
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("doc: ")
	counter.Open()
	dirsInit()
	err := do(os.Stdout, flag.CommandLine, os.Args[1:])
	if err != nil {
		log.Fatal(err)
	}
}

// do is the workhorse, broken out of main to make testing easier.
func do(writer io.Writer, flagSet *flag.FlagSet, args []string) (err error) {
	flagSet.Usage = usage
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
	counter.Inc("doc/invocations")
	counter.CountFlags("doc/flag:", *flag.CommandLine)
	if chdir != "" {
		if err := os.Chdir(chdir); err != nil {
			return err
		}
	}
	if serveHTTP {
		// Special case: if there are no arguments to go doc -http, allow
		// there to be no package in the current directory. We'll still try
		// to open the page for the documentation of the package in the current
		// directory, but if one doesn't exist, fall back to opening the home page.
		if len(flagSet.Args()) == 0 {
			var path string
			if importPath, err := runCmd("go", "list"); err == nil {
				path = importPath
			}
			return doPkgsite(path)
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
		buildPackage, userPath, sym, more := parseArgs(flagSet.Args())
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

		symbol, method = parseSymbol(sym)
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

func runCmd(cmdline ...string) (string, error) {
	var stdout, stderr strings.Builder
	cmd := exec.Command(cmdline[0], cmdline[1:]...)
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
		path, err = runCmd("go", "list", userPath)
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

func doPkgsite(urlPath string) error {
	port, err := pickUnusedPort()
	if err != nil {
		return fmt.Errorf("failed to find port for documentation server: %v", err)
	}
	addr := fmt.Sprintf("localhost:%d", port)
	path := path.Join("http://"+addr, urlPath)

	// Turn off the default signal handler for SIGINT (and SIGQUIT on Unix)
	// and instead wait for the child process to handle the signal and
	// exit before exiting ourselves.
	signal.Ignore(signalsToIgnore...)

	const version = "v0.0.0-20250520201116-40659211760d"
	cmd := exec.Command("go", "run", "golang.org/x/pkgsite/cmd/internal/doc@"+version,
		"-gorepo", buildCtx.GOROOT,
		"-http", addr,
		"-open", path)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			// Exit with the same exit status as pkgsite to avoid
			// printing of "exit status" error messages.
			// Any relevant messages have already been printed
			// to stdout or stderr.
			os.Exit(ee.ExitCode())
		}
		return err
	}

	return nil
}

// pickUnusedPort finds an unused port by trying to listen on port 0
// and letting the OS pick a port, then closing that connection and
// returning that port number.
// This is inherently racy.
func pickUnusedPort() (int, error) {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		return 0, err
	}
	port := l.Addr().(*net.TCPAddr).Port
	if err := l.Close(); err != nil {
		return 0, err
	}
	return port, nil
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
func parseArgs(args []string) (pkg *build.Package, path, symbol string, more bool) {
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
		usage()
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
func parseSymbol(str string) (symbol, method string) {
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
		usage()
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
