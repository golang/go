// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package generate implements the “go generate” command.
package generate

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"
	"cmd/go/internal/work"
)

var CmdGenerate = &base.Command{
	Run:       runGenerate,
	UsageLine: "go generate [-run regexp] [-n] [-v] [-x] [build flags] [file.go... | packages]",
	Short:     "generate Go files by processing source",
	Long: `
Generate runs commands described by directives within existing
files. Those commands can run any process but the intent is to
create or update Go source files.

Go generate is never run automatically by go build, go test,
and so on. It must be run explicitly.

Go generate scans the file for directives, which are lines of
the form,

	//go:generate command argument...

(note: no leading spaces and no space in "//go") where command
is the generator to be run, corresponding to an executable file
that can be run locally. It must either be in the shell path
(gofmt), a fully qualified path (/usr/you/bin/mytool), or a
command alias, described below.

Note that go generate does not parse the file, so lines that look
like directives in comments or multiline strings will be treated
as directives.

The arguments to the directive are space-separated tokens or
double-quoted strings passed to the generator as individual
arguments when it is run.

Quoted strings use Go syntax and are evaluated before execution; a
quoted string appears as a single argument to the generator.

To convey to humans and machine tools that code is generated,
generated source should have a line that matches the following
regular expression (in Go syntax):

	^// Code generated .* DO NOT EDIT\.$

This line must appear before the first non-comment, non-blank
text in the file.

Go generate sets several variables when it runs the generator:

	$GOARCH
		The execution architecture (arm, amd64, etc.)
	$GOOS
		The execution operating system (linux, windows, etc.)
	$GOFILE
		The base name of the file.
	$GOLINE
		The line number of the directive in the source file.
	$GOPACKAGE
		The name of the package of the file containing the directive.
	$GOROOT
		The GOROOT directory for the 'go' command that invoked the
		generator, containing the Go toolchain and standard library.
	$DOLLAR
		A dollar sign.
	$PATH
		The $PATH of the parent process, with $GOROOT/bin
		placed at the beginning. This causes generators
		that execute 'go' commands to use the same 'go'
		as the parent 'go generate' command.

Other than variable substitution and quoted-string evaluation, no
special processing such as "globbing" is performed on the command
line.

As a last step before running the command, any invocations of any
environment variables with alphanumeric names, such as $GOFILE or
$HOME, are expanded throughout the command line. The syntax for
variable expansion is $NAME on all operating systems. Due to the
order of evaluation, variables are expanded even inside quoted
strings. If the variable NAME is not set, $NAME expands to the
empty string.

A directive of the form,

	//go:generate -command xxx args...

specifies, for the remainder of this source file only, that the
string xxx represents the command identified by the arguments. This
can be used to create aliases or to handle multiword generators.
For example,

	//go:generate -command foo go tool foo

specifies that the command "foo" represents the generator
"go tool foo".

Generate processes packages in the order given on the command line,
one at a time. If the command line lists .go files from a single directory,
they are treated as a single package. Within a package, generate processes the
source files in a package in file name order, one at a time. Within
a source file, generate runs generators in the order they appear
in the file, one at a time. The go generate tool also sets the build
tag "generate" so that files may be examined by go generate but ignored
during build.

For packages with invalid code, generate processes only source files with a
valid package clause.

If any generator returns an error exit status, "go generate" skips
all further processing for that package.

The generator is run in the package's source directory.

Go generate accepts two specific flags:

	-run=""
		if non-empty, specifies a regular expression to select
		directives whose full original source text (excluding
		any trailing spaces and final newline) matches the
		expression.

	-skip=""
		if non-empty, specifies a regular expression to suppress
		directives whose full original source text (excluding
		any trailing spaces and final newline) matches the
		expression. If a directive matches both the -run and
		the -skip arguments, it is skipped.

It also accepts the standard build flags including -v, -n, and -x.
The -v flag prints the names of packages and files as they are
processed.
The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.

For more about build flags, see 'go help build'.

For more about specifying packages, see 'go help packages'.
	`,
}

var (
	generateRunFlag string         // generate -run flag
	generateRunRE   *regexp.Regexp // compiled expression for -run

	generateSkipFlag string         // generate -skip flag
	generateSkipRE   *regexp.Regexp // compiled expression for -skip
)

func init() {
	work.AddBuildFlags(CmdGenerate, work.DefaultBuildFlags)
	CmdGenerate.Flag.StringVar(&generateRunFlag, "run", "", "")
	CmdGenerate.Flag.StringVar(&generateSkipFlag, "skip", "", "")
}

func runGenerate(ctx context.Context, cmd *base.Command, args []string) {
	modload.InitWorkfile()

	if generateRunFlag != "" {
		var err error
		generateRunRE, err = regexp.Compile(generateRunFlag)
		if err != nil {
			log.Fatalf("generate: %s", err)
		}
	}
	if generateSkipFlag != "" {
		var err error
		generateSkipRE, err = regexp.Compile(generateSkipFlag)
		if err != nil {
			log.Fatalf("generate: %s", err)
		}
	}

	cfg.BuildContext.BuildTags = append(cfg.BuildContext.BuildTags, "generate")

	// Even if the arguments are .go files, this loop suffices.
	printed := false
	pkgOpts := load.PackageOpts{IgnoreImports: true}
	for _, pkg := range load.PackagesAndErrors(ctx, pkgOpts, args) {
		if modload.Enabled() && pkg.Module != nil && !pkg.Module.Main {
			if !printed {
				fmt.Fprintf(os.Stderr, "go: not generating in packages in dependency modules\n")
				printed = true
			}
			continue
		}

		if pkg.Error != nil && len(pkg.InternalAllGoFiles()) == 0 {
			// A directory only contains a Go package if it has at least
			// one .go source file, so the fact that there are no files
			// implies that the package couldn't be found.
			base.Errorf("%v", pkg.Error)
		}

		for _, file := range pkg.InternalGoFiles() {
			if !generate(file) {
				break
			}
		}

		for _, file := range pkg.InternalXGoFiles() {
			if !generate(file) {
				break
			}
		}
	}
	base.ExitIfErrors()
}

// generate runs the generation directives for a single file.
func generate(absFile string) bool {
	src, err := os.ReadFile(absFile)
	if err != nil {
		log.Fatalf("generate: %s", err)
	}

	// Parse package clause
	filePkg, err := parser.ParseFile(token.NewFileSet(), "", src, parser.PackageClauseOnly)
	if err != nil {
		// Invalid package clause - ignore file.
		return true
	}

	g := &Generator{
		r:        bytes.NewReader(src),
		path:     absFile,
		pkg:      filePkg.Name.String(),
		commands: make(map[string][]string),
	}
	return g.run()
}

// A Generator represents the state of a single Go source file
// being scanned for generator commands.
type Generator struct {
	r        io.Reader
	path     string // full rooted path name.
	dir      string // full rooted directory of file.
	file     string // base name of file.
	pkg      string
	commands map[string][]string
	lineNum  int // current line number.
	env      []string
}

// run runs the generators in the current file.
func (g *Generator) run() (ok bool) {
	// Processing below here calls g.errorf on failure, which does panic(stop).
	// If we encounter an error, we abort the package.
	defer func() {
		e := recover()
		if e != nil {
			ok = false
			if e != stop {
				panic(e)
			}
			base.SetExitStatus(1)
		}
	}()
	g.dir, g.file = filepath.Split(g.path)
	g.dir = filepath.Clean(g.dir) // No final separator please.
	if cfg.BuildV {
		fmt.Fprintf(os.Stderr, "%s\n", base.ShortPath(g.path))
	}

	// Scan for lines that start "//go:generate".
	// Can't use bufio.Scanner because it can't handle long lines,
	// which are likely to appear when using generate.
	input := bufio.NewReader(g.r)
	var err error
	// One line per loop.
	for {
		g.lineNum++ // 1-indexed.
		var buf []byte
		buf, err = input.ReadSlice('\n')
		if errors.Is(err, bufio.ErrBufferFull) {
			// Line too long - consume and ignore.
			if isGoGenerate(buf) {
				g.errorf("directive too long")
			}
			for errors.Is(err, bufio.ErrBufferFull) {
				_, err = input.ReadSlice('\n')
			}
			if err != nil {
				break
			}
			continue
		}

		if err != nil {
			// Check for marker at EOF without final \n.
			if err == io.EOF && isGoGenerate(buf) {
				err = io.ErrUnexpectedEOF
			}
			break
		}

		if !isGoGenerate(buf) {
			continue
		}
		if generateRunFlag != "" && !generateRunRE.Match(bytes.TrimSpace(buf)) {
			continue
		}
		if generateSkipFlag != "" && generateSkipRE.Match(bytes.TrimSpace(buf)) {
			continue
		}

		g.setEnv()
		words := g.split(string(buf))
		if len(words) == 0 {
			g.errorf("no arguments to directive")
		}
		if words[0] == "-command" {
			g.setShorthand(words)
			continue
		}
		// Run the command line.
		if cfg.BuildN || cfg.BuildX {
			fmt.Fprintf(os.Stderr, "%s\n", strings.Join(words, " "))
		}
		if cfg.BuildN {
			continue
		}
		g.exec(words)
	}
	if err != nil && err != io.EOF {
		g.errorf("error reading %s: %s", base.ShortPath(g.path), err)
	}
	return true
}

func isGoGenerate(buf []byte) bool {
	return bytes.HasPrefix(buf, []byte("//go:generate ")) || bytes.HasPrefix(buf, []byte("//go:generate\t"))
}

// setEnv sets the extra environment variables used when executing a
// single go:generate command.
func (g *Generator) setEnv() {
	env := []string{
		"GOROOT=" + cfg.GOROOT,
		"GOARCH=" + cfg.BuildContext.GOARCH,
		"GOOS=" + cfg.BuildContext.GOOS,
		"GOFILE=" + g.file,
		"GOLINE=" + strconv.Itoa(g.lineNum),
		"GOPACKAGE=" + g.pkg,
		"DOLLAR=" + "$",
	}
	env = base.AppendPATH(env)
	env = base.AppendPWD(env, g.dir)
	g.env = env
}

// split breaks the line into words, evaluating quoted
// strings and evaluating environment variables.
// The initial //go:generate element is present in line.
func (g *Generator) split(line string) []string {
	// Parse line, obeying quoted strings.
	var words []string
	line = line[len("//go:generate ") : len(line)-1] // Drop preamble and final newline.
	// There may still be a carriage return.
	if len(line) > 0 && line[len(line)-1] == '\r' {
		line = line[:len(line)-1]
	}
	// One (possibly quoted) word per iteration.
Words:
	for {
		line = strings.TrimLeft(line, " \t")
		if len(line) == 0 {
			break
		}
		if line[0] == '"' {
			for i := 1; i < len(line); i++ {
				c := line[i] // Only looking for ASCII so this is OK.
				switch c {
				case '\\':
					if i+1 == len(line) {
						g.errorf("bad backslash")
					}
					i++ // Absorb next byte (If it's a multibyte we'll get an error in Unquote).
				case '"':
					word, err := strconv.Unquote(line[0 : i+1])
					if err != nil {
						g.errorf("bad quoted string")
					}
					words = append(words, word)
					line = line[i+1:]
					// Check the next character is space or end of line.
					if len(line) > 0 && line[0] != ' ' && line[0] != '\t' {
						g.errorf("expect space after quoted argument")
					}
					continue Words
				}
			}
			g.errorf("mismatched quoted string")
		}
		i := strings.IndexAny(line, " \t")
		if i < 0 {
			i = len(line)
		}
		words = append(words, line[0:i])
		line = line[i:]
	}
	// Substitute command if required.
	if len(words) > 0 && g.commands[words[0]] != nil {
		// Replace 0th word by command substitution.
		//
		// Force a copy of the command definition to
		// ensure words doesn't end up as a reference
		// to the g.commands content.
		tmpCmdWords := append([]string(nil), (g.commands[words[0]])...)
		words = append(tmpCmdWords, words[1:]...)
	}
	// Substitute environment variables.
	for i, word := range words {
		words[i] = os.Expand(word, g.expandVar)
	}
	return words
}

var stop = fmt.Errorf("error in generation")

// errorf logs an error message prefixed with the file and line number.
// It then exits the program (with exit status 1) because generation stops
// at the first error.
func (g *Generator) errorf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "%s:%d: %s\n", base.ShortPath(g.path), g.lineNum,
		fmt.Sprintf(format, args...))
	panic(stop)
}

// expandVar expands the $XXX invocation in word. It is called
// by os.Expand.
func (g *Generator) expandVar(word string) string {
	w := word + "="
	for _, e := range g.env {
		if strings.HasPrefix(e, w) {
			return e[len(w):]
		}
	}
	return os.Getenv(word)
}

// setShorthand installs a new shorthand as defined by a -command directive.
func (g *Generator) setShorthand(words []string) {
	// Create command shorthand.
	if len(words) == 1 {
		g.errorf("no command specified for -command")
	}
	command := words[1]
	if g.commands[command] != nil {
		g.errorf("command %q multiply defined", command)
	}
	g.commands[command] = slices.Clip(words[2:])
}

// exec runs the command specified by the argument. The first word is
// the command name itself.
func (g *Generator) exec(words []string) {
	path := words[0]
	if path != "" && !strings.Contains(path, string(os.PathSeparator)) {
		// If a generator says '//go:generate go run <blah>' it almost certainly
		// intends to use the same 'go' as 'go generate' itself.
		// Prefer to resolve the binary from GOROOT/bin, and for consistency
		// prefer to resolve any other commands there too.
		gorootBinPath, err := cfg.LookPath(filepath.Join(cfg.GOROOTbin, path))
		if err == nil {
			path = gorootBinPath
		}
	}
	cmd := exec.Command(path, words[1:]...)
	cmd.Args[0] = words[0] // Overwrite with the original in case it was rewritten above.

	// Standard in and out of generator should be the usual.
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	// Run the command in the package directory.
	cmd.Dir = g.dir
	cmd.Env = str.StringList(cfg.OrigEnv, g.env)
	err := cmd.Run()
	if err != nil {
		g.errorf("running %q: %s", words[0], err)
	}
}
