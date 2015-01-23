// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lex

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/scanner"

	"cmd/asm/internal/flags"
	"cmd/internal/obj"
)

// Input is the main input: a stack of readers and some macro definitions.
// It also handles #include processing (by pushing onto the input stack)
// and parses and instantiates macro definitions.
type Input struct {
	Stack
	includes        []string
	beginningOfLine bool
	ifdefStack      []bool
	macros          map[string]*Macro
}

// NewInput returns a
func NewInput(name string) *Input {
	return &Input{
		// include directories: look in source dir, then -I directories.
		includes:        append([]string{filepath.Dir(name)}, flags.I...),
		beginningOfLine: true,
		macros:          predefine(flags.D),
	}
}

// predefine installs the macros set by the -D flag on the command line.
func predefine(defines flags.MultiFlag) map[string]*Macro {
	macros := make(map[string]*Macro)
	for _, name := range defines {
		value := "1"
		i := strings.IndexRune(name, '=')
		if i > 0 {
			name, value = name[:i], name[i+1:]
		}
		tokens := tokenize(name)
		if len(tokens) != 1 || tokens[0].ScanToken != scanner.Ident {
			fmt.Fprintf(os.Stderr, "asm: parsing -D: %q is not a valid identifier name\n", tokens[0])
			flags.Usage()
		}
		macros[name] = &Macro{
			name:   name,
			args:   nil,
			tokens: tokenize(value),
		}
	}
	return macros
}

func (in *Input) Error(args ...interface{}) {
	fmt.Fprintf(os.Stderr, "%s:%d: %s", in.File(), in.Line(), fmt.Sprintln(args...))
	os.Exit(1)
}

// expectText is like Error but adds "got XXX" where XXX is a quoted representation of the most recent token.
func (in *Input) expectText(args ...interface{}) {
	in.Error(append(args, "; got", strconv.Quote(in.Text()))...)
}

// enabled reports whether the input is enabled by an ifdef, or is at the top level.
func (in *Input) enabled() bool {
	return len(in.ifdefStack) == 0 || in.ifdefStack[len(in.ifdefStack)-1]
}

func (in *Input) expectNewline(directive string) {
	tok := in.Stack.Next()
	if tok != '\n' {
		in.expectText("expected newline after", directive)
	}
}

func (in *Input) Next() ScanToken {
	for {
		tok := in.Stack.Next()
		switch tok {
		case '#':
			if !in.beginningOfLine {
				in.Error("'#' must be first item on line")
			}
			in.beginningOfLine = in.hash()
		case scanner.Ident:
			// Is it a macro name?
			name := in.Stack.Text()
			macro := in.macros[name]
			if macro != nil {
				in.invokeMacro(macro)
				continue
			}
			fallthrough
		default:
			in.beginningOfLine = tok == '\n'
			if in.enabled() {
				return tok
			}
		}
	}
	in.Error("recursive macro invocation")
	return 0
}

// hash processes a # preprocessor directive. It returns true iff it completes.
func (in *Input) hash() bool {
	// We have a '#'; it must be followed by a known word (define, include, etc.).
	tok := in.Stack.Next()
	if tok != scanner.Ident {
		in.expectText("expected identifier after '#'")
	}
	if !in.enabled() {
		// Can only start including again if we are at #else or #endif.
		// We let #line through because it might affect errors.
		switch in.Text() {
		case "else", "endif", "line":
			// Press on.
		default:
			return false
		}
	}
	switch in.Text() {
	case "define":
		in.define()
	case "else":
		in.else_()
	case "endif":
		in.endif()
	case "ifdef":
		in.ifdef(true)
	case "ifndef":
		in.ifdef(false)
	case "include":
		in.include()
	case "line":
		in.line()
	case "undef":
		in.undef()
	default:
		in.Error("unexpected identifier after '#':", in.Text())
	}
	return true
}

// macroName returns the name for the macro being referenced.
func (in *Input) macroName() string {
	// We use the Stack's input method; no macro processing at this stage.
	tok := in.Stack.Next()
	if tok != scanner.Ident {
		in.expectText("expected identifier after # directive")
	}
	// Name is alphanumeric by definition.
	return in.Text()
}

// #define processing.
func (in *Input) define() {
	name := in.macroName()
	args, tokens := in.macroDefinition(name)
	in.defineMacro(name, args, tokens)
}

// defineMacro stores the macro definition in the Input.
func (in *Input) defineMacro(name string, args []string, tokens []Token) {
	if in.macros[name] != nil {
		in.Error("redefinition of macro:", name)
	}
	in.macros[name] = &Macro{
		name:   name,
		args:   args,
		tokens: tokens,
	}
}

// macroDefinition returns the list of formals and the tokens of the definition.
// The argument list is nil for no parens on the definition; otherwise a list of
// formal argument names.
func (in *Input) macroDefinition(name string) ([]string, []Token) {
	tok := in.Stack.Next()
	if tok == '\n' || tok == scanner.EOF {
		in.Error("no definition for macro:", name)
	}
	var args []string
	if tok == '(' {
		// Macro has arguments. Scan list of formals.
		acceptArg := true
		args = []string{} // Zero length but not nil.
	Loop:
		for {
			tok = in.Stack.Next()
			switch tok {
			case ')':
				tok = in.Stack.Next() // First token of macro definition.
				break Loop
			case ',':
				if acceptArg {
					in.Error("bad syntax in definition for macro:", name)
				}
				acceptArg = true
			case scanner.Ident:
				if !acceptArg {
					in.Error("bad syntax in definition for macro:", name)
				}
				arg := in.Stack.Text()
				if i := lookup(args, arg); i >= 0 {
					in.Error("duplicate argument", arg, "in definition for macro:", name)
				}
				args = append(args, arg)
				acceptArg = false
			default:
				in.Error("bad definition for macro:", name)
			}
		}
	}
	var tokens []Token
	// Scan to newline. Backslashes escape newlines.
	for tok != '\n' {
		if tok == '\\' {
			tok = in.Stack.Next()
			if tok != '\n' && tok != '\\' {
				in.Error(`can only escape \ or \n in definition for macro:`, name)
			}
			if tok == '\n' { // backslash-newline is discarded
				tok = in.Stack.Next()
				continue
			}
		}
		tokens = append(tokens, Make(tok, in.Text()))
		tok = in.Stack.Next()
	}
	return args, tokens
}

func lookup(args []string, arg string) int {
	for i, a := range args {
		if a == arg {
			return i
		}
	}
	return -1
}

// invokeMacro pushes onto the input Stack a Slice that holds the macro definition with the actual
// parameters substituted for the formals.
// Invoking a macro does not touch the PC/line history.
func (in *Input) invokeMacro(macro *Macro) {
	actuals := in.argsFor(macro)
	var tokens []Token
	for _, tok := range macro.tokens {
		if tok.ScanToken != scanner.Ident {
			tokens = append(tokens, tok)
			continue
		}
		substitution := actuals[tok.text]
		if substitution == nil {
			tokens = append(tokens, tok)
			continue
		}
		tokens = append(tokens, substitution...)
	}
	in.Push(NewSlice(in.File(), in.Line(), tokens))
}

// argsFor returns a map from formal name to actual value for this macro invocation.
func (in *Input) argsFor(macro *Macro) map[string][]Token {
	if macro.args == nil {
		return nil
	}
	tok := in.Stack.Next()
	if tok != '(' {
		in.Error("missing arguments for invocation of macro:", macro.name)
	}
	var tokens []Token
	args := make(map[string][]Token)
	argNum := 0
	for {
		tok = in.Stack.Next()
		switch tok {
		case scanner.EOF, '\n':
			in.Error("unterminated arg list invoking macro:", macro.name)
		case ',', ')':
			if argNum >= len(macro.args) {
				in.Error("too many arguments for macro:", macro.name)
			}
			if len(macro.args) == 0 && argNum == 0 && len(tokens) == 0 {
				// Zero-argument macro invoked with no arguments.
				return args
			}
			args[macro.args[argNum]] = tokens
			tokens = nil
			argNum++
			if tok == ')' {
				if argNum != len(macro.args) {
					in.Error("too few arguments for macro:", macro.name)
				}
				return args
			}
		default:
			tokens = append(tokens, Make(tok, in.Stack.Text()))
		}
	}
}

// #ifdef and #ifndef processing.
func (in *Input) ifdef(truth bool) {
	name := in.macroName()
	in.expectNewline("#if[n]def")
	if _, defined := in.macros[name]; !defined {
		truth = !truth
	}
	in.ifdefStack = append(in.ifdefStack, truth)
}

// #else processing
func (in *Input) else_() {
	in.expectNewline("#else")
	if len(in.ifdefStack) == 0 {
		in.Error("unmatched #else")
	}
	in.ifdefStack[len(in.ifdefStack)-1] = !in.ifdefStack[len(in.ifdefStack)-1]
}

// #endif processing.
func (in *Input) endif() {
	in.expectNewline("#endif")
	if len(in.ifdefStack) == 0 {
		in.Error("unmatched #endif")
	}
	in.ifdefStack = in.ifdefStack[:len(in.ifdefStack)-1]
}

// #include processing.
func (in *Input) include() {
	// Find and parse string.
	tok := in.Stack.Next()
	if tok != scanner.String {
		in.expectText("expected string after #include")
	}
	name, err := strconv.Unquote(in.Text())
	if err != nil {
		in.Error("unquoting include file name: ", err)
	}
	in.expectNewline("#include")
	// Push tokenizer for file onto stack.
	fd, err := os.Open(name)
	if err != nil {
		for _, dir := range in.includes {
			fd, err = os.Open(filepath.Join(dir, name))
			if err == nil {
				break
			}
		}
		if err != nil {
			in.Error("#include:", err)
		}
	}
	in.Push(NewTokenizer(name, fd, fd))
}

// #line processing.
func (in *Input) line() {
	// Only need to handle Plan 9 format: #line 337 "filename"
	tok := in.Stack.Next()
	if tok != scanner.Int {
		in.expectText("expected line number after #line")
	}
	line, err := strconv.Atoi(in.Stack.Text())
	if err != nil {
		in.Error("error parsing #line (cannot happen):", err)
	}
	tok = in.Stack.Next()
	if tok != scanner.String {
		in.expectText("expected file name in #line")
	}
	file, err := strconv.Unquote(in.Stack.Text())
	if err != nil {
		in.Error("unquoting #line file name: ", err)
	}
	obj.Linklinehist(linkCtxt, histLine, file, line)
	in.Stack.SetPos(line, file)
}

// #undef processing
func (in *Input) undef() {
	name := in.macroName()
	if in.macros[name] == nil {
		in.Error("#undef for undefined macro:", name)
	}
	// Newline must be next.
	tok := in.Stack.Next()
	if tok != '\n' {
		in.Error("syntax error in #undef for macro:", name)
	}
	delete(in.macros, name)
}

func (in *Input) Push(r TokenReader) {
	if len(in.tr) > 100 {
		in.Error("input recursion")
	}
	in.Stack.Push(r)
}

func (in *Input) Close() {
}
