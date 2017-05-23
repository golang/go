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
	text            string // Text of last token returned by Next.
	peek            bool
	peekToken       ScanToken
	peekText        string
}

// NewInput returns an Input from the given path.
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
		tokens := Tokenize(name)
		if len(tokens) != 1 || tokens[0].ScanToken != scanner.Ident {
			fmt.Fprintf(os.Stderr, "asm: parsing -D: %q is not a valid identifier name\n", tokens[0])
			flags.Usage()
		}
		macros[name] = &Macro{
			name:   name,
			args:   nil,
			tokens: Tokenize(value),
		}
	}
	return macros
}

var panicOnError bool // For testing.

func (in *Input) Error(args ...interface{}) {
	if panicOnError {
		panic(fmt.Errorf("%s:%d: %s", in.File(), in.Line(), fmt.Sprintln(args...)))
	}
	fmt.Fprintf(os.Stderr, "%s:%d: %s", in.File(), in.Line(), fmt.Sprintln(args...))
	os.Exit(1)
}

// expectText is like Error but adds "got XXX" where XXX is a quoted representation of the most recent token.
func (in *Input) expectText(args ...interface{}) {
	in.Error(append(args, "; got", strconv.Quote(in.Stack.Text()))...)
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
	if in.peek {
		in.peek = false
		tok := in.peekToken
		in.text = in.peekText
		return tok
	}
	// If we cannot generate a token after 100 macro invocations, we're in trouble.
	// The usual case is caught by Push, below, but be safe.
	for nesting := 0; nesting < 100; {
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
				nesting++
				in.invokeMacro(macro)
				continue
			}
			fallthrough
		default:
			if tok == scanner.EOF && len(in.ifdefStack) > 0 {
				// We're skipping text but have run out of input with no #endif.
				in.Error("unclosed #ifdef or #ifndef")
			}
			in.beginningOfLine = tok == '\n'
			if in.enabled() {
				in.text = in.Stack.Text()
				return tok
			}
		}
	}
	in.Error("recursive macro invocation")
	return 0
}

func (in *Input) Text() string {
	return in.text
}

// hash processes a # preprocessor directive. It returns true iff it completes.
func (in *Input) hash() bool {
	// We have a '#'; it must be followed by a known word (define, include, etc.).
	tok := in.Stack.Next()
	if tok != scanner.Ident {
		in.expectText("expected identifier after '#'")
	}
	if !in.enabled() {
		// Can only start including again if we are at #else or #endif but also
		// need to keep track of nested #if[n]defs.
		// We let #line through because it might affect errors.
		switch in.Stack.Text() {
		case "else", "endif", "ifdef", "ifndef", "line":
			// Press on.
		default:
			return false
		}
	}
	switch in.Stack.Text() {
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
		in.Error("unexpected token after '#':", in.Stack.Text())
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
	return in.Stack.Text()
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
	prevCol := in.Stack.Col()
	tok := in.Stack.Next()
	if tok == '\n' || tok == scanner.EOF {
		return nil, nil // No definition for macro
	}
	var args []string
	// The C preprocessor treats
	//	#define A(x)
	// and
	//	#define A (x)
	// distinctly: the first is a macro with arguments, the second without.
	// Distinguish these cases using the column number, since we don't
	// see the space itself. Note that text/scanner reports the position at the
	// end of the token. It's where you are now, and you just read this token.
	if tok == '(' && in.Stack.Col() == prevCol+1 {
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
		if tok == scanner.EOF {
			in.Error("missing newline in definition for macro:", name)
		}
		if tok == '\\' {
			tok = in.Stack.Next()
			if tok != '\n' && tok != '\\' {
				in.Error(`can only escape \ or \n in definition for macro:`, name)
			}
		}
		tokens = append(tokens, Make(tok, in.Stack.Text()))
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
	// If the macro has no arguments, just substitute the text.
	if macro.args == nil {
		in.Push(NewSlice(in.File(), in.Line(), macro.tokens))
		return
	}
	tok := in.Stack.Next()
	if tok != '(' {
		// If the macro has arguments but is invoked without them, all we push is the macro name.
		// First, put back the token.
		in.peekToken = tok
		in.peekText = in.text
		in.peek = true
		in.Push(NewSlice(in.File(), in.Line(), []Token{Make(macroName, macro.name)}))
		return
	}
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

// argsFor returns a map from formal name to actual value for this argumented macro invocation.
// The opening parenthesis has been absorbed.
func (in *Input) argsFor(macro *Macro) map[string][]Token {
	var args [][]Token
	// One macro argument per iteration. Collect them all and check counts afterwards.
	for argNum := 0; ; argNum++ {
		tokens, tok := in.collectArgument(macro)
		args = append(args, tokens)
		if tok == ')' {
			break
		}
	}
	// Zero-argument macros are tricky.
	if len(macro.args) == 0 && len(args) == 1 && args[0] == nil {
		args = nil
	} else if len(args) != len(macro.args) {
		in.Error("wrong arg count for macro", macro.name)
	}
	argMap := make(map[string][]Token)
	for i, arg := range args {
		argMap[macro.args[i]] = arg
	}
	return argMap
}

// collectArgument returns the actual tokens for a single argument of a macro.
// It also returns the token that terminated the argument, which will always
// be either ',' or ')'. The starting '(' has been scanned.
func (in *Input) collectArgument(macro *Macro) ([]Token, ScanToken) {
	nesting := 0
	var tokens []Token
	for {
		tok := in.Stack.Next()
		if tok == scanner.EOF || tok == '\n' {
			in.Error("unterminated arg list invoking macro:", macro.name)
		}
		if nesting == 0 && (tok == ')' || tok == ',') {
			return tokens, tok
		}
		if tok == '(' {
			nesting++
		}
		if tok == ')' {
			nesting--
		}
		tokens = append(tokens, Make(tok, in.Stack.Text()))
	}
}

// #ifdef and #ifndef processing.
func (in *Input) ifdef(truth bool) {
	name := in.macroName()
	in.expectNewline("#if[n]def")
	if !in.enabled() {
		truth = false
	} else if _, defined := in.macros[name]; !defined {
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
	if len(in.ifdefStack) == 1 || in.ifdefStack[len(in.ifdefStack)-2] {
		in.ifdefStack[len(in.ifdefStack)-1] = !in.ifdefStack[len(in.ifdefStack)-1]
	}
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
	name, err := strconv.Unquote(in.Stack.Text())
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
	tok = in.Stack.Next()
	if tok != '\n' {
		in.Error("unexpected token at end of #line: ", tok)
	}
	linkCtxt.LineHist.Update(histLine, file, line)
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
