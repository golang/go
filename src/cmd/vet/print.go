// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the printf-checker.

package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/token"
	"strings"
	"unicode/utf8"
)

var printfuncs = flag.String("printfuncs", "", "comma-separated list of print function names to check")

// printfList records the formatted-print functions. The value is the location
// of the format parameter. Names are lower-cased so the lookup is
// case insensitive.
var printfList = map[string]int{
	"errorf":  0,
	"fatalf":  0,
	"fprintf": 1,
	"panicf":  0,
	"printf":  0,
	"sprintf": 0,
}

// printList records the unformatted-print functions. The value is the location
// of the first parameter to be printed.  Names are lower-cased so the lookup is
// case insensitive.
var printList = map[string]int{
	"error":  0,
	"fatal":  0,
	"fprint": 1, "fprintln": 1,
	"panic": 0, "panicln": 0,
	"print": 0, "println": 0,
	"sprint": 0, "sprintln": 0,
}

// checkCall triggers the print-specific checks if the call invokes a print function.
func (f *File) checkFmtPrintfCall(call *ast.CallExpr, Name string) {
	name := strings.ToLower(Name)
	if skip, ok := printfList[name]; ok {
		f.checkPrintf(call, Name, skip)
		return
	}
	if skip, ok := printList[name]; ok {
		f.checkPrint(call, Name, skip)
		return
	}
}

// checkPrintf checks a call to a formatted print routine such as Printf.
// The skip argument records how many arguments to ignore; that is,
// call.Args[skip] is (well, should be) the format argument.
func (f *File) checkPrintf(call *ast.CallExpr, name string, skip int) {
	if len(call.Args) <= skip {
		return
	}
	// Common case: literal is first argument.
	arg := call.Args[skip]
	lit, ok := arg.(*ast.BasicLit)
	if !ok {
		// Too hard to check.
		if *verbose {
			f.Warn(call.Pos(), "can't check non-literal format in call to", name)
		}
		return
	}
	if lit.Kind == token.STRING {
		if !strings.Contains(lit.Value, "%") {
			if len(call.Args) > skip+1 {
				f.Badf(call.Pos(), "no formatting directive in %s call", name)
			}
			return
		}
	}
	// Hard part: check formats against args.
	// Trivial but useful test: count.
	numArgs := 0
	for i, w := 0, 0; i < len(lit.Value); i += w {
		w = 1
		if lit.Value[i] == '%' {
			nbytes, nargs := f.parsePrintfVerb(call, lit.Value[i:])
			w = nbytes
			numArgs += nargs
		}
	}
	expect := len(call.Args) - (skip + 1)
	if numArgs != expect {
		f.Badf(call.Pos(), "wrong number of args in %s call: %d needed but %d args", name, numArgs, expect)
	}
}

// parsePrintfVerb returns the number of bytes and number of arguments
// consumed by the Printf directive that begins s, including its percent sign
// and verb.
func (f *File) parsePrintfVerb(call *ast.CallExpr, s string) (nbytes, nargs int) {
	// There's guaranteed a percent sign.
	flags := make([]byte, 0, 5)
	nbytes = 1
	end := len(s)
	// There may be flags.
FlagLoop:
	for nbytes < end {
		switch s[nbytes] {
		case '#', '0', '+', '-', ' ':
			flags = append(flags, s[nbytes])
			nbytes++
		default:
			break FlagLoop
		}
	}
	getNum := func() {
		if nbytes < end && s[nbytes] == '*' {
			nbytes++
			nargs++
		} else {
			for nbytes < end && '0' <= s[nbytes] && s[nbytes] <= '9' {
				nbytes++
			}
		}
	}
	// There may be a width.
	getNum()
	// If there's a period, there may be a precision.
	if nbytes < end && s[nbytes] == '.' {
		flags = append(flags, '.') // Treat precision as a flag.
		nbytes++
		getNum()
	}
	// Now a verb.
	c, w := utf8.DecodeRuneInString(s[nbytes:])
	nbytes += w
	if c != '%' {
		nargs++
		f.checkPrintfVerb(call, c, flags)
	}
	return
}

type printVerb struct {
	verb  rune
	flags string // known flags are all ASCII
}

// Common flag sets for printf verbs.
const (
	numFlag      = " -+.0"
	sharpNumFlag = " -+.0#"
	allFlags     = " -+.0#"
)

// printVerbs identifies which flags are known to printf for each verb.
// TODO: A type that implements Formatter may do what it wants, and vet
// will complain incorrectly.
var printVerbs = []printVerb{
	// '-' is a width modifier, always valid.
	// '.' is a precision for float, max width for strings.
	// '+' is required sign for numbers, Go format for %v.
	// '#' is alternate format for several verbs.
	// ' ' is spacer for numbers
	{'b', numFlag},
	{'c', "-"},
	{'d', numFlag},
	{'e', numFlag},
	{'E', numFlag},
	{'f', numFlag},
	{'F', numFlag},
	{'g', numFlag},
	{'G', numFlag},
	{'o', sharpNumFlag},
	{'p', "-#"},
	{'q', "-+#."},
	{'s', "-."},
	{'t', "-"},
	{'T', "-"},
	{'U', "-#"},
	{'v', allFlags},
	{'x', sharpNumFlag},
	{'X', sharpNumFlag},
}

const printfVerbs = "bcdeEfFgGopqstTvxUX"

func (f *File) checkPrintfVerb(call *ast.CallExpr, verb rune, flags []byte) {
	// Linear scan is fast enough for a small list.
	for _, v := range printVerbs {
		if v.verb == verb {
			for _, flag := range flags {
				if !strings.ContainsRune(v.flags, rune(flag)) {
					f.Badf(call.Pos(), "unrecognized printf flag for verb %q: %q", verb, flag)
				}
			}
			return
		}
	}
	f.Badf(call.Pos(), "unrecognized printf verb %q", verb)
}

// checkPrint checks a call to an unformatted print routine such as Println.
// The skip argument records how many arguments to ignore; that is,
// call.Args[skip] is the first argument to be printed.
func (f *File) checkPrint(call *ast.CallExpr, name string, skip int) {
	isLn := strings.HasSuffix(name, "ln")
	isF := strings.HasPrefix(name, "F")
	args := call.Args
	// check for Println(os.Stderr, ...)
	if skip == 0 && !isF && len(args) > 0 {
		if sel, ok := args[0].(*ast.SelectorExpr); ok {
			if x, ok := sel.X.(*ast.Ident); ok {
				if x.Name == "os" && strings.HasPrefix(sel.Sel.Name, "Std") {
					f.Warnf(call.Pos(), "first argument to %s is %s.%s", name, x.Name, sel.Sel.Name)
				}
			}
		}
	}
	if len(args) <= skip {
		if *verbose && !isLn {
			f.Badf(call.Pos(), "no args in %s call", name)
		}
		return
	}
	arg := args[skip]
	if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
		if strings.Contains(lit.Value, "%") {
			f.Badf(call.Pos(), "possible formatting directive in %s call", name)
		}
	}
	if isLn {
		// The last item, if a string, should not have a newline.
		arg = args[len(call.Args)-1]
		if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
			if strings.HasSuffix(lit.Value, `\n"`) {
				f.Badf(call.Pos(), "%s call ends with newline", name)
			}
		}
	}
}

// This function never executes, but it serves as a simple test for the program.
// Test with make test.
func BadFunctionUsedInTests() {
	fmt.Println()                      // not an error
	fmt.Println("%s", "hi")            // ERROR "possible formatting directive in Println call"
	fmt.Printf("%s", "hi", 3)          // ERROR "wrong number of args in Printf call"
	fmt.Printf("%s%%%d", "hi", 3)      // correct
	fmt.Printf("%.*d", 3, 3)           // correct
	fmt.Printf("%.*d", 3, 3, 3)        // ERROR "wrong number of args in Printf call"
	printf("now is the time", "buddy") // ERROR "no formatting directive"
	Printf("now is the time", "buddy") // ERROR "no formatting directive"
	Printf("hi")                       // ok
	f := new(File)
	f.Warn(0, "%s", "hello", 3)  // ERROR "possible formatting directive in Warn call"
	f.Warnf(0, "%s", "hello", 3) // ERROR "wrong number of args in Warnf call"
	f.Warnf(0, "%r", "hello")    // ERROR "unrecognized printf verb"
	f.Warnf(0, "%#s", "hello")   // ERROR "unrecognized printf flag"
}

type BadTypeUsedInTests struct {
	X int "hello" // ERROR "struct field tag"
}

func (t *BadTypeUsedInTests) Scan(x fmt.ScanState, c byte) { // ERROR "method Scan[(]x fmt.ScanState, c byte[)] should have signature Scan[(]fmt.ScanState, rune[)] error"
}

type BadInterfaceUsedInTests interface {
	ReadByte() byte // ERROR "method ReadByte[(][)] byte should have signature ReadByte[(][)] [(]byte, error[)]"
}

// printf is used by the test.
func printf(format string, args ...interface{}) {
	panic("don't call - testing only")
}
