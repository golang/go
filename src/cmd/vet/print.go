// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the printf-checker.

package main

import (
	"bytes"
	"encoding/gob"
	"flag"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

var printfuncs = flag.String("printfuncs", "", "comma-separated list of print function names to check")

func init() {
	register("printf",
		"check printf-like invocations",
		checkFmtPrintfCall,
		funcDecl, callExpr)
	registerPkgCheck("printf", findPrintfLike)
	registerExport("printf", exportPrintfLike)
	gob.Register([]printfExport(nil))
}

func initPrintFlags() {
	if *printfuncs == "" {
		return
	}
	for _, name := range strings.Split(*printfuncs, ",") {
		if len(name) == 0 {
			flag.Usage()
		}

		// Backwards compatibility: skip optional first argument
		// index after the colon.
		if colon := strings.LastIndex(name, ":"); colon > 0 {
			name = name[:colon]
		}

		if !strings.Contains(name, ".") {
			name = strings.ToLower(name)
		}
		isPrint[name] = true
	}
}

var localPrintfLike = make(map[string]int)

type printfExport struct {
	Name string
	Kind int
}

// printfImported maps from package name to the printf vet data
// exported by that package.
var printfImported = make(map[string]map[string]int)

type printfWrapper struct {
	name    string
	fn      *ast.FuncDecl
	format  *ast.Field
	args    *ast.Field
	callers []printfCaller
	failed  bool // if true, not a printf wrapper
}

type printfCaller struct {
	w    *printfWrapper
	call *ast.CallExpr
}

// maybePrintfWrapper decides whether decl (a declared function) may be a wrapper
// around a fmt.Printf or fmt.Print function. If so it returns a printfWrapper
// function describing the declaration. Later processing will analyze the
// graph of potential printf wrappers to pick out the ones that are true wrappers.
// A function may be a Printf or Print wrapper if its last argument is ...interface{}.
// If the next-to-last argument is a string, then this may be a Printf wrapper.
// Otherwise it may be a Print wrapper.
func maybePrintfWrapper(decl ast.Decl) *printfWrapper {
	// Look for functions with final argument type ...interface{}.
	fn, ok := decl.(*ast.FuncDecl)
	if !ok || fn.Body == nil {
		return nil
	}
	name := fn.Name.Name
	if fn.Recv != nil {
		// For (*T).Name or T.name, use "T.name".
		rcvr := fn.Recv.List[0].Type
		if ptr, ok := rcvr.(*ast.StarExpr); ok {
			rcvr = ptr.X
		}
		id, ok := rcvr.(*ast.Ident)
		if !ok {
			return nil
		}
		name = id.Name + "." + name
	}
	params := fn.Type.Params.List
	if len(params) == 0 {
		return nil
	}
	args := params[len(params)-1]
	if len(args.Names) != 1 {
		return nil
	}
	ddd, ok := args.Type.(*ast.Ellipsis)
	if !ok {
		return nil
	}
	iface, ok := ddd.Elt.(*ast.InterfaceType)
	if !ok || len(iface.Methods.List) > 0 {
		return nil
	}
	var format *ast.Field
	if len(params) >= 2 {
		p := params[len(params)-2]
		if len(p.Names) == 1 {
			if id, ok := p.Type.(*ast.Ident); ok && id.Name == "string" {
				format = p
			}
		}
	}

	return &printfWrapper{
		name:   name,
		fn:     fn,
		format: format,
		args:   args,
	}
}

// findPrintfLike scans the entire package to find printf-like functions.
func findPrintfLike(pkg *Package) {
	if vcfg.ImportPath == "" { // no type or vetx information; don't bother
		return
	}

	// Gather potential wrappesr and call graph between them.
	byName := make(map[string]*printfWrapper)
	var wrappers []*printfWrapper
	for _, file := range pkg.files {
		if file.file == nil {
			continue
		}
		for _, decl := range file.file.Decls {
			w := maybePrintfWrapper(decl)
			if w == nil {
				continue
			}
			byName[w.name] = w
			wrappers = append(wrappers, w)
		}
	}

	// Walk the graph to figure out which are really printf wrappers.
	for _, w := range wrappers {
		// Scan function for calls that could be to other printf-like functions.
		ast.Inspect(w.fn.Body, func(n ast.Node) bool {
			if w.failed {
				return false
			}

			// TODO: Relax these checks; issue 26555.
			if assign, ok := n.(*ast.AssignStmt); ok {
				for _, lhs := range assign.Lhs {
					if match(lhs, w.format) || match(lhs, w.args) {
						// Modifies the format
						// string or args in
						// some way, so not a
						// simple wrapper.
						w.failed = true
						return false
					}
				}
			}
			if un, ok := n.(*ast.UnaryExpr); ok && un.Op == token.AND {
				if match(un.X, w.format) || match(un.X, w.args) {
					// Taking the address of the
					// format string or args,
					// so not a simple wrapper.
					w.failed = true
					return false
				}
			}

			call, ok := n.(*ast.CallExpr)
			if !ok || len(call.Args) == 0 || !match(call.Args[len(call.Args)-1], w.args) {
				return true
			}

			pkgpath, name, kind := printfNameAndKind(pkg, call.Fun)
			if kind != 0 {
				checkPrintfFwd(pkg, w, call, kind)
				return true
			}

			// If the call is to another function in this package,
			// maybe we will find out it is printf-like later.
			// Remember this call for later checking.
			if pkgpath == "" && byName[name] != nil {
				callee := byName[name]
				callee.callers = append(callee.callers, printfCaller{w, call})
			}

			return true
		})
	}
}

func match(arg ast.Expr, param *ast.Field) bool {
	id, ok := arg.(*ast.Ident)
	return ok && id.Obj != nil && id.Obj.Decl == param
}

const (
	kindPrintf = 1
	kindPrint  = 2
)

// printfLike reports whether a call to fn should be considered a call to a printf-like function.
// It returns 0 (indicating not a printf-like function), kindPrintf, or kindPrint.
func printfLike(pkg *Package, fn ast.Expr, byName map[string]*printfWrapper) int {
	if id, ok := fn.(*ast.Ident); ok && id.Obj != nil {
		if w := byName[id.Name]; w != nil && id.Obj.Decl == w.fn {
			// Found call to function in same package.
			return localPrintfLike[id.Name]
		}
	}
	if sel, ok := fn.(*ast.SelectorExpr); ok {
		if id, ok := sel.X.(*ast.Ident); ok && id.Name == "fmt" && strings.Contains(sel.Sel.Name, "rint") {
			if strings.HasSuffix(sel.Sel.Name, "f") {
				return kindPrintf
			}
			return kindPrint
		}
	}
	return 0
}

// checkPrintfFwd checks that a printf-forwarding wrapper is forwarding correctly.
// It diagnoses writing fmt.Printf(format, args) instead of fmt.Printf(format, args...).
func checkPrintfFwd(pkg *Package, w *printfWrapper, call *ast.CallExpr, kind int) {
	matched := kind == kindPrint ||
		kind == kindPrintf && len(call.Args) >= 2 && match(call.Args[len(call.Args)-2], w.format)
	if !matched {
		return
	}

	if !call.Ellipsis.IsValid() {
		typ, ok := pkg.types[call.Fun].Type.(*types.Signature)
		if !ok {
			return
		}
		if len(call.Args) > typ.Params().Len() {
			// If we're passing more arguments than what the
			// print/printf function can take, adding an ellipsis
			// would break the program. For example:
			//
			//   func foo(arg1 string, arg2 ...interface{} {
			//       fmt.Printf("%s %v", arg1, arg2)
			//   }
			return
		}
		if !vcfg.VetxOnly {
			desc := "printf"
			if kind == kindPrint {
				desc = "print"
			}
			pkg.files[0].Badf(call.Pos(), "missing ... in args forwarded to %s-like function", desc)
		}
		return
	}
	name := w.name
	if localPrintfLike[name] == 0 {
		localPrintfLike[name] = kind
		for _, caller := range w.callers {
			checkPrintfFwd(pkg, caller.w, caller.call, kind)
		}
	}
}

func exportPrintfLike() interface{} {
	out := make([]printfExport, 0, len(localPrintfLike))
	for name, kind := range localPrintfLike {
		out = append(out, printfExport{
			Name: name,
			Kind: kind,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})
	return out
}

// isPrint records the print functions.
// If a key ends in 'f' then it is assumed to be a formatted print.
var isPrint = map[string]bool{
	"fmt.Errorf":   true,
	"fmt.Fprint":   true,
	"fmt.Fprintf":  true,
	"fmt.Fprintln": true,
	"fmt.Print":    true,
	"fmt.Printf":   true,
	"fmt.Println":  true,
	"fmt.Sprint":   true,
	"fmt.Sprintf":  true,
	"fmt.Sprintln": true,

	// testing.B, testing.T not auto-detected
	// because the methods are picked up by embedding.
	"testing.B.Error":  true,
	"testing.B.Errorf": true,
	"testing.B.Fatal":  true,
	"testing.B.Fatalf": true,
	"testing.B.Log":    true,
	"testing.B.Logf":   true,
	"testing.B.Skip":   true,
	"testing.B.Skipf":  true,
	"testing.T.Error":  true,
	"testing.T.Errorf": true,
	"testing.T.Fatal":  true,
	"testing.T.Fatalf": true,
	"testing.T.Log":    true,
	"testing.T.Logf":   true,
	"testing.T.Skip":   true,
	"testing.T.Skipf":  true,

	// testing.TB is an interface, so can't detect wrapping.
	"testing.TB.Error":  true,
	"testing.TB.Errorf": true,
	"testing.TB.Fatal":  true,
	"testing.TB.Fatalf": true,
	"testing.TB.Log":    true,
	"testing.TB.Logf":   true,
	"testing.TB.Skip":   true,
	"testing.TB.Skipf":  true,
}

// formatString returns the format string argument and its index within
// the given printf-like call expression.
//
// The last parameter before variadic arguments is assumed to be
// a format string.
//
// The first string literal or string constant is assumed to be a format string
// if the call's signature cannot be determined.
//
// If it cannot find any format string parameter, it returns ("", -1).
func formatString(f *File, call *ast.CallExpr) (format string, idx int) {
	typ := f.pkg.types[call.Fun].Type
	if typ != nil {
		if sig, ok := typ.(*types.Signature); ok {
			if !sig.Variadic() {
				// Skip checking non-variadic functions.
				return "", -1
			}
			idx := sig.Params().Len() - 2
			if idx < 0 {
				// Skip checking variadic functions without
				// fixed arguments.
				return "", -1
			}
			s, ok := stringConstantArg(f, call, idx)
			if !ok {
				// The last argument before variadic args isn't a string.
				return "", -1
			}
			return s, idx
		}
	}

	// Cannot determine call's signature. Fall back to scanning for the first
	// string constant in the call.
	for idx := range call.Args {
		if s, ok := stringConstantArg(f, call, idx); ok {
			return s, idx
		}
		if f.pkg.types[call.Args[idx]].Type == types.Typ[types.String] {
			// Skip checking a call with a non-constant format
			// string argument, since its contents are unavailable
			// for validation.
			return "", -1
		}
	}
	return "", -1
}

// stringConstantArg returns call's string constant argument at the index idx.
//
// ("", false) is returned if call's argument at the index idx isn't a string
// constant.
func stringConstantArg(f *File, call *ast.CallExpr, idx int) (string, bool) {
	if idx >= len(call.Args) {
		return "", false
	}
	arg := call.Args[idx]
	lit := f.pkg.types[arg].Value
	if lit != nil && lit.Kind() == constant.String {
		return constant.StringVal(lit), true
	}
	return "", false
}

// checkCall triggers the print-specific checks if the call invokes a print function.
func checkFmtPrintfCall(f *File, node ast.Node) {
	if f.pkg.typesPkg == nil {
		// This check now requires type information.
		return
	}

	if d, ok := node.(*ast.FuncDecl); ok && isStringer(f, d) {
		// Remember we saw this.
		if f.stringerPtrs == nil {
			f.stringerPtrs = make(map[*ast.Object]bool)
		}
		if l := d.Recv.List; len(l) == 1 {
			if n := l[0].Names; len(n) == 1 {
				typ := f.pkg.types[l[0].Type]
				_, ptrRecv := typ.Type.(*types.Pointer)
				f.stringerPtrs[n[0].Obj] = ptrRecv
			}
		}
		return
	}

	call, ok := node.(*ast.CallExpr)
	if !ok {
		return
	}

	// Construct name like pkg.Printf or pkg.Type.Printf for lookup.
	_, name, kind := printfNameAndKind(f.pkg, call.Fun)
	if kind == kindPrintf {
		f.checkPrintf(call, name)
	}
	if kind == kindPrint {
		f.checkPrint(call, name)
	}
}

func printfName(pkg *Package, called ast.Expr) (pkgpath, name string) {
	switch x := called.(type) {
	case *ast.Ident:
		if fn, ok := pkg.uses[x].(*types.Func); ok {
			if fn.Pkg() == nil || fn.Pkg() == pkg.typesPkg {
				pkgpath = ""
			} else {
				pkgpath = fn.Pkg().Path()
			}
			return pkgpath, x.Name
		}

	case *ast.SelectorExpr:
		// Check for "fmt.Printf".
		if id, ok := x.X.(*ast.Ident); ok {
			if pkgName, ok := pkg.uses[id].(*types.PkgName); ok {
				return pkgName.Imported().Path(), x.Sel.Name
			}
		}

		// Check for t.Logf where t is a *testing.T.
		if sel := pkg.selectors[x]; sel != nil {
			recv := sel.Recv()
			if p, ok := recv.(*types.Pointer); ok {
				recv = p.Elem()
			}
			if named, ok := recv.(*types.Named); ok {
				obj := named.Obj()
				if obj.Pkg() == nil || obj.Pkg() == pkg.typesPkg {
					pkgpath = ""
				} else {
					pkgpath = obj.Pkg().Path()
				}
				return pkgpath, obj.Name() + "." + x.Sel.Name
			}
		}
	}
	return "", ""
}

func printfNameAndKind(pkg *Package, called ast.Expr) (pkgpath, name string, kind int) {
	pkgpath, name = printfName(pkg, called)
	if name == "" {
		return pkgpath, name, 0
	}

	if pkgpath == "" {
		kind = localPrintfLike[name]
	} else if m, ok := printfImported[pkgpath]; ok {
		kind = m[name]
	} else {
		var m map[string]int
		if out, ok := readVetx(pkgpath, "printf").([]printfExport); ok {
			m = make(map[string]int)
			for _, x := range out {
				m[x.Name] = x.Kind
			}
		}
		printfImported[pkgpath] = m
		kind = m[name]
	}

	if kind == 0 {
		_, ok := isPrint[pkgpath+"."+name]
		if !ok {
			// Next look up just "printf", for use with -printfuncs.
			short := name[strings.LastIndex(name, ".")+1:]
			_, ok = isPrint[strings.ToLower(short)]
		}
		if ok {
			if strings.HasSuffix(name, "f") {
				kind = kindPrintf
			} else {
				kind = kindPrint
			}
		}
	}
	return pkgpath, name, kind
}

// isStringer returns true if the provided declaration is a "String() string"
// method, an implementation of fmt.Stringer.
func isStringer(f *File, d *ast.FuncDecl) bool {
	return d.Recv != nil && d.Name.Name == "String" && d.Type.Results != nil &&
		len(d.Type.Params.List) == 0 && len(d.Type.Results.List) == 1 &&
		f.pkg.types[d.Type.Results.List[0].Type].Type == types.Typ[types.String]
}

// isFormatter reports whether t satisfies fmt.Formatter.
// Unlike fmt.Stringer, it's impossible to satisfy fmt.Formatter without importing fmt.
func (f *File) isFormatter(t types.Type) bool {
	return formatterType != nil && types.Implements(t, formatterType)
}

// formatState holds the parsed representation of a printf directive such as "%3.*[4]d".
// It is constructed by parsePrintfVerb.
type formatState struct {
	verb     rune   // the format verb: 'd' for "%d"
	format   string // the full format directive from % through verb, "%.3d".
	name     string // Printf, Sprintf etc.
	flags    []byte // the list of # + etc.
	argNums  []int  // the successive argument numbers that are consumed, adjusted to refer to actual arg in call
	firstArg int    // Index of first argument after the format in the Printf call.
	// Used only during parse.
	file         *File
	call         *ast.CallExpr
	argNum       int  // Which argument we're expecting to format now.
	hasIndex     bool // Whether the argument is indexed.
	indexPending bool // Whether we have an indexed argument that has not resolved.
	nbytes       int  // number of bytes of the format string consumed.
}

// checkPrintf checks a call to a formatted print routine such as Printf.
func (f *File) checkPrintf(call *ast.CallExpr, name string) {
	format, idx := formatString(f, call)
	if idx < 0 {
		if *verbose {
			f.Warn(call.Pos(), "can't check non-constant format in call to", name)
		}
		return
	}

	firstArg := idx + 1 // Arguments are immediately after format string.
	if !strings.Contains(format, "%") {
		if len(call.Args) > firstArg {
			f.Badf(call.Pos(), "%s call has arguments but no formatting directives", name)
		}
		return
	}
	// Hard part: check formats against args.
	argNum := firstArg
	maxArgNum := firstArg
	anyIndex := false
	for i, w := 0, 0; i < len(format); i += w {
		w = 1
		if format[i] != '%' {
			continue
		}
		state := f.parsePrintfVerb(call, name, format[i:], firstArg, argNum)
		if state == nil {
			return
		}
		w = len(state.format)
		if !f.okPrintfArg(call, state) { // One error per format is enough.
			return
		}
		if state.hasIndex {
			anyIndex = true
		}
		if len(state.argNums) > 0 {
			// Continue with the next sequential argument.
			argNum = state.argNums[len(state.argNums)-1] + 1
		}
		for _, n := range state.argNums {
			if n >= maxArgNum {
				maxArgNum = n + 1
			}
		}
	}
	// Dotdotdot is hard.
	if call.Ellipsis.IsValid() && maxArgNum >= len(call.Args)-1 {
		return
	}
	// If any formats are indexed, extra arguments are ignored.
	if anyIndex {
		return
	}
	// There should be no leftover arguments.
	if maxArgNum != len(call.Args) {
		expect := maxArgNum - firstArg
		numArgs := len(call.Args) - firstArg
		f.Badf(call.Pos(), "%s call needs %v but has %v", name, count(expect, "arg"), count(numArgs, "arg"))
	}
}

// parseFlags accepts any printf flags.
func (s *formatState) parseFlags() {
	for s.nbytes < len(s.format) {
		switch c := s.format[s.nbytes]; c {
		case '#', '0', '+', '-', ' ':
			s.flags = append(s.flags, c)
			s.nbytes++
		default:
			return
		}
	}
}

// scanNum advances through a decimal number if present.
func (s *formatState) scanNum() {
	for ; s.nbytes < len(s.format); s.nbytes++ {
		c := s.format[s.nbytes]
		if c < '0' || '9' < c {
			return
		}
	}
}

// parseIndex scans an index expression. It returns false if there is a syntax error.
func (s *formatState) parseIndex() bool {
	if s.nbytes == len(s.format) || s.format[s.nbytes] != '[' {
		return true
	}
	// Argument index present.
	s.nbytes++ // skip '['
	start := s.nbytes
	s.scanNum()
	ok := true
	if s.nbytes == len(s.format) || s.nbytes == start || s.format[s.nbytes] != ']' {
		ok = false
		s.nbytes = strings.Index(s.format, "]")
		if s.nbytes < 0 {
			s.file.Badf(s.call.Pos(), "%s format %s is missing closing ]", s.name, s.format)
			return false
		}
	}
	arg32, err := strconv.ParseInt(s.format[start:s.nbytes], 10, 32)
	if err != nil || !ok || arg32 <= 0 || arg32 > int64(len(s.call.Args)-s.firstArg) {
		s.file.Badf(s.call.Pos(), "%s format has invalid argument index [%s]", s.name, s.format[start:s.nbytes])
		return false
	}
	s.nbytes++ // skip ']'
	arg := int(arg32)
	arg += s.firstArg - 1 // We want to zero-index the actual arguments.
	s.argNum = arg
	s.hasIndex = true
	s.indexPending = true
	return true
}

// parseNum scans a width or precision (or *). It returns false if there's a bad index expression.
func (s *formatState) parseNum() bool {
	if s.nbytes < len(s.format) && s.format[s.nbytes] == '*' {
		if s.indexPending { // Absorb it.
			s.indexPending = false
		}
		s.nbytes++
		s.argNums = append(s.argNums, s.argNum)
		s.argNum++
	} else {
		s.scanNum()
	}
	return true
}

// parsePrecision scans for a precision. It returns false if there's a bad index expression.
func (s *formatState) parsePrecision() bool {
	// If there's a period, there may be a precision.
	if s.nbytes < len(s.format) && s.format[s.nbytes] == '.' {
		s.flags = append(s.flags, '.') // Treat precision as a flag.
		s.nbytes++
		if !s.parseIndex() {
			return false
		}
		if !s.parseNum() {
			return false
		}
	}
	return true
}

// parsePrintfVerb looks the formatting directive that begins the format string
// and returns a formatState that encodes what the directive wants, without looking
// at the actual arguments present in the call. The result is nil if there is an error.
func (f *File) parsePrintfVerb(call *ast.CallExpr, name, format string, firstArg, argNum int) *formatState {
	state := &formatState{
		format:   format,
		name:     name,
		flags:    make([]byte, 0, 5),
		argNum:   argNum,
		argNums:  make([]int, 0, 1),
		nbytes:   1, // There's guaranteed to be a percent sign.
		firstArg: firstArg,
		file:     f,
		call:     call,
	}
	// There may be flags.
	state.parseFlags()
	// There may be an index.
	if !state.parseIndex() {
		return nil
	}
	// There may be a width.
	if !state.parseNum() {
		return nil
	}
	// There may be a precision.
	if !state.parsePrecision() {
		return nil
	}
	// Now a verb, possibly prefixed by an index (which we may already have).
	if !state.indexPending && !state.parseIndex() {
		return nil
	}
	if state.nbytes == len(state.format) {
		f.Badf(call.Pos(), "%s format %s is missing verb at end of string", name, state.format)
		return nil
	}
	verb, w := utf8.DecodeRuneInString(state.format[state.nbytes:])
	state.verb = verb
	state.nbytes += w
	if verb != '%' {
		state.argNums = append(state.argNums, state.argNum)
	}
	state.format = state.format[:state.nbytes]
	return state
}

// printfArgType encodes the types of expressions a printf verb accepts. It is a bitmask.
type printfArgType int

const (
	argBool printfArgType = 1 << iota
	argInt
	argRune
	argString
	argFloat
	argComplex
	argPointer
	anyType printfArgType = ^0
)

type printVerb struct {
	verb  rune   // User may provide verb through Formatter; could be a rune.
	flags string // known flags are all ASCII
	typ   printfArgType
}

// Common flag sets for printf verbs.
const (
	noFlag       = ""
	numFlag      = " -+.0"
	sharpNumFlag = " -+.0#"
	allFlags     = " -+.0#"
)

// printVerbs identifies which flags are known to printf for each verb.
var printVerbs = []printVerb{
	// '-' is a width modifier, always valid.
	// '.' is a precision for float, max width for strings.
	// '+' is required sign for numbers, Go format for %v.
	// '#' is alternate format for several verbs.
	// ' ' is spacer for numbers
	{'%', noFlag, 0},
	{'b', numFlag, argInt | argFloat | argComplex},
	{'c', "-", argRune | argInt},
	{'d', numFlag, argInt | argPointer},
	{'e', sharpNumFlag, argFloat | argComplex},
	{'E', sharpNumFlag, argFloat | argComplex},
	{'f', sharpNumFlag, argFloat | argComplex},
	{'F', sharpNumFlag, argFloat | argComplex},
	{'g', sharpNumFlag, argFloat | argComplex},
	{'G', sharpNumFlag, argFloat | argComplex},
	{'o', sharpNumFlag, argInt},
	{'p', "-#", argPointer},
	{'q', " -+.0#", argRune | argInt | argString},
	{'s', " -+.0", argString},
	{'t', "-", argBool},
	{'T', "-", anyType},
	{'U', "-#", argRune | argInt},
	{'v', allFlags, anyType},
	{'x', sharpNumFlag, argRune | argInt | argString | argPointer},
	{'X', sharpNumFlag, argRune | argInt | argString | argPointer},
}

// okPrintfArg compares the formatState to the arguments actually present,
// reporting any discrepancies it can discern. If the final argument is ellipsissed,
// there's little it can do for that.
func (f *File) okPrintfArg(call *ast.CallExpr, state *formatState) (ok bool) {
	var v printVerb
	found := false
	// Linear scan is fast enough for a small list.
	for _, v = range printVerbs {
		if v.verb == state.verb {
			found = true
			break
		}
	}

	// Does current arg implement fmt.Formatter?
	formatter := false
	if state.argNum < len(call.Args) {
		if tv, ok := f.pkg.types[call.Args[state.argNum]]; ok {
			formatter = f.isFormatter(tv.Type)
		}
	}

	if !formatter {
		if !found {
			f.Badf(call.Pos(), "%s format %s has unknown verb %c", state.name, state.format, state.verb)
			return false
		}
		for _, flag := range state.flags {
			// TODO: Disable complaint about '0' for Go 1.10. To be fixed properly in 1.11.
			// See issues 23598 and 23605.
			if flag == '0' {
				continue
			}
			if !strings.ContainsRune(v.flags, rune(flag)) {
				f.Badf(call.Pos(), "%s format %s has unrecognized flag %c", state.name, state.format, flag)
				return false
			}
		}
	}
	// Verb is good. If len(state.argNums)>trueArgs, we have something like %.*s and all
	// but the final arg must be an integer.
	trueArgs := 1
	if state.verb == '%' {
		trueArgs = 0
	}
	nargs := len(state.argNums)
	for i := 0; i < nargs-trueArgs; i++ {
		argNum := state.argNums[i]
		if !f.argCanBeChecked(call, i, state) {
			return
		}
		arg := call.Args[argNum]
		if !f.matchArgType(argInt, nil, arg) {
			f.Badf(call.Pos(), "%s format %s uses non-int %s as argument of *", state.name, state.format, f.gofmt(arg))
			return false
		}
	}
	if state.verb == '%' || formatter {
		return true
	}
	argNum := state.argNums[len(state.argNums)-1]
	if !f.argCanBeChecked(call, len(state.argNums)-1, state) {
		return false
	}
	arg := call.Args[argNum]
	if f.isFunctionValue(arg) && state.verb != 'p' && state.verb != 'T' {
		f.Badf(call.Pos(), "%s format %s arg %s is a func value, not called", state.name, state.format, f.gofmt(arg))
		return false
	}
	if !f.matchArgType(v.typ, nil, arg) {
		typeString := ""
		if typ := f.pkg.types[arg].Type; typ != nil {
			typeString = typ.String()
		}
		f.Badf(call.Pos(), "%s format %s has arg %s of wrong type %s", state.name, state.format, f.gofmt(arg), typeString)
		return false
	}
	if v.typ&argString != 0 && v.verb != 'T' && !bytes.Contains(state.flags, []byte{'#'}) && f.recursiveStringer(arg) {
		f.Badf(call.Pos(), "%s format %s with arg %s causes recursive String method call", state.name, state.format, f.gofmt(arg))
		return false
	}
	return true
}

// recursiveStringer reports whether the provided argument is r or &r for the
// fmt.Stringer receiver identifier r.
func (f *File) recursiveStringer(e ast.Expr) bool {
	if len(f.stringerPtrs) == 0 {
		return false
	}
	ptr := false
	var obj *ast.Object
	switch e := e.(type) {
	case *ast.Ident:
		obj = e.Obj
	case *ast.UnaryExpr:
		if id, ok := e.X.(*ast.Ident); ok && e.Op == token.AND {
			obj = id.Obj
			ptr = true
		}
	}

	// It's unlikely to be a recursive stringer if it has a Format method.
	if typ := f.pkg.types[e].Type; typ != nil {
		if f.isFormatter(typ) {
			return false
		}
	}

	// We compare the underlying Object, which checks that the identifier
	// is the one we declared as the receiver for the String method in
	// which this printf appears.
	ptrRecv, exist := f.stringerPtrs[obj]
	if !exist {
		return false
	}
	// We also need to check that using &t when we declared String
	// on (t *T) is ok; in such a case, the address is printed.
	if ptr && ptrRecv {
		return false
	}
	return true
}

// isFunctionValue reports whether the expression is a function as opposed to a function call.
// It is almost always a mistake to print a function value.
func (f *File) isFunctionValue(e ast.Expr) bool {
	if typ := f.pkg.types[e].Type; typ != nil {
		_, ok := typ.(*types.Signature)
		return ok
	}
	return false
}

// argCanBeChecked reports whether the specified argument is statically present;
// it may be beyond the list of arguments or in a terminal slice... argument, which
// means we can't see it.
func (f *File) argCanBeChecked(call *ast.CallExpr, formatArg int, state *formatState) bool {
	argNum := state.argNums[formatArg]
	if argNum <= 0 {
		// Shouldn't happen, so catch it with prejudice.
		panic("negative arg num")
	}
	if argNum < len(call.Args)-1 {
		return true // Always OK.
	}
	if call.Ellipsis.IsValid() {
		return false // We just can't tell; there could be many more arguments.
	}
	if argNum < len(call.Args) {
		return true
	}
	// There are bad indexes in the format or there are fewer arguments than the format needs.
	// This is the argument number relative to the format: Printf("%s", "hi") will give 1 for the "hi".
	arg := argNum - state.firstArg + 1 // People think of arguments as 1-indexed.
	f.Badf(call.Pos(), "%s format %s reads arg #%d, but call has %v", state.name, state.format, arg, count(len(call.Args)-state.firstArg, "arg"))
	return false
}

// printFormatRE is the regexp we match and report as a possible format string
// in the first argument to unformatted prints like fmt.Print.
// We exclude the space flag, so that printing a string like "x % y" is not reported as a format.
var printFormatRE = regexp.MustCompile(`%` + flagsRE + numOptRE + `\.?` + numOptRE + indexOptRE + verbRE)

const (
	flagsRE    = `[+\-#]*`
	indexOptRE = `(\[[0-9]+\])?`
	numOptRE   = `([0-9]+|` + indexOptRE + `\*)?`
	verbRE     = `[bcdefgopqstvxEFGTUX]`
)

// checkPrint checks a call to an unformatted print routine such as Println.
func (f *File) checkPrint(call *ast.CallExpr, name string) {
	firstArg := 0
	typ := f.pkg.types[call.Fun].Type
	if typ == nil {
		// Skip checking functions with unknown type.
		return
	}
	if sig, ok := typ.(*types.Signature); ok {
		if !sig.Variadic() {
			// Skip checking non-variadic functions.
			return
		}
		params := sig.Params()
		firstArg = params.Len() - 1

		typ := params.At(firstArg).Type()
		typ = typ.(*types.Slice).Elem()
		it, ok := typ.(*types.Interface)
		if !ok || !it.Empty() {
			// Skip variadic functions accepting non-interface{} args.
			return
		}
	}
	args := call.Args
	if len(args) <= firstArg {
		// Skip calls without variadic args.
		return
	}
	args = args[firstArg:]

	if firstArg == 0 {
		if sel, ok := call.Args[0].(*ast.SelectorExpr); ok {
			if x, ok := sel.X.(*ast.Ident); ok {
				if x.Name == "os" && strings.HasPrefix(sel.Sel.Name, "Std") {
					f.Badf(call.Pos(), "%s does not take io.Writer but has first arg %s", name, f.gofmt(call.Args[0]))
				}
			}
		}
	}

	arg := args[0]
	if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
		// Ignore trailing % character in lit.Value.
		// The % in "abc 0.0%" couldn't be a formatting directive.
		s := strings.TrimSuffix(lit.Value, `%"`)
		if strings.Contains(s, "%") {
			m := printFormatRE.FindStringSubmatch(s)
			if m != nil {
				f.Badf(call.Pos(), "%s call has possible formatting directive %s", name, m[0])
			}
		}
	}
	if strings.HasSuffix(name, "ln") {
		// The last item, if a string, should not have a newline.
		arg = args[len(args)-1]
		if lit, ok := arg.(*ast.BasicLit); ok && lit.Kind == token.STRING {
			str, _ := strconv.Unquote(lit.Value)
			if strings.HasSuffix(str, "\n") {
				f.Badf(call.Pos(), "%s arg list ends with redundant newline", name)
			}
		}
	}
	for _, arg := range args {
		if f.isFunctionValue(arg) {
			f.Badf(call.Pos(), "%s arg %s is a func value, not called", name, f.gofmt(arg))
		}
		if f.recursiveStringer(arg) {
			f.Badf(call.Pos(), "%s arg %s causes recursive call to String method", name, f.gofmt(arg))
		}
	}
}

// count(n, what) returns "1 what" or "N whats"
// (assuming the plural of what is whats).
func count(n int, what string) string {
	if n == 1 {
		return "1 " + what
	}
	return fmt.Sprintf("%d %ss", n, what)
}
