// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Annotate Ref in Prog with C types by parsing gcc debug output.
// Conversion of debug output to Go types.

package main

import (
	"bytes"
	"debug/dwarf"
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

var debugDefine = flag.Bool("debug-define", false, "print relevant #defines")
var debugGcc = flag.Bool("debug-gcc", false, "print gcc invocations")

var nameToC = map[string]string{
	"schar":         "signed char",
	"uchar":         "unsigned char",
	"ushort":        "unsigned short",
	"uint":          "unsigned int",
	"ulong":         "unsigned long",
	"longlong":      "long long",
	"ulonglong":     "unsigned long long",
	"complexfloat":  "float complex",
	"complexdouble": "double complex",
}

// cname returns the C name to use for C.s.
// The expansions are listed in nameToC and also
// struct_foo becomes "struct foo", and similarly for
// union and enum.
func cname(s string) string {
	if t, ok := nameToC[s]; ok {
		return t
	}

	if strings.HasPrefix(s, "struct_") {
		return "struct " + s[len("struct_"):]
	}
	if strings.HasPrefix(s, "union_") {
		return "union " + s[len("union_"):]
	}
	if strings.HasPrefix(s, "enum_") {
		return "enum " + s[len("enum_"):]
	}
	if strings.HasPrefix(s, "sizeof_") {
		return "sizeof(" + cname(s[len("sizeof_"):]) + ")"
	}
	return s
}

// DiscardCgoDirectives processes the import C preamble, and discards
// all #cgo CFLAGS and LDFLAGS directives, so they don't make their
// way into _cgo_export.h.
func (f *File) DiscardCgoDirectives() {
	linesIn := strings.Split(f.Preamble, "\n")
	linesOut := make([]string, 0, len(linesIn))
	for _, line := range linesIn {
		l := strings.TrimSpace(line)
		if len(l) < 5 || l[:4] != "#cgo" || !unicode.IsSpace(rune(l[4])) {
			linesOut = append(linesOut, line)
		} else {
			linesOut = append(linesOut, "")
		}
	}
	f.Preamble = strings.Join(linesOut, "\n")
}

// addToFlag appends args to flag.  All flags are later written out onto the
// _cgo_flags file for the build system to use.
func (p *Package) addToFlag(flag string, args []string) {
	p.CgoFlags[flag] = append(p.CgoFlags[flag], args...)
	if flag == "CFLAGS" {
		// We'll also need these when preprocessing for dwarf information.
		p.GccOptions = append(p.GccOptions, args...)
	}
}

// splitQuoted splits the string s around each instance of one or more consecutive
// white space characters while taking into account quotes and escaping, and
// returns an array of substrings of s or an empty list if s contains only white space.
// Single quotes and double quotes are recognized to prevent splitting within the
// quoted region, and are removed from the resulting substrings. If a quote in s
// isn't closed err will be set and r will have the unclosed argument as the
// last element.  The backslash is used for escaping.
//
// For example, the following string:
//
//     `a b:"c d" 'e''f'  "g\""`
//
// Would be parsed as:
//
//     []string{"a", "b:c d", "ef", `g"`}
//
func splitQuoted(s string) (r []string, err error) {
	var args []string
	arg := make([]rune, len(s))
	escaped := false
	quoted := false
	quote := '\x00'
	i := 0
	for _, r := range s {
		switch {
		case escaped:
			escaped = false
		case r == '\\':
			escaped = true
			continue
		case quote != 0:
			if r == quote {
				quote = 0
				continue
			}
		case r == '"' || r == '\'':
			quoted = true
			quote = r
			continue
		case unicode.IsSpace(r):
			if quoted || i > 0 {
				quoted = false
				args = append(args, string(arg[:i]))
				i = 0
			}
			continue
		}
		arg[i] = r
		i++
	}
	if quoted || i > 0 {
		args = append(args, string(arg[:i]))
	}
	if quote != 0 {
		err = errors.New("unclosed quote")
	} else if escaped {
		err = errors.New("unfinished escaping")
	}
	return args, err
}

// Translate rewrites f.AST, the original Go input, to remove
// references to the imported package C, replacing them with
// references to the equivalent Go types, functions, and variables.
func (p *Package) Translate(f *File) {
	for _, cref := range f.Ref {
		// Convert C.ulong to C.unsigned long, etc.
		cref.Name.C = cname(cref.Name.Go)
	}
	p.loadDefines(f)
	needType := p.guessKinds(f)
	if len(needType) > 0 {
		p.loadDWARF(f, needType)
	}
	p.rewriteRef(f)
}

// loadDefines coerces gcc into spitting out the #defines in use
// in the file f and saves relevant renamings in f.Name[name].Define.
func (p *Package) loadDefines(f *File) {
	var b bytes.Buffer
	b.WriteString(f.Preamble)
	b.WriteString(builtinProlog)
	stdout := p.gccDefines(b.Bytes())

	for _, line := range strings.Split(stdout, "\n") {
		if len(line) < 9 || line[0:7] != "#define" {
			continue
		}

		line = strings.TrimSpace(line[8:])

		var key, val string
		spaceIndex := strings.Index(line, " ")
		tabIndex := strings.Index(line, "\t")

		if spaceIndex == -1 && tabIndex == -1 {
			continue
		} else if tabIndex == -1 || (spaceIndex != -1 && spaceIndex < tabIndex) {
			key = line[0:spaceIndex]
			val = strings.TrimSpace(line[spaceIndex:])
		} else {
			key = line[0:tabIndex]
			val = strings.TrimSpace(line[tabIndex:])
		}

		if n := f.Name[key]; n != nil {
			if *debugDefine {
				fmt.Fprintf(os.Stderr, "#define %s %s\n", key, val)
			}
			n.Define = val
		}
	}
}

// guessKinds tricks gcc into revealing the kind of each
// name xxx for the references C.xxx in the Go input.
// The kind is either a constant, type, or variable.
func (p *Package) guessKinds(f *File) []*Name {
	// Determine kinds for names we already know about,
	// like #defines or 'struct foo', before bothering with gcc.
	var names, needType []*Name
	for _, key := range nameKeys(f.Name) {
		n := f.Name[key]
		// If we've already found this name as a #define
		// and we can translate it as a constant value, do so.
		if n.Define != "" {
			isConst := false
			if _, err := strconv.Atoi(n.Define); err == nil {
				isConst = true
			} else if n.Define[0] == '"' || n.Define[0] == '\'' {
				if _, err := parser.ParseExpr(n.Define); err == nil {
					isConst = true
				}
			}
			if isConst {
				n.Kind = "const"
				// Turn decimal into hex, just for consistency
				// with enum-derived constants.  Otherwise
				// in the cgo -godefs output half the constants
				// are in hex and half are in whatever the #define used.
				i, err := strconv.ParseInt(n.Define, 0, 64)
				if err == nil {
					n.Const = fmt.Sprintf("%#x", i)
				} else {
					n.Const = n.Define
				}
				continue
			}

			if isName(n.Define) {
				n.C = n.Define
			}
		}

		needType = append(needType, n)

		// If this is a struct, union, or enum type name, no need to guess the kind.
		if strings.HasPrefix(n.C, "struct ") || strings.HasPrefix(n.C, "union ") || strings.HasPrefix(n.C, "enum ") {
			n.Kind = "type"
			continue
		}

		// Otherwise, we'll need to find out from gcc.
		names = append(names, n)
	}

	// Bypass gcc if there's nothing left to find out.
	if len(names) == 0 {
		return needType
	}

	// Coerce gcc into telling us whether each name is a type, a value, or undeclared.
	// For names, find out whether they are integer constants.
	// We used to look at specific warning or error messages here, but that tied the
	// behavior too closely to specific versions of the compilers.
	// Instead, arrange that we can infer what we need from only the presence or absence
	// of an error on a specific line.
	//
	// For each name, we generate these lines, where xxx is the index in toSniff plus one.
	//
	//	#line xxx "not-declared"
	//	void __cgo_f_xxx_1(void) { __typeof__(name) *__cgo_undefined__; }
	//	#line xxx "not-type"
	//	void __cgo_f_xxx_2(void) { name *__cgo_undefined__; }
	//	#line xxx "not-const"
	//	void __cgo_f_xxx_3(void) { enum { __cgo_undefined__ = (name)*1 }; }
	//
	// If we see an error at not-declared:xxx, the corresponding name is not declared.
	// If we see an error at not-type:xxx, the corresponding name is a type.
	// If we see an error at not-const:xxx, the corresponding name is not an integer constant.
	// If we see no errors, we assume the name is an expression but not a constant
	// (so a variable or a function).
	//
	// The specific input forms are chosen so that they are valid C syntax regardless of
	// whether name denotes a type or an expression.

	var b bytes.Buffer
	b.WriteString(f.Preamble)
	b.WriteString(builtinProlog)

	for i, n := range names {
		fmt.Fprintf(&b, "#line %d \"not-declared\"\n"+
			"void __cgo_f_%d_1(void) { __typeof__(%s) *__cgo_undefined__; }\n"+
			"#line %d \"not-type\"\n"+
			"void __cgo_f_%d_2(void) { %s *__cgo_undefined__; }\n"+
			"#line %d \"not-const\"\n"+
			"void __cgo_f_%d_3(void) { enum { __cgo__undefined__ = (%s)*1 }; }\n",
			i+1, i+1, n.C,
			i+1, i+1, n.C,
			i+1, i+1, n.C)
	}
	fmt.Fprintf(&b, "#line 1 \"completed\"\n"+
		"int __cgo__1 = __cgo__2;\n")

	stderr := p.gccErrors(b.Bytes())
	if stderr == "" {
		fatalf("%s produced no output\non input:\n%s", p.gccBaseCmd()[0], b.Bytes())
	}

	completed := false
	sniff := make([]int, len(names))
	const (
		notType = 1 << iota
		notConst
		notDeclared
	)
	for _, line := range strings.Split(stderr, "\n") {
		if !strings.Contains(line, ": error:") {
			// we only care about errors.
			// we tried to turn off warnings on the command line, but one never knows.
			continue
		}

		c1 := strings.Index(line, ":")
		if c1 < 0 {
			continue
		}
		c2 := strings.Index(line[c1+1:], ":")
		if c2 < 0 {
			continue
		}
		c2 += c1 + 1

		filename := line[:c1]
		i, _ := strconv.Atoi(line[c1+1 : c2])
		i--
		if i < 0 || i >= len(names) {
			continue
		}

		switch filename {
		case "completed":
			// Strictly speaking, there is no guarantee that seeing the error at completed:1
			// (at the end of the file) means we've seen all the errors from earlier in the file,
			// but usually it does. Certainly if we don't see the completed:1 error, we did
			// not get all the errors we expected.
			completed = true

		case "not-declared":
			sniff[i] |= notDeclared
		case "not-type":
			sniff[i] |= notType
		case "not-const":
			sniff[i] |= notConst
		}
	}

	if !completed {
		fatalf("%s did not produce error at completed:1\non input:\n%s\nfull error output:\n%s", p.gccBaseCmd()[0], b.Bytes(), stderr)
	}

	for i, n := range names {
		switch sniff[i] {
		default:
			error_(token.NoPos, "could not determine kind of name for C.%s", fixGo(n.Go))
		case notType:
			n.Kind = "const"
		case notConst:
			n.Kind = "type"
		case notConst | notType:
			n.Kind = "not-type"
		}
	}
	if nerrors > 0 {
		// Check if compiling the preamble by itself causes any errors,
		// because the messages we've printed out so far aren't helpful
		// to users debugging preamble mistakes.  See issue 8442.
		preambleErrors := p.gccErrors([]byte(f.Preamble))
		if len(preambleErrors) > 0 {
			error_(token.NoPos, "\n%s errors for preamble:\n%s", p.gccBaseCmd()[0], preambleErrors)
		}

		fatalf("unresolved names")
	}

	needType = append(needType, names...)
	return needType
}

// loadDWARF parses the DWARF debug information generated
// by gcc to learn the details of the constants, variables, and types
// being referred to as C.xxx.
func (p *Package) loadDWARF(f *File, names []*Name) {
	// Extract the types from the DWARF section of an object
	// from a well-formed C program.  Gcc only generates DWARF info
	// for symbols in the object file, so it is not enough to print the
	// preamble and hope the symbols we care about will be there.
	// Instead, emit
	//	__typeof__(names[i]) *__cgo__i;
	// for each entry in names and then dereference the type we
	// learn for __cgo__i.
	var b bytes.Buffer
	b.WriteString(f.Preamble)
	b.WriteString(builtinProlog)
	for i, n := range names {
		fmt.Fprintf(&b, "__typeof__(%s) *__cgo__%d;\n", n.C, i)
		if n.Kind == "const" {
			fmt.Fprintf(&b, "enum { __cgo_enum__%d = %s };\n", i, n.C)
		}
	}

	// Apple's LLVM-based gcc does not include the enumeration
	// names and values in its DWARF debug output.  In case we're
	// using such a gcc, create a data block initialized with the values.
	// We can read them out of the object file.
	fmt.Fprintf(&b, "long long __cgodebug_data[] = {\n")
	for _, n := range names {
		if n.Kind == "const" {
			fmt.Fprintf(&b, "\t%s,\n", n.C)
		} else {
			fmt.Fprintf(&b, "\t0,\n")
		}
	}
	// for the last entry, we can not use 0, otherwise
	// in case all __cgodebug_data is zero initialized,
	// LLVM-based gcc will place the it in the __DATA.__common
	// zero-filled section (our debug/macho doesn't support
	// this)
	fmt.Fprintf(&b, "\t1\n")
	fmt.Fprintf(&b, "};\n")

	d, bo, debugData := p.gccDebug(b.Bytes())
	enumVal := make([]int64, len(debugData)/8)
	for i := range enumVal {
		enumVal[i] = int64(bo.Uint64(debugData[i*8:]))
	}

	// Scan DWARF info for top-level TagVariable entries with AttrName __cgo__i.
	types := make([]dwarf.Type, len(names))
	enums := make([]dwarf.Offset, len(names))
	nameToIndex := make(map[*Name]int)
	for i, n := range names {
		nameToIndex[n] = i
	}
	nameToRef := make(map[*Name]*Ref)
	for _, ref := range f.Ref {
		nameToRef[ref.Name] = ref
	}
	r := d.Reader()
	for {
		e, err := r.Next()
		if err != nil {
			fatalf("reading DWARF entry: %s", err)
		}
		if e == nil {
			break
		}
		switch e.Tag {
		case dwarf.TagEnumerationType:
			offset := e.Offset
			for {
				e, err := r.Next()
				if err != nil {
					fatalf("reading DWARF entry: %s", err)
				}
				if e.Tag == 0 {
					break
				}
				if e.Tag == dwarf.TagEnumerator {
					entryName := e.Val(dwarf.AttrName).(string)
					if strings.HasPrefix(entryName, "__cgo_enum__") {
						n, _ := strconv.Atoi(entryName[len("__cgo_enum__"):])
						if 0 <= n && n < len(names) {
							enums[n] = offset
						}
					}
				}
			}
		case dwarf.TagVariable:
			name, _ := e.Val(dwarf.AttrName).(string)
			typOff, _ := e.Val(dwarf.AttrType).(dwarf.Offset)
			if name == "" || typOff == 0 {
				fatalf("malformed DWARF TagVariable entry")
			}
			if !strings.HasPrefix(name, "__cgo__") {
				break
			}
			typ, err := d.Type(typOff)
			if err != nil {
				fatalf("loading DWARF type: %s", err)
			}
			t, ok := typ.(*dwarf.PtrType)
			if !ok || t == nil {
				fatalf("internal error: %s has non-pointer type", name)
			}
			i, err := strconv.Atoi(name[7:])
			if err != nil {
				fatalf("malformed __cgo__ name: %s", name)
			}
			if enums[i] != 0 {
				t, err := d.Type(enums[i])
				if err != nil {
					fatalf("loading DWARF type: %s", err)
				}
				types[i] = t
			} else {
				types[i] = t.Type
			}
		}
		if e.Tag != dwarf.TagCompileUnit {
			r.SkipChildren()
		}
	}

	// Record types and typedef information.
	var conv typeConv
	conv.Init(p.PtrSize, p.IntSize)
	for i, n := range names {
		if types[i] == nil {
			continue
		}
		pos := token.NoPos
		if ref, ok := nameToRef[n]; ok {
			pos = ref.Pos()
		}
		f, fok := types[i].(*dwarf.FuncType)
		if n.Kind != "type" && fok {
			n.Kind = "func"
			n.FuncType = conv.FuncType(f, pos)
		} else {
			n.Type = conv.Type(types[i], pos)
			if enums[i] != 0 && n.Type.EnumValues != nil {
				k := fmt.Sprintf("__cgo_enum__%d", i)
				n.Kind = "const"
				n.Const = fmt.Sprintf("%#x", n.Type.EnumValues[k])
				// Remove injected enum to ensure the value will deep-compare
				// equally in future loads of the same constant.
				delete(n.Type.EnumValues, k)
			}
			// Prefer debug data over DWARF debug output, if we have it.
			if n.Kind == "const" && i < len(enumVal) {
				n.Const = fmt.Sprintf("%#x", enumVal[i])
			}
		}
		conv.FinishType(pos)
	}
}

// mangleName does name mangling to translate names
// from the original Go source files to the names
// used in the final Go files generated by cgo.
func (p *Package) mangleName(n *Name) {
	// When using gccgo variables have to be
	// exported so that they become global symbols
	// that the C code can refer to.
	prefix := "_C"
	if *gccgo && n.IsVar() {
		prefix = "C"
	}
	n.Mangle = prefix + n.Kind + "_" + n.Go
}

// rewriteRef rewrites all the C.xxx references in f.AST to refer to the
// Go equivalents, now that we have figured out the meaning of all
// the xxx.  In *godefs mode, rewriteRef replaces the names
// with full definitions instead of mangled names.
func (p *Package) rewriteRef(f *File) {
	// Keep a list of all the functions, to remove the ones
	// only used as expressions and avoid generating bridge
	// code for them.
	functions := make(map[string]bool)

	// Assign mangled names.
	for _, n := range f.Name {
		if n.Kind == "not-type" {
			n.Kind = "var"
		}
		if n.Mangle == "" {
			p.mangleName(n)
		}
		if n.Kind == "func" {
			functions[n.Go] = false
		}
	}

	// Now that we have all the name types filled in,
	// scan through the Refs to identify the ones that
	// are trying to do a ,err call.  Also check that
	// functions are only used in calls.
	for _, r := range f.Ref {
		if r.Name.Kind == "const" && r.Name.Const == "" {
			error_(r.Pos(), "unable to find value of constant C.%s", fixGo(r.Name.Go))
		}
		var expr ast.Expr = ast.NewIdent(r.Name.Mangle) // default
		switch r.Context {
		case "call", "call2":
			if r.Name.Kind != "func" {
				if r.Name.Kind == "type" {
					r.Context = "type"
					expr = r.Name.Type.Go
					break
				}
				error_(r.Pos(), "call of non-function C.%s", fixGo(r.Name.Go))
				break
			}
			functions[r.Name.Go] = true
			if r.Context == "call2" {
				if r.Name.Go == "_CMalloc" {
					error_(r.Pos(), "no two-result form for C.malloc")
					break
				}
				// Invent new Name for the two-result function.
				n := f.Name["2"+r.Name.Go]
				if n == nil {
					n = new(Name)
					*n = *r.Name
					n.AddError = true
					n.Mangle = "_C2func_" + n.Go
					f.Name["2"+r.Name.Go] = n
				}
				expr = ast.NewIdent(n.Mangle)
				r.Name = n
				break
			}
		case "expr":
			if r.Name.Kind == "func" {
				// Function is being used in an expression, to e.g. pass around a C function pointer.
				// Create a new Name for this Ref which causes the variable to be declared in Go land.
				fpName := "fp_" + r.Name.Go
				name := f.Name[fpName]
				if name == nil {
					name = &Name{
						Go:   fpName,
						C:    r.Name.C,
						Kind: "fpvar",
						Type: &Type{Size: p.PtrSize, Align: p.PtrSize, C: c("void*"), Go: ast.NewIdent("unsafe.Pointer")},
					}
					p.mangleName(name)
					f.Name[fpName] = name
				}
				r.Name = name
				// Rewrite into call to _Cgo_ptr to prevent assignments.  The _Cgo_ptr
				// function is defined in out.go and simply returns its argument. See
				// issue 7757.
				expr = &ast.CallExpr{
					Fun:  &ast.Ident{NamePos: (*r.Expr).Pos(), Name: "_Cgo_ptr"},
					Args: []ast.Expr{ast.NewIdent(name.Mangle)},
				}
			} else if r.Name.Kind == "type" {
				// Okay - might be new(T)
				expr = r.Name.Type.Go
			} else if r.Name.Kind == "var" {
				expr = &ast.StarExpr{Star: (*r.Expr).Pos(), X: expr}
			}

		case "type":
			if r.Name.Kind != "type" {
				error_(r.Pos(), "expression C.%s used as type", fixGo(r.Name.Go))
			} else if r.Name.Type == nil {
				// Use of C.enum_x, C.struct_x or C.union_x without C definition.
				// GCC won't raise an error when using pointers to such unknown types.
				error_(r.Pos(), "type C.%s: undefined C type '%s'", fixGo(r.Name.Go), r.Name.C)
			} else {
				expr = r.Name.Type.Go
			}
		default:
			if r.Name.Kind == "func" {
				error_(r.Pos(), "must call C.%s", fixGo(r.Name.Go))
			}
		}
		if *godefs {
			// Substitute definition for mangled type name.
			if id, ok := expr.(*ast.Ident); ok {
				if t := typedef[id.Name]; t != nil {
					expr = t.Go
				}
				if id.Name == r.Name.Mangle && r.Name.Const != "" {
					expr = ast.NewIdent(r.Name.Const)
				}
			}
		}

		// Copy position information from old expr into new expr,
		// in case expression being replaced is first on line.
		// See golang.org/issue/6563.
		pos := (*r.Expr).Pos()
		switch x := expr.(type) {
		case *ast.Ident:
			expr = &ast.Ident{NamePos: pos, Name: x.Name}
		}

		*r.Expr = expr
	}

	// Remove functions only used as expressions, so their respective
	// bridge functions are not generated.
	for name, used := range functions {
		if !used {
			delete(f.Name, name)
		}
	}
}

// gccBaseCmd returns the start of the compiler command line.
// It uses $CC if set, or else $GCC, or else the compiler recorded
// during the initial build as defaultCC.
// defaultCC is defined in zdefaultcc.go, written by cmd/dist.
func (p *Package) gccBaseCmd() []string {
	// Use $CC if set, since that's what the build uses.
	if ret := strings.Fields(os.Getenv("CC")); len(ret) > 0 {
		return ret
	}
	// Try $GCC if set, since that's what we used to use.
	if ret := strings.Fields(os.Getenv("GCC")); len(ret) > 0 {
		return ret
	}
	return strings.Fields(defaultCC)
}

// gccMachine returns the gcc -m flag to use, either "-m32", "-m64" or "-marm".
func (p *Package) gccMachine() []string {
	switch goarch {
	case "amd64":
		return []string{"-m64"}
	case "386":
		return []string{"-m32"}
	case "arm":
		return []string{"-marm"} // not thumb
	}
	return nil
}

func gccTmp() string {
	return *objDir + "_cgo_.o"
}

// gccCmd returns the gcc command line to use for compiling
// the input.
func (p *Package) gccCmd() []string {
	c := append(p.gccBaseCmd(),
		"-w",          // no warnings
		"-Wno-error",  // warnings are not errors
		"-o"+gccTmp(), // write object to tmp
		"-gdwarf-2",   // generate DWARF v2 debugging symbols
		"-c",          // do not link
		"-xc",         // input language is C
	)
	if strings.Contains(c[0], "clang") {
		c = append(c,
			"-ferror-limit=0",
			// Apple clang version 1.7 (tags/Apple/clang-77) (based on LLVM 2.9svn)
			// doesn't have -Wno-unneeded-internal-declaration, so we need yet another
			// flag to disable the warning. Yes, really good diagnostics, clang.
			"-Wno-unknown-warning-option",
			"-Wno-unneeded-internal-declaration",
			"-Wno-unused-function",
			"-Qunused-arguments",
			// Clang embeds prototypes for some builtin functions,
			// like malloc and calloc, but all size_t parameters are
			// incorrectly typed unsigned long. We work around that
			// by disabling the builtin functions (this is safe as
			// it won't affect the actual compilation of the C code).
			// See: http://golang.org/issue/6506.
			"-fno-builtin",
		)
	}

	c = append(c, p.GccOptions...)
	c = append(c, p.gccMachine()...)
	c = append(c, "-") //read input from standard input
	return c
}

// gccDebug runs gcc -gdwarf-2 over the C program stdin and
// returns the corresponding DWARF data and, if present, debug data block.
func (p *Package) gccDebug(stdin []byte) (*dwarf.Data, binary.ByteOrder, []byte) {
	runGcc(stdin, p.gccCmd())

	isDebugData := func(s string) bool {
		// Some systems use leading _ to denote non-assembly symbols.
		return s == "__cgodebug_data" || s == "___cgodebug_data"
	}

	if f, err := macho.Open(gccTmp()); err == nil {
		defer f.Close()
		d, err := f.DWARF()
		if err != nil {
			fatalf("cannot load DWARF output from %s: %v", gccTmp(), err)
		}
		var data []byte
		if f.Symtab != nil {
			for i := range f.Symtab.Syms {
				s := &f.Symtab.Syms[i]
				if isDebugData(s.Name) {
					// Found it.  Now find data section.
					if i := int(s.Sect) - 1; 0 <= i && i < len(f.Sections) {
						sect := f.Sections[i]
						if sect.Addr <= s.Value && s.Value < sect.Addr+sect.Size {
							if sdat, err := sect.Data(); err == nil {
								data = sdat[s.Value-sect.Addr:]
							}
						}
					}
				}
			}
		}
		return d, f.ByteOrder, data
	}

	if f, err := elf.Open(gccTmp()); err == nil {
		defer f.Close()
		d, err := f.DWARF()
		if err != nil {
			fatalf("cannot load DWARF output from %s: %v", gccTmp(), err)
		}
		var data []byte
		symtab, err := f.Symbols()
		if err == nil {
			for i := range symtab {
				s := &symtab[i]
				if isDebugData(s.Name) {
					// Found it.  Now find data section.
					if i := int(s.Section); 0 <= i && i < len(f.Sections) {
						sect := f.Sections[i]
						if sect.Addr <= s.Value && s.Value < sect.Addr+sect.Size {
							if sdat, err := sect.Data(); err == nil {
								data = sdat[s.Value-sect.Addr:]
							}
						}
					}
				}
			}
		}
		return d, f.ByteOrder, data
	}

	if f, err := pe.Open(gccTmp()); err == nil {
		defer f.Close()
		d, err := f.DWARF()
		if err != nil {
			fatalf("cannot load DWARF output from %s: %v", gccTmp(), err)
		}
		var data []byte
		for _, s := range f.Symbols {
			if isDebugData(s.Name) {
				if i := int(s.SectionNumber) - 1; 0 <= i && i < len(f.Sections) {
					sect := f.Sections[i]
					if s.Value < sect.Size {
						if sdat, err := sect.Data(); err == nil {
							data = sdat[s.Value:]
						}
					}
				}
			}
		}
		return d, binary.LittleEndian, data
	}

	fatalf("cannot parse gcc output %s as ELF, Mach-O, PE object", gccTmp())
	panic("not reached")
}

// gccDefines runs gcc -E -dM -xc - over the C program stdin
// and returns the corresponding standard output, which is the
// #defines that gcc encountered while processing the input
// and its included files.
func (p *Package) gccDefines(stdin []byte) string {
	base := append(p.gccBaseCmd(), "-E", "-dM", "-xc")
	base = append(base, p.gccMachine()...)
	stdout, _ := runGcc(stdin, append(append(base, p.GccOptions...), "-"))
	return stdout
}

// gccErrors runs gcc over the C program stdin and returns
// the errors that gcc prints.  That is, this function expects
// gcc to fail.
func (p *Package) gccErrors(stdin []byte) string {
	// TODO(rsc): require failure
	args := p.gccCmd()

	if *debugGcc {
		fmt.Fprintf(os.Stderr, "$ %s <<EOF\n", strings.Join(args, " "))
		os.Stderr.Write(stdin)
		fmt.Fprint(os.Stderr, "EOF\n")
	}
	stdout, stderr, _ := run(stdin, args)
	if *debugGcc {
		os.Stderr.Write(stdout)
		os.Stderr.Write(stderr)
	}
	return string(stderr)
}

// runGcc runs the gcc command line args with stdin on standard input.
// If the command exits with a non-zero exit status, runGcc prints
// details about what was run and exits.
// Otherwise runGcc returns the data written to standard output and standard error.
// Note that for some of the uses we expect useful data back
// on standard error, but for those uses gcc must still exit 0.
func runGcc(stdin []byte, args []string) (string, string) {
	if *debugGcc {
		fmt.Fprintf(os.Stderr, "$ %s <<EOF\n", strings.Join(args, " "))
		os.Stderr.Write(stdin)
		fmt.Fprint(os.Stderr, "EOF\n")
	}
	stdout, stderr, ok := run(stdin, args)
	if *debugGcc {
		os.Stderr.Write(stdout)
		os.Stderr.Write(stderr)
	}
	if !ok {
		os.Stderr.Write(stderr)
		os.Exit(2)
	}
	return string(stdout), string(stderr)
}

// A typeConv is a translator from dwarf types to Go types
// with equivalent memory layout.
type typeConv struct {
	// Cache of already-translated or in-progress types.
	m       map[dwarf.Type]*Type
	typedef map[string]ast.Expr

	// Map from types to incomplete pointers to those types.
	ptrs map[dwarf.Type][]*Type
	// Keys of ptrs in insertion order (deterministic worklist)
	ptrKeys []dwarf.Type

	// Predeclared types.
	bool                                   ast.Expr
	byte                                   ast.Expr // denotes padding
	int8, int16, int32, int64              ast.Expr
	uint8, uint16, uint32, uint64, uintptr ast.Expr
	float32, float64                       ast.Expr
	complex64, complex128                  ast.Expr
	void                                   ast.Expr
	string                                 ast.Expr
	goVoid                                 ast.Expr // _Ctype_void, denotes C's void
	goVoidPtr                              ast.Expr // unsafe.Pointer or *byte

	ptrSize int64
	intSize int64
}

var tagGen int
var typedef = make(map[string]*Type)
var goIdent = make(map[string]*ast.Ident)

func (c *typeConv) Init(ptrSize, intSize int64) {
	c.ptrSize = ptrSize
	c.intSize = intSize
	c.m = make(map[dwarf.Type]*Type)
	c.ptrs = make(map[dwarf.Type][]*Type)
	c.bool = c.Ident("bool")
	c.byte = c.Ident("byte")
	c.int8 = c.Ident("int8")
	c.int16 = c.Ident("int16")
	c.int32 = c.Ident("int32")
	c.int64 = c.Ident("int64")
	c.uint8 = c.Ident("uint8")
	c.uint16 = c.Ident("uint16")
	c.uint32 = c.Ident("uint32")
	c.uint64 = c.Ident("uint64")
	c.uintptr = c.Ident("uintptr")
	c.float32 = c.Ident("float32")
	c.float64 = c.Ident("float64")
	c.complex64 = c.Ident("complex64")
	c.complex128 = c.Ident("complex128")
	c.void = c.Ident("void")
	c.string = c.Ident("string")
	c.goVoid = c.Ident("_Ctype_void")

	// Normally cgo translates void* to unsafe.Pointer,
	// but for historical reasons -godefs uses *byte instead.
	if *godefs {
		c.goVoidPtr = &ast.StarExpr{X: c.byte}
	} else {
		c.goVoidPtr = c.Ident("unsafe.Pointer")
	}
}

// base strips away qualifiers and typedefs to get the underlying type
func base(dt dwarf.Type) dwarf.Type {
	for {
		if d, ok := dt.(*dwarf.QualType); ok {
			dt = d.Type
			continue
		}
		if d, ok := dt.(*dwarf.TypedefType); ok {
			dt = d.Type
			continue
		}
		break
	}
	return dt
}

// Map from dwarf text names to aliases we use in package "C".
var dwarfToName = map[string]string{
	"long int":               "long",
	"long unsigned int":      "ulong",
	"unsigned int":           "uint",
	"short unsigned int":     "ushort",
	"short int":              "short",
	"long long int":          "longlong",
	"long long unsigned int": "ulonglong",
	"signed char":            "schar",
	"float complex":          "complexfloat",
	"double complex":         "complexdouble",
}

const signedDelta = 64

// String returns the current type representation.  Format arguments
// are assembled within this method so that any changes in mutable
// values are taken into account.
func (tr *TypeRepr) String() string {
	if len(tr.Repr) == 0 {
		return ""
	}
	if len(tr.FormatArgs) == 0 {
		return tr.Repr
	}
	return fmt.Sprintf(tr.Repr, tr.FormatArgs...)
}

// Empty returns true if the result of String would be "".
func (tr *TypeRepr) Empty() bool {
	return len(tr.Repr) == 0
}

// Set modifies the type representation.
// If fargs are provided, repr is used as a format for fmt.Sprintf.
// Otherwise, repr is used unprocessed as the type representation.
func (tr *TypeRepr) Set(repr string, fargs ...interface{}) {
	tr.Repr = repr
	tr.FormatArgs = fargs
}

// FinishType completes any outstanding type mapping work.
// In particular, it resolves incomplete pointer types.
func (c *typeConv) FinishType(pos token.Pos) {
	// Completing one pointer type might produce more to complete.
	// Keep looping until they're all done.
	for len(c.ptrKeys) > 0 {
		dtype := c.ptrKeys[0]
		c.ptrKeys = c.ptrKeys[1:]

		// Note Type might invalidate c.ptrs[dtype].
		t := c.Type(dtype, pos)
		for _, ptr := range c.ptrs[dtype] {
			ptr.Go.(*ast.StarExpr).X = t.Go
			ptr.C.Set("%s*", t.C)
		}
		c.ptrs[dtype] = nil // retain the map key
	}
}

// Type returns a *Type with the same memory layout as
// dtype when used as the type of a variable or a struct field.
func (c *typeConv) Type(dtype dwarf.Type, pos token.Pos) *Type {
	if t, ok := c.m[dtype]; ok {
		if t.Go == nil {
			fatalf("%s: type conversion loop at %s", lineno(pos), dtype)
		}
		return t
	}

	t := new(Type)
	t.Size = dtype.Size() // note: wrong for array of pointers, corrected below
	t.Align = -1
	t.C = &TypeRepr{Repr: dtype.Common().Name}
	c.m[dtype] = t

	switch dt := dtype.(type) {
	default:
		fatalf("%s: unexpected type: %s", lineno(pos), dtype)

	case *dwarf.AddrType:
		if t.Size != c.ptrSize {
			fatalf("%s: unexpected: %d-byte address type - %s", lineno(pos), t.Size, dtype)
		}
		t.Go = c.uintptr
		t.Align = t.Size

	case *dwarf.ArrayType:
		if dt.StrideBitSize > 0 {
			// Cannot represent bit-sized elements in Go.
			t.Go = c.Opaque(t.Size)
			break
		}
		count := dt.Count
		if count == -1 {
			// Indicates flexible array member, which Go doesn't support.
			// Translate to zero-length array instead.
			count = 0
		}
		sub := c.Type(dt.Type, pos)
		t.Align = sub.Align
		t.Go = &ast.ArrayType{
			Len: c.intExpr(count),
			Elt: sub.Go,
		}
		// Recalculate t.Size now that we know sub.Size.
		t.Size = count * sub.Size
		t.C.Set("__typeof__(%s[%d])", sub.C, dt.Count)

	case *dwarf.BoolType:
		t.Go = c.bool
		t.Align = 1

	case *dwarf.CharType:
		if t.Size != 1 {
			fatalf("%s: unexpected: %d-byte char type - %s", lineno(pos), t.Size, dtype)
		}
		t.Go = c.int8
		t.Align = 1

	case *dwarf.EnumType:
		if t.Align = t.Size; t.Align >= c.ptrSize {
			t.Align = c.ptrSize
		}
		t.C.Set("enum " + dt.EnumName)
		signed := 0
		t.EnumValues = make(map[string]int64)
		for _, ev := range dt.Val {
			t.EnumValues[ev.Name] = ev.Val
			if ev.Val < 0 {
				signed = signedDelta
			}
		}
		switch t.Size + int64(signed) {
		default:
			fatalf("%s: unexpected: %d-byte enum type - %s", lineno(pos), t.Size, dtype)
		case 1:
			t.Go = c.uint8
		case 2:
			t.Go = c.uint16
		case 4:
			t.Go = c.uint32
		case 8:
			t.Go = c.uint64
		case 1 + signedDelta:
			t.Go = c.int8
		case 2 + signedDelta:
			t.Go = c.int16
		case 4 + signedDelta:
			t.Go = c.int32
		case 8 + signedDelta:
			t.Go = c.int64
		}

	case *dwarf.FloatType:
		switch t.Size {
		default:
			fatalf("%s: unexpected: %d-byte float type - %s", lineno(pos), t.Size, dtype)
		case 4:
			t.Go = c.float32
		case 8:
			t.Go = c.float64
		}
		if t.Align = t.Size; t.Align >= c.ptrSize {
			t.Align = c.ptrSize
		}

	case *dwarf.ComplexType:
		switch t.Size {
		default:
			fatalf("%s: unexpected: %d-byte complex type - %s", lineno(pos), t.Size, dtype)
		case 8:
			t.Go = c.complex64
		case 16:
			t.Go = c.complex128
		}
		if t.Align = t.Size; t.Align >= c.ptrSize {
			t.Align = c.ptrSize
		}

	case *dwarf.FuncType:
		// No attempt at translation: would enable calls
		// directly between worlds, but we need to moderate those.
		t.Go = c.uintptr
		t.Align = c.ptrSize

	case *dwarf.IntType:
		if dt.BitSize > 0 {
			fatalf("%s: unexpected: %d-bit int type - %s", lineno(pos), dt.BitSize, dtype)
		}
		switch t.Size {
		default:
			fatalf("%s: unexpected: %d-byte int type - %s", lineno(pos), t.Size, dtype)
		case 1:
			t.Go = c.int8
		case 2:
			t.Go = c.int16
		case 4:
			t.Go = c.int32
		case 8:
			t.Go = c.int64
		}
		if t.Align = t.Size; t.Align >= c.ptrSize {
			t.Align = c.ptrSize
		}

	case *dwarf.PtrType:
		// Clang doesn't emit DW_AT_byte_size for pointer types.
		if t.Size != c.ptrSize && t.Size != -1 {
			fatalf("%s: unexpected: %d-byte pointer type - %s", lineno(pos), t.Size, dtype)
		}
		t.Size = c.ptrSize
		t.Align = c.ptrSize

		if _, ok := base(dt.Type).(*dwarf.VoidType); ok {
			t.Go = c.goVoidPtr
			t.C.Set("void*")
			break
		}

		// Placeholder initialization; completed in FinishType.
		t.Go = &ast.StarExpr{}
		t.C.Set("<incomplete>*")
		if _, ok := c.ptrs[dt.Type]; !ok {
			c.ptrKeys = append(c.ptrKeys, dt.Type)
		}
		c.ptrs[dt.Type] = append(c.ptrs[dt.Type], t)

	case *dwarf.QualType:
		// Ignore qualifier.
		t = c.Type(dt.Type, pos)
		c.m[dtype] = t
		return t

	case *dwarf.StructType:
		// Convert to Go struct, being careful about alignment.
		// Have to give it a name to simulate C "struct foo" references.
		tag := dt.StructName
		if dt.ByteSize < 0 && tag == "" { // opaque unnamed struct - should not be possible
			break
		}
		if tag == "" {
			tag = "__" + strconv.Itoa(tagGen)
			tagGen++
		} else if t.C.Empty() {
			t.C.Set(dt.Kind + " " + tag)
		}
		name := c.Ident("_Ctype_" + dt.Kind + "_" + tag)
		t.Go = name // publish before recursive calls
		goIdent[name.Name] = name
		if dt.ByteSize < 0 {
			// Size calculation in c.Struct/c.Opaque will die with size=-1 (unknown),
			// so execute the basic things that the struct case would do
			// other than try to determine a Go representation.
			tt := *t
			tt.C = &TypeRepr{"%s %s", []interface{}{dt.Kind, tag}}
			tt.Go = c.Ident("struct{}")
			typedef[name.Name] = &tt
			break
		}
		switch dt.Kind {
		case "class", "union":
			t.Go = c.Opaque(t.Size)
			if t.C.Empty() {
				t.C.Set("__typeof__(unsigned char[%d])", t.Size)
			}
			t.Align = 1 // TODO: should probably base this on field alignment.
			typedef[name.Name] = t
		case "struct":
			g, csyntax, align := c.Struct(dt, pos)
			if t.C.Empty() {
				t.C.Set(csyntax)
			}
			t.Align = align
			tt := *t
			if tag != "" {
				tt.C = &TypeRepr{"struct %s", []interface{}{tag}}
			}
			tt.Go = g
			typedef[name.Name] = &tt
		}

	case *dwarf.TypedefType:
		// Record typedef for printing.
		if dt.Name == "_GoString_" {
			// Special C name for Go string type.
			// Knows string layout used by compilers: pointer plus length,
			// which rounds up to 2 pointers after alignment.
			t.Go = c.string
			t.Size = c.ptrSize * 2
			t.Align = c.ptrSize
			break
		}
		if dt.Name == "_GoBytes_" {
			// Special C name for Go []byte type.
			// Knows slice layout used by compilers: pointer, length, cap.
			t.Go = c.Ident("[]byte")
			t.Size = c.ptrSize + 4 + 4
			t.Align = c.ptrSize
			break
		}
		name := c.Ident("_Ctype_" + dt.Name)
		goIdent[name.Name] = name
		sub := c.Type(dt.Type, pos)
		t.Go = name
		t.Size = sub.Size
		t.Align = sub.Align
		oldType := typedef[name.Name]
		if oldType == nil {
			tt := *t
			tt.Go = sub.Go
			typedef[name.Name] = &tt
		}

		// If sub.Go.Name is "_Ctype_struct_foo" or "_Ctype_union_foo" or "_Ctype_class_foo",
		// use that as the Go form for this typedef too, so that the typedef will be interchangeable
		// with the base type.
		// In -godefs mode, do this for all typedefs.
		if isStructUnionClass(sub.Go) || *godefs {
			t.Go = sub.Go

			if isStructUnionClass(sub.Go) {
				// Use the typedef name for C code.
				typedef[sub.Go.(*ast.Ident).Name].C = t.C
			}

			// If we've seen this typedef before, and it
			// was an anonymous struct/union/class before
			// too, use the old definition.
			// TODO: it would be safer to only do this if
			// we verify that the types are the same.
			if oldType != nil && isStructUnionClass(oldType.Go) {
				t.Go = oldType.Go
			}
		}

	case *dwarf.UcharType:
		if t.Size != 1 {
			fatalf("%s: unexpected: %d-byte uchar type - %s", lineno(pos), t.Size, dtype)
		}
		t.Go = c.uint8
		t.Align = 1

	case *dwarf.UintType:
		if dt.BitSize > 0 {
			fatalf("%s: unexpected: %d-bit uint type - %s", lineno(pos), dt.BitSize, dtype)
		}
		switch t.Size {
		default:
			fatalf("%s: unexpected: %d-byte uint type - %s", lineno(pos), t.Size, dtype)
		case 1:
			t.Go = c.uint8
		case 2:
			t.Go = c.uint16
		case 4:
			t.Go = c.uint32
		case 8:
			t.Go = c.uint64
		}
		if t.Align = t.Size; t.Align >= c.ptrSize {
			t.Align = c.ptrSize
		}

	case *dwarf.VoidType:
		t.Go = c.goVoid
		t.C.Set("void")
		t.Align = 1
	}

	switch dtype.(type) {
	case *dwarf.AddrType, *dwarf.BoolType, *dwarf.CharType, *dwarf.IntType, *dwarf.FloatType, *dwarf.UcharType, *dwarf.UintType:
		s := dtype.Common().Name
		if s != "" {
			if ss, ok := dwarfToName[s]; ok {
				s = ss
			}
			s = strings.Join(strings.Split(s, " "), "") // strip spaces
			name := c.Ident("_Ctype_" + s)
			tt := *t
			typedef[name.Name] = &tt
			if !*godefs {
				t.Go = name
			}
		}
	}

	if t.Size < 0 {
		// Unsized types are [0]byte, unless they're typedefs of other types
		// or structs with tags.
		// if so, use the name we've already defined.
		t.Size = 0
		switch dt := dtype.(type) {
		case *dwarf.TypedefType:
			// ok
		case *dwarf.StructType:
			if dt.StructName != "" {
				break
			}
			t.Go = c.Opaque(0)
		default:
			t.Go = c.Opaque(0)
		}
		if t.C.Empty() {
			t.C.Set("void")
		}
	}

	if t.C.Empty() {
		fatalf("%s: internal error: did not create C name for %s", lineno(pos), dtype)
	}

	return t
}

// isStructUnionClass reports whether the type described by the Go syntax x
// is a struct, union, or class with a tag.
func isStructUnionClass(x ast.Expr) bool {
	id, ok := x.(*ast.Ident)
	if !ok {
		return false
	}
	name := id.Name
	return strings.HasPrefix(name, "_Ctype_struct_") ||
		strings.HasPrefix(name, "_Ctype_union_") ||
		strings.HasPrefix(name, "_Ctype_class_")
}

// FuncArg returns a Go type with the same memory layout as
// dtype when used as the type of a C function argument.
func (c *typeConv) FuncArg(dtype dwarf.Type, pos token.Pos) *Type {
	t := c.Type(dtype, pos)
	switch dt := dtype.(type) {
	case *dwarf.ArrayType:
		// Arrays are passed implicitly as pointers in C.
		// In Go, we must be explicit.
		tr := &TypeRepr{}
		tr.Set("%s*", t.C)
		return &Type{
			Size:  c.ptrSize,
			Align: c.ptrSize,
			Go:    &ast.StarExpr{X: t.Go},
			C:     tr,
		}
	case *dwarf.TypedefType:
		// C has much more relaxed rules than Go for
		// implicit type conversions.  When the parameter
		// is type T defined as *X, simulate a little of the
		// laxness of C by making the argument *X instead of T.
		if ptr, ok := base(dt.Type).(*dwarf.PtrType); ok {
			// Unless the typedef happens to point to void* since
			// Go has special rules around using unsafe.Pointer.
			if _, void := base(ptr.Type).(*dwarf.VoidType); void {
				break
			}

			t = c.Type(ptr, pos)
			if t == nil {
				return nil
			}

			// Remember the C spelling, in case the struct
			// has __attribute__((unavailable)) on it.  See issue 2888.
			t.Typedef = dt.Name
		}
	}
	return t
}

// FuncType returns the Go type analogous to dtype.
// There is no guarantee about matching memory layout.
func (c *typeConv) FuncType(dtype *dwarf.FuncType, pos token.Pos) *FuncType {
	p := make([]*Type, len(dtype.ParamType))
	gp := make([]*ast.Field, len(dtype.ParamType))
	for i, f := range dtype.ParamType {
		// gcc's DWARF generator outputs a single DotDotDotType parameter for
		// function pointers that specify no parameters (e.g. void
		// (*__cgo_0)()).  Treat this special case as void.  This case is
		// invalid according to ISO C anyway (i.e. void (*__cgo_1)(...) is not
		// legal).
		if _, ok := f.(*dwarf.DotDotDotType); ok && i == 0 {
			p, gp = nil, nil
			break
		}
		p[i] = c.FuncArg(f, pos)
		gp[i] = &ast.Field{Type: p[i].Go}
	}
	var r *Type
	var gr []*ast.Field
	if _, ok := dtype.ReturnType.(*dwarf.VoidType); ok {
		gr = []*ast.Field{{Type: c.goVoid}}
	} else if dtype.ReturnType != nil {
		r = c.Type(dtype.ReturnType, pos)
		gr = []*ast.Field{{Type: r.Go}}
	}
	return &FuncType{
		Params: p,
		Result: r,
		Go: &ast.FuncType{
			Params:  &ast.FieldList{List: gp},
			Results: &ast.FieldList{List: gr},
		},
	}
}

// Identifier
func (c *typeConv) Ident(s string) *ast.Ident {
	return ast.NewIdent(s)
}

// Opaque type of n bytes.
func (c *typeConv) Opaque(n int64) ast.Expr {
	return &ast.ArrayType{
		Len: c.intExpr(n),
		Elt: c.byte,
	}
}

// Expr for integer n.
func (c *typeConv) intExpr(n int64) ast.Expr {
	return &ast.BasicLit{
		Kind:  token.INT,
		Value: strconv.FormatInt(n, 10),
	}
}

// Add padding of given size to fld.
func (c *typeConv) pad(fld []*ast.Field, size int64) []*ast.Field {
	n := len(fld)
	fld = fld[0 : n+1]
	fld[n] = &ast.Field{Names: []*ast.Ident{c.Ident("_")}, Type: c.Opaque(size)}
	return fld
}

// Struct conversion: return Go and (6g) C syntax for type.
func (c *typeConv) Struct(dt *dwarf.StructType, pos token.Pos) (expr *ast.StructType, csyntax string, align int64) {
	// Minimum alignment for a struct is 1 byte.
	align = 1

	var buf bytes.Buffer
	buf.WriteString("struct {")
	fld := make([]*ast.Field, 0, 2*len(dt.Field)+1) // enough for padding around every field
	off := int64(0)

	// Rename struct fields that happen to be named Go keywords into
	// _{keyword}.  Create a map from C ident -> Go ident.  The Go ident will
	// be mangled.  Any existing identifier that already has the same name on
	// the C-side will cause the Go-mangled version to be prefixed with _.
	// (e.g. in a struct with fields '_type' and 'type', the latter would be
	// rendered as '__type' in Go).
	ident := make(map[string]string)
	used := make(map[string]bool)
	for _, f := range dt.Field {
		ident[f.Name] = f.Name
		used[f.Name] = true
	}

	if !*godefs {
		for cid, goid := range ident {
			if token.Lookup(goid).IsKeyword() {
				// Avoid keyword
				goid = "_" + goid

				// Also avoid existing fields
				for _, exist := used[goid]; exist; _, exist = used[goid] {
					goid = "_" + goid
				}

				used[goid] = true
				ident[cid] = goid
			}
		}
	}

	anon := 0
	for _, f := range dt.Field {
		if f.ByteOffset > off {
			fld = c.pad(fld, f.ByteOffset-off)
			off = f.ByteOffset
		}

		name := f.Name
		ft := f.Type

		// In godefs mode, if this field is a C11
		// anonymous union then treat the first field in the
		// union as the field in the struct.  This handles
		// cases like the glibc <sys/resource.h> file; see
		// issue 6677.
		if *godefs {
			if st, ok := f.Type.(*dwarf.StructType); ok && name == "" && st.Kind == "union" && len(st.Field) > 0 && !used[st.Field[0].Name] {
				name = st.Field[0].Name
				ident[name] = name
				ft = st.Field[0].Type
			}
		}

		// TODO: Handle fields that are anonymous structs by
		// promoting the fields of the inner struct.

		t := c.Type(ft, pos)
		tgo := t.Go
		size := t.Size
		talign := t.Align
		if f.BitSize > 0 {
			if f.BitSize%8 != 0 {
				continue
			}
			size = f.BitSize / 8
			name := tgo.(*ast.Ident).String()
			if strings.HasPrefix(name, "int") {
				name = "int"
			} else {
				name = "uint"
			}
			tgo = ast.NewIdent(name + fmt.Sprint(f.BitSize))
			talign = size
		}

		if talign > 0 && f.ByteOffset%talign != 0 {
			// Drop misaligned fields, the same way we drop integer bit fields.
			// The goal is to make available what can be made available.
			// Otherwise one bad and unneeded field in an otherwise okay struct
			// makes the whole program not compile. Much of the time these
			// structs are in system headers that cannot be corrected.
			continue
		}
		n := len(fld)
		fld = fld[0 : n+1]
		if name == "" {
			name = fmt.Sprintf("anon%d", anon)
			anon++
			ident[name] = name
		}
		fld[n] = &ast.Field{Names: []*ast.Ident{c.Ident(ident[name])}, Type: tgo}
		off += size
		buf.WriteString(t.C.String())
		buf.WriteString(" ")
		buf.WriteString(name)
		buf.WriteString("; ")
		if talign > align {
			align = talign
		}
	}
	if off < dt.ByteSize {
		fld = c.pad(fld, dt.ByteSize-off)
		off = dt.ByteSize
	}
	if off != dt.ByteSize {
		fatalf("%s: struct size calculation error off=%d bytesize=%d", lineno(pos), off, dt.ByteSize)
	}
	buf.WriteString("}")
	csyntax = buf.String()

	if *godefs {
		godefsFields(fld)
	}
	expr = &ast.StructType{Fields: &ast.FieldList{List: fld}}
	return
}

func upper(s string) string {
	if s == "" {
		return ""
	}
	r, size := utf8.DecodeRuneInString(s)
	if r == '_' {
		return "X" + s
	}
	return string(unicode.ToUpper(r)) + s[size:]
}

// godefsFields rewrites field names for use in Go or C definitions.
// It strips leading common prefixes (like tv_ in tv_sec, tv_usec)
// converts names to upper case, and rewrites _ into Pad_godefs_n,
// so that all fields are exported.
func godefsFields(fld []*ast.Field) {
	prefix := fieldPrefix(fld)
	npad := 0
	for _, f := range fld {
		for _, n := range f.Names {
			if n.Name != prefix {
				n.Name = strings.TrimPrefix(n.Name, prefix)
			}
			if n.Name == "_" {
				// Use exported name instead.
				n.Name = "Pad_cgo_" + strconv.Itoa(npad)
				npad++
			}
			n.Name = upper(n.Name)
		}
	}
}

// fieldPrefix returns the prefix that should be removed from all the
// field names when generating the C or Go code.  For generated
// C, we leave the names as is (tv_sec, tv_usec), since that's what
// people are used to seeing in C.  For generated Go code, such as
// package syscall's data structures, we drop a common prefix
// (so sec, usec, which will get turned into Sec, Usec for exporting).
func fieldPrefix(fld []*ast.Field) string {
	prefix := ""
	for _, f := range fld {
		for _, n := range f.Names {
			// Ignore field names that don't have the prefix we're
			// looking for.  It is common in C headers to have fields
			// named, say, _pad in an otherwise prefixed header.
			// If the struct has 3 fields tv_sec, tv_usec, _pad1, then we
			// still want to remove the tv_ prefix.
			// The check for "orig_" here handles orig_eax in the
			// x86 ptrace register sets, which otherwise have all fields
			// with reg_ prefixes.
			if strings.HasPrefix(n.Name, "orig_") || strings.HasPrefix(n.Name, "_") {
				continue
			}
			i := strings.Index(n.Name, "_")
			if i < 0 {
				continue
			}
			if prefix == "" {
				prefix = n.Name[:i+1]
			} else if prefix != n.Name[:i+1] {
				return ""
			}
		}
	}
	return prefix
}
