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
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"runtime"
	"strconv"
	"strings"
	"unicode"
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
	return s
}

// ParseFlags extracts #cgo CFLAGS and LDFLAGS options from the file
// preamble. Multiple occurrences are concatenated with a separating space,
// even across files.
func (p *Package) ParseFlags(f *File, srcfile string) {
	linesIn := strings.Split(f.Preamble, "\n", -1)
	linesOut := make([]string, 0, len(linesIn))

NextLine:
	for _, line := range linesIn {
		l := strings.TrimSpace(line)
		if len(l) < 5 || l[:4] != "#cgo" || !unicode.IsSpace(int(l[4])) {
			linesOut = append(linesOut, line)
			continue
		}

		l = strings.TrimSpace(l[4:])
		fields := strings.Split(l, ":", 2)
		if len(fields) != 2 {
			fatal("%s: bad #cgo line: %s", srcfile, line)
		}

		var k string
		kf := strings.Fields(fields[0])
		switch len(kf) {
		case 1:
			k = kf[0]
		case 2:
			k = kf[1]
			switch kf[0] {
			case runtime.GOOS:
			case runtime.GOARCH:
			case runtime.GOOS + "/" + runtime.GOARCH:
			default:
				continue NextLine
			}
		default:
			fatal("%s: bad #cgo option: %s", srcfile, fields[0])
		}

		if k != "CFLAGS" && k != "LDFLAGS" {
			fatal("%s: unsupported #cgo option %s", srcfile, k)
		}

		v := strings.TrimSpace(fields[1])
		args, err := splitQuoted(v)
		if err != nil {
			fatal("%s: bad #cgo option %s: %s", srcfile, k, err.String())
		}
		if oldv, ok := p.CgoFlags[k]; ok {
			p.CgoFlags[k] = oldv + " " + v
		} else {
			p.CgoFlags[k] = v
		}
		if k == "CFLAGS" {
			p.GccOptions = append(p.GccOptions, args...)
		}
	}
	f.Preamble = strings.Join(linesOut, "\n")
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
func splitQuoted(s string) (r []string, err os.Error) {
	var args []string
	arg := make([]int, len(s))
	escaped := false
	quoted := false
	quote := 0
	i := 0
	for _, rune := range s {
		switch {
		case escaped:
			escaped = false
		case rune == '\\':
			escaped = true
			continue
		case quote != 0:
			if rune == quote {
				quote = 0
				continue
			}
		case rune == '"' || rune == '\'':
			quoted = true
			quote = rune
			continue
		case unicode.IsSpace(rune):
			if quoted || i > 0 {
				quoted = false
				args = append(args, string(arg[:i]))
				i = 0
			}
			continue
		}
		arg[i] = rune
		i++
	}
	if quoted || i > 0 {
		args = append(args, string(arg[:i]))
	}
	if quote != 0 {
		err = os.ErrorString("unclosed quote")
	} else if escaped {
		err = os.ErrorString("unfinished escaping")
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
	b.WriteString(builtinProlog)
	b.WriteString(f.Preamble)
	stdout := p.gccDefines(b.Bytes())

	for _, line := range strings.Split(stdout, "\n", -1) {
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
	// Coerce gcc into telling us whether each name is
	// a type, a value, or undeclared.  We compile a function
	// containing the line:
	//	name;
	// If name is a type, gcc will print:
	//	cgo-test:2: warning: useless type name in empty declaration
	// If name is a value, gcc will print
	//	cgo-test:2: warning: statement with no effect
	// If name is undeclared, gcc will print
	//	cgo-test:2: error: 'name' undeclared (first use in this function)
	// A line number directive causes the line number to
	// correspond to the index in the names array.
	//
	// The line also has an enum declaration:
	//	name; enum { _cgo_enum_1 = name };
	// If name is not a constant, gcc will print:
	//	cgo-test:4: error: enumerator value for '_cgo_enum_4' is not an integer constant
	// we assume lines without that error are constants.

	// Make list of names that need sniffing, type lookup.
	toSniff := make([]*Name, 0, len(f.Name))
	needType := make([]*Name, 0, len(f.Name))

	for _, n := range f.Name {
		// If we've already found this name as a #define
		// and we can translate it as a constant value, do so.
		if n.Define != "" {
			ok := false
			if _, err := strconv.Atoi(n.Define); err == nil {
				ok = true
			} else if n.Define[0] == '"' || n.Define[0] == '\'' {
				_, err := parser.ParseExpr(fset, "", n.Define)
				if err == nil {
					ok = true
				}
			}
			if ok {
				n.Kind = "const"
				n.Const = n.Define
				continue
			}

			if isName(n.Define) {
				n.C = n.Define
			}
		}

		// If this is a struct, union, or enum type name,
		// record the kind but also that we need type information.
		if strings.HasPrefix(n.C, "struct ") || strings.HasPrefix(n.C, "union ") || strings.HasPrefix(n.C, "enum ") {
			n.Kind = "type"
			i := len(needType)
			needType = needType[0 : i+1]
			needType[i] = n
			continue
		}

		i := len(toSniff)
		toSniff = toSniff[0 : i+1]
		toSniff[i] = n
	}

	if len(toSniff) == 0 {
		return needType
	}

	var b bytes.Buffer
	b.WriteString(builtinProlog)
	b.WriteString(f.Preamble)
	b.WriteString("void __cgo__f__(void) {\n")
	b.WriteString("#line 0 \"cgo-test\"\n")
	for i, n := range toSniff {
		fmt.Fprintf(&b, "%s; enum { _cgo_enum_%d = %s }; /* cgo-test:%d */\n", n.C, i, n.C, i)
	}
	b.WriteString("}\n")
	stderr := p.gccErrors(b.Bytes())
	if stderr == "" {
		fatal("gcc produced no output\non input:\n%s", b.Bytes())
	}

	names := make([]*Name, len(toSniff))
	copy(names, toSniff)

	isConst := make([]bool, len(toSniff))
	for i := range isConst {
		isConst[i] = true // until proven otherwise
	}

	for _, line := range strings.Split(stderr, "\n", -1) {
		if len(line) < 9 || line[0:9] != "cgo-test:" {
			// the user will see any compiler errors when the code is compiled later.
			continue
		}
		line = line[9:]
		colon := strings.Index(line, ":")
		if colon < 0 {
			continue
		}
		i, err := strconv.Atoi(line[0:colon])
		if err != nil {
			continue
		}
		what := ""
		switch {
		default:
			continue
		case strings.Contains(line, ": useless type name in empty declaration"):
			what = "type"
			isConst[i] = false
		case strings.Contains(line, ": statement with no effect"):
			what = "not-type" // const or func or var
		case strings.Contains(line, "undeclared"):
			error(token.NoPos, "%s", strings.TrimSpace(line[colon+1:]))
		case strings.Contains(line, "is not an integer constant"):
			isConst[i] = false
			continue
		}
		n := toSniff[i]
		if n == nil {
			continue
		}
		toSniff[i] = nil
		n.Kind = what

		j := len(needType)
		needType = needType[0 : j+1]
		needType[j] = n
	}
	for i, b := range isConst {
		if b {
			names[i].Kind = "const"
		}
	}
	for _, n := range toSniff {
		if n == nil {
			continue
		}
		if n.Kind != "" {
			continue
		}
		error(token.NoPos, "could not determine kind of name for C.%s", n.Go)
	}
	if nerrors > 0 {
		fatal("unresolved names")
	}
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
	//	typeof(names[i]) *__cgo__i;
	// for each entry in names and then dereference the type we
	// learn for __cgo__i.
	var b bytes.Buffer
	b.WriteString(builtinProlog)
	b.WriteString(f.Preamble)
	for i, n := range names {
		fmt.Fprintf(&b, "typeof(%s) *__cgo__%d;\n", n.C, i)
		if n.Kind == "const" {
			fmt.Fprintf(&b, "enum { __cgo_enum__%d = %s };\n", i, n.C)
		}
	}
	d := p.gccDebug(b.Bytes())

	// Scan DWARF info for top-level TagVariable entries with AttrName __cgo__i.
	types := make([]dwarf.Type, len(names))
	enums := make([]dwarf.Offset, len(names))
	nameToIndex := make(map[*Name]int)
	for i, n := range names {
		nameToIndex[n] = i
	}
	r := d.Reader()
	for {
		e, err := r.Next()
		if err != nil {
			fatal("reading DWARF entry: %s", err)
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
					fatal("reading DWARF entry: %s", err)
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
				fatal("malformed DWARF TagVariable entry")
			}
			if !strings.HasPrefix(name, "__cgo__") {
				break
			}
			typ, err := d.Type(typOff)
			if err != nil {
				fatal("loading DWARF type: %s", err)
			}
			t, ok := typ.(*dwarf.PtrType)
			if !ok || t == nil {
				fatal("internal error: %s has non-pointer type", name)
			}
			i, err := strconv.Atoi(name[7:])
			if err != nil {
				fatal("malformed __cgo__ name: %s", name)
			}
			if enums[i] != 0 {
				t, err := d.Type(enums[i])
				if err != nil {
					fatal("loading DWARF type: %s", err)
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
	conv.Init(p.PtrSize)
	for i, n := range names {
		f, fok := types[i].(*dwarf.FuncType)
		if n.Kind != "type" && fok {
			n.Kind = "func"
			n.FuncType = conv.FuncType(f)
		} else {
			n.Type = conv.Type(types[i])
			if enums[i] != 0 && n.Type.EnumValues != nil {
				k := fmt.Sprintf("__cgo_enum__%d", i)
				n.Kind = "const"
				n.Const = strconv.Itoa64(n.Type.EnumValues[k])
				// Remove injected enum to ensure the value will deep-compare
				// equally in future loads of the same constant.
				n.Type.EnumValues[k] = 0, false
			}
		}
	}
}

// rewriteRef rewrites all the C.xxx references in f.AST to refer to the
// Go equivalents, now that we have figured out the meaning of all
// the xxx.
func (p *Package) rewriteRef(f *File) {
	// Assign mangled names.
	for _, n := range f.Name {
		if n.Kind == "not-type" {
			n.Kind = "var"
		}
		if n.Mangle == "" {
			n.Mangle = "_C" + n.Kind + "_" + n.Go
		}
	}

	// Now that we have all the name types filled in,
	// scan through the Refs to identify the ones that
	// are trying to do a ,err call.  Also check that
	// functions are only used in calls.
	for _, r := range f.Ref {
		var expr ast.Expr = ast.NewIdent(r.Name.Mangle) // default
		switch r.Context {
		case "call", "call2":
			if r.Name.Kind != "func" {
				if r.Name.Kind == "type" {
					r.Context = "type"
					expr = r.Name.Type.Go
					break
				}
				error(r.Pos(), "call of non-function C.%s", r.Name.Go)
				break
			}
			if r.Context == "call2" {
				if r.Name.FuncType.Result == nil {
					error(r.Pos(), "assignment count mismatch: 2 = 0")
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
				error(r.Pos(), "must call C.%s", r.Name.Go)
			}
			if r.Name.Kind == "type" {
				// Okay - might be new(T)
				expr = r.Name.Type.Go
			}
			if r.Name.Kind == "var" {
				expr = &ast.StarExpr{X: expr}
			}

		case "type":
			if r.Name.Kind != "type" {
				error(r.Pos(), "expression C.%s used as type", r.Name.Go)
			} else {
				expr = r.Name.Type.Go
			}
		default:
			if r.Name.Kind == "func" {
				error(r.Pos(), "must call C.%s", r.Name.Go)
			}
		}
		*r.Expr = expr
	}
}

// gccName returns the name of the compiler to run.  Use $GCC if set in
// the environment, otherwise just "gcc".

func (p *Package) gccName() (ret string) {
	if ret = os.Getenv("GCC"); ret == "" {
		ret = "gcc"
	}
	return
}

// gccMachine returns the gcc -m flag to use, either "-m32" or "-m64".
func (p *Package) gccMachine() string {
	if p.PtrSize == 8 {
		return "-m64"
	}
	return "-m32"
}

const gccTmp = "_obj/_cgo_.o"

// gccCmd returns the gcc command line to use for compiling
// the input.
func (p *Package) gccCmd() []string {
	return []string{
		p.gccName(),
		p.gccMachine(),
		"-Wall",                             // many warnings
		"-Werror",                           // warnings are errors
		"-o" + gccTmp,                       // write object to tmp
		"-gdwarf-2",                         // generate DWARF v2 debugging symbols
		"-fno-eliminate-unused-debug-types", // gets rid of e.g. untyped enum otherwise
		"-c",                                // do not link
		"-xc",                               // input language is C
		"-",                                 // read input from standard input
	}
}

// gccDebug runs gcc -gdwarf-2 over the C program stdin and
// returns the corresponding DWARF data and any messages
// printed to standard error.
func (p *Package) gccDebug(stdin []byte) *dwarf.Data {
	runGcc(stdin, append(p.gccCmd(), p.GccOptions...))

	// Try to parse f as ELF and Mach-O and hope one works.
	var f interface {
		DWARF() (*dwarf.Data, os.Error)
	}
	var err os.Error
	if f, err = elf.Open(gccTmp); err != nil {
		if f, err = macho.Open(gccTmp); err != nil {
			if f, err = pe.Open(gccTmp); err != nil {
				fatal("cannot parse gcc output %s as ELF or Mach-O or PE object", gccTmp)
			}
		}
	}

	d, err := f.DWARF()
	if err != nil {
		fatal("cannot load DWARF debug information from %s: %s", gccTmp, err)
	}
	return d
}

// gccDefines runs gcc -E -dM -xc - over the C program stdin
// and returns the corresponding standard output, which is the
// #defines that gcc encountered while processing the input
// and its included files.
func (p *Package) gccDefines(stdin []byte) string {
	base := []string{p.gccName(), p.gccMachine(), "-E", "-dM", "-xc", "-"}
	stdout, _ := runGcc(stdin, append(base, p.GccOptions...))
	return stdout
}

// gccErrors runs gcc over the C program stdin and returns
// the errors that gcc prints.  That is, this function expects
// gcc to fail.
func (p *Package) gccErrors(stdin []byte) string {
	// TODO(rsc): require failure
	args := append(p.gccCmd(), p.GccOptions...)
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

	// Predeclared types.
	bool                                   ast.Expr
	byte                                   ast.Expr // denotes padding
	int8, int16, int32, int64              ast.Expr
	uint8, uint16, uint32, uint64, uintptr ast.Expr
	float32, float64                       ast.Expr
	complex64, complex128                  ast.Expr
	void                                   ast.Expr
	unsafePointer                          ast.Expr
	string                                 ast.Expr

	ptrSize int64
}

var tagGen int
var typedef = make(map[string]ast.Expr)

func (c *typeConv) Init(ptrSize int64) {
	c.ptrSize = ptrSize
	c.m = make(map[dwarf.Type]*Type)
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
	c.unsafePointer = c.Ident("unsafe.Pointer")
	c.void = c.Ident("void")
	c.string = c.Ident("string")
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

// Type returns a *Type with the same memory layout as
// dtype when used as the type of a variable or a struct field.
func (c *typeConv) Type(dtype dwarf.Type) *Type {
	if t, ok := c.m[dtype]; ok {
		if t.Go == nil {
			fatal("type conversion loop at %s", dtype)
		}
		return t
	}

	t := new(Type)
	t.Size = dtype.Size()
	t.Align = -1
	t.C = &TypeRepr{Repr: dtype.Common().Name}
	c.m[dtype] = t

	if t.Size < 0 {
		// Unsized types are [0]byte
		t.Size = 0
		t.Go = c.Opaque(0)
		if t.C.Empty() {
			t.C.Set("void")
		}
		return t
	}

	switch dt := dtype.(type) {
	default:
		fatal("unexpected type: %s", dtype)

	case *dwarf.AddrType:
		if t.Size != c.ptrSize {
			fatal("unexpected: %d-byte address type - %s", t.Size, dtype)
		}
		t.Go = c.uintptr
		t.Align = t.Size

	case *dwarf.ArrayType:
		if dt.StrideBitSize > 0 {
			// Cannot represent bit-sized elements in Go.
			t.Go = c.Opaque(t.Size)
			break
		}
		gt := &ast.ArrayType{
			Len: c.intExpr(dt.Count),
		}
		t.Go = gt // publish before recursive call
		sub := c.Type(dt.Type)
		t.Align = sub.Align
		gt.Elt = sub.Go
		t.C.Set("typeof(%s[%d])", sub.C, dt.Count)

	case *dwarf.BoolType:
		t.Go = c.bool
		t.Align = c.ptrSize

	case *dwarf.CharType:
		if t.Size != 1 {
			fatal("unexpected: %d-byte char type - %s", t.Size, dtype)
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
			fatal("unexpected: %d-byte enum type - %s", t.Size, dtype)
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
			fatal("unexpected: %d-byte float type - %s", t.Size, dtype)
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
			fatal("unexpected: %d-byte complex type - %s", t.Size, dtype)
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
			fatal("unexpected: %d-bit int type - %s", dt.BitSize, dtype)
		}
		switch t.Size {
		default:
			fatal("unexpected: %d-byte int type - %s", t.Size, dtype)
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
		t.Align = c.ptrSize

		// Translate void* as unsafe.Pointer
		if _, ok := base(dt.Type).(*dwarf.VoidType); ok {
			t.Go = c.unsafePointer
			t.C.Set("void*")
			break
		}

		gt := &ast.StarExpr{}
		t.Go = gt // publish before recursive call
		sub := c.Type(dt.Type)
		gt.X = sub.Go
		t.C.Set("%s*", sub.C)

	case *dwarf.QualType:
		// Ignore qualifier.
		t = c.Type(dt.Type)
		c.m[dtype] = t
		return t

	case *dwarf.StructType:
		// Convert to Go struct, being careful about alignment.
		// Have to give it a name to simulate C "struct foo" references.
		tag := dt.StructName
		if tag == "" {
			tag = "__" + strconv.Itoa(tagGen)
			tagGen++
		} else if t.C.Empty() {
			t.C.Set(dt.Kind + " " + tag)
		}
		name := c.Ident("_Ctype_" + dt.Kind + "_" + tag)
		t.Go = name // publish before recursive calls
		switch dt.Kind {
		case "union", "class":
			typedef[name.Name] = c.Opaque(t.Size)
			if t.C.Empty() {
				t.C.Set("typeof(unsigned char[%d])", t.Size)
			}
		case "struct":
			g, csyntax, align := c.Struct(dt)
			if t.C.Empty() {
				t.C.Set(csyntax)
			}
			t.Align = align
			typedef[name.Name] = g
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
		name := c.Ident("_Ctypedef_" + dt.Name)
		t.Go = name // publish before recursive call
		sub := c.Type(dt.Type)
		t.Size = sub.Size
		t.Align = sub.Align
		if _, ok := typedef[name.Name]; !ok {
			typedef[name.Name] = sub.Go
		}

	case *dwarf.UcharType:
		if t.Size != 1 {
			fatal("unexpected: %d-byte uchar type - %s", t.Size, dtype)
		}
		t.Go = c.uint8
		t.Align = 1

	case *dwarf.UintType:
		if dt.BitSize > 0 {
			fatal("unexpected: %d-bit uint type - %s", dt.BitSize, dtype)
		}
		switch t.Size {
		default:
			fatal("unexpected: %d-byte uint type - %s", t.Size, dtype)
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
		t.Go = c.void
		t.C.Set("void")
	}

	switch dtype.(type) {
	case *dwarf.AddrType, *dwarf.BoolType, *dwarf.CharType, *dwarf.IntType, *dwarf.FloatType, *dwarf.UcharType, *dwarf.UintType:
		s := dtype.Common().Name
		if s != "" {
			if ss, ok := dwarfToName[s]; ok {
				s = ss
			}
			s = strings.Join(strings.Split(s, " ", -1), "") // strip spaces
			name := c.Ident("_Ctype_" + s)
			typedef[name.Name] = t.Go
			t.Go = name
		}
	}

	if t.C.Empty() {
		fatal("internal error: did not create C name for %s", dtype)
	}

	return t
}

// FuncArg returns a Go type with the same memory layout as
// dtype when used as the type of a C function argument.
func (c *typeConv) FuncArg(dtype dwarf.Type) *Type {
	t := c.Type(dtype)
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
			if _, void := base(ptr.Type).(*dwarf.VoidType); !void {
				return c.Type(ptr)
			}
		}
	}
	return t
}

// FuncType returns the Go type analogous to dtype.
// There is no guarantee about matching memory layout.
func (c *typeConv) FuncType(dtype *dwarf.FuncType) *FuncType {
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
		p[i] = c.FuncArg(f)
		gp[i] = &ast.Field{Type: p[i].Go}
	}
	var r *Type
	var gr []*ast.Field
	if _, ok := dtype.ReturnType.(*dwarf.VoidType); !ok && dtype.ReturnType != nil {
		r = c.Type(dtype.ReturnType)
		gr = []*ast.Field{&ast.Field{Type: r.Go}}
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
		Value: strconv.Itoa64(n),
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
func (c *typeConv) Struct(dt *dwarf.StructType) (expr *ast.StructType, csyntax string, align int64) {
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
	for cid, goid := range ident {
		if token.Lookup([]byte(goid)).IsKeyword() {
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

	for _, f := range dt.Field {
		if f.BitSize > 0 && f.BitSize != f.ByteSize*8 {
			continue
		}
		if f.ByteOffset > off {
			fld = c.pad(fld, f.ByteOffset-off)
			off = f.ByteOffset
		}
		t := c.Type(f.Type)
		n := len(fld)
		fld = fld[0 : n+1]

		fld[n] = &ast.Field{Names: []*ast.Ident{c.Ident(ident[f.Name])}, Type: t.Go}
		off += t.Size
		buf.WriteString(t.C.String())
		buf.WriteString(" ")
		buf.WriteString(f.Name)
		buf.WriteString("; ")
		if t.Align > align {
			align = t.Align
		}
	}
	if off < dt.ByteSize {
		fld = c.pad(fld, dt.ByteSize-off)
		off = dt.ByteSize
	}
	if off != dt.ByteSize {
		fatal("struct size calculation error")
	}
	buf.WriteString("}")
	csyntax = buf.String()
	expr = &ast.StructType{Fields: &ast.FieldList{List: fld}}
	return
}
