// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Annotate Crefs in Prog with C types by parsing gcc debug output.
// Conversion of debug output to Go types.

package main

import (
	"bytes";
	"debug/dwarf";
	"debug/elf";
	"debug/macho";
	"fmt";
	"go/ast";
	"go/token";
	"os";
	"strconv";
	"strings";
)

func (p *Prog) loadDebugInfo() {
	// Construct a slice of unique names from p.Crefs.
	m := make(map[string]int);
	for _, c := range p.Crefs {
		m[c.Name] = -1;
	}
	names := make([]string, 0, len(m));
	for name, _ := range m {
		i := len(names);
		names = names[0:i+1];
		names[i] = name;
		m[name] = i;
	}

	// Coerce gcc into telling us whether each name is
	// a type, a value, or undeclared.  We compile a function
	// containing the line:
	//	name;
	// If name is a type, gcc will print:
	//	x.c:2: warning: useless type name in empty declaration
	// If name is a value, gcc will print
	//	x.c:2: warning: statement with no effect
	// If name is undeclared, gcc will print
	//	x.c:2: error: 'name' undeclared (first use in this function)
	// A line number directive causes the line number to
	// correspond to the index in the names array.
	var b bytes.Buffer;
	b.WriteString(p.Preamble);
	b.WriteString("void f(void) {\n");
	b.WriteString("#line 0 \"cgo-test\"\n");
	for _, n := range names {
		b.WriteString(n);
		b.WriteString(";\n");
	}
	b.WriteString("}\n");

	kind := make(map[string]string);
	_, stderr := p.gccDebug(b.Bytes());
	if stderr == "" {
		fatal("gcc produced no output");
	}
	for _, line := range strings.Split(stderr, "\n", 0) {
		if len(line) < 9 || line[0:9] != "cgo-test:" {
			continue;
		}
		line = line[9:len(line)];
		colon := strings.Index(line, ":");
		if colon < 0 {
			continue;
		}
		i, err := strconv.Atoi(line[0:colon]);
		if err != nil {
			continue;
		}
		what := "";
		switch {
		default:
			continue;
		case strings.Index(line, "warning: useless type name in empty declaration") >= 0:
			what = "type";
		case strings.Index(line, "warning: statement with no effect") >= 0:
			what = "value";
		case strings.Index(line, "undeclared") >= 0:
			what = "error";
		}
		if old, ok := kind[names[i]]; ok && old != what {
			error(noPos, "inconsistent gcc output about C.%s", names[i]);
		}
		kind[names[i]] = what;
	}
	for _, n := range names {
		if _, ok := kind[n]; !ok {
			error(noPos, "could not determine kind of name for C.%s", n);
		}
	}

	// Extract the types from the DWARF section of an object
	// from a well-formed C program.  Gcc only generates DWARF info
	// for symbols in the object file, so it is not enough to print the
	// preamble and hope the symbols we care about will be there.
	// Instead, emit
	//	typeof(names[i]) *__cgo__i;
	// for each entry in names and then dereference the type we
	// learn for __cgo__i.
	b.Reset();
	b.WriteString(p.Preamble);
	for i, n := range names {
		fmt.Fprintf(&b, "typeof(%s) *__cgo__%d;\n", n, i);
	}
	d, stderr := p.gccDebug(b.Bytes());
	if d == nil {
		fatal("gcc failed:\n%s\non input:\n%s", stderr, b.Bytes());
	}

	// Scan DWARF info for  top-level TagVariable entries with AttrName __cgo__i.
	types := make([]dwarf.Type, len(names));
	r := d.Reader();
	for {
		e, err := r.Next();
		if err != nil {
			fatal("reading DWARF entry: %s", err);
		}
		if e == nil {
			break;
		}
		if e.Tag != dwarf.TagVariable {
			goto Continue;
		}
		name, _ := e.Val(dwarf.AttrName).(string);
		typOff, _ := e.Val(dwarf.AttrType).(dwarf.Offset);
		if name == "" || typOff == 0 {
			fatal("malformed DWARF TagVariable entry");
		}
		if !strings.HasPrefix(name, "__cgo__") {
			goto Continue;
		}
		typ, err := d.Type(typOff);
		if err != nil {
			fatal("loading DWARF type: %s", err);
		}
		t, ok := typ.(*dwarf.PtrType);
		if !ok || t == nil {
			fatal("internal error: %s has non-pointer type", name);
		}
		i, err := strconv.Atoi(name[7:len(name)]);
		if err != nil {
			fatal("malformed __cgo__ name: %s", name);
		}
		types[i] = t.Type;

	Continue:
		if e.Tag != dwarf.TagCompileUnit {
			r.SkipChildren();
		}
	}

	// Record types and typedef information in Crefs.
	var conv typeConv;
	conv.Init(p.PtrSize);
	for _, c := range p.Crefs {
		i := m[c.Name];
		c.TypeName = kind[c.Name] == "type";
		f, fok := types[i].(*dwarf.FuncType);
		if c.Context == "call" && !c.TypeName && fok {
			c.FuncType = conv.FuncType(f);
		} else {
			c.Type = conv.Type(types[i]);
		}
	}
	p.Typedef = conv.typedef;
}

func concat(a, b []string) []string {
	c := make([]string, len(a)+len(b));
	for i, s := range a {
		c[i] = s;
	}
	for i, s := range b {
		c[i+len(a)] = s;
	}
	return c;
}

// gccDebug runs gcc -gdwarf-2 over the C program stdin and
// returns the corresponding DWARF data and any messages
// printed to standard error.
func (p *Prog) gccDebug(stdin []byte) (*dwarf.Data, string) {
	machine := "-m32";
	if p.PtrSize == 8 {
		machine = "-m64";
	}

	tmp := "_cgo_.o";
	base := []string{
		"gcc",
		machine,
		"-Wall",	// many warnings
		"-Werror",	// warnings are errors
		"-o"+tmp, 	// write object to tmp
		"-gdwarf-2", 	// generate DWARF v2 debugging symbols
		"-c",	// do not link
		"-xc", 	// input language is C
		"-",	// read input from standard input
	};
	_, stderr, ok := run(stdin, concat(base, p.GccOptions));
	if !ok {
		return nil, string(stderr);
	}

	// Try to parse f as ELF and Mach-O and hope one works.
	var f interface{DWARF() (*dwarf.Data, os.Error)};
	var err os.Error;
	if f, err = elf.Open(tmp); err != nil {
		if f, err = macho.Open(tmp); err != nil {
			fatal("cannot parse gcc output %s as ELF or Mach-O object", tmp);
		}
	}

	d, err := f.DWARF();
	if err != nil {
		fatal("cannot load DWARF debug information from %s: %s", tmp, err);
	}
	return d, "";
}

// A typeConv is a translator from dwarf types to Go types
// with equivalent memory layout.
type typeConv struct {
	// Cache of already-translated or in-progress types.
	m map[dwarf.Type]*Type;
	typedef map[string]ast.Expr;

	// Predeclared types.
	byte ast.Expr;	// denotes padding
	int8, int16, int32, int64 ast.Expr;
	uint8, uint16, uint32, uint64, uintptr ast.Expr;
	float32, float64 ast.Expr;
	void ast.Expr;
	unsafePointer ast.Expr;
	string ast.Expr;

	ptrSize int64;

	tagGen int;
}

func (c *typeConv) Init(ptrSize int64) {
	c.ptrSize = ptrSize;
	c.m = make(map[dwarf.Type]*Type);
	c.typedef = make(map[string]ast.Expr);
	c.byte = c.Ident("byte");
	c.int8 = c.Ident("int8");
	c.int16 = c.Ident("int16");
	c.int32 = c.Ident("int32");
	c.int64 = c.Ident("int64");
	c.uint8 = c.Ident("uint8");
	c.uint16 = c.Ident("uint16");
	c.uint32 = c.Ident("uint32");
	c.uint64 = c.Ident("uint64");
	c.uintptr = c.Ident("uintptr");
	c.float32 = c.Ident("float32");
	c.float64 = c.Ident("float64");
	c.unsafePointer = c.Ident("unsafe.Pointer");
	c.void = c.Ident("void");
	c.string = c.Ident("string");
}

// base strips away qualifiers and typedefs to get the underlying type
func base(dt dwarf.Type) dwarf.Type {
	for {
		if d, ok := dt.(*dwarf.QualType); ok {
			dt = d.Type;
			continue;
		}
		if d, ok := dt.(*dwarf.TypedefType); ok {
			dt = d.Type;
			continue;
		}
		break;
	}
	return dt;
}

// Map from dwarf text names to aliases we use in package "C".
var cnameMap = map[string] string {
	"long int": "long",
	"long unsigned int": "ulong",
	"unsigned int": "uint",
	"short unsigned int": "ushort",
	"short int": "short",
	"long long int": "longlong",
	"long long unsigned int": "ulonglong",
	"signed char": "schar",
};

// Type returns a *Type with the same memory layout as
// dtype when used as the type of a variable or a struct field.
func (c *typeConv) Type(dtype dwarf.Type) *Type {
	if t, ok := c.m[dtype]; ok {
		if t.Go == nil {
			fatal("type conversion loop at %s", dtype);
		}
		return t;
	}

	t := new(Type);
	t.Size = dtype.Size();
	t.Align = -1;
	t.C = dtype.Common().Name;
	if t.Size < 0 {
		fatal("dwarf.Type %s reports unknown size", dtype);
	}

	c.m[dtype] = t;
	switch dt := dtype.(type) {
	default:
		fatal("unexpected type: %s", dtype);

	case *dwarf.AddrType:
		if t.Size != c.ptrSize {
			fatal("unexpected: %d-byte address type - %s", t.Size, dtype);
		}
		t.Go = c.uintptr;
		t.Align = t.Size;

	case *dwarf.ArrayType:
		if dt.StrideBitSize > 0 {
			// Cannot represent bit-sized elements in Go.
			t.Go = c.Opaque(t.Size);
			break;
		}
		gt := &ast.ArrayType{
			Len: c.intExpr(dt.Count),
		};
		t.Go = gt;	// publish before recursive call
		sub := c.Type(dt.Type);
		t.Align = sub.Align;
		gt.Elt = sub.Go;
		t.C = fmt.Sprintf("typeof(%s[%d])", sub.C, dt.Count);

	case *dwarf.CharType:
		if t.Size != 1 {
			fatal("unexpected: %d-byte char type - %s", t.Size, dtype);
		}
		t.Go = c.int8;
		t.Align = 1;

	case *dwarf.EnumType:
		switch t.Size {
		default:
			fatal("unexpected: %d-byte enum type - %s", t.Size, dtype);
		case 1:
			t.Go = c.uint8;
		case 2:
			t.Go = c.uint16;
		case 4:
			t.Go = c.uint32;
		case 8:
			t.Go = c.uint64;
		}
		if t.Align = t.Size; t.Align >= c.ptrSize {
			t.Align = c.ptrSize;
		}
		t.C = "enum " + dt.EnumName;

	case *dwarf.FloatType:
		switch t.Size {
		default:
			fatal("unexpected: %d-byte float type - %s", t.Size, dtype);
		case 4:
			t.Go = c.float32;
		case 8:
			t.Go = c.float64;
		}
		if t.Align = t.Size; t.Align >= c.ptrSize {
			t.Align = c.ptrSize;
		}

	case *dwarf.FuncType:
		// No attempt at translation: would enable calls
		// directly between worlds, but we need to moderate those.
		t.Go = c.uintptr;
		t.Align = c.ptrSize;

	case *dwarf.IntType:
		if dt.BitSize > 0 {
			fatal("unexpected: %d-bit int type - %s", dt.BitSize, dtype);
		}
		switch t.Size {
		default:
			fatal("unexpected: %d-byte int type - %s", t.Size, dtype);
		case 1:
			t.Go = c.int8;
		case 2:
			t.Go = c.int16;
		case 4:
			t.Go = c.int32;
		case 8:
			t.Go = c.int64;
		}
		if t.Align = t.Size; t.Align >= c.ptrSize {
			t.Align = c.ptrSize;
		}

	case *dwarf.PtrType:
		t.Align = c.ptrSize;

		// Translate void* as unsafe.Pointer
		if _, ok := base(dt.Type).(*dwarf.VoidType); ok {
			t.Go = c.unsafePointer;
			t.C = "void*";
			break;
		}

		gt := &ast.StarExpr{};
		t.Go = gt;	// publish before recursive call
		sub := c.Type(dt.Type);
		gt.X = sub.Go;
		t.C = sub.C + "*";

	case *dwarf.QualType:
		// Ignore qualifier.
		t = c.Type(dt.Type);
		c.m[dtype] = t;
		return t;

	case *dwarf.StructType:
		// Convert to Go struct, being careful about alignment.
		// Have to give it a name to simulate C "struct foo" references.
		tag := dt.StructName;
		if tag == "" {
			tag = "__" + strconv.Itoa(c.tagGen);
			c.tagGen++;
		} else if t.C == "" {
			t.C = dt.Kind + " " + tag;
		}
		name := c.Ident("_C" + dt.Kind + "_" + tag);
		t.Go = name;	// publish before recursive calls
		switch dt.Kind {
		case "union", "class":
			c.typedef[name.Value] = c.Opaque(t.Size);
			if t.C == "" {
				t.C = fmt.Sprintf("typeof(unsigned char[%d])", t.Size);
			}
		case "struct":
			g, csyntax, align := c.Struct(dt);
			if t.C == "" {
				t.C = csyntax;
			}
			t.Align = align;
			c.typedef[name.Value] = g;
		}

	case *dwarf.TypedefType:
		// Record typedef for printing.
		if dt.Name == "_GoString_" {
			// Special C name for Go string type.
			// Knows string layout used by compilers: pointer plus length,
			// which rounds up to 2 pointers after alignment.
			t.Go = c.string;
			t.Size = c.ptrSize * 2;
			t.Align = c.ptrSize;
			break;
		}
		name := c.Ident("_C_" + dt.Name);
		t.Go = name;	// publish before recursive call
		sub := c.Type(dt.Type);
		t.Size = sub.Size;
		t.Align = sub.Align;
		if _, ok := c.typedef[name.Value]; !ok {
			c.typedef[name.Value] = sub.Go;
		}

	case *dwarf.UcharType:
		if t.Size != 1 {
			fatal("unexpected: %d-byte uchar type - %s", t.Size, dtype);
		}
		t.Go = c.uint8;
		t.Align = 1;

	case *dwarf.UintType:
		if dt.BitSize > 0 {
			fatal("unexpected: %d-bit uint type - %s", dt.BitSize, dtype);
		}
		switch t.Size {
		default:
			fatal("unexpected: %d-byte uint type - %s", t.Size, dtype);
		case 1:
			t.Go = c.uint8;
		case 2:
			t.Go = c.uint16;
		case 4:
			t.Go = c.uint32;
		case 8:
			t.Go = c.uint64;
		}
		if t.Align = t.Size; t.Align >= c.ptrSize {
			t.Align = c.ptrSize;
		}

	case *dwarf.VoidType:
		t.Go = c.void;
		t.C = "void";
	}

	switch dtype.(type) {
	case *dwarf.AddrType, *dwarf.CharType, *dwarf.IntType, *dwarf.FloatType, *dwarf.UcharType, *dwarf.UintType:
		s := dtype.Common().Name;
		if s != "" {
			if ss, ok := cnameMap[s]; ok {
				s = ss;
			}
			s = strings.Join(strings.Split(s, " ", 0), "");	// strip spaces
			name := c.Ident("_C_" + s);
			c.typedef[name.Value] = t.Go;
			t.Go = name;
		}
	}

	if t.C == "" {
		fatal("internal error: did not create C name for %s", dtype);
	}

	return t;
}

// FuncArg returns a Go type with the same memory layout as
// dtype when used as the type of a C function argument.
func (c *typeConv) FuncArg(dtype dwarf.Type) *Type {
	t := c.Type(dtype);
	switch dt := dtype.(type) {
	case *dwarf.ArrayType:
		// Arrays are passed implicitly as pointers in C.
		// In Go, we must be explicit.
		return &Type{
			Size: c.ptrSize,
			Align: c.ptrSize,
			Go: &ast.StarExpr{X: t.Go},
			C: t.C + "*"
		};
	case *dwarf.TypedefType:
		// C has much more relaxed rules than Go for
		// implicit type conversions.  When the parameter
		// is type T defined as *X, simulate a little of the
		// laxness of C by making the argument *X instead of T.
		if ptr, ok := base(dt.Type).(*dwarf.PtrType); ok {
			return c.Type(ptr);
		}
	}
	return t;
}

// FuncType returns the Go type analogous to dtype.
// There is no guarantee about matching memory layout.
func (c *typeConv) FuncType(dtype *dwarf.FuncType) *FuncType {
	p := make([]*Type, len(dtype.ParamType));
	gp := make([]*ast.Field, len(dtype.ParamType));
	for i, f := range dtype.ParamType {
		p[i] = c.FuncArg(f);
		gp[i] = &ast.Field{Type: p[i].Go};
	}
	var r *Type;
	var gr []*ast.Field;
	if _, ok := dtype.ReturnType.(*dwarf.VoidType); !ok && dtype.ReturnType != nil {
		r = c.Type(dtype.ReturnType);
		gr = []*ast.Field{&ast.Field{Type: r.Go}};
	}
	return &FuncType{
		Params: p,
		Result: r,
		Go: &ast.FuncType{
			Params: gp,
			Results: gr
		}
	};
}

// Identifier
func (c *typeConv) Ident(s string) *ast.Ident {
	return &ast.Ident{Value: s};
}

// Opaque type of n bytes.
func (c *typeConv) Opaque(n int64) ast.Expr {
	return &ast.ArrayType{
		Len: c.intExpr(n),
		Elt: c.byte
	};
}

// Expr for integer n.
func (c *typeConv) intExpr(n int64) ast.Expr {
	return &ast.BasicLit{
		Kind: token.INT,
		Value: strings.Bytes(strconv.Itoa64(n)),
	}
}

// Add padding of given size to fld.
func (c *typeConv) pad(fld []*ast.Field, size int64) []*ast.Field {
	n := len(fld);
	fld = fld[0:n+1];
	fld[n] = &ast.Field{Names: []*ast.Ident{c.Ident("_")}, Type: c.Opaque(size)};
	return fld;
}

// Struct conversion
func (c *typeConv) Struct(dt *dwarf.StructType) (expr *ast.StructType, csyntax string, align int64)  {
	csyntax = "struct { ";
	fld := make([]*ast.Field, 0, 2*len(dt.Field)+1);	// enough for padding around every field
	off := int64(0);
	for _, f := range dt.Field {
		if f.ByteOffset > off {
			fld = c.pad(fld, f.ByteOffset - off);
			off = f.ByteOffset;
		}
		t := c.Type(f.Type);
		n := len(fld);
		fld = fld[0:n+1];
		fld[n] = &ast.Field{Names: []*ast.Ident{c.Ident(f.Name)}, Type: t.Go};
		off += t.Size;
		csyntax += t.C + " " + f.Name + "; ";
		if t.Align > align {
			align = t.Align;
		}
	}
	if off < dt.ByteSize {
		fld = c.pad(fld, dt.ByteSize - off);
		off = dt.ByteSize;
	}
	if off != dt.ByteSize {
		fatal("struct size calculation error");
	}
	csyntax += "}";
	expr = &ast.StructType{Fields: fld};
	return;
}
