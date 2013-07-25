// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"os"
	"strings"
)

// godefs returns the output for -godefs mode.
func (p *Package) godefs(f *File, srcfile string) string {
	var buf bytes.Buffer

	fmt.Fprintf(&buf, "// Created by cgo -godefs - DO NOT EDIT\n")
	fmt.Fprintf(&buf, "// %s\n", strings.Join(os.Args, " "))
	fmt.Fprintf(&buf, "\n")

	override := make(map[string]string)

	// Allow source file to specify override mappings.
	// For example, the socket data structures refer
	// to in_addr and in_addr6 structs but we want to be
	// able to treat them as byte arrays, so the godefs
	// inputs in package syscall say
	//
	//	// +godefs map struct_in_addr [4]byte
	//	// +godefs map struct_in_addr6 [16]byte
	//
	for _, g := range f.Comments {
		for _, c := range g.List {
			i := strings.Index(c.Text, "+godefs map")
			if i < 0 {
				continue
			}
			s := strings.TrimSpace(c.Text[i+len("+godefs map"):])
			i = strings.Index(s, " ")
			if i < 0 {
				fmt.Fprintf(os.Stderr, "invalid +godefs map comment: %s\n", c.Text)
				continue
			}
			override["_Ctype_"+strings.TrimSpace(s[:i])] = strings.TrimSpace(s[i:])
		}
	}
	for _, n := range f.Name {
		if s := override[n.Go]; s != "" {
			override[n.Mangle] = s
		}
	}

	// Otherwise, if the source file says type T C.whatever,
	// use "T" as the mangling of C.whatever,
	// except in the definition (handled at end of function).
	refName := make(map[*ast.Expr]*Name)
	for _, r := range f.Ref {
		refName[r.Expr] = r.Name
	}
	for _, d := range f.AST.Decls {
		d, ok := d.(*ast.GenDecl)
		if !ok || d.Tok != token.TYPE {
			continue
		}
		for _, s := range d.Specs {
			s := s.(*ast.TypeSpec)
			n := refName[&s.Type]
			if n != nil && n.Mangle != "" {
				override[n.Mangle] = s.Name.Name
			}
		}
	}

	// Extend overrides using typedefs:
	// If we know that C.xxx should format as T
	// and xxx is a typedef for yyy, make C.yyy format as T.
	for typ, def := range typedef {
		if new := override[typ]; new != "" {
			if id, ok := def.Go.(*ast.Ident); ok {
				override[id.Name] = new
			}
		}
	}

	// Apply overrides.
	for old, new := range override {
		if id := goIdent[old]; id != nil {
			id.Name = new
		}
	}

	// Any names still using the _C syntax are not going to compile,
	// although in general we don't know whether they all made it
	// into the file, so we can't warn here.
	//
	// The most common case is union types, which begin with
	// _Ctype_union and for which typedef[name] is a Go byte
	// array of the appropriate size (such as [4]byte).
	// Substitute those union types with byte arrays.
	for name, id := range goIdent {
		if id.Name == name && strings.Contains(name, "_Ctype_union") {
			if def := typedef[name]; def != nil {
				id.Name = gofmt(def)
			}
		}
	}

	conf.Fprint(&buf, fset, f.AST)

	return buf.String()
}

// cdefs returns the output for -cdefs mode.
// The easiest way to do this is to translate the godefs Go to C.
func (p *Package) cdefs(f *File, srcfile string) string {
	godefsOutput := p.godefs(f, srcfile)

	lines := strings.Split(godefsOutput, "\n")
	lines[0] = "// Created by cgo -cdefs - DO NOT EDIT"

	for i, line := range lines {
		lines[i] = strings.TrimSpace(line)
	}

	var out bytes.Buffer
	printf := func(format string, args ...interface{}) { fmt.Fprintf(&out, format, args...) }

	didTypedef := false
	for i := 0; i < len(lines); i++ {
		line := lines[i]

		// Delete
		//	package x
		if strings.HasPrefix(line, "package ") {
			continue
		}

		// Convert
		//	const (
		//		A = 1
		//		B = 2
		//	)
		//
		// to
		//
		//	enum {
		//		A = 1,
		//		B = 2,
		//	};
		if line == "const (" {
			printf("enum {\n")
			for i++; i < len(lines) && lines[i] != ")"; i++ {
				line = lines[i]
				if line != "" {
					printf("\t%s,", line)
				}
				printf("\n")
			}
			printf("};\n")
			continue
		}

		// Convert
		//	const A = 1
		// to
		//	enum { A = 1 };
		if strings.HasPrefix(line, "const ") {
			printf("enum { %s };\n", line[len("const "):])
			continue
		}

		// On first type definition, typedef all the structs
		// in case there are dependencies between them.
		if !didTypedef && strings.HasPrefix(line, "type ") {
			didTypedef = true
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "type ") && strings.HasSuffix(line, " struct {") {
					s := strings.TrimSuffix(strings.TrimPrefix(line, "type "), " struct {")
					printf("typedef struct %s %s;\n", s, s)
				}
			}
			printf("\n")
			printf("#pragma pack on\n")
			printf("\n")
		}

		// Convert
		//	type T struct {
		//		X int64
		//		Y *int32
		//		Z [4]byte
		//	}
		//
		// to
		//
		//	struct T {
		//		int64 X;
		//		int32 *Y;
		//		byte Z[4];
		//	}
		if strings.HasPrefix(line, "type ") && strings.HasSuffix(line, " struct {") {
			if len(lines) > i+1 && lines[i+1] == "}" {
				// do not output empty struct
				i++
				continue
			}
			s := line[len("type ") : len(line)-len(" struct {")]
			printf("struct %s {\n", s)
			for i++; i < len(lines) && lines[i] != "}"; i++ {
				line := lines[i]
				if line != "" {
					f := strings.Fields(line)
					if len(f) != 2 {
						fmt.Fprintf(os.Stderr, "cgo: cannot parse struct field: %s\n", line)
						nerrors++
						continue
					}
					printf("\t%s;", cdecl(f[0], f[1]))
				}
				printf("\n")
			}
			printf("};\n")
			continue
		}

		// Convert
		//	type T int
		// to
		//	typedef int T;
		if strings.HasPrefix(line, "type ") {
			f := strings.Fields(line[len("type "):])
			if len(f) != 2 {
				fmt.Fprintf(os.Stderr, "cgo: cannot parse type definition: %s\n", line)
				nerrors++
				continue
			}
			printf("typedef\t%s;\n", cdecl(f[0], f[1]))
			continue
		}

		printf("%s\n", line)
	}

	if didTypedef {
		printf("\n")
		printf("#pragma pack off\n")
	}

	return out.String()
}

// cdecl returns the C declaration for the given Go name and type.
// It only handles the specific cases necessary for converting godefs output.
func cdecl(name, typ string) string {
	// X *[0]byte -> X *void
	if strings.HasPrefix(typ, "*[0]") {
		typ = "*void"
	}
	// X [4]byte -> X[4] byte
	for strings.HasPrefix(typ, "[") {
		i := strings.Index(typ, "]") + 1
		name = name + typ[:i]
		typ = typ[i:]
	}
	// X *byte -> *X byte
	for strings.HasPrefix(typ, "*") {
		name = "*" + name
		typ = typ[1:]
	}
	// X T -> T X
	// Handle the special case: 'unsafe.Pointer' is 'void *'
	if typ == "unsafe.Pointer" {
		typ = "void"
		name = "*" + name
	}
	return typ + "\t" + name
}

var gofmtBuf bytes.Buffer

// gofmt returns the gofmt-formatted string for an AST node.
func gofmt(n interface{}) string {
	gofmtBuf.Reset()
	err := printer.Fprint(&gofmtBuf, fset, n)
	if err != nil {
		return "<" + err.Error() + ">"
	}
	return gofmtBuf.String()
}
