// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt";
	"go/ast";
	"go/printer";
	"os";
)

// writeOutput creates output files to be compiled by 6g, 6c, and gcc.
// (The comments here say 6g and 6c but the code applies to the 8 and 5 tools too.)
func (p *Prog) writeOutput(srcfile, go_, c, gcc string) {
	fgo, err := os.Open(go_, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0666);
	if err != nil {
		fatal("%s", err);
	}
	fc, err := os.Open(c, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0666);
	if err != nil {
		fatal("%s", err);
	}
	fgcc, err := os.Open(gcc, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0666);
	if err != nil {
		fatal("%s", err);
	}

	// Write Go output: Go input with rewrites of C.xxx to _C_xxx,
	// then append the definitions of the _C_xxx types and vars and funcs.
	fmt.Fprintf(fgo, "//line %s:1\n", srcfile);
	printer.Fprint(fgo, p.AST, 0, 8);
	fmt.Fprintf(fgo, "\n\n// Added by cgo\n");

	for name, def := range p.Typedef {
		fmt.Fprintf(fgo, "type %s ", name);
		printer.Fprint(fgo, def, 0, 8);
		fmt.Fprintf(fgo, "\n");
	}
	fmt.Fprintf(fgo, "type _C_void [0]byte\n");

	// While we process the vars and funcs, also write 6c and gcc output.
	// Gcc output starts with the preamble.
	fmt.Fprintf(fgcc, "%s\n", p.Preamble);
	fmt.Fprintf(fgcc, "%s\n", gccProlog);

	fmt.Fprintf(fc, cProlog, p.Package, p.Package);

	for name, def := range p.Vardef {
		fmt.Fprintf(fc, "#pragma dynld %s·_C_%s %s \"%s.so\"\n", p.Package, name, name, p.PackagePath);
		fmt.Fprintf(fgo, "var _C_%s ", name);
		printer.Fprint(fgo, &ast.StarExpr{X: def.Go}, 0, 8);
		fmt.Fprintf(fgo, "\n");
	}
	fmt.Fprintf(fc, "\n");

	for name, def := range p.Funcdef {
		// Go func declaration.
		d := &ast.FuncDecl{
			Name: &ast.Ident{Value: "_C_" + name},
			Type: def.Go,
		};
		printer.Fprint(fgo, d, 0, 8);
		fmt.Fprintf(fgo, "\n");

		if name == "CString" || name == "GoString" {
			// The builtins are already defined in the C prolog.
			continue;
		}

		// Construct a gcc struct matching the 6c argument frame.
		// Assumes that in gcc, char is 1 byte, short 2 bytes, int 4 bytes, long long 8 bytes.
		// These assumptions are checked by the gccProlog.
		// Also assumes that 6c convention is to word-align the
		// input and output parameters.
		structType := "struct {\n";
		off := int64(0);
		npad := 0;
		for i, t := range def.Params {
			if off%t.Align != 0 {
				pad := t.Align - off%t.Align;
				structType += fmt.Sprintf("\t\tchar __pad%d[%d];\n", npad, pad);
				off += pad;
				npad++;
			}
			structType += fmt.Sprintf("\t\t%s p%d;\n", t.C, i);
			off += t.Size;
		}
		if off%ptrSize != 0 {
			pad := ptrSize - off%ptrSize;
			structType += fmt.Sprintf("\t\tchar __pad%d[%d];\n", npad, pad);
			off += pad;
			npad++;
		}
		if t := def.Result; t != nil {
			if off%t.Align != 0 {
				pad := t.Align - off%t.Align;
				structType += fmt.Sprintf("\t\tchar __pad%d[%d];\n", npad, pad);
				off += pad;
				npad++;
			}
			structType += fmt.Sprintf("\t\t%s r;\n", t.C);
			off += t.Size;
		}
		if off%ptrSize != 0 {
			pad := ptrSize - off%ptrSize;
			structType += fmt.Sprintf("\t\tchar __pad%d[%d];\n", npad, pad);
			off += pad;
			npad++;
		}
		if len(def.Params) == 0 && def.Result == nil {
			structType += "\t\tchar unused;\n";	// avoid empty struct
			off++;
		}
		structType += "\t}";
		argSize := off;

		// C wrapper calls into gcc, passing a pointer to the argument frame.
		// Also emit #pragma to get a pointer to the gcc wrapper.
		fmt.Fprintf(fc, "#pragma dynld _cgo_%s _cgo_%s \"%s.so\"\n", name, name, p.PackagePath);
		fmt.Fprintf(fc, "void (*_cgo_%s)(void*);\n", name);
		fmt.Fprintf(fc, "\n");
		fmt.Fprintf(fc, "void\n");
		fmt.Fprintf(fc, "%s·_C_%s(struct{uint8 x[%d];}p)\n", p.Package, name, argSize);
		fmt.Fprintf(fc, "{\n");
		fmt.Fprintf(fc, "\tcgocall(_cgo_%s, &p);\n", name);
		fmt.Fprintf(fc, "}\n");
		fmt.Fprintf(fc, "\n");

		// Gcc wrapper unpacks the C argument struct
		// and calls the actual C function.
		fmt.Fprintf(fgcc, "void\n");
		fmt.Fprintf(fgcc, "_cgo_%s(void *v)\n", name);
		fmt.Fprintf(fgcc, "{\n");
		fmt.Fprintf(fgcc, "\t%s *a = v;\n", structType);
		fmt.Fprintf(fgcc, "\t");
		if def.Result != nil {
			fmt.Fprintf(fgcc, "a->r = ");
		}
		fmt.Fprintf(fgcc, "%s(", name);
		for i := range def.Params {
			if i > 0 {
				fmt.Fprintf(fgcc, ", ");
			}
			fmt.Fprintf(fgcc, "a->p%d", i);
		}
		fmt.Fprintf(fgcc, ");\n");
		fmt.Fprintf(fgcc, "}\n");
		fmt.Fprintf(fgcc, "\n");
	}
}

const gccProlog = `
// Usual nonsense: if x and y are not equal, the type will be invalid
// (have a negative array count) and an inscrutable error will come
// out of the compiler and hopefully mention "name".
#define __cgo_compile_assert_eq(x, y, name) typedef char name[(x-y)*(x-y)*-2+1];

// Check at compile time that the sizes we use match our expectations.
#define __cgo_size_assert(t, n) __cgo_compile_assert_eq(sizeof(t), n, _cgo_sizeof_##t##_is_not_##n)

__cgo_size_assert(char, 1)
__cgo_size_assert(short, 2)
__cgo_size_assert(int, 4)
typedef long long __cgo_long_long;
__cgo_size_assert(__cgo_long_long, 8)
__cgo_size_assert(float, 4)
__cgo_size_assert(double, 8)
`

const builtinProlog = `
typedef struct { char *p; int n; } _GoString_;
_GoString_ GoString(char *p);
char *CString(_GoString_);
`

const cProlog = `
#include "runtime.h"
#include "cgocall.h"

#pragma dynld initcgo initcgo "libcgo.so"
#pragma dynld cgo cgo "libcgo.so"
#pragma dynld _cgo_malloc _cgo_malloc "libcgo.so"
#pragma dynld _cgo_free free "libcgo.so"

void
%s·_C_GoString(int8 *p, String s)
{
	s = gostring((byte*)p);
	FLUSH(&s);
}

void
%s·_C_CString(String s, int8 *p)
{
	p = cmalloc(s.len+1);
	mcpy((byte*)p, s.str, s.len);
	p[s.len] = 0;
	FLUSH(&p);
}
`

