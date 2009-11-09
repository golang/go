// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cgo; see gmp.go for an overview.

// TODO(rsc):
//	Emit correct line number annotations.
//	Make 6g understand the annotations.

package main

import (
	"fmt";
	"go/ast";
	"os";
)

func usage()	{ fmt.Fprint(os.Stderr, "usage: cgo [compiler options] file.go\n") }

var ptrSizeMap = map[string]int64{
	"386": 4,
	"amd64": 8,
	"arm": 4,
}

var expandName = map[string]string{
	"schar": "signed char",
	"uchar": "unsigned char",
	"ushort": "unsigned short",
	"uint": "unsigned int",
	"ulong": "unsigned long",
	"longlong": "long long",
	"ulonglong": "unsigned long long",
}

func main() {
	args := os.Args;
	if len(args) < 2 {
		usage();
		os.Exit(2);
	}
	gccOptions := args[1 : len(args)-1];
	input := args[len(args)-1];

	arch := os.Getenv("GOARCH");
	if arch == "" {
		fatal("$GOARCH is not set")
	}
	ptrSize, ok := ptrSizeMap[arch];
	if !ok {
		fatal("unknown architecture %s", arch)
	}

	p := openProg(input);
	for _, cref := range p.Crefs {
		// Convert C.ulong to C.unsigned long, etc.
		if expand, ok := expandName[cref.Name]; ok {
			cref.Name = expand
		}
	}

	p.PtrSize = ptrSize;
	p.Preamble = p.Preamble + "\n" + builtinProlog;
	p.GccOptions = gccOptions;
	p.loadDebugInfo();
	p.Vardef = make(map[string]*Type);
	p.Funcdef = make(map[string]*FuncType);

	for _, cref := range p.Crefs {
		switch cref.Context {
		case "call":
			if !cref.TypeName {
				// Is an actual function call.
				*cref.Expr = &ast.Ident{Value: "_C_" + cref.Name};
				p.Funcdef[cref.Name] = cref.FuncType;
				break;
			}
			*cref.Expr = cref.Type.Go;
		case "expr":
			if cref.TypeName {
				error((*cref.Expr).Pos(), "type C.%s used as expression", cref.Name)
			}
			// Reference to C variable.
			// We declare a pointer and arrange to have it filled in.
			*cref.Expr = &ast.StarExpr{X: &ast.Ident{Value: "_C_" + cref.Name}};
			p.Vardef[cref.Name] = cref.Type;
		case "type":
			if !cref.TypeName {
				error((*cref.Expr).Pos(), "expression C.%s used as type", cref.Name)
			}
			*cref.Expr = cref.Type.Go;
		}
	}
	if nerrors > 0 {
		os.Exit(2)
	}

	p.PackagePath = p.Package;
	p.writeOutput(input);
}
