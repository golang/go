// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cgo; see gmp.go for an overview.

// TODO(rsc):
//	Emit correct line number annotations.
//	Make 6g understand the annotations.

package main

import (
	"flag";
	"fmt";
	"go/ast";
	"os";
)

func usage() {
	fmt.Fprint(os.Stderr, "usage: cgo file.cgo\n");
	flag.PrintDefaults();
}

var ptrSizeMap = map[string]int64 {
	"386": 4,
	"amd64": 8,
	"arm": 4
}

func main() {
	flag.Usage = usage;
	flag.Parse();

	arch := os.Getenv("GOARCH");
	if arch == "" {
		fatal("$GOARCH is not set");
	}
	ptrSize, ok := ptrSizeMap[arch];
	if !ok {
		fatal("unknown architecture %s", arch);
	}

	args := flag.Args();
	if len(args) != 1 {
		usage();
		os.Exit(2);
	}
	p := openProg(args[0]);
	p.PtrSize = ptrSize;
	p.Preamble = p.Preamble + "\n" + builtinProlog;
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
				error((*cref.Expr).Pos(), "type C.%s used as expression", cref.Name);
			}
			// Reference to C variable.
			// We declare a pointer and arrange to have it filled in.
			*cref.Expr = &ast.StarExpr{X: &ast.Ident{Value: "_C_" + cref.Name}};
			p.Vardef[cref.Name] = cref.Type;
		case "type":
			if !cref.TypeName {
				error((*cref.Expr).Pos(), "expression C.%s used as type", cref.Name);
			}
			*cref.Expr = cref.Type.Go;
		}
	}
	if nerrors > 0 {
		os.Exit(2);
	}

	p.PackagePath = p.Package;
	p.writeOutput(args[0]);
}
