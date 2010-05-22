// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cgo; see gmp.go for an overview.

// TODO(rsc):
//	Emit correct line number annotations.
//	Make 6g understand the annotations.

package main

import (
	"fmt"
	"go/ast"
	"os"
	"strings"
)

func usage() { fmt.Fprint(os.Stderr, "usage: cgo [compiler options] file.go ...\n") }

var ptrSizeMap = map[string]int64{
	"386":   4,
	"amd64": 8,
	"arm":   4,
}

var expandName = map[string]string{
	"schar":     "signed char",
	"uchar":     "unsigned char",
	"ushort":    "unsigned short",
	"uint":      "unsigned int",
	"ulong":     "unsigned long",
	"longlong":  "long long",
	"ulonglong": "unsigned long long",
}

func main() {
	args := os.Args
	if len(args) < 2 {
		usage()
		os.Exit(2)
	}

	// Find first arg that looks like a go file and assume everything before
	// that are options to pass to gcc.
	var i int
	for i = len(args) - 1; i > 0; i-- {
		if !strings.HasSuffix(args[i], ".go") {
			break
		}
	}

	i += 1

	gccOptions, goFiles := args[1:i], args[i:]

	arch := os.Getenv("GOARCH")
	if arch == "" {
		fatal("$GOARCH is not set")
	}
	ptrSize, ok := ptrSizeMap[arch]
	if !ok {
		fatal("unknown architecture %s", arch)
	}

	// Clear locale variables so gcc emits English errors [sic].
	os.Setenv("LANG", "en_US.UTF-8")
	os.Setenv("LC_ALL", "C")
	os.Setenv("LC_CTYPE", "C")

	p := new(Prog)

	p.PtrSize = ptrSize
	p.GccOptions = gccOptions
	p.Vardef = make(map[string]*Type)
	p.Funcdef = make(map[string]*FuncType)
	p.Enumdef = make(map[string]int64)
	p.Constdef = make(map[string]string)
	p.OutDefs = make(map[string]bool)

	for _, input := range goFiles {
		// Reset p.Preamble so that we don't end up with conflicting headers / defines
		p.Preamble = builtinProlog
		openProg(input, p)
		for _, cref := range p.Crefs {
			// Convert C.ulong to C.unsigned long, etc.
			if expand, ok := expandName[cref.Name]; ok {
				cref.Name = expand
			}
		}
		p.loadDebugInfo()
		for _, cref := range p.Crefs {
			switch cref.Context {
			case "const":
				// This came from a #define and we'll output it later.
				*cref.Expr = ast.NewIdent(cref.Name)
				break
			case "call":
				if !cref.TypeName {
					// Is an actual function call.
					pos := (*cref.Expr).Pos()
					*cref.Expr = &ast.Ident{Position: pos, Obj: ast.NewObj(ast.Err, pos, "_C_"+cref.Name)}
					p.Funcdef[cref.Name] = cref.FuncType
					break
				}
				*cref.Expr = cref.Type.Go
			case "expr":
				if cref.TypeName {
					error((*cref.Expr).Pos(), "type C.%s used as expression", cref.Name)
				}
				// If the expression refers to an enumerated value, then
				// place the identifier for the value and add it to Enumdef so
				// it will be declared as a constant in the later stage.
				if cref.Type.EnumValues != nil {
					*cref.Expr = ast.NewIdent(cref.Name)
					p.Enumdef[cref.Name] = cref.Type.EnumValues[cref.Name]
					break
				}
				// Reference to C variable.
				// We declare a pointer and arrange to have it filled in.
				*cref.Expr = &ast.StarExpr{X: ast.NewIdent("_C_" + cref.Name)}
				p.Vardef[cref.Name] = cref.Type
			case "type":
				if !cref.TypeName {
					error((*cref.Expr).Pos(), "expression C.%s used as type", cref.Name)
				}
				*cref.Expr = cref.Type.Go
			}
		}
		if nerrors > 0 {
			os.Exit(2)
		}
		pkg := p.Package
		if dir := os.Getenv("CGOPKGPATH"); dir != "" {
			pkg = dir + "/" + pkg
		}
		p.PackagePath = pkg
		p.writeOutput(input)
	}

	p.writeDefs()
}
