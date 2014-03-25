// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io"

	"code.google.com/p/go.tools/go/types"
)

// TODO(gri) handle indentation
// TODO(gri) filter unexported fields of struct types?
// TODO(gri) use tabwriter for alignment?

func print(w io.Writer, pkg *types.Package, filter func(types.Object) bool) {
	var p printer
	p.pkg = pkg
	p.printPackage(pkg, filter)
	io.Copy(w, &p.buf)
}

type printer struct {
	pkg    *types.Package
	buf    bytes.Buffer
	indent int
}

func (p *printer) print(s string) {
	p.buf.WriteString(s)
}

func (p *printer) printf(format string, args ...interface{}) {
	fmt.Fprintf(&p.buf, format, args...)
}

func (p *printer) printPackage(pkg *types.Package, filter func(types.Object) bool) {
	// collect objects by kind
	var (
		consts []*types.Const
		typez  []*types.TypeName // types without methods
		typem  []*types.TypeName // types with methods
		vars   []*types.Var
		funcs  []*types.Func
	)
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if !filter(obj) {
			continue
		}
		switch obj := obj.(type) {
		case *types.Const:
			consts = append(consts, obj)
		case *types.TypeName:
			if obj.Type().(*types.Named).NumMethods() > 0 {
				typem = append(typem, obj)
			} else {
				typez = append(typez, obj)
			}
		case *types.Var:
			vars = append(vars, obj)
		case *types.Func:
			funcs = append(funcs, obj)
		}
	}

	p.printf("package %s\n\n", pkg.Name())

	if len(consts) > 0 {
		p.print("const (\n")
		for _, obj := range consts {
			p.printObj(obj)
			p.print("\n")
		}
		p.print(")\n\n")
	}

	if len(vars) > 0 {
		p.print("var (\n")
		for _, obj := range vars {
			p.printObj(obj)
			p.print("\n")
		}
		p.print(")\n\n")
	}

	if len(typez) > 0 {
		p.print("type (\n")
		for _, obj := range typez {
			p.printf("\t%s ", obj.Name())
			types.WriteType(&p.buf, p.pkg, obj.Type().Underlying())
			p.print("\n")
		}
		p.print(")\n\n")
	}

	for _, obj := range typem {
		p.printf("type %s ", obj.Name())
		typ := obj.Type().(*types.Named)
		types.WriteType(&p.buf, p.pkg, typ.Underlying())
		p.print("\n")
		for i, n := 0, typ.NumMethods(); i < n; i++ {
			p.printFunc(typ.Method(i))
			p.print("\n")
		}
		p.print("\n")
	}

	for _, obj := range funcs {
		p.printFunc(obj)
		p.print("\n")
	}

	p.print("\n")
}

func (p *printer) printObj(obj types.Object) {
	p.printf("\t %s", obj.Name())
	// don't write untyped types (for constants)
	if typ := obj.Type(); typed(typ) {
		p.print(" ")
		types.WriteType(&p.buf, p.pkg, typ)
	}
	// write constant value
	if obj, ok := obj.(*types.Const); ok {
		p.printf(" = %s", obj.Val())
	}
}

func (p *printer) printFunc(obj *types.Func) {
	p.print("func ")
	sig := obj.Type().(*types.Signature)
	if recv := sig.Recv(); recv != nil {
		p.print("(")
		if name := recv.Name(); name != "" {
			p.print(name)
			p.print(" ")
		}
		types.WriteType(&p.buf, p.pkg, recv.Type())
		p.print(") ")
	}
	p.print(obj.Name())
	types.WriteSignature(&p.buf, p.pkg, sig)
}

func typed(typ types.Type) bool {
	if t, ok := typ.Underlying().(*types.Basic); ok {
		return t.Info()&types.IsUntyped == 0
	}
	return true
}
