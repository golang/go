// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/types"
	"strings"
)

// formatCompletion creates a completion item for a given types.Object.
func (c *completer) item(obj types.Object, score float64) CompletionItem {
	label := obj.Name()
	detail := types.TypeString(obj.Type(), c.qf)
	var kind CompletionItemKind

	switch o := obj.(type) {
	case *types.TypeName:
		detail, kind = formatType(o.Type(), c.qf)
		if obj.Parent() == types.Universe {
			detail = ""
		}
	case *types.Const:
		if obj.Parent() == types.Universe {
			detail = ""
		} else {
			val := o.Val().ExactString()
			if !strings.Contains(val, "\\n") { // skip any multiline constants
				label += " = " + o.Val().ExactString()
			}
		}
		kind = ConstantCompletionItem
	case *types.Var:
		if _, ok := o.Type().(*types.Struct); ok {
			detail = "struct{...}" // for anonymous structs
		}
		if o.IsField() {
			kind = FieldCompletionItem
		} else if c.isParameter(o) {
			kind = ParameterCompletionItem
		} else {
			kind = VariableCompletionItem
		}
	case *types.Func:
		if sig, ok := o.Type().(*types.Signature); ok {
			label += formatParams(sig, c.qf)
			detail = strings.Trim(types.TypeString(sig.Results(), c.qf), "()")
			kind = FunctionCompletionItem
			if sig.Recv() != nil {
				kind = MethodCompletionItem
			}
		}
	case *types.Builtin:
		item, ok := builtinDetails[obj.Name()]
		if !ok {
			break
		}
		label, detail = item.label, item.detail
		kind = FunctionCompletionItem
	case *types.PkgName:
		kind = PackageCompletionItem
		detail = fmt.Sprintf("\"%s\"", o.Imported().Path())
	case *types.Nil:
		kind = VariableCompletionItem
		detail = ""
	}
	detail = strings.TrimPrefix(detail, "untyped ")

	return CompletionItem{
		Label:  label,
		Detail: detail,
		Kind:   kind,
		Score:  score,
	}
}

// isParameter returns true if the given *types.Var is a parameter
// of the enclosingFunction.
func (c *completer) isParameter(v *types.Var) bool {
	if c.enclosingFunction == nil {
		return false
	}
	for i := 0; i < c.enclosingFunction.Params().Len(); i++ {
		if c.enclosingFunction.Params().At(i) == v {
			return true
		}
	}
	return false
}

// formatType returns the detail and kind for an object of type *types.TypeName.
func formatType(typ types.Type, qf types.Qualifier) (detail string, kind CompletionItemKind) {
	if types.IsInterface(typ) {
		detail = "interface{...}"
		kind = InterfaceCompletionItem
	} else if _, ok := typ.(*types.Struct); ok {
		detail = "struct{...}"
		kind = StructCompletionItem
	} else if typ != typ.Underlying() {
		detail, kind = formatType(typ.Underlying(), qf)
	} else {
		detail = types.TypeString(typ, qf)
		kind = TypeCompletionItem
	}
	return detail, kind
}

// formatParams correctly format the parameters of a function.
func formatParams(sig *types.Signature, qf types.Qualifier) string {
	var b bytes.Buffer
	b.WriteByte('(')
	for i := 0; i < sig.Params().Len(); i++ {
		if i > 0 {
			b.WriteString(", ")
		}
		el := sig.Params().At(i)
		typ := types.TypeString(el.Type(), qf)
		// Handle a variadic parameter (can only be the final parameter).
		if sig.Variadic() && i == sig.Params().Len()-1 {
			typ = strings.Replace(typ, "[]", "...", 1)
		}
		if el.Name() == "" {
			fmt.Fprintf(&b, "%v", typ)
		} else {
			fmt.Fprintf(&b, "%v %v", el.Name(), typ)
		}
	}
	b.WriteByte(')')
	return b.String()
}

// qualifier returns a function that appropriately formats a types.PkgName
// appearing in a *ast.File.
func qualifier(f *ast.File, pkg *types.Package, info *types.Info) types.Qualifier {
	// Construct mapping of import paths to their defined or implicit names.
	imports := make(map[*types.Package]string)
	for _, imp := range f.Imports {
		var obj types.Object
		if imp.Name != nil {
			obj = info.Defs[imp.Name]
		} else {
			obj = info.Implicits[imp]
		}
		if pkgname, ok := obj.(*types.PkgName); ok {
			imports[pkgname.Imported()] = pkgname.Name()
		}
	}
	// Define qualifier to replace full package paths with names of the imports.
	return func(p *types.Package) string {
		if p == pkg {
			return ""
		}
		if name, ok := imports[p]; ok {
			return name
		}
		return p.Name()
	}
}

type itemDetails struct {
	label, detail string
}

var builtinDetails = map[string]itemDetails{
	"append": { // append(slice []T, elems ...T)
		label:  "append(slice []T, elems ...T)",
		detail: "[]T",
	},
	"cap": { // cap(v []T) int
		label:  "cap(v []T)",
		detail: "int",
	},
	"close": { // close(c chan<- T)
		label: "close(c chan<- T)",
	},
	"complex": { // complex(r, i float64) complex128
		label:  "complex(real float64, imag float64)",
		detail: "complex128",
	},
	"copy": { // copy(dst, src []T) int
		label:  "copy(dst []T, src []T)",
		detail: "int",
	},
	"delete": { // delete(m map[T]T1, key T)
		label: "delete(m map[K]V, key K)",
	},
	"imag": { // imag(c complex128) float64
		label:  "imag(complex128)",
		detail: "float64",
	},
	"len": { // len(v T) int
		label:  "len(T)",
		detail: "int",
	},
	"make": { // make(t T, size ...int) T
		label:  "make(t T, size ...int)",
		detail: "T",
	},
	"new": { // new(T) *T
		label:  "new(T)",
		detail: "*T",
	},
	"panic": { // panic(v interface{})
		label: "panic(interface{})",
	},
	"print": { // print(args ...T)
		label: "print(args ...T)",
	},
	"println": { // println(args ...T)
		label: "println(args ...T)",
	},
	"real": { // real(c complex128) float64
		label:  "real(complex128)",
		detail: "float64",
	},
	"recover": { // recover() interface{}
		label:  "recover()",
		detail: "interface{}",
	},
}
