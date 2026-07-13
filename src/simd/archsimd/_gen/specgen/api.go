// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"fmt"
	"go/types"
	"simd/archsimd/_gen/specgen/specexpr"
	"strings"
)

// Func represents a function or method in the SIMD API.
type Func struct {
	Name string

	// Doc is the function documentation, without any leading comment markers
	Doc string

	// Recv, if non-zero, is the shape of the receiver. The name of the receiver
	// is always "x".
	Recv Arg

	In  []Arg
	Out []Arg

	// specFunc and instance describe the underlying spec function and its
	// instantiation that led to this API function.
	specFunc      *specFunc
	typeParamVars map[*types.TypeParam]specexpr.Variable
	instance      *specexpr.Bindings
}

type Arg struct {
	Name string
	Type specexpr.Type
}

func (f *Func) Signature() string {
	var buf strings.Builder
	buf.WriteString("func ")
	argList := func(args []Arg, canShort bool) {
		if canShort {
			if len(args) == 0 {
				return
			} else if len(args) == 1 && args[0].Name == "" {
				buf.WriteString(args[0].Type.String())
				return
			}
		}
		buf.WriteByte('(')
		for i, arg := range args {
			if i > 0 {
				buf.WriteString(", ")
			}
			if arg.Name == "" {
				panic("empty parameter/result name")
			}
			fmt.Fprintf(&buf, "%s %s", arg.Name, arg.Type)
		}
		buf.WriteByte(')')
	}
	if f.Recv.Type != nil {
		fmt.Fprintf(&buf, "(%s %s) ", f.Recv.Name, f.Recv.Type)
	}
	buf.WriteString(f.Name)
	argList(f.In, false)
	if len(f.Out) > 0 {
		buf.WriteByte(' ')
		argList(f.Out, true)
	}
	return buf.String()
}

func (f *Func) Decl() string {
	var buf strings.Builder
	if f.Doc != "" {
		for line := range strings.SplitSeq(strings.TrimRight(f.Doc, "\n"), "\n") {
			fmt.Fprintf(&buf, "// %s\n", line)
		}
	}
	buf.WriteString(f.Signature())
	return buf.String()
}

// SpecFunc returns information about the spec package function that generated
// f. name is the name of the function in the spec package, sig is its
// uninstantiated signature type, and typeArgs is a slice of the type arguments
// it was instantiated on to construct f.
func (f *Func) SpecFunc() (name string, sig *types.Signature, typeArgs []types.Type) {
	sFn := f.specFunc

	// Build instantiated signature
	for _, tparam := range sFn.TypeParams {
		val := f.instance.Get(f.typeParamVars[tparam])
		switch val := val.(type) {
		case specexpr.Type:
			typeArgs = append(typeArgs, specTypeToType(sFn.Pkg, val))
		case specexpr.Num:
			wt := sFn.Pkg.WidthTypes[val]
			if wt == nil {
				panic(fmt.Sprintf("no spec package type for width %s", val))
			}
			typeArgs = append(typeArgs, wt)
		default:
			panic("unexpected type parameter value")
		}
	}

	return sFn.Name, sFn.Sig, typeArgs
}
