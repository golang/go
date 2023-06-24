// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Indexed package export.
//
// The indexed export data format is an evolution of the previous
// binary export data format. Its chief contribution is introducing an
// index table, which allows efficient random access of individual
// declarations and inline function bodies. In turn, this allows
// avoiding unnecessary work for compilation units that import large
// packages.
//
//
// The top-level data format is structured as:
//
//     Header struct {
//         Tag        byte   // 'i'
//         Version    uvarint
//         StringSize uvarint
//         DataSize   uvarint
//     }
//
//     Strings [StringSize]byte
//     Data    [DataSize]byte
//
//     MainIndex []struct{
//         PkgPath   stringOff
//         PkgName   stringOff
//         PkgHeight uvarint
//
//         Decls []struct{
//             Name   stringOff
//             Offset declOff
//         }
//     }
//
//     Fingerprint [8]byte
//
// uvarint means a uint64 written out using uvarint encoding.
//
// []T means a uvarint followed by that many T objects. In other
// words:
//
//     Len   uvarint
//     Elems [Len]T
//
// stringOff means a uvarint that indicates an offset within the
// Strings section. At that offset is another uvarint, followed by
// that many bytes, which form the string value.
//
// declOff means a uvarint that indicates an offset within the Data
// section where the associated declaration can be found.
//
//
// There are five kinds of declarations, distinguished by their first
// byte:
//
//     type Var struct {
//         Tag  byte // 'V'
//         Pos  Pos
//         Type typeOff
//     }
//
//     type Func struct {
//         Tag       byte // 'F' or 'G'
//         Pos       Pos
//         TypeParams []typeOff  // only present if Tag == 'G'
//         Signature Signature
//     }
//
//     type Const struct {
//         Tag   byte // 'C'
//         Pos   Pos
//         Value Value
//     }
//
//     type Type struct {
//         Tag        byte // 'T' or 'U'
//         Pos        Pos
//         TypeParams []typeOff  // only present if Tag == 'U'
//         Underlying typeOff
//
//         Methods []struct{  // omitted if Underlying is an interface type
//             Pos       Pos
//             Name      stringOff
//             Recv      Param
//             Signature Signature
//         }
//     }
//
//     type Alias struct {
//         Tag  byte // 'A'
//         Pos  Pos
//         Type typeOff
//     }
//
//     // "Automatic" declaration of each typeparam
//     type TypeParam struct {
//         Tag        byte // 'P'
//         Pos        Pos
//         Implicit   bool
//         Constraint typeOff
//     }
//
// typeOff means a uvarint that either indicates a predeclared type,
// or an offset into the Data section. If the uvarint is less than
// predeclReserved, then it indicates the index into the predeclared
// types list (see predeclared in bexport.go for order). Otherwise,
// subtracting predeclReserved yields the offset of a type descriptor.
//
// Value means a type, kind, and type-specific value. See
// (*exportWriter).value for details.
//
//
// There are twelve kinds of type descriptors, distinguished by an itag:
//
//     type DefinedType struct {
//         Tag     itag // definedType
//         Name    stringOff
//         PkgPath stringOff
//     }
//
//     type PointerType struct {
//         Tag  itag // pointerType
//         Elem typeOff
//     }
//
//     type SliceType struct {
//         Tag  itag // sliceType
//         Elem typeOff
//     }
//
//     type ArrayType struct {
//         Tag  itag // arrayType
//         Len  uint64
//         Elem typeOff
//     }
//
//     type ChanType struct {
//         Tag  itag   // chanType
//         Dir  uint64 // 1 RecvOnly; 2 SendOnly; 3 SendRecv
//         Elem typeOff
//     }
//
//     type MapType struct {
//         Tag  itag // mapType
//         Key  typeOff
//         Elem typeOff
//     }
//
//     type FuncType struct {
//         Tag       itag // signatureType
//         PkgPath   stringOff
//         Signature Signature
//     }
//
//     type StructType struct {
//         Tag     itag // structType
//         PkgPath stringOff
//         Fields []struct {
//             Pos      Pos
//             Name     stringOff
//             Type     typeOff
//             Embedded bool
//             Note     stringOff
//         }
//     }
//
//     type InterfaceType struct {
//         Tag     itag // interfaceType
//         PkgPath stringOff
//         Embeddeds []struct {
//             Pos  Pos
//             Type typeOff
//         }
//         Methods []struct {
//             Pos       Pos
//             Name      stringOff
//             Signature Signature
//         }
//     }
//
//     // Reference to a type param declaration
//     type TypeParamType struct {
//         Tag     itag // typeParamType
//         Name    stringOff
//         PkgPath stringOff
//     }
//
//     // Instantiation of a generic type (like List[T2] or List[int])
//     type InstanceType struct {
//         Tag     itag // instanceType
//         Pos     pos
//         TypeArgs []typeOff
//         BaseType typeOff
//     }
//
//     type UnionType struct {
//         Tag     itag // interfaceType
//         Terms   []struct {
//             tilde bool
//             Type  typeOff
//         }
//     }
//
//
//
//     type Signature struct {
//         Params   []Param
//         Results  []Param
//         Variadic bool  // omitted if Results is empty
//     }
//
//     type Param struct {
//         Pos  Pos
//         Name stringOff
//         Type typOff
//     }
//
//
// Pos encodes a file:line:column triple, incorporating a simple delta
// encoding scheme within a data object. See exportWriter.pos for
// details.
//
//
// Compiler-specific details.
//
// cmd/compile writes out a second index for inline bodies and also
// appends additional compiler-specific details after declarations.
// Third-party tools are not expected to depend on these details and
// they're expected to change much more rapidly, so they're omitted
// here. See exportWriter's varExt/funcExt/etc methods for details.

package typecheck

import (
	"go/constant"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// predeclReserved is the number of type offsets reserved for types
// implicitly declared in the universe block.
const predeclReserved = 32

// An itag distinguishes the kind of type that was written into the
// indexed export format.
type itag uint64

const (
	// Types
	definedType itag = iota
	pointerType
	sliceType
	arrayType
	chanType
	mapType
	signatureType
	structType
	interfaceType
	typeParamType
	instanceType // Instantiation of a generic type
	unionType
)

const (
	debug = false
	magic = 0x6742937dc293105
)

// exportPath returns the path for pkg as it appears in the iexport
// file format. For historical reasons (before cmd/compile required
// the -p flag), the local package is represented as the empty string,
// instead of its actual path.
func exportPath(pkg *types.Pkg) string {
	if pkg == types.LocalPkg {
		return ""
	}
	return pkg.Path
}

const blankMarker = "$"

// TparamExportName creates a unique name for type param in a method or a generic
// type, using the specified unique prefix and the index of the type param. The index
// is only used if the type param is blank, in which case the blank is replace by
// "$<index>". A unique name is needed for later substitution in the compiler and
// export/import that keeps blank type params associated with the correct constraint.
func TparamExportName(prefix string, name string, index int) string {
	if name == "_" {
		name = blankMarker + strconv.Itoa(index)
	}
	return prefix + "." + name
}

// TparamName returns the real name of a type parameter, after stripping its
// qualifying prefix and reverting blank-name encoding. See TparamExportName
// for details.
func TparamName(exportName string) string {
	// Remove the "path" from the type param name that makes it unique.
	ix := strings.LastIndex(exportName, ".")
	if ix < 0 {
		return ""
	}
	name := exportName[ix+1:]
	if strings.HasPrefix(name, blankMarker) {
		return "_"
	}
	return name
}

func constTypeOf(typ *types.Type) constant.Kind {
	switch typ {
	case types.UntypedInt, types.UntypedRune:
		return constant.Int
	case types.UntypedFloat:
		return constant.Float
	case types.UntypedComplex:
		return constant.Complex
	}

	switch typ.Kind() {
	case types.TBOOL:
		return constant.Bool
	case types.TSTRING:
		return constant.String
	case types.TINT, types.TINT8, types.TINT16, types.TINT32, types.TINT64,
		types.TUINT, types.TUINT8, types.TUINT16, types.TUINT32, types.TUINT64, types.TUINTPTR:
		return constant.Int
	case types.TFLOAT32, types.TFLOAT64:
		return constant.Float
	case types.TCOMPLEX64, types.TCOMPLEX128:
		return constant.Complex
	}

	base.Fatalf("unexpected constant type: %v", typ)
	return 0
}

func intSize(typ *types.Type) (signed bool, maxBytes uint) {
	if typ.IsUntyped() {
		return true, ir.ConstPrec / 8
	}

	switch typ.Kind() {
	case types.TFLOAT32, types.TCOMPLEX64:
		return true, 3
	case types.TFLOAT64, types.TCOMPLEX128:
		return true, 7
	}

	signed = typ.IsSigned()
	maxBytes = uint(typ.Size())

	// The go/types API doesn't expose sizes to importers, so they
	// don't know how big these types are.
	switch typ.Kind() {
	case types.TINT, types.TUINT, types.TUINTPTR:
		maxBytes = 8
	}

	return
}

func isNonEmptyAssign(n ir.Node) bool {
	switch n.Op() {
	case ir.OAS:
		if n.(*ir.AssignStmt).Y != nil {
			return true
		}
	case ir.OAS2, ir.OAS2DOTTYPE, ir.OAS2FUNC, ir.OAS2MAPR, ir.OAS2RECV:
		return true
	}
	return false
}
func isNamedTypeSwitch(x ir.Node) bool {
	guard, ok := x.(*ir.TypeSwitchGuard)
	return ok && guard.Tag != nil
}

func simplifyForExport(n ir.Node) ir.Node {
	switch n.Op() {
	case ir.OPAREN:
		n := n.(*ir.ParenExpr)
		return simplifyForExport(n.X)
	}
	return n
}

// The name used for dictionary parameters or local variables.
const LocalDictName = ".dict"
