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
	"bytes"
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"go/constant"
	"io"
	"math/big"
	"sort"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/goobj"
	"cmd/internal/src"
)

// Current indexed export format version. Increase with each format change.
// 0: Go1.11 encoding
// 1: added column details to Pos
// 2: added information for generic function/types.  The export of non-generic
// functions/types remains largely backward-compatible.  Breaking changes include:
//    - a 'kind' byte is added to constant values
const (
	iexportVersionGo1_11   = 0
	iexportVersionPosCol   = 1
	iexportVersionGenerics = 2
	iexportVersionGo1_18   = 2

	iexportVersionCurrent = 2
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

// WriteExports writes the indexed export format to out. If extensions
// is true, then the compiler-only extensions are included.
func WriteExports(out io.Writer, extensions bool) {
	if extensions {
		// If we're exporting inline bodies, invoke the crawler to mark
		// which bodies to include.
		crawlExports(Target.Exports)
	}

	p := iexporter{
		allPkgs:     map[*types.Pkg]bool{},
		stringIndex: map[string]uint64{},
		declIndex:   map[*types.Sym]uint64{},
		inlineIndex: map[*types.Sym]uint64{},
		typIndex:    map[*types.Type]uint64{},
		extensions:  extensions,
	}

	for i, pt := range predeclared() {
		p.typIndex[pt] = uint64(i)
	}
	if len(p.typIndex) > predeclReserved {
		base.Fatalf("too many predeclared types: %d > %d", len(p.typIndex), predeclReserved)
	}

	// Initialize work queue with exported declarations.
	for _, n := range Target.Exports {
		p.pushDecl(n)
	}

	// Loop until no more work. We use a queue because while
	// writing out inline bodies, we may discover additional
	// declarations that are needed.
	for !p.declTodo.Empty() {
		p.doDecl(p.declTodo.PopLeft())
	}

	// Append indices to data0 section.
	dataLen := uint64(p.data0.Len())
	w := p.newWriter()
	w.writeIndex(p.declIndex, true)
	w.writeIndex(p.inlineIndex, false)
	w.flush()

	if *base.Flag.LowerV {
		fmt.Printf("export: hdr strings %v, data %v, index %v\n", p.strings.Len(), dataLen, p.data0.Len())
	}

	// Assemble header.
	var hdr intWriter
	hdr.WriteByte('i')
	hdr.uint64(iexportVersionCurrent)
	hdr.uint64(uint64(p.strings.Len()))
	hdr.uint64(dataLen)

	// Flush output.
	h := md5.New()
	wr := io.MultiWriter(out, h)
	io.Copy(wr, &hdr)
	io.Copy(wr, &p.strings)
	io.Copy(wr, &p.data0)

	// Add fingerprint (used by linker object file).
	// Attach this to the end, so tools (e.g. gcimporter) don't care.
	copy(base.Ctxt.Fingerprint[:], h.Sum(nil)[:])
	out.Write(base.Ctxt.Fingerprint[:])
}

// writeIndex writes out a symbol index. mainIndex indicates whether
// we're writing out the main index, which is also read by
// non-compiler tools and includes a complete package description
// (i.e., name and height).
func (w *exportWriter) writeIndex(index map[*types.Sym]uint64, mainIndex bool) {
	// Build a map from packages to symbols from that package.
	pkgSyms := map[*types.Pkg][]*types.Sym{}

	// For the main index, make sure to include every package that
	// we reference, even if we're not exporting (or reexporting)
	// any symbols from it.
	if mainIndex {
		pkgSyms[types.LocalPkg] = nil
		for pkg := range w.p.allPkgs {
			pkgSyms[pkg] = nil
		}
	}

	// Group symbols by package.
	for sym := range index {
		pkgSyms[sym.Pkg] = append(pkgSyms[sym.Pkg], sym)
	}

	// Sort packages by path.
	var pkgs []*types.Pkg
	for pkg := range pkgSyms {
		pkgs = append(pkgs, pkg)
	}
	sort.Slice(pkgs, func(i, j int) bool {
		return pkgs[i].Path < pkgs[j].Path
	})

	w.uint64(uint64(len(pkgs)))
	for _, pkg := range pkgs {
		w.string(pkg.Path)
		if mainIndex {
			w.string(pkg.Name)
			w.uint64(uint64(pkg.Height))
		}

		// Sort symbols within a package by name.
		syms := pkgSyms[pkg]
		sort.Slice(syms, func(i, j int) bool {
			return syms[i].Name < syms[j].Name
		})

		w.uint64(uint64(len(syms)))
		for _, sym := range syms {
			w.string(sym.Name)
			w.uint64(index[sym])
		}
	}
}

type iexporter struct {
	// allPkgs tracks all packages that have been referenced by
	// the export data, so we can ensure to include them in the
	// main index.
	allPkgs map[*types.Pkg]bool

	declTodo ir.NameQueue

	strings     intWriter
	stringIndex map[string]uint64

	data0       intWriter
	declIndex   map[*types.Sym]uint64
	inlineIndex map[*types.Sym]uint64
	typIndex    map[*types.Type]uint64

	extensions bool
}

// stringOff returns the offset of s within the string section.
// If not already present, it's added to the end.
func (p *iexporter) stringOff(s string) uint64 {
	off, ok := p.stringIndex[s]
	if !ok {
		off = uint64(p.strings.Len())
		p.stringIndex[s] = off

		if *base.Flag.LowerV {
			fmt.Printf("export: str %v %.40q\n", off, s)
		}

		p.strings.uint64(uint64(len(s)))
		p.strings.WriteString(s)
	}
	return off
}

// pushDecl adds n to the declaration work queue, if not already present.
func (p *iexporter) pushDecl(n *ir.Name) {
	if n.Sym() == nil || n.Sym().Def != n && n.Op() != ir.OTYPE {
		base.Fatalf("weird Sym: %v, %v", n, n.Sym())
	}

	// Don't export predeclared declarations.
	if n.Sym().Pkg == types.BuiltinPkg || n.Sym().Pkg == types.UnsafePkg {
		return
	}

	if _, ok := p.declIndex[n.Sym()]; ok {
		return
	}

	p.declIndex[n.Sym()] = ^uint64(0) // mark n present in work queue
	p.declTodo.PushRight(n)
}

// exportWriter handles writing out individual data section chunks.
type exportWriter struct {
	p *iexporter

	data       intWriter
	currPkg    *types.Pkg
	prevFile   string
	prevLine   int64
	prevColumn int64

	// dclIndex maps function-scoped declarations to an int used to refer to
	// them later in the function. For local variables/params, the int is
	// non-negative and in order of the appearance in the Func's Dcl list. For
	// closure variables, the index is negative starting at -2.
	dclIndex           map[*ir.Name]int
	maxDclIndex        int
	maxClosureVarIndex int
}

func (p *iexporter) doDecl(n *ir.Name) {
	w := p.newWriter()
	w.setPkg(n.Sym().Pkg, false)

	switch n.Op() {
	case ir.ONAME:
		switch n.Class {
		case ir.PEXTERN:
			// Variable.
			w.tag('V')
			w.pos(n.Pos())
			w.typ(n.Type())
			if w.p.extensions {
				w.varExt(n)
			}

		case ir.PFUNC:
			if ir.IsMethod(n) {
				base.Fatalf("unexpected method: %v", n)
			}

			// Function.
			if n.Type().TParams().NumFields() == 0 {
				w.tag('F')
			} else {
				w.tag('G')
			}
			w.pos(n.Pos())
			// The tparam list of the function type is the
			// declaration of the type params. So, write out the type
			// params right now. Then those type params will be
			// referenced via their type offset (via typOff) in all
			// other places in the signature and function that they
			// are used.
			if n.Type().TParams().NumFields() > 0 {
				w.tparamList(n.Type().TParams().FieldSlice())
			}
			w.signature(n.Type())
			if w.p.extensions {
				w.funcExt(n)
			}

		default:
			base.Fatalf("unexpected class: %v, %v", n, n.Class)
		}

	case ir.OLITERAL:
		// TODO(mdempsky): Extend check to all declarations.
		if n.Typecheck() == 0 {
			base.FatalfAt(n.Pos(), "missed typecheck: %v", n)
		}

		// Constant.
		w.tag('C')
		w.pos(n.Pos())
		w.value(n.Type(), n.Val())
		if w.p.extensions {
			w.constExt(n)
		}

	case ir.OTYPE:
		if n.Type().IsTypeParam() && n.Type().Underlying() == n.Type() {
			// Even though it has local scope, a typeparam requires a
			// declaration via its package and unique name, because it
			// may be referenced within its type bound during its own
			// definition.
			w.tag('P')
			// A typeparam has a name, and has a type bound rather
			// than an underlying type.
			w.pos(n.Pos())
			if iexportVersionCurrent >= iexportVersionGo1_18 {
				implicit := n.Type().Bound().IsImplicit()
				w.bool(implicit)
			}
			w.typ(n.Type().Bound())
			break
		}

		if n.Alias() {
			// Alias.
			w.tag('A')
			w.pos(n.Pos())
			w.typ(n.Type())
			break
		}

		// Defined type.
		if len(n.Type().RParams()) == 0 {
			w.tag('T')
		} else {
			w.tag('U')
		}
		w.pos(n.Pos())

		if len(n.Type().RParams()) > 0 {
			// Export type parameters, if any, needed for this type
			w.typeList(n.Type().RParams())
		}

		underlying := n.Type().Underlying()
		if underlying == types.ErrorType.Underlying() {
			// For "type T error", use error as the
			// underlying type instead of error's own
			// underlying anonymous interface. This
			// ensures consistency with how importers may
			// declare error (e.g., go/types uses nil Pkg
			// for predeclared objects).
			underlying = types.ErrorType
		}
		if underlying == types.ComparableType.Underlying() {
			// Do same for ComparableType as for ErrorType.
			underlying = types.ComparableType
		}
		if base.Flag.G > 0 && underlying == types.AnyType.Underlying() {
			// Do same for AnyType as for ErrorType.
			underlying = types.AnyType
		}
		w.typ(underlying)

		t := n.Type()
		if t.IsInterface() {
			if w.p.extensions {
				w.typeExt(t)
			}
			break
		}

		// Sort methods, for consistency with types2.
		methods := append([]*types.Field(nil), t.Methods().Slice()...)
		if base.Debug.UnifiedQuirks != 0 {
			sort.Sort(types.MethodsByName(methods))
		}

		w.uint64(uint64(len(methods)))
		for _, m := range methods {
			w.pos(m.Pos)
			w.selector(m.Sym)
			w.param(m.Type.Recv())
			w.signature(m.Type)
		}

		if w.p.extensions {
			w.typeExt(t)
			for _, m := range methods {
				w.methExt(m)
			}
		}

	default:
		base.Fatalf("unexpected node: %v", n)
	}

	w.finish("dcl", p.declIndex, n.Sym())
}

func (w *exportWriter) tag(tag byte) {
	w.data.WriteByte(tag)
}

func (w *exportWriter) finish(what string, index map[*types.Sym]uint64, sym *types.Sym) {
	off := w.flush()
	if *base.Flag.LowerV {
		fmt.Printf("export: %v %v %v\n", what, off, sym)
	}
	index[sym] = off
}

func (p *iexporter) doInline(f *ir.Name) {
	w := p.newWriter()
	w.setPkg(fnpkg(f), false)

	w.dclIndex = make(map[*ir.Name]int, len(f.Func.Inl.Dcl))
	w.funcBody(f.Func)

	w.finish("inl", p.inlineIndex, f.Sym())
}

func (w *exportWriter) pos(pos src.XPos) {
	p := base.Ctxt.PosTable.Pos(pos)
	file := p.Base().AbsFilename()
	line := int64(p.RelLine())
	column := int64(p.RelCol())

	// Encode position relative to the last position: column
	// delta, then line delta, then file name. We reserve the
	// bottom bit of the column and line deltas to encode whether
	// the remaining fields are present.
	//
	// Note: Because data objects may be read out of order (or not
	// at all), we can only apply delta encoding within a single
	// object. This is handled implicitly by tracking prevFile,
	// prevLine, and prevColumn as fields of exportWriter.

	deltaColumn := (column - w.prevColumn) << 1
	deltaLine := (line - w.prevLine) << 1

	if file != w.prevFile {
		deltaLine |= 1
	}
	if deltaLine != 0 {
		deltaColumn |= 1
	}

	w.int64(deltaColumn)
	if deltaColumn&1 != 0 {
		w.int64(deltaLine)
		if deltaLine&1 != 0 {
			w.string(file)
		}
	}

	w.prevFile = file
	w.prevLine = line
	w.prevColumn = column
}

func (w *exportWriter) pkg(pkg *types.Pkg) {
	// TODO(mdempsky): Add flag to types.Pkg to mark pseudo-packages.
	if pkg == ir.Pkgs.Go {
		base.Fatalf("export of pseudo-package: %q", pkg.Path)
	}

	// Ensure any referenced packages are declared in the main index.
	w.p.allPkgs[pkg] = true

	w.string(pkg.Path)
}

func (w *exportWriter) qualifiedIdent(n *ir.Name) {
	// Ensure any referenced declarations are written out too.
	w.p.pushDecl(n)

	s := n.Sym()
	w.string(s.Name)
	w.pkg(s.Pkg)
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

func (w *exportWriter) selector(s *types.Sym) {
	if w.currPkg == nil {
		base.Fatalf("missing currPkg")
	}

	// If the selector being written is unexported, it comes with a package qualifier.
	// If the selector being written is exported, it is not package-qualified.
	// See the spec: https://golang.org/ref/spec#Uniqueness_of_identifiers
	// As an optimization, we don't actually write the package every time - instead we
	// call setPkg before a group of selectors (all of which must have the same package qualifier).
	pkg := w.currPkg
	if types.IsExported(s.Name) {
		pkg = types.LocalPkg
	}
	if s.Pkg != pkg {
		base.Fatalf("package mismatch in selector: %v in package %q, but want %q", s, s.Pkg.Path, pkg.Path)
	}

	w.string(s.Name)
}

func (w *exportWriter) typ(t *types.Type) {
	w.data.uint64(w.p.typOff(t))
}

// The "exotic" functions in this section encode a wider range of
// items than the standard encoding functions above. These include
// types that do not appear in declarations, only in code, such as
// method types. These methods need to be separate from the standard
// encoding functions because we don't want to modify the encoding
// generated by the standard functions (because that exported
// information is read by tools besides the compiler).

// exoticType exports a type to the writer.
func (w *exportWriter) exoticType(t *types.Type) {
	switch {
	case t == nil:
		// Calls-as-statements have no type.
		w.data.uint64(exoticTypeNil)
	case t.IsStruct() && t.StructType().Funarg != types.FunargNone:
		// These are weird structs for representing tuples of types returned
		// by multi-return functions.
		// They don't fit the standard struct type mold. For instance,
		// they don't have any package info.
		w.data.uint64(exoticTypeTuple)
		w.uint64(uint64(t.StructType().Funarg))
		w.uint64(uint64(t.NumFields()))
		for _, f := range t.FieldSlice() {
			w.pos(f.Pos)
			s := f.Sym
			if s == nil {
				w.uint64(0)
			} else if s.Pkg == nil {
				w.uint64(exoticTypeSymNoPkg)
				w.string(s.Name)
			} else {
				w.uint64(exoticTypeSymWithPkg)
				w.pkg(s.Pkg)
				w.string(s.Name)
			}
			w.typ(f.Type)
			if f.Embedded != 0 || f.Note != "" {
				panic("extra info in funarg struct field")
			}
		}
	case t.Kind() == types.TFUNC && t.Recv() != nil:
		w.data.uint64(exoticTypeRecv)
		// interface method types have a fake receiver type.
		isFakeRecv := t.Recv().Type == types.FakeRecvType()
		w.bool(isFakeRecv)
		if !isFakeRecv {
			w.exoticParam(t.Recv())
		}
		w.exoticSignature(t)

	default:
		// A regular type.
		w.data.uint64(exoticTypeRegular)
		w.typ(t)
	}
}

const (
	exoticTypeNil = iota
	exoticTypeTuple
	exoticTypeRecv
	exoticTypeRegular
)
const (
	exoticTypeSymNil = iota
	exoticTypeSymNoPkg
	exoticTypeSymWithPkg
)

// Export a selector, but one whose package may not match
// the package being compiled. This is a separate function
// because the standard selector() serialization format is fixed
// by the go/types reader. This one can only be used during
// inline/generic body exporting.
func (w *exportWriter) exoticSelector(s *types.Sym) {
	pkg := w.currPkg
	if types.IsExported(s.Name) {
		pkg = types.LocalPkg
	}

	w.string(s.Name)
	if s.Pkg == pkg {
		w.uint64(0)
	} else {
		w.uint64(1)
		w.pkg(s.Pkg)
	}
}

func (w *exportWriter) exoticSignature(t *types.Type) {
	hasPkg := t.Pkg() != nil
	w.bool(hasPkg)
	if hasPkg {
		w.pkg(t.Pkg())
	}
	w.exoticParamList(t.Params().FieldSlice())
	w.exoticParamList(t.Results().FieldSlice())
}

func (w *exportWriter) exoticParamList(fs []*types.Field) {
	w.uint64(uint64(len(fs)))
	for _, f := range fs {
		w.exoticParam(f)
	}

}
func (w *exportWriter) exoticParam(f *types.Field) {
	w.pos(f.Pos)
	w.exoticSym(f.Sym)
	w.uint64(uint64(f.Offset))
	w.exoticType(f.Type)
	w.bool(f.IsDDD())
}

func (w *exportWriter) exoticField(f *types.Field) {
	w.pos(f.Pos)
	w.exoticSym(f.Sym)
	w.uint64(uint64(f.Offset))
	w.exoticType(f.Type)
	w.string(f.Note)
}

func (w *exportWriter) exoticSym(s *types.Sym) {
	if s == nil {
		w.string("")
		return
	}
	if s.Name == "" {
		base.Fatalf("empty symbol name")
	}
	w.string(s.Name)
	if !types.IsExported(s.Name) {
		w.pkg(s.Pkg)
	}
}

func (p *iexporter) newWriter() *exportWriter {
	return &exportWriter{p: p}
}

func (w *exportWriter) flush() uint64 {
	off := uint64(w.p.data0.Len())
	io.Copy(&w.p.data0, &w.data)
	return off
}

func (p *iexporter) typOff(t *types.Type) uint64 {
	off, ok := p.typIndex[t]
	if !ok {
		w := p.newWriter()
		w.doTyp(t)
		rawOff := w.flush()
		if *base.Flag.LowerV {
			fmt.Printf("export: typ %v %v\n", rawOff, t)
		}
		off = predeclReserved + rawOff
		p.typIndex[t] = off
	}
	return off
}

func (w *exportWriter) startType(k itag) {
	w.data.uint64(uint64(k))
}

func (w *exportWriter) doTyp(t *types.Type) {
	s := t.Sym()
	if s != nil && t.OrigSym() != nil {
		assert(base.Flag.G > 0)
		// This is an instantiated type - could be a re-instantiation like
		// Value[T2] or a full instantiation like Value[int].
		if strings.Index(s.Name, "[") < 0 {
			base.Fatalf("incorrect name for instantiated type")
		}
		w.startType(instanceType)
		w.pos(t.Pos())
		// Export the type arguments for the instantiated type. The
		// instantiated type could be in a method header (e.g. "func (v
		// *Value[T2]) set (...) { ... }"), so the type args are "new"
		// typeparams. Or the instantiated type could be in a
		// function/method body, so the type args are either concrete
		// types or existing typeparams from the function/method header.
		w.typeList(t.RParams())
		// Export a reference to the base type.
		baseType := t.OrigSym().Def.(*ir.Name).Type()
		w.typ(baseType)
		return
	}

	// The 't.Underlying() == t' check is to confirm this is a base typeparam
	// type, rather than a defined type with typeparam underlying type, like:
	// type orderedAbs[T any] T
	if t.IsTypeParam() && t.Underlying() == t {
		assert(base.Flag.G > 0)
		if s.Pkg == types.BuiltinPkg || s.Pkg == types.UnsafePkg {
			base.Fatalf("builtin type missing from typIndex: %v", t)
		}
		// Write out the first use of a type param as a qualified ident.
		// This will force a "declaration" of the type param.
		w.startType(typeParamType)
		w.qualifiedIdent(t.Obj().(*ir.Name))
		return
	}

	if s != nil {
		if s.Pkg == types.BuiltinPkg || s.Pkg == types.UnsafePkg {
			base.Fatalf("builtin type missing from typIndex: %v", t)
		}

		w.startType(definedType)
		w.qualifiedIdent(t.Obj().(*ir.Name))
		return
	}

	switch t.Kind() {
	case types.TPTR:
		w.startType(pointerType)
		w.typ(t.Elem())

	case types.TSLICE:
		w.startType(sliceType)
		w.typ(t.Elem())

	case types.TARRAY:
		w.startType(arrayType)
		w.uint64(uint64(t.NumElem()))
		w.typ(t.Elem())

	case types.TCHAN:
		w.startType(chanType)
		w.uint64(uint64(t.ChanDir()))
		w.typ(t.Elem())

	case types.TMAP:
		w.startType(mapType)
		w.typ(t.Key())
		w.typ(t.Elem())

	case types.TFUNC:
		w.startType(signatureType)
		w.setPkg(t.Pkg(), true)
		w.signature(t)

	case types.TSTRUCT:
		w.startType(structType)
		w.setPkg(t.Pkg(), true)

		w.uint64(uint64(t.NumFields()))
		for _, f := range t.FieldSlice() {
			w.pos(f.Pos)
			w.selector(f.Sym)
			w.typ(f.Type)
			w.bool(f.Embedded != 0)
			w.string(f.Note)
		}

	case types.TINTER:
		var embeddeds, methods []*types.Field
		for _, m := range t.Methods().Slice() {
			if m.Sym != nil {
				methods = append(methods, m)
			} else {
				embeddeds = append(embeddeds, m)
			}
		}

		// Sort methods and embedded types, for consistency with types2.
		// Note: embedded types may be anonymous, and types2 sorts them
		// with sort.Stable too.
		if base.Debug.UnifiedQuirks != 0 {
			sort.Sort(types.MethodsByName(methods))
			sort.Stable(types.EmbeddedsByName(embeddeds))
		}

		w.startType(interfaceType)
		w.setPkg(t.Pkg(), true)

		w.uint64(uint64(len(embeddeds)))
		for _, f := range embeddeds {
			w.pos(f.Pos)
			w.typ(f.Type)
		}

		w.uint64(uint64(len(methods)))
		for _, f := range methods {
			w.pos(f.Pos)
			w.selector(f.Sym)
			w.signature(f.Type)
		}

	case types.TUNION:
		assert(base.Flag.G > 0)
		// TODO(danscales): possibly put out the tilde bools in more
		// compact form.
		w.startType(unionType)
		nt := t.NumTerms()
		w.uint64(uint64(nt))
		for i := 0; i < nt; i++ {
			typ, tilde := t.Term(i)
			w.bool(tilde)
			w.typ(typ)
		}

	default:
		base.Fatalf("unexpected type: %v", t)
	}
}

func (w *exportWriter) setPkg(pkg *types.Pkg, write bool) {
	if pkg == types.NoPkg {
		base.Fatalf("missing pkg")
	}

	if write {
		w.pkg(pkg)
	}

	w.currPkg = pkg
}

func (w *exportWriter) signature(t *types.Type) {
	w.paramList(t.Params().FieldSlice())
	w.paramList(t.Results().FieldSlice())
	if n := t.Params().NumFields(); n > 0 {
		w.bool(t.Params().Field(n - 1).IsDDD())
	}
}

func (w *exportWriter) typeList(ts []*types.Type) {
	w.uint64(uint64(len(ts)))
	for _, rparam := range ts {
		w.typ(rparam)
	}
}

func (w *exportWriter) tparamList(fs []*types.Field) {
	w.uint64(uint64(len(fs)))
	for _, f := range fs {
		if !f.Type.IsTypeParam() {
			base.Fatalf("unexpected non-typeparam")
		}
		w.typ(f.Type)
	}
}

func (w *exportWriter) paramList(fs []*types.Field) {
	w.uint64(uint64(len(fs)))
	for _, f := range fs {
		w.param(f)
	}
}

func (w *exportWriter) param(f *types.Field) {
	w.pos(f.Pos)
	w.localIdent(types.OrigSym(f.Sym))
	w.typ(f.Type)
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

func (w *exportWriter) value(typ *types.Type, v constant.Value) {
	w.typ(typ)

	if iexportVersionCurrent >= iexportVersionGo1_18 {
		w.int64(int64(v.Kind()))
	}

	var kind constant.Kind
	var valType *types.Type

	if typ.IsTypeParam() {
		kind = v.Kind()
		if iexportVersionCurrent < iexportVersionGo1_18 {
			// A constant will have a TYPEPARAM type if it appears in a place
			// where it must match that typeparam type (e.g. in a binary
			// operation with a variable of that typeparam type). If so, then
			// we must write out its actual constant kind as well, so its
			// constant val can be read in properly during import.
			w.int64(int64(kind))
		}

		switch kind {
		case constant.Int:
			valType = types.Types[types.TINT64]
		case constant.Float:
			valType = types.Types[types.TFLOAT64]
		case constant.Complex:
			valType = types.Types[types.TCOMPLEX128]
		}
	} else {
		ir.AssertValidTypeForConst(typ, v)
		kind = constTypeOf(typ)
		valType = typ
	}

	// Each type has only one admissible constant representation, so we could
	// type switch directly on v.Kind() here. However, switching on the type
	// (in the non-typeparam case) increases symmetry with import logic and
	// provides a useful consistency check.

	switch kind {
	case constant.Bool:
		w.bool(constant.BoolVal(v))
	case constant.String:
		w.string(constant.StringVal(v))
	case constant.Int:
		w.mpint(v, valType)
	case constant.Float:
		w.mpfloat(v, valType)
	case constant.Complex:
		w.mpfloat(constant.Real(v), valType)
		w.mpfloat(constant.Imag(v), valType)
	}
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

// mpint exports a multi-precision integer.
//
// For unsigned types, small values are written out as a single
// byte. Larger values are written out as a length-prefixed big-endian
// byte string, where the length prefix is encoded as its complement.
// For example, bytes 0, 1, and 2 directly represent the integer
// values 0, 1, and 2; while bytes 255, 254, and 253 indicate a 1-,
// 2-, and 3-byte big-endian string follow.
//
// Encoding for signed types use the same general approach as for
// unsigned types, except small values use zig-zag encoding and the
// bottom bit of length prefix byte for large values is reserved as a
// sign bit.
//
// The exact boundary between small and large encodings varies
// according to the maximum number of bytes needed to encode a value
// of type typ. As a special case, 8-bit types are always encoded as a
// single byte.
func (w *exportWriter) mpint(x constant.Value, typ *types.Type) {
	signed, maxBytes := intSize(typ)

	negative := constant.Sign(x) < 0
	if !signed && negative {
		base.Fatalf("negative unsigned integer; type %v, value %v", typ, x)
	}

	b := constant.Bytes(x) // little endian
	for i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {
		b[i], b[j] = b[j], b[i]
	}

	if len(b) > 0 && b[0] == 0 {
		base.Fatalf("leading zeros")
	}
	if uint(len(b)) > maxBytes {
		base.Fatalf("bad mpint length: %d > %d (type %v, value %v)", len(b), maxBytes, typ, x)
	}

	maxSmall := 256 - maxBytes
	if signed {
		maxSmall = 256 - 2*maxBytes
	}
	if maxBytes == 1 {
		maxSmall = 256
	}

	// Check if x can use small value encoding.
	if len(b) <= 1 {
		var ux uint
		if len(b) == 1 {
			ux = uint(b[0])
		}
		if signed {
			ux <<= 1
			if negative {
				ux--
			}
		}
		if ux < maxSmall {
			w.data.WriteByte(byte(ux))
			return
		}
	}

	n := 256 - uint(len(b))
	if signed {
		n = 256 - 2*uint(len(b))
		if negative {
			n |= 1
		}
	}
	if n < maxSmall || n >= 256 {
		base.Fatalf("encoding mistake: %d, %v, %v => %d", len(b), signed, negative, n)
	}

	w.data.WriteByte(byte(n))
	w.data.Write(b)
}

// mpfloat exports a multi-precision floating point number.
//
// The number's value is decomposed into mantissa × 2**exponent, where
// mantissa is an integer. The value is written out as mantissa (as a
// multi-precision integer) and then the exponent, except exponent is
// omitted if mantissa is zero.
func (w *exportWriter) mpfloat(v constant.Value, typ *types.Type) {
	f := ir.BigFloat(v)
	if f.IsInf() {
		base.Fatalf("infinite constant")
	}

	// Break into f = mant × 2**exp, with 0.5 <= mant < 1.
	var mant big.Float
	exp := int64(f.MantExp(&mant))

	// Scale so that mant is an integer.
	prec := mant.MinPrec()
	mant.SetMantExp(&mant, int(prec))
	exp -= int64(prec)

	manti, acc := mant.Int(nil)
	if acc != big.Exact {
		base.Fatalf("mantissa scaling failed for %f (%s)", f, acc)
	}
	w.mpint(constant.Make(manti), typ)
	if manti.Sign() != 0 {
		w.int64(exp)
	}
}

func (w *exportWriter) mprat(v constant.Value) {
	r, ok := constant.Val(v).(*big.Rat)
	if !w.bool(ok) {
		return
	}
	// TODO(mdempsky): Come up with a more efficient binary
	// encoding before bumping iexportVersion to expose to
	// gcimporter.
	w.string(r.String())
}

func (w *exportWriter) bool(b bool) bool {
	var x uint64
	if b {
		x = 1
	}
	w.uint64(x)
	return b
}

func (w *exportWriter) int64(x int64)   { w.data.int64(x) }
func (w *exportWriter) uint64(x uint64) { w.data.uint64(x) }
func (w *exportWriter) string(s string) { w.uint64(w.p.stringOff(s)) }

// Compiler-specific extensions.

func (w *exportWriter) constExt(n *ir.Name) {
	// Internally, we now represent untyped float and complex
	// constants with infinite-precision rational numbers using
	// go/constant, but the "public" export data format known to
	// gcimporter only supports 512-bit floating point constants.
	// In case rationals turn out to be a bad idea and we want to
	// switch back to fixed-precision constants, for now we
	// continue writing out the 512-bit truncation in the public
	// data section, and write the exact, rational constant in the
	// compiler's extension data. Also, we only need to worry
	// about exporting rationals for declared constants, because
	// constants that appear in an expression will already have
	// been coerced to a concrete, fixed-precision type.
	//
	// Eventually, assuming we stick with using rationals, we
	// should bump iexportVersion to support rationals, and do the
	// whole gcimporter update song-and-dance.
	//
	// TODO(mdempsky): Prepare vocals for that.

	switch n.Type() {
	case types.UntypedFloat:
		w.mprat(n.Val())
	case types.UntypedComplex:
		v := n.Val()
		w.mprat(constant.Real(v))
		w.mprat(constant.Imag(v))
	}
}

func (w *exportWriter) varExt(n *ir.Name) {
	w.linkname(n.Sym())
	w.symIdx(n.Sym())
}

func (w *exportWriter) funcExt(n *ir.Name) {
	w.linkname(n.Sym())
	w.symIdx(n.Sym())

	// Record definition ABI so cross-ABI calls can be direct.
	// This is important for the performance of calling some
	// common functions implemented in assembly (e.g., bytealg).
	w.uint64(uint64(n.Func.ABI))

	w.uint64(uint64(n.Func.Pragma))

	// Escape analysis.
	for _, fs := range &types.RecvsParams {
		for _, f := range fs(n.Type()).FieldSlice() {
			w.string(f.Note)
		}
	}

	// Write out inline body or body of a generic function/method.
	if n.Type().HasTParam() && n.Func.Body != nil && n.Func.Inl == nil {
		base.FatalfAt(n.Pos(), "generic function is not marked inlineable")
	}
	if n.Func.Inl != nil {
		w.uint64(1 + uint64(n.Func.Inl.Cost))
		w.bool(n.Func.Inl.CanDelayResults)
		if n.Func.ExportInline() || n.Type().HasTParam() {
			if n.Type().HasTParam() {
				// If this generic function/method is from another
				// package, but we didn't use for instantiation in
				// this package, we may not yet have imported it.
				ImportedBody(n.Func)
			}
			w.p.doInline(n)
		}

		// Endlineno for inlined function.
		w.pos(n.Func.Endlineno)
	} else {
		w.uint64(0)
	}
}

func (w *exportWriter) methExt(m *types.Field) {
	w.bool(m.Nointerface())
	w.funcExt(m.Nname.(*ir.Name))
}

func (w *exportWriter) linkname(s *types.Sym) {
	w.string(s.Linkname)
}

func (w *exportWriter) symIdx(s *types.Sym) {
	lsym := s.Linksym()
	if lsym.PkgIdx > goobj.PkgIdxSelf || (lsym.PkgIdx == goobj.PkgIdxInvalid && !lsym.Indexed()) || s.Linkname != "" {
		// Don't export index for non-package symbols, linkname'd symbols,
		// and symbols without an index. They can only be referenced by
		// name.
		w.int64(-1)
	} else {
		// For a defined symbol, export its index.
		// For re-exporting an imported symbol, pass its index through.
		w.int64(int64(lsym.SymIdx))
	}
}

func (w *exportWriter) typeExt(t *types.Type) {
	// Export whether this type is marked notinheap.
	w.bool(t.NotInHeap())
	// For type T, export the index of type descriptor symbols of T and *T.
	if i, ok := typeSymIdx[t]; ok {
		w.int64(i[0])
		w.int64(i[1])
		return
	}
	w.symIdx(types.TypeSym(t))
	w.symIdx(types.TypeSym(t.PtrTo()))
}

// Inline bodies.

func (w *exportWriter) writeNames(dcl []*ir.Name) {
	w.int64(int64(len(dcl)))
	for i, n := range dcl {
		w.pos(n.Pos())
		w.localIdent(n.Sym())
		w.typ(n.Type())
		w.dclIndex[n] = w.maxDclIndex + i
	}
	w.maxDclIndex += len(dcl)
}

func (w *exportWriter) funcBody(fn *ir.Func) {
	//fmt.Printf("Exporting %s\n", fn.Nname.Sym().Name)
	w.writeNames(fn.Inl.Dcl)

	w.stmtList(fn.Inl.Body)
}

func (w *exportWriter) stmtList(list []ir.Node) {
	for _, n := range list {
		w.node(n)
	}
	w.op(ir.OEND)
}

func (w *exportWriter) node(n ir.Node) {
	if ir.OpPrec[n.Op()] < 0 {
		w.stmt(n)
	} else {
		w.expr(n)
	}
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

// Caution: stmt will emit more than one node for statement nodes n that have a
// non-empty n.Ninit and where n is not a non-empty assignment or a node with a natural init
// section (such as in "if", "for", etc.).
func (w *exportWriter) stmt(n ir.Node) {
	if len(n.Init()) > 0 && !ir.StmtWithInit(n.Op()) && !isNonEmptyAssign(n) && n.Op() != ir.ORANGE {
		// can't use stmtList here since we don't want the final OEND
		for _, n := range n.Init() {
			w.stmt(n)
		}
	}

	switch n.Op() {
	case ir.OBLOCK:
		// No OBLOCK in export data.
		// Inline content into this statement list,
		// like the init list above.
		// (At the moment neither the parser nor the typechecker
		// generate OBLOCK nodes except to denote an empty
		// function body, although that may change.)
		n := n.(*ir.BlockStmt)
		for _, n := range n.List {
			w.stmt(n)
		}

	case ir.ODCL:
		n := n.(*ir.Decl)
		if ir.IsBlank(n.X) {
			return // blank declarations not useful to importers
		}
		w.op(ir.ODCL)
		w.localName(n.X)

	case ir.OAS:
		// Don't export "v = <N>" initializing statements, hope they're always
		// preceded by the DCL which will be re-parsed and typecheck to reproduce
		// the "v = <N>" again.
		n := n.(*ir.AssignStmt)
		if n.Y != nil {
			w.op(ir.OAS)
			w.pos(n.Pos())
			w.stmtList(n.Init())
			w.expr(n.X)
			w.expr(n.Y)
			w.bool(n.Def)
		}

	case ir.OASOP:
		n := n.(*ir.AssignOpStmt)
		w.op(ir.OASOP)
		w.pos(n.Pos())
		w.op(n.AsOp)
		w.expr(n.X)
		if w.bool(!n.IncDec) {
			w.expr(n.Y)
		}

	case ir.OAS2, ir.OAS2DOTTYPE, ir.OAS2FUNC, ir.OAS2MAPR, ir.OAS2RECV:
		n := n.(*ir.AssignListStmt)
		if go117ExportTypes {
			w.op(n.Op())
		} else {
			w.op(ir.OAS2)
		}
		w.pos(n.Pos())
		w.stmtList(n.Init())
		w.exprList(n.Lhs)
		w.exprList(n.Rhs)
		w.bool(n.Def)

	case ir.ORETURN:
		n := n.(*ir.ReturnStmt)
		w.op(ir.ORETURN)
		w.pos(n.Pos())
		w.exprList(n.Results)

	// case ORETJMP:
	// 	unreachable - generated by compiler for trampoline routines

	case ir.OGO, ir.ODEFER:
		n := n.(*ir.GoDeferStmt)
		w.op(n.Op())
		w.pos(n.Pos())
		w.expr(n.Call)

	case ir.OIF:
		n := n.(*ir.IfStmt)
		w.op(ir.OIF)
		w.pos(n.Pos())
		w.stmtList(n.Init())
		w.expr(n.Cond)
		w.stmtList(n.Body)
		w.stmtList(n.Else)

	case ir.OFOR:
		n := n.(*ir.ForStmt)
		w.op(ir.OFOR)
		w.pos(n.Pos())
		w.stmtList(n.Init())
		w.exprsOrNil(n.Cond, n.Post)
		w.stmtList(n.Body)

	case ir.ORANGE:
		n := n.(*ir.RangeStmt)
		w.op(ir.ORANGE)
		w.pos(n.Pos())
		w.stmtList(n.Init())
		w.exprsOrNil(n.Key, n.Value)
		w.expr(n.X)
		w.stmtList(n.Body)

	case ir.OSELECT:
		n := n.(*ir.SelectStmt)
		w.op(n.Op())
		w.pos(n.Pos())
		w.stmtList(n.Init())
		w.commList(n.Cases)

	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)
		w.op(n.Op())
		w.pos(n.Pos())
		w.stmtList(n.Init())
		w.exprsOrNil(n.Tag, nil)
		w.caseList(n.Cases, isNamedTypeSwitch(n.Tag))

	// case OCASE:
	//	handled by caseList

	case ir.OFALL:
		n := n.(*ir.BranchStmt)
		w.op(ir.OFALL)
		w.pos(n.Pos())

	case ir.OBREAK, ir.OCONTINUE, ir.OGOTO, ir.OLABEL:
		w.op(n.Op())
		w.pos(n.Pos())
		label := ""
		if sym := n.Sym(); sym != nil {
			label = sym.Name
		}
		w.string(label)

	default:
		base.Fatalf("exporter: CANNOT EXPORT: %v\nPlease notify gri@\n", n.Op())
	}
}

func isNamedTypeSwitch(x ir.Node) bool {
	guard, ok := x.(*ir.TypeSwitchGuard)
	return ok && guard.Tag != nil
}

func (w *exportWriter) caseList(cases []*ir.CaseClause, namedTypeSwitch bool) {
	w.uint64(uint64(len(cases)))
	for _, cas := range cases {
		w.pos(cas.Pos())
		w.stmtList(cas.List)
		if namedTypeSwitch {
			w.localName(cas.Var)
		}
		w.stmtList(cas.Body)
	}
}

func (w *exportWriter) commList(cases []*ir.CommClause) {
	w.uint64(uint64(len(cases)))
	for _, cas := range cases {
		w.pos(cas.Pos())
		defaultCase := cas.Comm == nil
		w.bool(defaultCase)
		if !defaultCase {
			// Only call w.node for non-default cause (cas.Comm is non-nil)
			w.node(cas.Comm)
		}
		w.stmtList(cas.Body)
	}
}

func (w *exportWriter) exprList(list ir.Nodes) {
	for _, n := range list {
		w.expr(n)
	}
	w.op(ir.OEND)
}

func simplifyForExport(n ir.Node) ir.Node {
	switch n.Op() {
	case ir.OPAREN:
		n := n.(*ir.ParenExpr)
		return simplifyForExport(n.X)
	}
	return n
}

func (w *exportWriter) expr(n ir.Node) {
	n = simplifyForExport(n)
	switch n.Op() {
	// expressions
	// (somewhat closely following the structure of exprfmt in fmt.go)
	case ir.ONIL:
		n := n.(*ir.NilExpr)
		// If n is a typeparam, it will have already been checked
		// for proper use by the types2 typechecker.
		if !n.Type().IsTypeParam() && !n.Type().HasNil() {
			base.Fatalf("unexpected type for nil: %v", n.Type())
		}
		w.op(ir.ONIL)
		w.pos(n.Pos())
		w.typ(n.Type())

	case ir.OLITERAL:
		w.op(ir.OLITERAL)
		if ir.HasUniquePos(n) {
			w.pos(n.Pos())
		} else {
			w.pos(src.NoXPos)
		}
		w.value(n.Type(), n.Val())

	case ir.ONAME:
		// Package scope name.
		n := n.(*ir.Name)
		if (n.Class == ir.PEXTERN || n.Class == ir.PFUNC) && !ir.IsBlank(n) {
			w.op(ir.ONONAME)
			// Indicate that this is not an OKEY entry.
			w.bool(false)
			w.qualifiedIdent(n)
			if go117ExportTypes {
				w.typ(n.Type())
			}
			break
		}

		// Function scope name.
		// We don't need a type here, as the type will be provided at the
		// declaration of n.
		w.op(ir.ONAME)

		// This handles the case where we haven't yet transformed a call
		// to a builtin, so we must write out the builtin as a name in the
		// builtin package.
		isBuiltin := n.BuiltinOp != ir.OXXX
		w.bool(isBuiltin)
		if isBuiltin {
			w.bool(n.Sym().Pkg == types.UnsafePkg)
			w.string(n.Sym().Name)
			break
		}
		w.localName(n)

	case ir.ONONAME:
		w.op(ir.ONONAME)
		// This can only be for OKEY nodes in generic functions. Mark it
		// as a key entry.
		w.bool(true)
		s := n.Sym()
		w.string(s.Name)
		w.pkg(s.Pkg)
		if go117ExportTypes {
			w.typ(n.Type())
		}

	// case OPACK:
	// 	should have been resolved by typechecking - handled by default case

	case ir.OTYPE:
		w.op(ir.OTYPE)
		w.typ(n.Type())

	case ir.ODYNAMICTYPE:
		n := n.(*ir.DynamicType)
		w.op(ir.ODYNAMICTYPE)
		w.pos(n.Pos())
		w.expr(n.X)
		if n.ITab != nil {
			w.bool(true)
			w.expr(n.ITab)
		} else {
			w.bool(false)
		}
		w.typ(n.Type())

	case ir.OTYPESW:
		n := n.(*ir.TypeSwitchGuard)
		w.op(ir.OTYPESW)
		w.pos(n.Pos())
		var s *types.Sym
		if n.Tag != nil {
			if n.Tag.Op() != ir.ONONAME {
				base.Fatalf("expected ONONAME, got %v", n.Tag)
			}
			s = n.Tag.Sym()
		}
		w.localIdent(s) // declared pseudo-variable, if any
		w.expr(n.X)

	// case OTARRAY, OTMAP, OTCHAN, OTSTRUCT, OTINTER, OTFUNC:
	// 	should have been resolved by typechecking - handled by default case

	case ir.OCLOSURE:
		n := n.(*ir.ClosureExpr)
		w.op(ir.OCLOSURE)
		w.pos(n.Pos())
		old := w.currPkg
		w.setPkg(n.Type().Pkg(), true)
		w.signature(n.Type())
		w.setPkg(old, true)

		// Write out id for the Outer of each conditional variable. The
		// conditional variable itself for this closure will be re-created
		// during import.
		w.int64(int64(len(n.Func.ClosureVars)))
		for i, cv := range n.Func.ClosureVars {
			w.pos(cv.Pos())
			w.localName(cv.Outer)
			// Closure variable (which will be re-created during
			// import) is given via a negative id, starting at -2,
			// which is used to refer to it later in the function
			// during export. -1 represents blanks.
			w.dclIndex[cv] = -(i + 2) - w.maxClosureVarIndex
		}
		w.maxClosureVarIndex += len(n.Func.ClosureVars)

		// like w.funcBody(n.Func), but not for .Inl
		w.writeNames(n.Func.Dcl)
		w.stmtList(n.Func.Body)

	// case OCOMPLIT:
	// 	should have been resolved by typechecking - handled by default case

	case ir.OPTRLIT:
		n := n.(*ir.AddrExpr)
		if go117ExportTypes {
			w.op(ir.OPTRLIT)
		} else {
			w.op(ir.OADDR)
		}
		w.pos(n.Pos())
		w.expr(n.X)
		if go117ExportTypes {
			w.typ(n.Type())
		}

	case ir.OSTRUCTLIT:
		n := n.(*ir.CompLitExpr)
		w.op(ir.OSTRUCTLIT)
		w.pos(n.Pos())
		w.typ(n.Type())
		w.fieldList(n.List) // special handling of field names

	case ir.OCOMPLIT, ir.OARRAYLIT, ir.OSLICELIT, ir.OMAPLIT:
		n := n.(*ir.CompLitExpr)
		if go117ExportTypes {
			w.op(n.Op())
		} else {
			w.op(ir.OCOMPLIT)
		}
		w.pos(n.Pos())
		w.typ(n.Type())
		w.exprList(n.List)
		if go117ExportTypes && n.Op() == ir.OSLICELIT {
			w.uint64(uint64(n.Len))
		}
	case ir.OKEY:
		n := n.(*ir.KeyExpr)
		w.op(ir.OKEY)
		w.pos(n.Pos())
		w.expr(n.Key)
		w.expr(n.Value)

	// case OSTRUCTKEY:
	//	unreachable - handled in case OSTRUCTLIT by elemList

	case ir.OXDOT, ir.ODOT, ir.ODOTPTR, ir.ODOTINTER, ir.ODOTMETH, ir.OMETHVALUE, ir.OMETHEXPR:
		n := n.(*ir.SelectorExpr)
		if go117ExportTypes {
			// For go117ExportTypes, we usually see all ops except
			// OXDOT, but we can see OXDOT for generic functions.
			w.op(n.Op())
		} else {
			w.op(ir.OXDOT)
		}
		w.pos(n.Pos())
		w.expr(n.X)
		w.exoticSelector(n.Sel)
		if go117ExportTypes {
			w.exoticType(n.Type())
			if n.Op() == ir.OXDOT {
				// n.Selection for method references will be
				// reconstructed during import.
				w.bool(n.Selection != nil)
			} else if n.Op() == ir.ODOT || n.Op() == ir.ODOTPTR || n.Op() == ir.ODOTINTER {
				w.exoticField(n.Selection)
			}
			// n.Selection is not required for OMETHEXPR, ODOTMETH, and OMETHVALUE. It will
			// be reconstructed during import.  n.Selection is computed during
			// transformDot() for OXDOT.
		}

	case ir.ODOTTYPE, ir.ODOTTYPE2:
		n := n.(*ir.TypeAssertExpr)
		if go117ExportTypes {
			w.op(n.Op())
		} else {
			w.op(ir.ODOTTYPE)
		}
		w.pos(n.Pos())
		w.expr(n.X)
		w.typ(n.Type())

	case ir.ODYNAMICDOTTYPE, ir.ODYNAMICDOTTYPE2:
		n := n.(*ir.DynamicTypeAssertExpr)
		w.op(n.Op())
		w.pos(n.Pos())
		w.expr(n.X)
		w.expr(n.T)
		w.typ(n.Type())

	case ir.OINDEX, ir.OINDEXMAP:
		n := n.(*ir.IndexExpr)
		if go117ExportTypes {
			w.op(n.Op())
		} else {
			w.op(ir.OINDEX)
		}
		w.pos(n.Pos())
		w.expr(n.X)
		w.expr(n.Index)
		if go117ExportTypes {
			w.exoticType(n.Type())
			if n.Op() == ir.OINDEXMAP {
				w.bool(n.Assigned)
			}
		}

	case ir.OSLICE, ir.OSLICESTR, ir.OSLICEARR:
		n := n.(*ir.SliceExpr)
		if go117ExportTypes {
			w.op(n.Op())
		} else {
			w.op(ir.OSLICE)
		}
		w.pos(n.Pos())
		w.expr(n.X)
		w.exprsOrNil(n.Low, n.High)
		if go117ExportTypes {
			w.typ(n.Type())
		}

	case ir.OSLICE3, ir.OSLICE3ARR:
		n := n.(*ir.SliceExpr)
		if go117ExportTypes {
			w.op(n.Op())
		} else {
			w.op(ir.OSLICE3)
		}
		w.pos(n.Pos())
		w.expr(n.X)
		w.exprsOrNil(n.Low, n.High)
		w.expr(n.Max)
		if go117ExportTypes {
			w.typ(n.Type())
		}

	case ir.OCOPY, ir.OCOMPLEX, ir.OUNSAFEADD, ir.OUNSAFESLICE:
		// treated like other builtin calls (see e.g., OREAL)
		n := n.(*ir.BinaryExpr)
		w.op(n.Op())
		w.pos(n.Pos())
		w.expr(n.X)
		w.expr(n.Y)
		if go117ExportTypes {
			w.typ(n.Type())
		} else {
			w.op(ir.OEND)
		}

	case ir.OCONV, ir.OCONVIFACE, ir.OCONVIDATA, ir.OCONVNOP, ir.OBYTES2STR, ir.ORUNES2STR, ir.OSTR2BYTES, ir.OSTR2RUNES, ir.ORUNESTR, ir.OSLICE2ARRPTR:
		n := n.(*ir.ConvExpr)
		if go117ExportTypes {
			w.op(n.Op())
		} else {
			w.op(ir.OCONV)
		}
		w.pos(n.Pos())
		w.typ(n.Type())
		w.expr(n.X)

	case ir.OREAL, ir.OIMAG, ir.OCAP, ir.OCLOSE, ir.OLEN, ir.ONEW, ir.OPANIC:
		n := n.(*ir.UnaryExpr)
		w.op(n.Op())
		w.pos(n.Pos())
		w.expr(n.X)
		if go117ExportTypes {
			if n.Op() != ir.OPANIC {
				w.typ(n.Type())
			}
		} else {
			w.op(ir.OEND)
		}

	case ir.OAPPEND, ir.ODELETE, ir.ORECOVER, ir.OPRINT, ir.OPRINTN:
		n := n.(*ir.CallExpr)
		w.op(n.Op())
		w.pos(n.Pos())
		w.exprList(n.Args) // emits terminating OEND
		// only append() calls may contain '...' arguments
		if n.Op() == ir.OAPPEND {
			w.bool(n.IsDDD)
		} else if n.IsDDD {
			base.Fatalf("exporter: unexpected '...' with %v call", n.Op())
		}
		if go117ExportTypes {
			if n.Op() != ir.ODELETE && n.Op() != ir.OPRINT && n.Op() != ir.OPRINTN {
				w.typ(n.Type())
			}
		}

	case ir.OCALL, ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER, ir.OGETG:
		n := n.(*ir.CallExpr)
		if go117ExportTypes {
			w.op(n.Op())
		} else {
			w.op(ir.OCALL)
		}
		w.pos(n.Pos())
		w.stmtList(n.Init())
		w.expr(n.X)
		w.exprList(n.Args)
		w.bool(n.IsDDD)
		if go117ExportTypes {
			w.exoticType(n.Type())
		}

	case ir.OMAKEMAP, ir.OMAKECHAN, ir.OMAKESLICE:
		n := n.(*ir.MakeExpr)
		w.op(n.Op()) // must keep separate from OMAKE for importer
		w.pos(n.Pos())
		w.typ(n.Type())
		switch {
		default:
			// empty list
			w.op(ir.OEND)
		case n.Cap != nil:
			w.expr(n.Len)
			w.expr(n.Cap)
			w.op(ir.OEND)
		case n.Len != nil && (n.Op() == ir.OMAKESLICE || !n.Len.Type().IsUntyped()):
			// Note: the extra conditional exists because make(T) for
			// T a map or chan type, gets an untyped zero added as
			// an argument. Don't serialize that argument here.
			w.expr(n.Len)
			w.op(ir.OEND)
		case n.Len != nil && go117ExportTypes:
			w.expr(n.Len)
			w.op(ir.OEND)
		}

	case ir.OLINKSYMOFFSET:
		n := n.(*ir.LinksymOffsetExpr)
		w.op(ir.OLINKSYMOFFSET)
		w.pos(n.Pos())
		w.string(n.Linksym.Name)
		w.uint64(uint64(n.Offset_))
		w.typ(n.Type())

	// unary expressions
	case ir.OPLUS, ir.ONEG, ir.OBITNOT, ir.ONOT, ir.ORECV, ir.OIDATA:
		n := n.(*ir.UnaryExpr)
		w.op(n.Op())
		w.pos(n.Pos())
		w.expr(n.X)
		if go117ExportTypes {
			w.typ(n.Type())
		}

	case ir.OADDR:
		n := n.(*ir.AddrExpr)
		w.op(n.Op())
		w.pos(n.Pos())
		w.expr(n.X)
		if go117ExportTypes {
			w.typ(n.Type())
		}

	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		w.op(n.Op())
		w.pos(n.Pos())
		w.expr(n.X)
		if go117ExportTypes {
			w.typ(n.Type())
		}

	case ir.OSEND:
		n := n.(*ir.SendStmt)
		w.op(n.Op())
		w.pos(n.Pos())
		w.expr(n.Chan)
		w.expr(n.Value)

	// binary expressions
	case ir.OADD, ir.OAND, ir.OANDNOT, ir.ODIV, ir.OEQ, ir.OGE, ir.OGT, ir.OLE, ir.OLT,
		ir.OLSH, ir.OMOD, ir.OMUL, ir.ONE, ir.OOR, ir.ORSH, ir.OSUB, ir.OXOR, ir.OEFACE:
		n := n.(*ir.BinaryExpr)
		w.op(n.Op())
		w.pos(n.Pos())
		w.expr(n.X)
		w.expr(n.Y)
		if go117ExportTypes {
			w.typ(n.Type())
		}

	case ir.OANDAND, ir.OOROR:
		n := n.(*ir.LogicalExpr)
		w.op(n.Op())
		w.pos(n.Pos())
		w.expr(n.X)
		w.expr(n.Y)
		if go117ExportTypes {
			w.typ(n.Type())
		}

	case ir.OADDSTR:
		n := n.(*ir.AddStringExpr)
		w.op(ir.OADDSTR)
		w.pos(n.Pos())
		w.exprList(n.List)
		if go117ExportTypes {
			w.typ(n.Type())
		}

	case ir.ODCLCONST:
		// if exporting, DCLCONST should just be removed as its usage
		// has already been replaced with literals

	case ir.OFUNCINST:
		n := n.(*ir.InstExpr)
		w.op(ir.OFUNCINST)
		w.pos(n.Pos())
		w.expr(n.X)
		w.uint64(uint64(len(n.Targs)))
		for _, targ := range n.Targs {
			w.typ(targ.Type())
		}
		if go117ExportTypes {
			w.typ(n.Type())
		}

	case ir.OSELRECV2:
		n := n.(*ir.AssignListStmt)
		w.op(ir.OSELRECV2)
		w.pos(n.Pos())
		w.stmtList(n.Init())
		w.exprList(n.Lhs)
		w.exprList(n.Rhs)
		w.bool(n.Def)

	default:
		base.Fatalf("cannot export %v (%d) node\n"+
			"\t==> please file an issue and assign to gri@", n.Op(), int(n.Op()))
	}
}

func (w *exportWriter) op(op ir.Op) {
	if debug {
		w.uint64(magic)
	}
	w.uint64(uint64(op))
}

func (w *exportWriter) exprsOrNil(a, b ir.Node) {
	ab := 0
	if a != nil {
		ab |= 1
	}
	if b != nil {
		ab |= 2
	}
	w.uint64(uint64(ab))
	if ab&1 != 0 {
		w.expr(a)
	}
	if ab&2 != 0 {
		w.node(b)
	}
}

func (w *exportWriter) fieldList(list ir.Nodes) {
	w.uint64(uint64(len(list)))
	for _, n := range list {
		n := n.(*ir.StructKeyExpr)
		w.pos(n.Pos())
		w.exoticField(n.Field)
		w.expr(n.Value)
	}
}

func (w *exportWriter) localName(n *ir.Name) {
	if ir.IsBlank(n) {
		w.int64(-1)
		return
	}

	i, ok := w.dclIndex[n]
	if !ok {
		base.FatalfAt(n.Pos(), "missing from dclIndex: %+v", n)
	}
	w.int64(int64(i))
}

func (w *exportWriter) localIdent(s *types.Sym) {
	if w.currPkg == nil {
		base.Fatalf("missing currPkg")
	}

	// Anonymous parameters.
	if s == nil {
		w.string("")
		return
	}

	name := s.Name
	if name == "_" {
		w.string("_")
		return
	}

	// The name of autotmp variables isn't important; they just need to
	// be unique. To stabilize the export data, simply write out "$" as
	// a marker and let the importer generate its own unique name.
	if strings.HasPrefix(name, ".autotmp_") {
		w.string("$autotmp")
		return
	}

	if i := strings.LastIndex(name, "."); i >= 0 && !strings.HasPrefix(name, LocalDictName) {
		base.Fatalf("unexpected dot in identifier: %v", name)
	}

	if s.Pkg != w.currPkg {
		base.Fatalf("weird package in name: %v => %v from %q, not %q", s, name, s.Pkg.Path, w.currPkg.Path)
	}

	w.string(name)
}

type intWriter struct {
	bytes.Buffer
}

func (w *intWriter) int64(x int64) {
	var buf [binary.MaxVarintLen64]byte
	n := binary.PutVarint(buf[:], x)
	w.Write(buf[:n])
}

func (w *intWriter) uint64(x uint64) {
	var buf [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(buf[:], x)
	w.Write(buf[:n])
}

// If go117ExportTypes is true, then we write type information when
// exporting function bodies, so those function bodies don't need to
// be re-typechecked on import.
// This flag adds some other info to the serialized stream as well
// which was previously recomputed during typechecking, like
// specializing opcodes (e.g. OXDOT to ODOTPTR) and ancillary
// information (e.g. length field for OSLICELIT).
const go117ExportTypes = true
const Go117ExportTypes = go117ExportTypes

// The name used for dictionary parameters or local variables.
const LocalDictName = ".dict"
