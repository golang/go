// Copyright 2019 The Go Authors. All rights reserved.
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
//         Tag  byte // 'A' or 'B'
//         Pos  Pos
//         TypeParams []typeOff  // only present if Tag == 'B'
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

package gcimporter

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
	"io"
	"math/big"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/go/types/objectpath"
)

// IExportShallow encodes "shallow" export data for the specified package.
//
// For types, we use "shallow" export data. Historically, the Go
// compiler always produced a summary of the types for a given package
// that included types from other packages that it indirectly
// referenced: "deep" export data. This had the advantage that the
// compiler (and analogous tools such as gopls) need only load one
// file per direct import.  However, it meant that the files tended to
// get larger based on the level of the package in the import
// graph. For example, higher-level packages in the kubernetes module
// have over 1MB of "deep" export data, even when they have almost no
// content of their own, merely because they mention a major type that
// references many others. In pathological cases the export data was
// 300x larger than the source for a package due to this quadratic
// growth.
//
// "Shallow" export data means that the serialized types describe only
// a single package. If those types mention types from other packages,
// the type checker may need to request additional packages beyond
// just the direct imports. Type information for the entire transitive
// closure of imports is provided (lazily) by the DAG.
//
// No promises are made about the encoding other than that it can be decoded by
// the same version of IIExportShallow. If you plan to save export data in the
// file system, be sure to include a cryptographic digest of the executable in
// the key to avoid version skew.
//
// If the provided reportf func is non-nil, it is used for reporting
// bugs (e.g. recovered panics) encountered during export, enabling us
// to obtain via telemetry the stack that would otherwise be lost by
// merely returning an error.
func IExportShallow(fset *token.FileSet, pkg *types.Package, reportf ReportFunc) ([]byte, error) {
	// In principle this operation can only fail if out.Write fails,
	// but that's impossible for bytes.Buffer---and as a matter of
	// fact iexportCommon doesn't even check for I/O errors.
	// TODO(adonovan): handle I/O errors properly.
	// TODO(adonovan): use byte slices throughout, avoiding copying.
	const bundle, shallow = false, true
	var out bytes.Buffer
	err := iexportCommon(&out, fset, bundle, shallow, iexportVersion, []*types.Package{pkg}, reportf)
	return out.Bytes(), err
}

// IImportShallow decodes "shallow" types.Package data encoded by
// [IExportShallow] in the same executable. This function cannot import data
// from cmd/compile or gcexportdata.Write.
//
// The importer calls getPackages to obtain package symbols for all
// packages mentioned in the export data, including the one being
// decoded.
//
// If the provided reportf func is non-nil, it will be used for reporting bugs
// encountered during import.
// TODO(rfindley): remove reportf when we are confident enough in the new
// objectpath encoding.
func IImportShallow(fset *token.FileSet, getPackages GetPackagesFunc, data []byte, path string, reportf ReportFunc) (*types.Package, error) {
	const bundle = false
	const shallow = true
	pkgs, err := iimportCommon(fset, getPackages, data, bundle, path, shallow, reportf)
	if err != nil {
		return nil, err
	}
	return pkgs[0], nil
}

// ReportFunc is the type of a function used to report formatted bugs.
type ReportFunc = func(string, ...any)

// Current bundled export format version. Increase with each format change.
// 0: initial implementation
const bundleVersion = 0

// IExportData writes indexed export data for pkg to out.
//
// If no file set is provided, position info will be missing.
// The package path of the top-level package will not be recorded,
// so that calls to IImportData can override with a provided package path.
func IExportData(out io.Writer, fset *token.FileSet, pkg *types.Package) error {
	const bundle, shallow = false, false
	return iexportCommon(out, fset, bundle, shallow, iexportVersion, []*types.Package{pkg}, nil)
}

// IExportBundle writes an indexed export bundle for pkgs to out.
func IExportBundle(out io.Writer, fset *token.FileSet, pkgs []*types.Package) error {
	const bundle, shallow = true, false
	return iexportCommon(out, fset, bundle, shallow, iexportVersion, pkgs, nil)
}

func iexportCommon(out io.Writer, fset *token.FileSet, bundle, shallow bool, version int, pkgs []*types.Package, reportf ReportFunc) (err error) {
	if !debug {
		defer func() {
			if e := recover(); e != nil {
				// Report the stack via telemetry (see #71067).
				if reportf != nil {
					reportf("panic in exporter")
				}
				if ierr, ok := e.(internalError); ok {
					// internalError usually means we exported a
					// bad go/types data structure: a violation
					// of an implicit precondition of Export.
					err = ierr
					return
				}
				// Not an internal error; panic again.
				panic(e)
			}
		}()
	}

	p := iexporter{
		fset:        fset,
		version:     version,
		shallow:     shallow,
		allPkgs:     map[*types.Package]bool{},
		stringIndex: map[string]uint64{},
		declIndex:   map[types.Object]uint64{},
		tparamNames: map[types.Object]string{},
		typIndex:    map[types.Type]uint64{},
	}
	if !bundle {
		p.localpkg = pkgs[0]
	}

	for i, pt := range predeclared() {
		p.typIndex[pt] = uint64(i)
	}
	if len(p.typIndex) > predeclReserved {
		panic(internalErrorf("too many predeclared types: %d > %d", len(p.typIndex), predeclReserved))
	}

	// Initialize work queue with exported declarations.
	for _, pkg := range pkgs {
		scope := pkg.Scope()
		for _, name := range scope.Names() {
			if token.IsExported(name) {
				p.pushDecl(scope.Lookup(name))
			}
		}

		if bundle {
			// Ensure pkg and its imports are included in the index.
			p.allPkgs[pkg] = true
			for _, imp := range pkg.Imports() {
				p.allPkgs[imp] = true
			}
		}
	}

	// Loop until no more work.
	for !p.declTodo.empty() {
		p.doDecl(p.declTodo.popHead())
	}

	// Produce index of offset of each file record in files.
	var files intWriter
	var fileOffset []uint64 // fileOffset[i] is offset in files of file encoded as i
	if p.shallow {
		fileOffset = make([]uint64, len(p.fileInfos))
		for i, info := range p.fileInfos {
			fileOffset[i] = uint64(files.Len())
			p.encodeFile(&files, info.file, info.needed)
		}
	}

	// Append indices to data0 section.
	dataLen := uint64(p.data0.Len())
	w := p.newWriter()
	w.writeIndex(p.declIndex)

	if bundle {
		w.uint64(uint64(len(pkgs)))
		for _, pkg := range pkgs {
			w.pkg(pkg)
			imps := pkg.Imports()
			w.uint64(uint64(len(imps)))
			for _, imp := range imps {
				w.pkg(imp)
			}
		}
	}
	w.flush()

	// Assemble header.
	var hdr intWriter
	if bundle {
		hdr.uint64(bundleVersion)
	}
	hdr.uint64(uint64(p.version))
	hdr.uint64(uint64(p.strings.Len()))
	if p.shallow {
		hdr.uint64(uint64(files.Len()))
		hdr.uint64(uint64(len(fileOffset)))
		for _, offset := range fileOffset {
			hdr.uint64(offset)
		}
	}
	hdr.uint64(dataLen)

	// Flush output.
	io.Copy(out, &hdr)
	io.Copy(out, &p.strings)
	if p.shallow {
		io.Copy(out, &files)
	}
	io.Copy(out, &p.data0)

	return nil
}

// encodeFile writes to w a representation of the file sufficient to
// faithfully restore position information about all needed offsets.
// Mutates the needed array.
func (p *iexporter) encodeFile(w *intWriter, file *token.File, needed []uint64) {
	_ = needed[0] // precondition: needed is non-empty

	w.uint64(p.stringOff(file.Name()))

	size := uint64(file.Size())
	w.uint64(size)

	// Sort the set of needed offsets. Duplicates are harmless.
	slices.Sort(needed)

	lines := file.Lines() // byte offset of each line start
	w.uint64(uint64(len(lines)))

	// Rather than record the entire array of line start offsets,
	// we save only a sparse list of (index, offset) pairs for
	// the start of each line that contains a needed position.
	var sparse [][2]int // (index, offset) pairs
outer:
	for i, lineStart := range lines {
		lineEnd := size
		if i < len(lines)-1 {
			lineEnd = uint64(lines[i+1])
		}
		// Does this line contains a needed offset?
		if needed[0] < lineEnd {
			sparse = append(sparse, [2]int{i, lineStart})
			for needed[0] < lineEnd {
				needed = needed[1:]
				if len(needed) == 0 {
					break outer
				}
			}
		}
	}

	// Delta-encode the columns.
	w.uint64(uint64(len(sparse)))
	var prev [2]int
	for _, pair := range sparse {
		w.uint64(uint64(pair[0] - prev[0]))
		w.uint64(uint64(pair[1] - prev[1]))
		prev = pair
	}
}

// writeIndex writes out an object index. mainIndex indicates whether
// we're writing out the main index, which is also read by
// non-compiler tools and includes a complete package description
// (i.e., name and height).
func (w *exportWriter) writeIndex(index map[types.Object]uint64) {
	type pkgObj struct {
		obj  types.Object
		name string // qualified name; differs from obj.Name for type params
	}
	// Build a map from packages to objects from that package.
	pkgObjs := map[*types.Package][]pkgObj{}

	// For the main index, make sure to include every package that
	// we reference, even if we're not exporting (or reexporting)
	// any symbols from it.
	if w.p.localpkg != nil {
		pkgObjs[w.p.localpkg] = nil
	}
	for pkg := range w.p.allPkgs {
		pkgObjs[pkg] = nil
	}

	for obj := range index {
		name := w.p.exportName(obj)
		pkgObjs[obj.Pkg()] = append(pkgObjs[obj.Pkg()], pkgObj{obj, name})
	}

	var pkgs []*types.Package
	for pkg, objs := range pkgObjs {
		pkgs = append(pkgs, pkg)

		sort.Slice(objs, func(i, j int) bool {
			return objs[i].name < objs[j].name
		})
	}

	sort.Slice(pkgs, func(i, j int) bool {
		return w.exportPath(pkgs[i]) < w.exportPath(pkgs[j])
	})

	w.uint64(uint64(len(pkgs)))
	for _, pkg := range pkgs {
		w.string(w.exportPath(pkg))
		w.string(pkg.Name())
		w.uint64(uint64(0)) // package height is not needed for go/types

		objs := pkgObjs[pkg]
		w.uint64(uint64(len(objs)))
		for _, obj := range objs {
			w.string(obj.name)
			w.uint64(index[obj.obj])
		}
	}
}

// exportName returns the 'exported' name of an object. It differs from
// obj.Name() only for type parameters (see tparamExportName for details).
func (p *iexporter) exportName(obj types.Object) (res string) {
	if name := p.tparamNames[obj]; name != "" {
		return name
	}
	return obj.Name()
}

type iexporter struct {
	fset    *token.FileSet
	version int

	shallow    bool                // don't put types from other packages in the index
	objEncoder *objectpath.Encoder // encodes objects from other packages in shallow mode; lazily allocated
	localpkg   *types.Package      // (nil in bundle mode)

	// allPkgs tracks all packages that have been referenced by
	// the export data, so we can ensure to include them in the
	// main index.
	allPkgs map[*types.Package]bool

	declTodo objQueue

	strings     intWriter
	stringIndex map[string]uint64

	// In shallow mode, object positions are encoded as (file, offset).
	// Each file is recorded as a line-number table.
	// Only the lines of needed positions are saved faithfully.
	fileInfo  map[*token.File]uint64 // value is index in fileInfos
	fileInfos []*filePositions

	data0       intWriter
	declIndex   map[types.Object]uint64
	tparamNames map[types.Object]string // typeparam->exported name
	typIndex    map[types.Type]uint64

	indent int // for tracing support
}

type filePositions struct {
	file   *token.File
	needed []uint64 // unordered list of needed file offsets
}

func (p *iexporter) trace(format string, args ...any) {
	if !trace {
		// Call sites should also be guarded, but having this check here allows
		// easily enabling/disabling debug trace statements.
		return
	}
	fmt.Printf(strings.Repeat("..", p.indent)+format+"\n", args...)
}

// objectpathEncoder returns the lazily allocated objectpath.Encoder to use
// when encoding objects in other packages during shallow export.
//
// Using a shared Encoder amortizes some of cost of objectpath search.
func (p *iexporter) objectpathEncoder() *objectpath.Encoder {
	if p.objEncoder == nil {
		p.objEncoder = new(objectpath.Encoder)
	}
	return p.objEncoder
}

// stringOff returns the offset of s within the string section.
// If not already present, it's added to the end.
func (p *iexporter) stringOff(s string) uint64 {
	off, ok := p.stringIndex[s]
	if !ok {
		off = uint64(p.strings.Len())
		p.stringIndex[s] = off

		p.strings.uint64(uint64(len(s)))
		p.strings.WriteString(s)
	}
	return off
}

// fileIndexAndOffset returns the index of the token.File and the byte offset of pos within it.
func (p *iexporter) fileIndexAndOffset(file *token.File, pos token.Pos) (uint64, uint64) {
	index, ok := p.fileInfo[file]
	if !ok {
		index = uint64(len(p.fileInfo))
		p.fileInfos = append(p.fileInfos, &filePositions{file: file})
		if p.fileInfo == nil {
			p.fileInfo = make(map[*token.File]uint64)
		}
		p.fileInfo[file] = index
	}
	// Record each needed offset.
	info := p.fileInfos[index]
	offset := uint64(file.Offset(pos))
	info.needed = append(info.needed, offset)

	return index, offset
}

// pushDecl adds n to the declaration work queue, if not already present.
func (p *iexporter) pushDecl(obj types.Object) {
	// Package unsafe is known to the compiler and predeclared.
	// Caller should not ask us to do export it.
	if obj.Pkg() == types.Unsafe {
		panic("cannot export package unsafe")
	}

	// Shallow export data: don't index decls from other packages.
	if p.shallow && obj.Pkg() != p.localpkg {
		return
	}

	if _, ok := p.declIndex[obj]; ok {
		return
	}

	p.declIndex[obj] = ^uint64(0) // mark obj present in work queue
	p.declTodo.pushTail(obj)
}

// exportWriter handles writing out individual data section chunks.
type exportWriter struct {
	p *iexporter

	data       intWriter
	prevFile   string
	prevLine   int64
	prevColumn int64
}

func (w *exportWriter) exportPath(pkg *types.Package) string {
	if pkg == w.p.localpkg {
		return ""
	}
	return pkg.Path()
}

func (p *iexporter) doDecl(obj types.Object) {
	if trace {
		p.trace("exporting decl %v (%T)", obj, obj)
		p.indent++
		defer func() {
			p.indent--
			p.trace("=> %s", obj)
		}()
	}
	w := p.newWriter()

	switch obj := obj.(type) {
	case *types.Var:
		w.tag(varTag)
		w.pos(obj.Pos())
		w.typ(obj.Type(), obj.Pkg())

	case *types.Func:
		sig, _ := obj.Type().(*types.Signature)
		if sig.Recv() != nil {
			// We shouldn't see methods in the package scope,
			// but the type checker may repair "func () F() {}"
			// to "func (Invalid) F()" and then treat it like "func F()",
			// so allow that. See golang/go#57729.
			if sig.Recv().Type() != types.Typ[types.Invalid] {
				panic(internalErrorf("unexpected method: %v", sig))
			}
		}

		// Function.
		if sig.TypeParams().Len() == 0 {
			w.tag(funcTag)
		} else {
			w.tag(genericFuncTag)
		}
		w.pos(obj.Pos())
		// The tparam list of the function type is the declaration of the type
		// params. So, write out the type params right now. Then those type params
		// will be referenced via their type offset (via typOff) in all other
		// places in the signature and function where they are used.
		//
		// While importing the type parameters, tparamList computes and records
		// their export name, so that it can be later used when writing the index.
		if tparams := sig.TypeParams(); tparams.Len() > 0 {
			w.tparamList(obj.Name(), tparams, obj.Pkg())
		}
		w.signature(sig)

	case *types.Const:
		w.tag(constTag)
		w.pos(obj.Pos())
		w.value(obj.Type(), obj.Val())

	case *types.TypeName:
		t := obj.Type()

		if tparam, ok := types.Unalias(t).(*types.TypeParam); ok {
			w.tag(typeParamTag)
			w.pos(obj.Pos())
			constraint := tparam.Constraint()
			if p.version >= iexportVersionGo1_18 {
				implicit := false
				if iface, _ := types.Unalias(constraint).(*types.Interface); iface != nil {
					implicit = iface.IsImplicit()
				}
				w.bool(implicit)
			}
			w.typ(constraint, obj.Pkg())
			break
		}

		if obj.IsAlias() {
			alias, materialized := t.(*types.Alias) // perhaps false for certain built-ins?

			var tparams *types.TypeParamList
			if materialized {
				tparams = alias.TypeParams()
			}
			if tparams.Len() == 0 {
				w.tag(aliasTag)
			} else {
				w.tag(genericAliasTag)
			}
			w.pos(obj.Pos())
			if tparams.Len() > 0 {
				w.tparamList(obj.Name(), tparams, obj.Pkg())
			}
			if materialized {
				// Preserve materialized aliases,
				// even of non-exported types.
				t = alias.Rhs()
			}
			w.typ(t, obj.Pkg())
			break
		}

		// Defined type.
		named, ok := t.(*types.Named)
		if !ok {
			panic(internalErrorf("%s is not a defined type", t))
		}

		if named.TypeParams().Len() == 0 {
			w.tag(typeTag)
		} else {
			w.tag(genericTypeTag)
		}
		w.pos(obj.Pos())

		if named.TypeParams().Len() > 0 {
			// While importing the type parameters, tparamList computes and records
			// their export name, so that it can be later used when writing the index.
			w.tparamList(obj.Name(), named.TypeParams(), obj.Pkg())
		}

		underlying := named.Underlying()
		w.typ(underlying, obj.Pkg())

		if types.IsInterface(t) {
			break
		}

		n := named.NumMethods()
		w.uint64(uint64(n))
		for i := range n {
			m := named.Method(i)
			w.pos(m.Pos())
			w.string(m.Name())
			sig, _ := m.Type().(*types.Signature)
			if w.p.version >= iexportVersionGenericMethods && w.bool(sig.TypeParams().Len() > 0) {
				w.tparamList(obj.Name()+"."+m.Name(), sig.TypeParams(), obj.Pkg())
			}

			// Receiver type parameters are type arguments of the receiver type, so
			// their name must be qualified before exporting recv.
			if rparams := sig.RecvTypeParams(); rparams.Len() > 0 {
				prefix := obj.Name() + "." + m.Name()
				for rparam := range rparams.TypeParams() {
					name := tparamExportName(prefix, rparam)
					w.p.tparamNames[rparam.Obj()] = name
				}
			}
			w.param(sig.Recv())
			w.signature(sig)
		}

	default:
		panic(internalErrorf("unexpected object: %v", obj))
	}

	p.declIndex[obj] = w.flush()
}

func (w *exportWriter) tag(tag byte) {
	w.data.WriteByte(tag)
}

func (w *exportWriter) pos(pos token.Pos) {
	if w.p.shallow {
		w.posV2(pos)
	} else if w.p.version >= iexportVersionPosCol {
		w.posV1(pos)
	} else {
		w.posV0(pos)
	}
}

// posV2 encoding (used only in shallow mode) records positions as
// (file, offset), where file is the index in the token.File table
// (which records the file name and newline offsets) and offset is a
// byte offset. It effectively ignores //line directives.
func (w *exportWriter) posV2(pos token.Pos) {
	if pos == token.NoPos {
		w.uint64(0)
		return
	}
	file := w.p.fset.File(pos) // fset must be non-nil
	index, offset := w.p.fileIndexAndOffset(file, pos)
	w.uint64(1 + index)
	w.uint64(offset)
}

func (w *exportWriter) posV1(pos token.Pos) {
	if w.p.fset == nil {
		w.int64(0)
		return
	}

	p := w.p.fset.Position(pos)
	file := p.Filename
	line := int64(p.Line)
	column := int64(p.Column)

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

func (w *exportWriter) posV0(pos token.Pos) {
	if w.p.fset == nil {
		w.int64(0)
		return
	}

	p := w.p.fset.Position(pos)
	file := p.Filename
	line := int64(p.Line)

	// When file is the same as the last position (common case),
	// we can save a few bytes by delta encoding just the line
	// number.
	//
	// Note: Because data objects may be read out of order (or not
	// at all), we can only apply delta encoding within a single
	// object. This is handled implicitly by tracking prevFile and
	// prevLine as fields of exportWriter.

	if file == w.prevFile {
		delta := line - w.prevLine
		w.int64(delta)
		if delta == deltaNewFile {
			w.int64(-1)
		}
	} else {
		w.int64(deltaNewFile)
		w.int64(line) // line >= 0
		w.string(file)
		w.prevFile = file
	}
	w.prevLine = line
}

func (w *exportWriter) pkg(pkg *types.Package) {
	if pkg == nil {
		// [exportWriter.typ] accepts a nil pkg only for types
		// of constants, which cannot contain named objects
		// such as fields or methods and thus should never
		// reach this method (#76222).
		panic("nil package")
	}
	// Ensure any referenced packages are declared in the main index.
	w.p.allPkgs[pkg] = true

	w.string(w.exportPath(pkg))
}

func (w *exportWriter) qualifiedType(obj *types.TypeName) {
	name := w.p.exportName(obj)

	// Ensure any referenced declarations are written out too.
	w.p.pushDecl(obj)
	w.string(name)
	w.pkg(obj.Pkg())
}

// typ emits the specified type.
//
// Objects within the type (struct fields and interface methods) are
// qualified by pkg. It may be nil if the type cannot contain objects,
// such as the type of a constant.
func (w *exportWriter) typ(t types.Type, pkg *types.Package) {
	w.data.uint64(w.p.typOff(t, pkg))
}

func (p *iexporter) newWriter() *exportWriter {
	return &exportWriter{p: p}
}

func (w *exportWriter) flush() uint64 {
	off := uint64(w.p.data0.Len())
	io.Copy(&w.p.data0, &w.data)
	return off
}

func (p *iexporter) typOff(t types.Type, pkg *types.Package) uint64 {
	off, ok := p.typIndex[t]
	if !ok {
		w := p.newWriter()
		w.doTyp(t, pkg)
		off = predeclReserved + w.flush()
		p.typIndex[t] = off
	}
	return off
}

func (w *exportWriter) startType(k itag) {
	w.data.uint64(uint64(k))
}

// doTyp is the implementation of [exportWriter.typ].
func (w *exportWriter) doTyp(t types.Type, pkg *types.Package) {
	if trace {
		w.p.trace("exporting type %s (%T)", t, t)
		w.p.indent++
		defer func() {
			w.p.indent--
			w.p.trace("=> %s", t)
		}()
	}
	switch t := t.(type) {
	case *types.Alias:
		if targs := t.TypeArgs(); targs.Len() > 0 {
			w.startType(instanceType)
			w.pos(t.Obj().Pos())
			w.typeList(targs, pkg)
			w.typ(t.Origin(), pkg)
			return
		}
		w.startType(aliasType)
		w.qualifiedType(t.Obj())

	case *types.Named:
		if targs := t.TypeArgs(); targs.Len() > 0 {
			w.startType(instanceType)
			// TODO(rfindley): investigate if this position is correct, and if it
			// matters.
			w.pos(t.Obj().Pos())
			w.typeList(targs, pkg)
			w.typ(t.Origin(), pkg)
			return
		}
		w.startType(definedType)
		w.qualifiedType(t.Obj())

	case *types.TypeParam:
		w.startType(typeParamType)
		w.qualifiedType(t.Obj())

	case *types.Pointer:
		w.startType(pointerType)
		w.typ(t.Elem(), pkg)

	case *types.Slice:
		w.startType(sliceType)
		w.typ(t.Elem(), pkg)

	case *types.Array:
		w.startType(arrayType)
		w.uint64(uint64(t.Len()))
		w.typ(t.Elem(), pkg)

	case *types.Chan:
		w.startType(chanType)
		// 1 RecvOnly; 2 SendOnly; 3 SendRecv
		var dir uint64
		switch t.Dir() {
		case types.RecvOnly:
			dir = 1
		case types.SendOnly:
			dir = 2
		case types.SendRecv:
			dir = 3
		}
		w.uint64(dir)
		w.typ(t.Elem(), pkg)

	case *types.Map:
		w.startType(mapType)
		w.typ(t.Key(), pkg)
		w.typ(t.Elem(), pkg)

	case *types.Signature:
		w.startType(signatureType)
		w.pkg(pkg) // qualifies param/result vars
		w.signature(t)

	case *types.Struct:
		w.startType(structType)
		n := t.NumFields()
		// Even for struct{} we must emit some qualifying package, because that's
		// what the compiler does, and thus that's what the importer expects.
		fieldPkg := pkg
		if n > 0 {
			fieldPkg = t.Field(0).Pkg()
		}
		if fieldPkg == nil {
			// TODO(rfindley): improve this very hacky logic.
			//
			// The importer expects a package to be set for all struct types, even
			// those with no fields. A better encoding might be to set NumFields
			// before pkg. setPkg panics with a nil package, which may be possible
			// to reach with invalid packages (and perhaps valid packages, too?), so
			// (arbitrarily) set the localpkg if available.
			//
			// Alternatively, we may be able to simply guarantee that pkg != nil, by
			// reconsidering the encoding of constant values.
			if w.p.shallow {
				fieldPkg = w.p.localpkg
			} else {
				panic(internalErrorf("no package to set for empty struct"))
			}
		}
		w.pkg(fieldPkg)
		w.uint64(uint64(n))

		for i := range n {
			f := t.Field(i)
			if w.p.shallow {
				w.objectPath(f)
			}
			w.pos(f.Pos())
			w.string(f.Name()) // unexported fields implicitly qualified by prior setPkg
			w.typ(f.Type(), fieldPkg)
			w.bool(f.Anonymous())
			w.string(t.Tag(i)) // note (or tag)
		}

	case *types.Interface:
		w.startType(interfaceType)
		w.pkg(pkg) // qualifies unexported method funcs

		n := t.NumEmbeddeds()
		w.uint64(uint64(n))
		for i := 0; i < n; i++ {
			ft := t.EmbeddedType(i)
			if named, _ := types.Unalias(ft).(*types.Named); named != nil {
				w.pos(named.Obj().Pos())
			} else {
				// e.g. ~int
				w.pos(token.NoPos)
			}
			w.typ(ft, pkg)
		}

		// See comment for struct fields. In shallow mode we change the encoding
		// for interface methods that are promoted from other packages.

		n = t.NumExplicitMethods()
		w.uint64(uint64(n))
		for i := 0; i < n; i++ {
			m := t.ExplicitMethod(i)
			if w.p.shallow {
				w.objectPath(m)
			}
			w.pos(m.Pos())
			w.string(m.Name())
			sig, _ := m.Type().(*types.Signature)
			w.signature(sig)
		}

	case *types.Union:
		w.startType(unionType)
		nt := t.Len()
		w.uint64(uint64(nt))
		for i := range nt {
			term := t.Term(i)
			w.bool(term.Tilde())
			w.typ(term.Type(), pkg)
		}

	default:
		panic(internalErrorf("unexpected type: %v, %v", t, reflect.TypeOf(t)))
	}
}

// objectPath writes the package and objectPath to use to look up obj in a
// different package, when encoding in "shallow" mode.
//
// When doing a shallow import, the importer creates only the local package,
// and requests package symbols for dependencies from the client.
// However, certain types defined in the local package may hold objects defined
// (perhaps deeply) within another package.
//
// For example, consider the following:
//
//	package a
//	func F() chan * map[string] struct { X int }
//
//	package b
//	import "a"
//	var B = a.F()
//
// In this example, the type of b.B holds fields defined in package a.
// In order to have the correct canonical objects for the field defined in the
// type of B, they are encoded as objectPaths and later looked up in the
// importer. The same problem applies to interface methods.
func (w *exportWriter) objectPath(obj types.Object) {
	if obj.Pkg() == nil || obj.Pkg() == w.p.localpkg {
		// obj.Pkg() may be nil for the builtin error.Error.
		// In this case, or if obj is declared in the local package, no need to
		// encode.
		w.string("")
		return
	}
	objectPath, err := w.p.objectpathEncoder().For(obj)
	if err != nil {
		// Fall back to the empty string, which will cause the importer to create a
		// new object, which matches earlier behavior. Creating a new object is
		// sufficient for many purposes (such as type checking), but causes certain
		// references algorithms to fail (golang/go#60819). However, we didn't
		// notice this problem during months of gopls@v0.12.0 testing.
		//
		// TODO(golang/go#61674): this workaround is insufficient, as in the case
		// where the field forwarded from an instantiated type that may not appear
		// in the export data of the original package:
		//
		//  // package a
		//  type A[P any] struct{ F P }
		//
		//  // package b
		//  type B a.A[int]
		//
		// We need to update references algorithms not to depend on this
		// de-duplication, at which point we may want to simply remove the
		// workaround here.
		w.string("")
		return
	}
	w.string(string(objectPath))
	w.pkg(obj.Pkg())
}

func (w *exportWriter) signature(sig *types.Signature) {
	w.paramList(sig.Params())
	w.paramList(sig.Results())
	if sig.Params().Len() > 0 {
		w.bool(sig.Variadic())
	}
}

func (w *exportWriter) typeList(ts *types.TypeList, pkg *types.Package) {
	w.uint64(uint64(ts.Len()))
	for t := range ts.Types() {
		w.typ(t, pkg)
	}
}

func (w *exportWriter) tparamList(prefix string, list *types.TypeParamList, pkg *types.Package) {
	ll := uint64(list.Len())
	w.uint64(ll)
	for tparam := range list.TypeParams() {
		// Set the type parameter exportName before exporting its type.
		exportName := tparamExportName(prefix, tparam)
		w.p.tparamNames[tparam.Obj()] = exportName
		w.typ(tparam, pkg)
	}
}

const blankMarker = "$"

// tparamExportName returns the 'exported' name of a type parameter, which
// differs from its actual object name: it is prefixed with a qualifier, and
// blank type parameter names are disambiguated by their index in the type
// parameter list.
func tparamExportName(prefix string, tparam *types.TypeParam) string {
	assert(prefix != "")
	name := tparam.Obj().Name()
	if name == "_" {
		name = blankMarker + strconv.Itoa(tparam.Index())
	}
	return prefix + "." + name
}

// tparamName returns the real name of a type parameter, after stripping its
// qualifying prefix and reverting blank-name encoding. See tparamExportName
// for details.
func tparamName(exportName string) string {
	// Remove the "path" from the type param name that makes it unique.
	ix := strings.LastIndex(exportName, ".")
	if ix < 0 {
		errorf("malformed type parameter export name %s: missing prefix", exportName)
	}
	name := exportName[ix+1:]
	if strings.HasPrefix(name, blankMarker) {
		return "_"
	}
	return name
}

func (w *exportWriter) paramList(tup *types.Tuple) {
	n := tup.Len()
	w.uint64(uint64(n))
	for i := range n {
		w.param(tup.At(i))
	}
}

func (w *exportWriter) param(obj types.Object) {
	w.pos(obj.Pos())
	w.localIdent(obj)
	w.typ(obj.Type(), obj.Pkg())
}

func (w *exportWriter) value(typ types.Type, v constant.Value) {
	w.typ(typ, nil)
	if w.p.version >= iexportVersionGo1_18 {
		w.int64(int64(v.Kind()))
	}

	if v.Kind() == constant.Unknown {
		// golang/go#60605: treat unknown constant values as if they have invalid type
		//
		// This loses some fidelity over the package type-checked from source, but that
		// is acceptable.
		//
		// TODO(rfindley): we should switch on the recorded constant kind rather
		// than the constant type
		return
	}

	switch b := typ.Underlying().(*types.Basic); b.Info() & types.IsConstType {
	case types.IsBoolean:
		w.bool(constant.BoolVal(v))
	case types.IsInteger:
		var i big.Int
		if i64, exact := constant.Int64Val(v); exact {
			i.SetInt64(i64)
		} else if ui64, exact := constant.Uint64Val(v); exact {
			i.SetUint64(ui64)
		} else {
			i.SetString(v.ExactString(), 10)
		}
		w.mpint(&i, typ)
	case types.IsFloat:
		f := constantToFloat(v)
		w.mpfloat(f, typ)
	case types.IsComplex:
		w.mpfloat(constantToFloat(constant.Real(v)), typ)
		w.mpfloat(constantToFloat(constant.Imag(v)), typ)
	case types.IsString:
		w.string(constant.StringVal(v))
	default:
		if b.Kind() == types.Invalid {
			// package contains type errors
			break
		}
		panic(internalErrorf("unexpected type %v (%v)", typ, typ.Underlying()))
	}
}

// constantToFloat converts a constant.Value with kind constant.Float to a
// big.Float.
func constantToFloat(x constant.Value) *big.Float {
	x = constant.ToFloat(x)
	// Use the same floating-point precision (512) as cmd/compile
	// (see Mpprec in cmd/compile/internal/gc/mpfloat.go).
	const mpprec = 512
	var f big.Float
	f.SetPrec(mpprec)
	if v, exact := constant.Float64Val(x); exact {
		// float64
		f.SetFloat64(v)
	} else if num, denom := constant.Num(x), constant.Denom(x); num.Kind() == constant.Int {
		// TODO(gri): add big.Rat accessor to constant.Value.
		n := valueToRat(num)
		d := valueToRat(denom)
		f.SetRat(n.Quo(n, d))
	} else {
		// Value too large to represent as a fraction => inaccessible.
		// TODO(gri): add big.Float accessor to constant.Value.
		_, ok := f.SetString(x.ExactString())
		assert(ok)
	}
	return &f
}

func valueToRat(x constant.Value) *big.Rat {
	// Convert little-endian to big-endian.
	// I can't believe this is necessary.
	bytes := constant.Bytes(x)
	for i := 0; i < len(bytes)/2; i++ {
		bytes[i], bytes[len(bytes)-1-i] = bytes[len(bytes)-1-i], bytes[i]
	}
	return new(big.Rat).SetInt(new(big.Int).SetBytes(bytes))
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
//
// TODO(mdempsky): Is this level of complexity really worthwhile?
func (w *exportWriter) mpint(x *big.Int, typ types.Type) {
	basic, ok := typ.Underlying().(*types.Basic)
	if !ok {
		panic(internalErrorf("unexpected type %v (%T)", typ.Underlying(), typ.Underlying()))
	}

	signed, maxBytes := intSize(basic)

	negative := x.Sign() < 0
	if !signed && negative {
		panic(internalErrorf("negative unsigned integer; type %v, value %v", typ, x))
	}

	b := x.Bytes()
	if len(b) > 0 && b[0] == 0 {
		panic(internalErrorf("leading zeros"))
	}
	if uint(len(b)) > maxBytes {
		panic(internalErrorf("bad mpint length: %d > %d (type %v, value %v)", len(b), maxBytes, typ, x))
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
		panic(internalErrorf("encoding mistake: %d, %v, %v => %d", len(b), signed, negative, n))
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
func (w *exportWriter) mpfloat(f *big.Float, typ types.Type) {
	if f.IsInf() {
		panic("infinite constant")
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
		panic(internalErrorf("mantissa scaling failed for %f (%s)", f, acc))
	}
	w.mpint(manti, typ)
	if manti.Sign() != 0 {
		w.int64(exp)
	}
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

func (w *exportWriter) localIdent(obj types.Object) {
	// Anonymous parameters.
	if obj == nil {
		w.string("")
		return
	}

	name := obj.Name()
	if name == "_" {
		w.string("_")
		return
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

func assert(cond bool) {
	if !cond {
		panic("internal error: assertion failed")
	}
}

// The below is copied from go/src/cmd/compile/internal/gc/syntax.go.

// objQueue is a FIFO queue of types.Object. The zero value of objQueue is
// a ready-to-use empty queue.
type objQueue struct {
	ring       []types.Object
	head, tail int
}

// empty returns true if q contains no Nodes.
func (q *objQueue) empty() bool {
	return q.head == q.tail
}

// pushTail appends n to the tail of the queue.
func (q *objQueue) pushTail(obj types.Object) {
	if len(q.ring) == 0 {
		q.ring = make([]types.Object, 16)
	} else if q.head+len(q.ring) == q.tail {
		// Grow the ring.
		nring := make([]types.Object, len(q.ring)*2)
		// Copy the old elements.
		part := q.ring[q.head%len(q.ring):]
		if q.tail-q.head <= len(part) {
			part = part[:q.tail-q.head]
			copy(nring, part)
		} else {
			pos := copy(nring, part)
			copy(nring[pos:], q.ring[:q.tail%len(q.ring)])
		}
		q.ring, q.head, q.tail = nring, 0, q.tail-q.head
	}

	q.ring[q.tail%len(q.ring)] = obj
	q.tail++
}

// popHead pops a node from the head of the queue. It panics if q is empty.
func (q *objQueue) popHead() types.Object {
	if q.empty() {
		panic("dequeue empty")
	}
	obj := q.ring[q.head%len(q.ring)]
	q.head++
	return obj
}

// internalError represents an error generated inside this package.
type internalError string

func (e internalError) Error() string { return "gcimporter: " + string(e) }

// TODO(adonovan): make this call panic, so that it's symmetric with errorf.
// Otherwise it's easy to forget to do anything with the error.
//
// TODO(adonovan): also, consider switching the names "errorf" and
// "internalErrorf" as the former is used for bugs, whose cause is
// internal inconsistency, whereas the latter is used for ordinary
// situations like bad input, whose cause is external.
func internalErrorf(format string, args ...any) error {
	return internalError(fmt.Sprintf(format, args...))
}
