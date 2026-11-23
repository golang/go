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
//         Tag        byte    // 'i' = "indexed" format
//         Version    uvarint
//         Shallow    bool    // since iexportVersionShallowFlag
//         StringSize uvarint
//         FileSize   uvarint // since iexportVersionFullPos
//         DataSize   uvarint
//     }
//
//     Strings [StringSize]byte
//     Files   [FileSize]byte // since iexportVersionFullPos; see File below
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
//     FileIndex []fileOff // since iexportVersionFullPos
//
// uvarint (or varint) means a uint64 (or int64) written out using
// uvarint (or varint) encoding; bool means a uvarint 0 or 1.
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
// fileOff means a uvarint that indicates the offset within the Files
// section of a File record (see below).
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
// There are several kinds of type descriptors, distinguished by an itag:
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
//     // Reference to a struct or interface literal of another
//     // package (shallow mode only; see typepath.go).
//     type ExternalType struct {
//         Tag  itag // externalType
//         Ref  LiteralRef
//     }
//
//     type LiteralRef struct {
//         PkgPath stringOff
//         Root    stringOff // name of exported package-level object
//         Steps   []uvarint // component indices; see typePath
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
// Since iexportVersionFullPos, position encoding is the same in deep
// and shallow mode. A Pos encodes either NoPos or a (file, offset)
// pair, where file is the unsigned index into FileIndex, and offset
// is the unsigned byte offset. To exploit locality we delta-encode
// adjacent offsets within a declaration or type descriptor, with
// initial state (0, 0). Each Pos record has one of three forms:
//
//     none
//     absolute   (file, offset uint)
//     relative   (Δoffset int)
//
// represented as follows:
//
//     type Pos struct {
//         X      varint
//         Offset uvarint // present iff X is positive odd
//     }
//
//     X == -1        => none     (NoPos, no state change)
//     X even         => relative (same file, offset += X>>1)
//     X positive odd => absolute (file = X>>1, offset = Offset)
//
// Negative odd X values are invalid.
// (Both 'none' and zero 'relative' records are rare in practice.
//
// Each element FileIndex[i] gives the offset within Files of the ith
// File record:
//
//     type File struct {
//         Name      stringOff
//         Size      uvarint
//         LineCount uvarint
//         Lines     []struct {
//             DeltaIndex     uvarint
//             ResidualOffset varint
//         }
//         Infos     []struct {
//             DeltaOffset uvarint
//             Filename    stringOff
//             Line        uvarint
//             Column      uvarint
//         }
//     }
//
// Lines is a sparse table of the start offsets of (only) the lines
// that contain a position mentioned in the data, in increasing order.
// Each element encodes a (0-based) line index and its start offset
// relative to the previous element (initially (0, 0)): DeltaIndex is
// index - prev.index, and ResidualOffset is the residual of offset
// relative to the prediction prev.offset + DeltaIndex*avg, where
// avg = Size / max(1, LineCount) is the mean line length.
// See [writeLines].
//
// Infos is a synthesized list of //line directive information, in the
// form of arguments to [token.File.AddLineColumnInfo] (Column 0 means
// unknown), with offsets delta-encoded relative to the previous entry
// (initially 0). The entries do not generally correspond to the
// original directives, which may lie on unrecorded lines, but cause
// the decoder's token.File to compute the same adjusted position as
// the original for every mentioned position. See [writeLineInfos].
//
// See [exportWriter.posV3] and [iexporter.encodeFile] for the implementation.
//
// In earlier versions, deep mode used [exportWriter.posV1], in which
// Pos encodes a file:line:column triple, incorporating a simple delta
// encoding scheme within a data object. (Earlier shallow data,
// which has no compatibility guarantee, used an encoding that is no
// longer supported.)

package exportdata

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
	"io"
	"math"
	"math/big"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
	// This package is dependency-restricted; see x/tools/go/gcexportdata.TestDeps.
)

// iexportCommon encodes export data (without the leading 'i' tag byte).
// In principle this operation can only fail if out.Write fails,
// but that's impossible for bytes.Buffer---and as a matter of
// fact iexportCommon doesn't even check for I/O errors.
// TODO(adonovan): handle I/O errors properly.
func iexportCommon(out io.Writer, fset *token.FileSet, shallow bool, version int, pkg *types.Package) (err error) {
	if !debug {
		defer func() {
			if e := recover(); e != nil {
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
		localpkg:    pkg,
	}

	for i, pt := range predeclared() {
		p.typIndex[pt] = uint64(i)
	}
	if len(p.typIndex) > predeclReserved {
		panic(internalErrorf("too many predeclared types: %d > %d", len(p.typIndex), predeclReserved))
	}

	// Initialize work queue with exported declarations.
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		if token.IsExported(name) {
			p.pushDecl(scope.Lookup(name))
		}
	}

	// Loop until no more work.
	for !p.declTodo.empty() {
		p.doDecl(p.declTodo.popHead())
	}

	// Encode the table of files referenced by posV3 positions.
	// (p.files is empty in earlier versions.)
	var files intWriter
	fileOffset := make([]uint64, len(p.files)) // fileOffset[i] is offset in files of p.files[i]
	for i, fp := range p.files {
		fileOffset[i] = uint64(files.Len())
		p.encodeFile(&files, fp.file, fp.needed)
	}

	// Append indices to data0 section.
	dataLen := uint64(p.data0.Len())
	w := p.newWriter()
	w.writeIndex(p.declIndex)
	if p.version >= iexportVersionFullPos {
		w.uint64(uint64(len(fileOffset)))
		for _, offset := range fileOffset {
			w.uint64(offset)
		}
	}
	w.flush()

	// Assemble header.
	var hdr intWriter
	hdr.uint64(uint64(p.version))
	if p.version >= iexportVersionShallowFlag {
		hdr.bool(shallow)
	}
	hdr.uint64(uint64(p.strings.Len()))
	if p.version >= iexportVersionFullPos || p.shallow {
		hdr.uint64(uint64(files.Len()))
	}
	hdr.uint64(dataLen)

	// Flush output.
	if buf, ok := out.(*bytes.Buffer); ok {
		buf.Grow(hdr.Len() + p.strings.Len() + files.Len() + p.data0.Len())
	}
	io.Copy(out, &hdr)
	io.Copy(out, &p.strings)
	if p.version >= iexportVersionFullPos {
		io.Copy(out, &files)
	}
	io.Copy(out, &p.data0)

	return nil
}

// encodeFile writes to w a File record sufficient to faithfully
// restore position information (both raw and //line-adjusted) for all
// needed offsets in file.
// Sorts and compacts the underlying array of needed.
func (p *iexporter) encodeFile(w *intWriter, file *token.File, offsets []uint32) {
	_ = offsets[0] // precondition: some offsets are needed

	// Sort the set of needed offsets and remove duplicates.
	slices.Sort(offsets)
	offsets = slices.Compact(offsets)

	size := file.Size()
	allLines := file.Lines()

	w.uint64(p.stringOff(file.Name()))
	w.uint64(uint64(size))
	w.uint64(uint64(len(allLines)))

	avg := size / max(1, len(allLines)) // be defensive of synthetic positions in empty files
	lines := sparseLines(allLines, offsets)
	writeLines(w, avg, lines)
	writeLineInfos(p, w, file, offsets, lines)
}

// A lineStart records the byte offset of the start of a (0-based) line.
type lineStart struct{ index, offset int }

// sparseLines returns the start offsets of only those lines in allLines
// that contain at least one of the sorted needed offsets.
func sparseLines(allLines []int, needed []uint32) []lineStart {
	if len(allLines) == 0 {
		return nil
	}
	var sparse []lineStart
	from := 0 // index in allLines of the first line after sparse's last entry
	for _, off := range needed {
		o := int(off)
		if from > 0 && o < allLines[from] {
			continue // same line as previous offset
		}
		// Find the last line in allLines[from:] starting at or before o.
		i, exact := slices.BinarySearch(allLines[from:], o)
		if !exact {
			i--
		}
		if i < 0 {
			continue // o precedes allLines[0]
		}
		index := from + i
		sparse = append(sparse, lineStart{index: index, offset: allLines[index]})
		from = index + 1
		if from == len(allLines) {
			break // remaining offsets are all on the last line
		}
	}
	return sparse
}

// writeLines writes a sparse list of line starts, in increasing
// order, as (index, offset) pairs delta-encoded relative to the
// previous pair (initially (0, 0)). Each offset is encoded as the
// residual relative to the prediction that lines have length avg.
// See [readLines].
func writeLines(w *intWriter, avg int, lines []lineStart) {
	w.uint64(uint64(len(lines)))
	var prev lineStart
	for _, line := range lines {
		w.uint64(uint64(line.index - prev.index))
		w.int64(int64(line.offset - predictLineStart(prev, line.index, avg)))
		prev = line
	}
}

// predictLineStart predicts the start offset of the line with the
// specified (0-based) index, given the previous line start and the
// mean line length.
func predictLineStart(prev lineStart, index, avg int) int {
	return prev.offset + (index-prev.index)*avg
}

// mayHaveLineInfos reports whether file may have alternative
// position information (from //line directives). If it reports
// false, adjusted and unadjusted positions are equal throughout file.
//
// The go/token API provides no way to ask this question, so we use
// reflection to inspect the unexported field token.File.infos, if
// present; otherwise we conservatively report true. This avoids the
// cost of computing the adjusted position of every needed offset,
// which is substantial (+15% deep encoding time for std).
// (FileSet.Write would expose the infos without the hack, but it
// clones every line table: +8% time and +35% allocation.)
//
// The read of infos is not synchronized by token.File's mutex, but
// infos is appended to only during parsing, which is complete
// before export.
//
// TODO(adonovan): when this code moves to std, use go/token internals.
func mayHaveLineInfos(file *token.File) bool {
	if fileInfosField < 0 {
		return true // unexpected layout
	}
	return reflect.ValueOf(file).Elem().Field(fileInfosField).Len() > 0
}

// fileInfosField is the index of the field token.File.infos,
// or -1 if token.File has no such slice field.
// (Reflection is memory-safe, so there is no need to check
// the rest of the layout.)
var fileInfosField = func() int {
	f, ok := reflect.TypeFor[token.File]().FieldByName("infos")
	if !ok || f.Type.Kind() != reflect.Slice {
		return -1
	}
	return f.Index[0]
}()

// A lineInfo represents one synthesized [token.File.AddLineColumnInfo] entry:
// at unadjusted line rawLine (and byte offset alt.Offset), the adjusted
// position is alt.
type lineInfo struct {
	rawLine int
	alt     token.Position
}

// writeLineInfos synthesizes and writes the //line directive information
// (alternative positions; see [token.File.AddLineColumnInfo]) needed to
// reconstruct the adjusted position of each needed offset.
//
// Most files have no //line directives, and cost a single zero byte.
//
// The go/token API provides no way to enumerate a file's line infos;
// moreover, an actual directive's offset may lie on a line omitted from
// the sparse line table, which would skew line deltas in the decoder.
// Instead, we simulate the decoder's computation of the adjusted
// position at each needed offset, and emit an entry (at that offset)
// wherever the simulation disagrees with the actual adjusted position.
// Because entries are located only at needed offsets, whose lines are
// faithfully recorded by the sparse line table, the decoder's
// token.File computes the exact same adjusted positions.
func writeLineInfos(p *iexporter, w *intWriter, file *token.File, needed []uint32, lines []lineStart) {
	if len(lines) == 0 || !mayHaveLineInfos(file) {
		w.uint64(0)
		return
	}

	var (
		infos   []lineInfo
		prev    *lineInfo
		lineIdx = 0
		size    = file.Size()
		name    = file.Name()
	)
	for _, off := range needed {
		o := int(off)
		if o < lines[0].offset {
			continue // precedes first line
		}
		if o >= size {
			// Every object's Pos is at an identifier, so o < size in
			// valid Go. An EOF position (possible after a syntax error)
			// cannot be passed to AddLineColumnInfo, which requires o < size.
			break
		}
		for lineIdx+1 < len(lines) && o >= lines[lineIdx+1].offset {
			lineIdx++
		}
		line := lines[lineIdx]
		raw := token.Position{
			Filename: name,
			Offset:   o,
			Line:     line.index + 1,
			Column:   o - line.offset + 1,
		}
		want := file.PositionFor(file.Pos(o), true)
		if adjust(prev, raw) != want {
			infos = append(infos, lineInfo{rawLine: raw.Line, alt: want})
			prev = &infos[len(infos)-1]
		}
	}

	w.uint64(uint64(len(infos)))
	prevOffset := 0
	for _, info := range infos {
		w.uint64(uint64(info.alt.Offset - prevOffset))
		w.uint64(p.stringOff(info.alt.Filename))
		w.uint64(uint64(info.alt.Line))
		w.uint64(uint64(info.alt.Column))
		prevOffset = info.alt.Offset
	}
}

// adjust returns the adjusted position that the decoder's token.File
// will compute for an offset whose unadjusted position is raw, given
// that info (if non-nil) is the closest preceding lineInfo.
//
// It must agree with the private (*token.File).unpack method in
// $GOROOT/src/go/token/position.go; this is checked by TestPositions.
func adjust(info *lineInfo, raw token.Position) token.Position {
	if info == nil {
		return raw // no line info: adjusted == unadjusted
	}
	d := raw.Line - info.rawLine // lines since info
	pos := token.Position{
		Filename: info.alt.Filename,
		Offset:   raw.Offset,
		Line:     info.alt.Line + d,
		Column:   raw.Column,
	}
	switch {
	case info.alt.Column == 0:
		pos.Column = 0 // column unknown until next directive
	case d == 0:
		pos.Column = info.alt.Column + (raw.Offset - info.alt.Offset) // same line as info
	}
	return pos
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

	shallow     bool                                       // don't encode types from other packages; refer to them by name (or typePath)
	typeIndexes map[*types.Package]map[types.Type]typePath // paths to literals of other packages in shallow mode; lazily populated
	localpkg    *types.Package

	// allPkgs tracks all packages that have been referenced by
	// the export data, so we can ensure to include them in the
	// main index.
	allPkgs map[*types.Package]bool

	declTodo objQueue

	strings     intWriter
	stringIndex map[string]uint64

	// Since iexportVersionFullPos, object positions are encoded as
	// (file, offset) pairs referencing a table of files. Each file
	// records sparse line starts and //line info for needed offsets.
	files     []*filePositions
	fileIndex map[*token.File]*filePositions
	lastFile  *filePositions // single-entry cache for consecutive positions

	data0       intWriter
	declIndex   map[types.Object]uint64
	tparamNames map[types.Object]string // typeparam->exported name
	typIndex    map[types.Type]uint64

	indent int // for tracing support
}

type filePositions struct {
	file   *token.File
	index  int      // index in iexporter.files
	needed []uint32 // unordered list of needed file offsets (32 bits suffice; see fileIndexAndOffset)
}

func (p *iexporter) trace(format string, args ...any) {
	if !trace {
		// Call sites should also be guarded, but having this check here allows
		// easily enabling/disabling debug trace statements.
		return
	}
	fmt.Printf(strings.Repeat("..", p.indent)+format+"\n", args...)
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

// fileIndexAndOffset resolves pos to a file index in p.files and a byte
// offset within that file, recording the offset as needed.
// It returns (-1, 0) if pos is invalid or not found in p.fset.
func (p *iexporter) fileIndexAndOffset(pos token.Pos) (int, int64) {
	if !pos.IsValid() {
		return -1, 0
	}

	// Consecutive positions are usually in the same file.
	fp := p.lastFile
	if fp == nil || pos < token.Pos(fp.file.Base()) || pos > fp.file.End() {
		file := p.fset.File(pos)
		if file == nil {
			return -1, 0
		}
		fp = p.fileIndex[file]
		if fp == nil {
			// Offsets are recorded in 32 bits, to save memory.
			if uint64(file.Size()) > math.MaxUint32 {
				panic(internalErrorf("file %s is too large (%d bytes)", file.Name(), file.Size()))
			}
			fp = &filePositions{file: file, index: len(p.files)}
			p.files = append(p.files, fp)
			if p.fileIndex == nil {
				p.fileIndex = make(map[*token.File]*filePositions)
			}
			p.fileIndex[file] = fp
		}
		p.lastFile = fp
	}

	offset := uint32(fp.file.Offset(pos))
	if n := len(fp.needed); n == 0 || fp.needed[n-1] != offset {
		fp.needed = append(fp.needed, offset)
	}
	return fp.index, int64(offset)
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
	p    *iexporter
	data intWriter

	// posV0/posV1 state
	prevFile   string
	prevLine   int64
	prevColumn int64

	// posV3 state (initially file 0, offset 0)
	prevFileIdx int
	prevOffset  int64
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
		sig := obj.Type().(*types.Signature)
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
		w.value(obj.Type(), obj.Val(), obj.Pkg())

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
			sig := m.Type().(*types.Signature)
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
	if w.p.version >= iexportVersionFullPos {
		w.posV3(pos)
	} else if w.p.version >= iexportVersionPosCol {
		w.posV1(pos)
	} else {
		w.posV0(pos)
	}
}

// posV3 encodes a position (used in both modes since
// iexportVersionFullPos) as a (file, offset) pair, delta-encoded
// within a declaration or type descriptor. See the description of Pos
// in the format comment at the top of this file, and
// [importReader.posV3] for the decoder.
func (w *exportWriter) posV3(pos token.Pos) {
	file, offset := w.p.fileIndexAndOffset(pos)
	switch {
	case file < 0: // NoPos
		w.int64(-1)
	case file == w.prevFileIdx: // same file
		w.int64((offset - w.prevOffset) << 1)
		w.prevOffset = offset
	default: // new file
		w.int64(int64(file)<<1 | 1)
		w.uint64(uint64(offset))
		w.prevFileIdx = file
		w.prevOffset = offset
	}
}

func (w *exportWriter) posV1(pos token.Pos) {
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
// qualified by pkg.
func (w *exportWriter) typ(t types.Type, pkg *types.Package) {
	if pkg == nil {
		pkg = w.p.localpkg
	}
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
		// In shallow mode, a literal of another package is
		// encoded as a reference to it (see typepath.go).
		if w.p.shallow && w.externalLiteral(t) {
			return
		}

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
			w.pos(f.Pos())
			w.string(f.Name()) // unexported fields implicitly qualified by prior setPkg
			w.typ(f.Type(), fieldPkg)
			w.bool(f.Anonymous())
			w.string(t.Tag(i)) // note (or tag)
		}

	case *types.Interface:
		// (See comment for case *types.Struct.)
		if w.p.shallow && w.externalLiteral(t) {
			return
		}

		w.startType(interfaceType)
		methodPkg := pkg
		for i := range t.NumExplicitMethods() {
			if mp := t.ExplicitMethod(i).Pkg(); mp != nil {
				methodPkg = mp
				break
			}
		}
		w.pkg(methodPkg) // qualifies unexported method funcs

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

		n = t.NumExplicitMethods()
		w.uint64(uint64(n))
		for i := 0; i < n; i++ {
			m := t.ExplicitMethod(i)
			w.pos(m.Pos())
			w.string(m.Name())
			sig := m.Type().(*types.Signature)
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

// externalLiteral reports whether t, a struct or interface literal,
// should be encoded in shallow mode as a reference to a literal of
// another package, and if so, writes the reference.
//
// If t has members (fields or explicit methods) declared in a package
// other than the local one, and t is reachable from the API of that
// package, externalLiteral writes a reference to it (see typepath.go),
// so that the importer uses the canonical literal and members. It
// reports an internal error if t cannot be reached: the importer will
// create fresh member objects, which is a bug unless the package is
// ill-typed.
func (w *exportWriter) externalLiteral(t types.Type) bool {
	p := w.p

	// TODO(golang/go#61674): a literal created by instantiation (such
	// as the underlying type of B in "type B a.A[int]") is not
	// reachable from the API of a, so it is encoded in full, and the
	// importer creates fresh member objects: even members that the
	// instance shares with its origin, whose types do not depend on
	// type parameters, lose their identity, and the fresh members do
	// not record their origin. We plan to fix this once the real
	// implementation of this codec lives in the standard library,
	// where the importer can set Var.Origin and Func.Origin.
	var pkg *types.Package // package of first member
	for m := range literalMembers(t) {
		if pkg == nil {
			pkg = m.Pkg()
			if pkg == nil || pkg == p.localpkg {
				return false // local literal (or ill-typed)
			}
		}
		if origin(m) != m {
			return false // instance
		}
	}
	if pkg == nil {
		return false // no member objects whose identity must be preserved
	}

	// Memoize each package's newTypeIndex traversal.
	ix := p.typeIndexes[pkg]
	if ix == nil {
		ix = newTypeIndex(pkg)
		if p.typeIndexes == nil {
			p.typeIndexes = make(map[*types.Package]map[types.Type]typePath)
		}
		p.typeIndexes[pkg] = ix
	}
	path, ok := ix[t]
	if !ok {
		return false // e.g. literal created by instantiation (golang/go#61674)
	}

	w.startType(externalType)
	w.pkg(pkg)
	w.string(path.root)
	w.uint64(uint64(len(path.steps)))
	for _, i := range path.steps {
		w.uint64(uint64(i))
	}
	return true
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

// only called for constants
func (w *exportWriter) value(typ types.Type, v constant.Value, pkg *types.Package) {
	if pkg == nil {
		pkg = w.p.localpkg
	}
	w.typ(typ, pkg)
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
		w.float(v, typ)
	case types.IsComplex:
		w.float(constant.Real(v), typ)
		w.float(constant.Imag(v), typ)
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

// float exports a floating-point constant value.
func (w *exportWriter) float(x constant.Value, typ types.Type) {
	if w.p.version < iexportVersionRationalFloats {
		w.mpfloat(constantToFloat(x), typ) // 512-bit mantissa f.p.
		return
	}

	// lossless encoding
	switch v := constant.Val(constant.ToFloat(x)).(type) {
	case *big.Rat:
		if n := v.Num(); v.IsInt() && n.BitLen() <= 8*untypedMaxBytes {
			w.uint64(floatTagInt) // integer
			w.mpint(n, types.Typ[types.UntypedFloat])
		} else {
			w.uint64(floatTagRat) // rational
			w.bigInt(n)
			w.bigInt(v.Denom())
		}
	case *big.Float:
		w.uint64(floatTagFloat) // 512-bit mantissa f.p.
		w.mpfloat(v, typ)
	default:
		panic(internalErrorf("unexpected float constant %v (%T)", x, v))
	}
}

// bigInt exports an arbitrary-precision integer as its big-endian
// magnitude bytes (as a string) followed by a sign bool.
func (w *exportWriter) bigInt(v *big.Int) {
	w.string(string(v.Bytes())) // (big endian)
	w.bool(v.Sign() < 0)
}

func valueToRat(x constant.Value) *big.Rat {
	// Convert little-endian to big-endian.
	// I can't believe this is necessary.
	bytes := constant.Bytes(x)
	for i := 0; i < len(bytes)/2; i++ {
		bytes[i], bytes[len(bytes)-1-i] = bytes[len(bytes)-1-i], bytes[i]
	}
	i := new(big.Int).SetBytes(bytes)
	if constant.Sign(x) < 0 {
		i.Neg(i) // constant.Bytes returns the absolute value
	}
	return new(big.Rat).SetInt(i)
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

func (w *intWriter) bool(b bool) {
	var x uint64
	if b {
		x = 1
	}
	w.uint64(x)
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

func (e internalError) Error() string { return "exportdata: " + string(e) }

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
