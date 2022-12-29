// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package methodsets defines an incremental, serializable index of
// method-set information that allows efficient 'implements' queries
// across packages of the workspace without using the type checker.
//
// This package provides only the "global" (all workspace) search; the
// "local" search within a given package uses a different
// implementation based on type-checker data structures for a single
// package plus variants; see ../implementation2.go.
// The local algorithm is more precise as it tests function-local types too.
//
// A global index of function-local types is challenging since they
// may reference other local types, for which we would need to invent
// stable names, an unsolved problem described in passing in Go issue
// 57497. The global algorithm also does not index anonymous interface
// types, even outside function bodies.
//
// Consequently, global results are not symmetric: applying the
// operation twice may not get you back where you started.
package methodsets

// DESIGN
//
// See https://go.dev/cl/452060 for a minimal exposition of the algorithm.
//
// For each method, we compute a fingerprint: a string representing
// the method name and type such that equal fingerprint strings mean
// identical method types.
//
// For efficiency, the fingerprint is reduced to a single bit
// of a uint64, so that the method set can be represented as
// the union of those method bits (a uint64 bitmask).
// Assignability thus reduces to a subset check on bitmasks
// followed by equality checks on fingerprints.
//
// In earlier experiments, using 128-bit masks instead of 64 reduced
// the number of candidates by about 2x. Using (like a Bloom filter) a
// different hash function to compute a second 64-bit mask and
// performing a second mask test reduced it by about 4x.
// Neither had much effect on the running time, presumably because a
// single 64-bit mask is quite effective. See CL 452060 for details.

import (
	"fmt"
	"go/token"
	"go/types"
	"hash/crc32"
	"strconv"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/internal/typeparams"
)

// An Index records the non-empty method sets of all package-level
// types in a package in a form that permits assignability queries
// without the type checker.
type Index struct {
	pkg gobPackage
}

// NewIndex returns a new index of method-set information for all
// package-level types in the specified package.
func NewIndex(fset *token.FileSet, pkg *types.Package) *Index {
	return new(indexBuilder).build(fset, pkg)
}

// A Location records the extent of an identifier in byte-offset form.
//
// Conversion to protocol (UTF-16) form is done by the caller after a
// search, not during index construction.
// TODO(adonovan): opt: reconsider this choice, if FileHandles, not
// ParsedGoFiles were to provide ColumnMapper-like functionality.
// (Column mapping is currently associated with parsing,
// but non-parsed and even non-Go files need it too.)
// Since type checking requires reading (but not parsing) all
// dependencies' Go files, we could do the conversion at type-checking
// time at little extra cost in that case.
type Location struct {
	Filename   string
	Start, End int // byte offsets
}

// A Key represents the method set of a given type in a form suitable
// to pass to the (*Index).Search method of many different Indexes.
type Key struct {
	mset gobMethodSet // note: lacks position information
}

// KeyOf returns the search key for the method sets of a given type.
// It returns false if the type has no methods.
func KeyOf(t types.Type) (Key, bool) {
	mset := methodSetInfo(func(types.Object) (_ gobPosition) { return }, t, gobPosition{})
	if mset.Mask == 0 {
		return Key{}, false // no methods
	}
	return Key{mset}, true
}

// Search reports each type that implements (or is implemented by) the
// type that produced the search key. If methodID is nonempty, only
// that method of each type is reported.
// The result is the location of each type or method.
func (index *Index) Search(key Key, methodID string) []Location {
	var locs []Location
	for _, candidate := range index.pkg.MethodSets {
		// Traditionally this feature doesn't report
		// interface/interface elements of the relation.
		// I think that's a mistake.
		// TODO(adonovan): UX: change it, here and in the local implementation.
		if candidate.IsInterface && key.mset.IsInterface {
			continue
		}
		if !satisfies(candidate, key.mset) && !satisfies(key.mset, candidate) {
			continue
		}

		if candidate.Tricky {
			// If any interface method is tricky then extra
			// checking may be needed to eliminate a false positive.
			// TODO(adonovan): implement it.
		}

		if methodID == "" {
			locs = append(locs, index.location(candidate.Posn))
		} else {
			for _, m := range candidate.Methods {
				// Here we exploit knowledge of the shape of the fingerprint string.
				if strings.HasPrefix(m.Fingerprint, methodID) &&
					m.Fingerprint[len(methodID)] == '(' {
					locs = append(locs, index.location(m.Posn))
					break
				}
			}
		}
	}
	return locs
}

// satisfies does a fast check for whether x satisfies y.
func satisfies(x, y gobMethodSet) bool {
	return y.IsInterface && x.Mask&y.Mask == y.Mask && subset(y, x)
}

// subset reports whether method set x is a subset of y.
func subset(x, y gobMethodSet) bool {
outer:
	for _, mx := range x.Methods {
		for _, my := range y.Methods {
			if mx.Sum == my.Sum && mx.Fingerprint == my.Fingerprint {
				continue outer // found; try next x method
			}
		}
		return false // method of x not found in y
	}
	return true // all methods of x found in y
}

func (index *Index) location(posn gobPosition) Location {
	return Location{
		Filename: index.pkg.Filenames[posn.File],
		Start:    posn.Offset,
		End:      posn.Offset + posn.Len,
	}
}

// An indexBuilder builds an index for a single package.
type indexBuilder struct {
	gobPackage
	filenameIndex map[string]int
}

// build adds to the index all package-level named types of the specified package.
func (b *indexBuilder) build(fset *token.FileSet, pkg *types.Package) *Index {
	// We ignore aliases, though in principle they could define a
	// struct{...}  or interface{...} type, or an instantiation of
	// a generic, that has a novel method set.
	scope := pkg.Scope()
	for _, name := range scope.Names() {
		if tname, ok := scope.Lookup(name).(*types.TypeName); ok && !tname.IsAlias() {
			b.add(fset, tname)
		}
	}

	return &Index{pkg: b.gobPackage}
}

func (b *indexBuilder) add(fset *token.FileSet, tname *types.TypeName) {
	objectPos := func(obj types.Object) gobPosition {
		posn := safetoken.StartPosition(fset, obj.Pos())
		return gobPosition{b.fileIndex(posn.Filename), posn.Offset, len(obj.Name())}
	}
	if mset := methodSetInfo(objectPos, tname.Type(), objectPos(tname)); mset.Mask != 0 {
		// Only record types with non-trivial method sets.
		b.MethodSets = append(b.MethodSets, mset)
	}
}

// fileIndex returns a small integer that encodes the file name.
func (b *indexBuilder) fileIndex(filename string) int {
	i, ok := b.filenameIndex[filename]
	if !ok {
		i = len(b.Filenames)
		if b.filenameIndex == nil {
			b.filenameIndex = make(map[string]int)
		}
		b.filenameIndex[filename] = i
		b.Filenames = append(b.Filenames, filename)
	}
	return i
}

// methodSetInfo returns the method-set fingerprint
// of a type and records its position (typePosn)
// and the position of each of its methods m,
// as provided by objectPos(m).
func methodSetInfo(objectPos func(types.Object) gobPosition, t types.Type, typePosn gobPosition) gobMethodSet {
	// For non-interface types, use *T
	// (if T is not already a pointer)
	// since it may have more methods.
	mset := types.NewMethodSet(EnsurePointer(t))

	// Convert the method set into a compact summary.
	var mask uint64
	tricky := false
	methods := make([]gobMethod, mset.Len())
	for i := 0; i < mset.Len(); i++ {
		m := mset.At(i).Obj().(*types.Func)
		fp, isTricky := fingerprint(m)
		if isTricky {
			tricky = true
		}
		sum := crc32.ChecksumIEEE([]byte(fp))
		methods[i] = gobMethod{fp, sum, objectPos(m)}
		mask |= 1 << uint64(((sum>>24)^(sum>>16)^(sum>>8)^sum)&0x3f)
	}
	return gobMethodSet{typePosn, types.IsInterface(t), tricky, mask, methods}
}

// EnsurePointer wraps T in a types.Pointer if T is a named, non-interface type.
// This is useful to make sure you consider a named type's full method set.
func EnsurePointer(T types.Type) types.Type {
	if _, ok := T.(*types.Named); ok && !types.IsInterface(T) {
		return types.NewPointer(T)
	}

	return T
}

// fingerprint returns an encoding of a method signature such that two
// methods with equal encodings have identical types, except for a few
// tricky types whose encodings may spuriously match and whose exact
// identity computation requires the type checker to eliminate false
// positives (which are rare). The boolean result indicates whether
// the result was one of these tricky types.
//
// In the standard library, 99.8% of package-level types have a
// non-tricky method-set.  The most common exceptions are due to type
// parameters.
//
// The fingerprint string starts with method.Id() + "(".
func fingerprint(method *types.Func) (string, bool) {
	var buf strings.Builder
	tricky := false
	var fprint func(t types.Type)
	fprint = func(t types.Type) {
		switch t := t.(type) {
		case *types.Named:
			tname := t.Obj()
			if tname.Pkg() != nil {
				buf.WriteString(strconv.Quote(tname.Pkg().Path()))
				buf.WriteByte('.')
			} else if tname.Name() != "error" {
				panic(tname) // error is the only named type with no package
			}
			buf.WriteString(tname.Name())

		case *types.Array:
			fmt.Fprintf(&buf, "[%d]", t.Len())
			fprint(t.Elem())

		case *types.Slice:
			buf.WriteString("[]")
			fprint(t.Elem())

		case *types.Pointer:
			buf.WriteByte('*')
			fprint(t.Elem())

		case *types.Map:
			buf.WriteString("map[")
			fprint(t.Key())
			buf.WriteByte(']')
			fprint(t.Elem())

		case *types.Chan:
			switch t.Dir() {
			case types.SendRecv:
				buf.WriteString("chan ")
			case types.SendOnly:
				buf.WriteString("<-chan ")
			case types.RecvOnly:
				buf.WriteString("chan<- ")
			}
			fprint(t.Elem())

		case *types.Tuple:
			buf.WriteByte('(')
			for i := 0; i < t.Len(); i++ {
				if i > 0 {
					buf.WriteByte(',')
				}
				fprint(t.At(i).Type())
			}
			buf.WriteByte(')')

		case *types.Basic:
			// Use canonical names for uint8 and int32 aliases.
			switch t.Kind() {
			case types.Byte:
				buf.WriteString("byte")
			case types.Rune:
				buf.WriteString("rune")
			default:
				buf.WriteString(t.String())
			}

		case *types.Signature:
			buf.WriteString("func")
			fprint(t.Params())
			if t.Variadic() {
				buf.WriteString("...") // not quite Go syntax
			}
			fprint(t.Results())

		case *types.Struct:
			// Non-empty unnamed struct types in method
			// signatures are vanishingly rare.
			buf.WriteString("struct{")
			for i := 0; i < t.NumFields(); i++ {
				if i > 0 {
					buf.WriteByte(';')
				}
				f := t.Field(i)
				// This isn't quite right for embedded type aliases.
				// (See types.TypeString(StructType) and #44410 for context.)
				// But this is vanishingly rare.
				if !f.Embedded() {
					buf.WriteString(f.Id())
					buf.WriteByte(' ')
				}
				fprint(f.Type())
				if tag := t.Tag(i); tag != "" {
					buf.WriteByte(' ')
					buf.WriteString(strconv.Quote(tag))
				}
			}
			buf.WriteString("}")

		case *types.Interface:
			if t.NumMethods() == 0 {
				buf.WriteString("any") // common case
			} else {
				// Interface assignability is particularly
				// tricky due to the possibility of recursion.
				tricky = true
				// We could still give more disambiguating precision
				// than "..." if we wanted to.
				buf.WriteString("interface{...}")
			}

		case *typeparams.TypeParam:
			tricky = true
			// TODO(adonovan): refine this by adding a numeric suffix
			// indicating the index among the receiver type's parameters.
			buf.WriteByte('?')

		default: // incl. *types.Union
			panic(t)
		}
	}

	buf.WriteString(method.Id()) // e.g. "pkg.Type"
	sig := method.Type().(*types.Signature)
	fprint(sig.Params())
	fprint(sig.Results())
	return buf.String(), tricky
}

// -- serial format of index --

// The cost of gob encoding and decoding for most packages in x/tools
// is under 50us, with occasional peaks of around 1-3ms.
// The encoded indexes are around 1KB-50KB.

// A gobPackage records the method set of each package-level type for a single package.
type gobPackage struct {
	Filenames  []string // see gobPosition.File
	MethodSets []gobMethodSet
}

// A gobMethodSet records the method set of a single type.
type gobMethodSet struct {
	Posn        gobPosition
	IsInterface bool
	Tricky      bool   // at least one method is tricky; assignability requires go/types
	Mask        uint64 // mask with 1 bit from each of methods[*].sum
	Methods     []gobMethod
}

// A gobMethod records the name, type, and position of a single method.
type gobMethod struct {
	Fingerprint string      // string of form "methodID(params...)(results)"
	Sum         uint32      // checksum of fingerprint
	Posn        gobPosition // location of method declaration
}

// A gobPosition records the file, offset, and length of an identifier.
type gobPosition struct {
	File        int // index into Index.filenames
	Offset, Len int // in bytes
}
