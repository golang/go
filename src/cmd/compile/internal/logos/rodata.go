// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

// This file implements M5-S3 read-only placement (SPEC Sec 2.1, T-201):
// package-level vars holding const composites (slices, arrays, structs
// of constants) are emitted with obj.RODATA via ir.Name.MarkReadonly,
// so pointer-free data lands in .rodata and pointer-containing data in
// read-only-after-relocation memory. Slice backing temps inherit the
// mark at creation (staticinit OSLICELIT hook). Maps stay writable until
// M5-S4 lowers them to static tables; nested maps and opaque calls are
// never marked.
//
// Marker transport is textual, like the rest of this lowering: lowering
// registers created vars directly and splices a //logos:rodata line into
// the eject text; hand-written //logos:rodata lines above ungrouped
// package-level vars are picked up by scanning the source bytes, so no
// upstream parser, pragma, or unified-IR change is needed. The backend
// (staticinit schedule hook) marks the ir.Name readonly and stops the
// compile with an internal error if the initializer needs dynamic code.

import (
	"maps"
	"strings"
	"sync"

	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
)

// rodataMarker is the directive line lowering emits and scans.
const rodataMarker = "//logos:rodata"

var (
	rodataMu   sync.Mutex
	rodataVars = map[string]bool{}
)

func rodataKey(pkgPath, name string) string { return pkgPath + "\x00" + name }

// RegisterRodata records a package-level var for read-only placement.
// The backend consults it when scheduling package initialization.
func RegisterRodata(pkgPath, name string) {
	rodataMu.Lock()
	defer rodataMu.Unlock()
	rodataVars[rodataKey(pkgPath, name)] = true
}

// IsRodata reports whether name in pkgPath was registered.
func IsRodata(pkgPath, name string) bool {
	rodataMu.Lock()
	defer rodataMu.Unlock()
	return rodataVars[rodataKey(pkgPath, name)]
}

// clearRodataPackage drops registrations of one package. Lowering calls
// it on entry so a repeated in-process compile cannot inherit stale
// marks; hand-written markers are re-scanned every run.
func clearRodataPackage(pkgPath string) {
	rodataMu.Lock()
	defer rodataMu.Unlock()
	maps.DeleteFunc(rodataVars, func(k string, _ bool) bool {
		return strings.HasPrefix(k, pkgPath+"\x00")
	})
}

// rodataVar is one package-level var awaiting read-only emission.
type rodataVar struct {
	vd     *syntax.VarDecl
	splice bool // the eject text still needs a marker line above vd
}

// markRodataVar records an in-place lowering-created var; emitRodata adds
// its eject marker line.
func (fl *fileLower) markRodataVar(vd *syntax.VarDecl, v ctfe.Value) {
	if rodataStaticValue(v) {
		fl.rodataVars = append(fl.rodataVars, rodataVar{vd: vd, splice: true})
	}
}

// markRodataMoved records a var moved past its group. insertVarText already
// wrote its eject marker, so only the backend registration remains.
func (fl *fileLower) markRodataMoved(vd *syntax.VarDecl, v ctfe.Value) {
	if rodataStaticValue(v) {
		fl.rodataVars = append(fl.rodataVars, rodataVar{vd: vd})
	}
}

// rodataStaticValue reports whether v and every type nested in it
// (slice, array, struct fields, map key and element) avoid maps.
// Maps keep dynamic initialization until M5-S4 lowers them to static
// tables. Opaque or invalid types are not static.
func rodataStaticValue(v ctfe.Value) bool {
	return rodataStaticType(v.Type(), 0)
}

// rodataStaticType walks a descriptor tree. Depth is bounded because
// CTFE descriptors are finite trees; a cycle would also fail here.
func rodataStaticType(t *ctfe.Type, depth int) bool {
	if t == nil || depth > 64 || t.Opaque {
		return false
	}
	switch t.Kind {
	case ctfe.KMap:
		return false
	case ctfe.KSlice, ctfe.KArray:
		return rodataStaticType(t.Elem, depth+1)
	case ctfe.KStruct:
		for _, f := range t.Fields {
			if !rodataStaticType(f.Type, depth+1) {
				return false
			}
		}
		return true
	}
	return true
}

// scanRodataMarkers picks up hand-written //logos:rodata lines above
// ungrouped package-level vars. It runs before the const passes, while
// hand-written vars are still VarDecls and consts are still ConstDecls,
// so the two registration paths never overlap.
func (fl *fileLower) scanRodataMarkers() {
	for _, decl := range fl.file.DeclList {
		vd, ok := decl.(*syntax.VarDecl)
		if !ok || vd.Group != nil {
			continue
		}
		if _, found := rodataMarkerAbove(fl.src, fl.lineStart(vd)); !found {
			continue
		}
		for _, name := range vd.NameList {
			RegisterRodata(fl.pkg.tpkg.Path(), name.Value)
		}
	}
}

// rodataMarkerAbove walks up past the contiguous // comment block ending
// just above lineOff (a byte offset of a line start). It returns the
// block top (where a marker splice goes) and whether the block already
// holds the marker.
func rodataMarkerAbove(src []byte, lineOff int) (top int, found bool) {
	top = lineOff
	for top > 0 {
		prev := top - 1
		for prev > 0 && src[prev-1] != '\n' {
			prev--
		}
		trim := prev
		for trim < top-1 && (src[trim] == ' ' || src[trim] == '\t') {
			trim++
		}
		if trim+1 >= top || src[trim] != '/' || src[trim+1] != '/' {
			return top, false
		}
		if strings.HasPrefix(string(src[trim:top-1]), rodataMarker) {
			return top, true
		}
		top = prev
	}
	return top, false
}

// emitRodata registers every marked var for the backend and stages the
// marker line in the eject text. Moved vars (grouped consts rewritten
// past their group) already carry the marker in their inserted var line.
func (fl *fileLower) emitRodata() {
	if len(fl.rodataVars) == 0 {
		return
	}
	pkgPath := fl.pkg.tpkg.Path()
	for _, rv := range fl.rodataVars {
		for _, name := range rv.vd.NameList {
			RegisterRodata(pkgPath, name.Value)
		}
		if rv.splice {
			fl.addRodataSplice(rv.vd)
		}
	}
}

// addRodataSplice inserts the marker line above vd, hopping over a doc
// comment block so it stays attached, and never duplicating a
// hand-written marker.
func (fl *fileLower) addRodataSplice(vd *syntax.VarDecl) {
	top, found := rodataMarkerAbove(fl.src, fl.lineStart(vd))
	if found {
		return
	}
	fl.add(top, top, rodataMarker+"\n")
}
