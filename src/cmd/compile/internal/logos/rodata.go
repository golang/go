// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"maps"
	"strings"
	"sync"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
)

const rodataMarker = "//logos:rodata"

type rodataOrigin int

const (
	rodataGenerated rodataOrigin = iota + 1
	rodataWritten
)

var (
	rodataMu   sync.Mutex
	rodataVars = map[string]rodataOrigin{}
)

func rodataKey(pkgPath, name string) string { return pkgPath + "\x00" + name }

func RegisterRodata(pkgPath, name string) {
	rodataMu.Lock()
	defer rodataMu.Unlock()
	key := rodataKey(pkgPath, name)
	if rodataVars[key] == 0 {
		rodataVars[key] = rodataGenerated
	}
}

func RegisterWrittenRodata(pkgPath, name string) {
	rodataMu.Lock()
	defer rodataMu.Unlock()
	rodataVars[rodataKey(pkgPath, name)] = rodataWritten
}

func IsRodata(pkgPath, name string) bool {
	rodataMu.Lock()
	defer rodataMu.Unlock()
	return rodataVars[rodataKey(pkgPath, name)] != 0
}

func IsWrittenRodata(pkgPath, name string) bool {
	rodataMu.Lock()
	defer rodataMu.Unlock()
	return rodataVars[rodataKey(pkgPath, name)] == rodataWritten
}

func clearRodataPackage(pkgPath string) {
	rodataMu.Lock()
	defer rodataMu.Unlock()
	maps.DeleteFunc(rodataVars, func(k string, _ rodataOrigin) bool {
		return strings.HasPrefix(k, pkgPath+"\x00")
	})
}

type rodataVar struct {
	vd     *syntax.VarDecl
	splice bool
}

func (fl *fileLower) markRodataVar(vd *syntax.VarDecl, v ctfe.Value) {
	fl.markRodata(vd, v, true)
}

func (fl *fileLower) markRodataMoved(vd *syntax.VarDecl, v ctfe.Value) {
	fl.markRodata(vd, v, false)
}

func (fl *fileLower) markRodata(vd *syntax.VarDecl, v ctfe.Value, splice bool) {
	if rodataStaticValue(v) {
		fl.rodataVars = append(fl.rodataVars, rodataVar{vd: vd, splice: splice})
	}
}

func rodataStaticValue(v ctfe.Value) bool {
	return rodataStaticType(v.Type(), 0)
}

func rodataStaticType(t *ctfe.Type, depth int) bool {
	if t == nil || depth > 64 {
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

func (fl *fileLower) scanRodataMarkers() {
	attached := map[uint]bool{}
	for _, decl := range fl.file.DeclList {
		vd, ok := decl.(*syntax.VarDecl)
		if !ok {
			continue
		}
		top, found := rodataMarkerAbove(fl.src, fl.lineStart(vd))
		if !found {
			continue
		}
		attached[uint(bytes.Count(fl.src[:top], []byte("\n")))] = true
		for _, name := range vd.NameList {
			RegisterWrittenRodata(fl.pkg.tpkg.Path(), name.Value)
		}
	}
	fl.reportUnattachedRodata(attached)
}

func (fl *fileLower) reportUnattachedRodata(attached map[uint]bool) {
	posBase := fl.file.Pos().FileBase()
	syntax.CommentsDoMode(bytes.NewReader(fl.src), syntax.LogosMode, func(line, col uint, text string) {
		if _, ok := syntax.DirectiveArg(text, rodataMarker); ok && !attached[line] {
			base.Errorf("%s: [correctness::misplaced_directive] %s must be on its own line directly above a package-level var or a spec of a var group", syntax.MakePos(posBase, line, col).String(), rodataMarker)
		}
	})
}

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
		if _, ok := syntax.DirectiveArg(string(src[trim:top-1]), rodataMarker); ok {
			return top, true
		}
		top = prev
	}
	return top, false
}

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

func (fl *fileLower) addRodataSplice(vd *syntax.VarDecl) {
	top, found := rodataMarkerAbove(fl.src, fl.lineStart(vd))
	if found {
		return
	}
	fl.add(top, top, rodataMarker+"\n")
}
