// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmp"
	"encoding/binary"
	"flag"
	"hash/fnv"
	"slices"
	"strings"
)

// flagLogosICF enables identical code folding of text symbols. Cmd/go
// passes it for Os and Oz builds; direct link runs leave it off so the
// default link keeps upstream behavior.
var flagLogosICF = flag.Bool("logos.icf", false, "fold text symbols with identical machine code")

// logosIcfFolder holds the fold map from a duplicate text symbol to the
// kept symbol that takes its relocations and its layout entry, plus the
// name-addressed symbols that must keep their own entry.
type logosIcfFolder struct {
	ctxt   *Link
	ldr    *loader.Loader
	keep   map[loader.Sym]loader.Sym
	pinned map[loader.Sym]bool
}

// logosIcfFold merges reachable compiler-generated text symbols that
// carry identical machine code and identical relocations, keeping one
// copy whose name and line table serve every merged function. The hook
// line lives in main.go; every other line lives here.
func logosIcfFold(ctxt *Link) {
	if !*flagLogosICF {
		return
	}
	f := logosIcfFolder{ctxt: ctxt, ldr: ctxt.loader}
	f.pin()
	for f.pass() {
	}
}

// logosIcfSkip reports whether policy excludes a package from folding:
// the runtime tree keeps distinct addresses for traces and profiles,
// and assembly functions may rely on address identity.
func logosIcfSkip(pkg string, fromAsm bool) bool {
	if fromAsm {
		return true
	}
	return logosIcfRuntime(pkg)
}

// logosIcfRuntime reports whether pkg is the runtime tree.
func logosIcfRuntime(pkg string) bool {
	return pkg == "runtime" || strings.HasPrefix(pkg, "runtime/")
}

// logosIcfFind returns the kept symbol for s, following merges.
func (f *logosIcfFolder) find(s loader.Sym) loader.Sym {
	k, ok := f.keep[s]
	if !ok {
		return s
	}
	r := f.find(k)
	f.keep[s] = r
	return r
}

// logosIcfPin records the name-addressed symbols that keep their own
// entry: dynamic exports and wasm exports resolve by symbol, not by
// relocation, so folding them would strand their lookup.
func (f *logosIcfFolder) pin() {
	f.keep = make(map[loader.Sym]loader.Sym)
	f.pinned = make(map[loader.Sym]bool)
	for _, s := range f.ctxt.dynexp {
		f.pinned[s] = true
	}
	for _, s := range f.ldr.WasmExports {
		f.pinned[s] = true
	}
}

// logosIcfPass groups eligible text symbols by content hash, merges the
// groups whose bytes and relocations match, rewrites every reachable
// relocation from a duplicate to its kept symbol, and drops duplicates
// from the text layout. It reports whether it merged anything.
func (f *logosIcfFolder) pass() bool {
	merged := false
	for _, group := range f.groups(f.candidates()) {
		merged = f.merge(group) || merged
	}
	if !merged {
		return false
	}
	f.rewrite()
	f.drop()
	return true
}

// logosIcfCandidates lists the text symbols eligible for folding:
// reachable top-level functions outside the runtime tree, excluding
// assembly, the link entry, and name-addressed exports. It walks the
// loader because the text layout list only fills at linksetup.
func (f *logosIcfFolder) candidates() []loader.Sym {
	var out []loader.Sym
	for s := loader.Sym(1); s < loader.Sym(f.ldr.NSym()); s++ {
		if f.eligible(s) && f.foldable(s) {
			out = append(out, s)
		}
	}
	return out
}

// logosIcfEligible reports whether s is a live top-level text symbol
// with machine code to compare.
func (f *logosIcfFolder) eligible(s loader.Sym) bool {
	if !f.ldr.AttrReachable(s) {
		return false
	}
	if !f.ldr.SymType(s).IsText() {
		return false
	}
	if f.ldr.SymSize(s) == 0 {
		return false
	}
	if f.ldr.IsExternal(s) {
		return false
	}
	return f.ldr.TopLevelSym(s)
}

// logosIcfFoldable reports whether s survives the policy exclusions.
func (f *logosIcfFolder) foldable(s loader.Sym) bool {
	if logosIcfSkip(f.ldr.SymPkg(s), f.ldr.IsFromAssembly(s)) {
		return false
	}
	if f.pinned[s] {
		return false
	}
	return *flagEntrySymbol == "" || f.ldr.SymName(s) != *flagEntrySymbol
}

// logosIcfGroups buckets candidates by content hash so only
// byte-identical functions pay the full comparison. The buckets come
// out in deterministic order so repeated links fold the same way.
func (f *logosIcfFolder) groups(cands []loader.Sym) [][]loader.Sym {
	buckets := make(map[uint64][]loader.Sym)
	for _, s := range cands {
		h := f.hash(s)
		buckets[h] = append(buckets[h], s)
	}
	var out [][]loader.Sym
	for _, g := range buckets {
		if len(g) < 2 {
			continue
		}
		slices.Sort(g)
		out = append(out, g)
	}
	slices.SortFunc(out, func(a, b []loader.Sym) int { return cmp.Compare(a[0], b[0]) })
	return out
}

// logosIcfHash hashes a symbol's machine code bytes and the offset,
// size, type, addend, and canonical target of every relocation.
func (f *logosIcfFolder) hash(s loader.Sym) uint64 {
	h := fnv.New64a()
	h.Write(f.ldr.Data(s))
	relocs := f.ldr.Relocs(s)
	var buf [26]byte
	for i := range relocs.Count() {
		r := relocs.At(i)
		t, add := f.target(r)
		binary.LittleEndian.PutUint32(buf[0:4], uint32(r.Off()))
		buf[4] = r.Siz()
		binary.LittleEndian.PutUint32(buf[5:9], uint32(r.Type()))
		binary.LittleEndian.PutUint64(buf[9:17], uint64(add))
		binary.LittleEndian.PutUint32(buf[17:21], uint32(t))
		buf[21] = 0
		if r.Weak() {
			buf[21] = 1
		}
		binary.LittleEndian.PutUint32(buf[22:26], uint32(f.ldr.RelocVariant(s, i)))
		h.Write(buf[:])
	}
	return h.Sum64()
}

// logosIcfTarget canonicalizes a relocation target through interior
// symbols and earlier merges so duplicates of one function hash alike.
func (f *logosIcfFolder) target(r loader.Reloc) (loader.Sym, int64) {
	rs := r.Sym()
	if rs == 0 {
		return 0, r.Add()
	}
	outer, off := FoldSubSymbolOffset(f.ldr, rs)
	return f.find(outer), r.Add() + off
}

// logosIcfMerge folds every member of group that matches the lowest
// member byte for byte, recording victim to keeper redirections.
func (f *logosIcfFolder) merge(group []loader.Sym) bool {
	rep := group[0]
	merged := false
	for _, s := range group[1:] {
		if f.equal(rep, s) {
			f.keep[s] = rep
			merged = true
		}
	}
	return merged
}

// logosIcfEqual confirms two hash-equal symbols carry identical bytes
// and relocations before they share one address.
func (f *logosIcfFolder) equal(a, b loader.Sym) bool {
	if !bytes.Equal(f.ldr.Data(a), f.ldr.Data(b)) {
		return false
	}
	ra := f.ldr.Relocs(a)
	rb := f.ldr.Relocs(b)
	if ra.Count() != rb.Count() {
		return false
	}
	for i := range ra.Count() {
		if !f.relocEqual(ra.At(i), rb.At(i)) {
			return false
		}
		if f.ldr.RelocVariant(a, i) != f.ldr.RelocVariant(b, i) {
			return false
		}
	}
	return true
}

// logosIcfRelocEqual reports whether two relocations match after
// canonicalizing their targets through merges and interior offsets.
func (f *logosIcfFolder) relocEqual(x, y loader.Reloc) bool {
	if x.Off() != y.Off() || x.Siz() != y.Siz() || x.Type() != y.Type() {
		return false
	}
	if x.Weak() != y.Weak() {
		return false
	}
	tx, ax := f.target(x)
	ty, ay := f.target(y)
	return tx == ty && ax == ay
}

// logosIcfRewrite redirects every reachable relocation that targets a
// duplicate, interior offset included, to the kept symbol.
func (f *logosIcfFolder) rewrite() {
	for s := loader.Sym(1); s < loader.Sym(f.ldr.NSym()); s++ {
		if !f.ldr.AttrReachable(s) {
			continue
		}
		if f.ldr.IsExternal(s) {
			continue
		}
		f.rewriteSym(s)
	}
}

// logosIcfRewriteSym redirects one symbol's duplicate relocations to
// their kept symbols, carrying interior offsets into the addend.
func (f *logosIcfFolder) rewriteSym(s loader.Sym) {
	relocs := f.ldr.Relocs(s)
	var su *loader.SymbolBuilder
	for i := range relocs.Count() {
		r := relocs.At(i)
		rs := r.Sym()
		if rs == 0 {
			continue
		}
		outer, off := FoldSubSymbolOffset(f.ldr, rs)
		k := f.find(outer)
		if k == outer {
			continue
		}
		if su == nil {
			su = f.ldr.MakeSymbolUpdater(s)
			logosIcfKeepWeak(su, relocs)
		}
		su.SetRelocSym(i, k)
		if off != 0 {
			su.SetRelocAdd(i, r.Add()+off)
		}
	}
}

// logosIcfKeepWeak restores the weak bit on every relocation of su. The loader
// clones an object symbol into a payload with that bit masked off, and
// an itab's references to methods that dead-code elimination left
// unreachable must stay weak so they resolve to the unreachable-method
// stub instead of failing the link.
func logosIcfKeepWeak(su *loader.SymbolBuilder, relocs loader.Relocs) {
	for i := range relocs.Count() {
		r := relocs.At(i)
		if r.Weak() {
			su.SetRelocType(i, r.Type()|objabi.R_WEAK)
		}
	}
}

// logosIcfDrop marks duplicates unreachable so linksetup leaves them
// out of the text layout and later passes emit one copy only.
func (f *logosIcfFolder) drop() {
	for s := range f.keep {
		f.ldr.SetAttrReachable(s, false)
	}
}
