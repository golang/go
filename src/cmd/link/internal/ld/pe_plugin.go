// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"encoding/binary"
	"internal/buildcfg"
	"sort"
)

// This file contains the windows/amd64 implementation of plugin
// support, mirroring the dlopen-based path used on Linux/Darwin.
//
// The high-level flow is:
//
// 1. When linking a *host* EXE that imports the "plugin" package,
//    peAppendHostExports widens the PE export table to cover every
//    reachable Go symbol the runtime needs to share with plugins, so
//    plugins can resolve them via GetProcAddress(GetModuleHandle(NULL))
//    at load time. This makes the host's PE export table the
//    Windows analogue of the ELF dynamic symbol table.
//
// 2. When linking a *plugin* DLL (BuildModePlugin) on windows/amd64,
//    peMarkPluginImports builds a `go:plugin.resolveSlots` array (one
//    8-byte slot per host-shared sym) plus a `go:plugin.resolveMeta`
//    blob describing each slot (name + flags). Every R_PEIMPORT reloc
//    that targets a candidate host-shared sym is redirected to its
//    slot; each shared text body is replaced by a 6-byte
//    `JMP [rip+disp32]` thunk through its slot. Both symbols are
//    exported from the plugin DLL so the host-side plugin loader
//    (plugin/plugin_windows.go) can walk the meta table at load time,
//    resolve each name via GetProcAddress on the host EXE module, and
//    write the result into the slot — exactly what the ELF dynamic
//    linker does for us on Linux.
//
//    Data slots whose target is also defined locally in the plugin
//    are pre-filled with the local address as a fallback (resolved
//    via an R_ADDR reloc at link time); the runtime resolver only
//    overwrites them when the host actually exports the same name.
//    Text slots have no usable fallback (the original body was
//    replaced with the thunk), so they are required to resolve at
//    load time — the runtime resolver fails the load if a text
//    symbol is missing on the host.

// pluginResolveSlotsSym is the SDATA symbol holding the plugin's per-
// import slot array, exported from the plugin DLL as
// "go_plugin_resolveslots". Layout: [N]uintptr.
const pluginResolveSlotsSym = "go:plugin.resolveSlots"

// pluginResolveMetaSym is the SRODATA symbol holding the plugin's
// resolve metadata, exported from the plugin DLL as
// "go_plugin_resolvemeta". Layout:
//
//	header { magic [8]byte = "GORESV01"; count uint32; pad uint32 }
//	entries [count] { flags uint32; nameLen uint32; name [nameLen]byte; pad to 4 }
//
// flags bit 0: TEXT — slot must be resolved at load time; resolver
// fails if the host does not export the symbol. flags=0 means a data
// slot whose initial value is the local fallback address.
const pluginResolveMetaSym = "go:plugin.resolveMeta"

// peLinkPlugin is the windows/amd64 plugin-link entry point. Called
// from the PE writer when ctxt.BuildMode == BuildModePlugin, after
// peMarkPluginImports has already wired up the resolve table.
func peLinkPlugin(ctxt *Link) {
	if buildcfg.GOARCH != "amd64" {
		Exitf("buildmode=plugin on windows is only supported on amd64 (got %s)", buildcfg.GOARCH)
	}
	if ctxt.Debugvlog > 0 {
		ctxt.Logf("plugin: -buildmode=plugin output %s\n", *flagOutfile)
	}
}

// peMarkPluginImports runs from dope() before initdynimport. For each
// reachable Go symbol it walks R_PEIMPORT relocations (emitted by the
// compiler under -dynlink) and decides whether to redirect them
// through a resolve-slot table. Text bodies for shared functions are
// replaced with 6-byte JMP-via-slot thunks so direct R_CALL relocs
// from the plugin's own runtime copies also reach the host.
func peMarkPluginImports(ctxt *Link) {
	if ctxt.BuildMode != BuildModePlugin {
		return
	}
	if ctxt.HeadType != objabi.Hwindows {
		return
	}
	if buildcfg.GOARCH != "amd64" {
		return
	}

	ldr := ctxt.loader

	// Plugin-local package — symbols defined here never come from the
	// host and must keep their local definitions. Without this filter
	// any reachable text sym in the plugin's own package (anonymous
	// closures like pkg.Foo.func1, etc.) would get a JMP-via-slot
	// thunk pointing at a host symbol that does not exist, and the
	// runtime resolver would hard-fail the load.
	pluginPkg := ""
	if flagPluginPath != nil {
		pluginPkg = *flagPluginPath
	}

	classify := func(s loader.Sym) (text, ok bool) {
		name := ldr.SymName(s)
		if name == "" {
			return false, false
		}
		t := ldr.SymType(s)
		if pluginShouldKeepLocal(name, t) {
			return false, false
		}
		if pluginPkg != "" && ldr.SymPkg(s) == pluginPkg {
			return false, false
		}
		if t == sym.SDYNIMPORT || t == sym.SHOSTOBJ || t == sym.SUNDEFEXT {
			return false, false
		}
		switch {
		case t.IsText():
			return true, true
		case t == sym.SRODATA, t == sym.STYPE,
			sym.SNOPTRDATA <= t && t <= sym.SNOPTRBSS:
			return false, true
		}
		return false, false
	}

	// Pass 1: discover targets. Under -dynlink the compiler emits an
	// R_PEIMPORT reloc for every cross-package load (data or function
	// address take); intra-package calls remain direct R_CALL. So we
	// only need to walk R_PEIMPORT relocs — any sym that the plugin
	// reaches through one of them is a candidate for a host-resolved
	// slot, subject to classify().
	slotIdx := make(map[loader.Sym]int)
	var slotOrder []loader.Sym
	addSlot := func(s loader.Sym) {
		if _, ok := slotIdx[s]; ok {
			return
		}
		slotIdx[s] = len(slotOrder)
		slotOrder = append(slotOrder, s)
	}
	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if !ldr.AttrReachable(s) {
			continue
		}
		// Pre-scan reloc list cheaply (Relocs() does not clone the
		// sym to external; MakeSymbolUpdater would and that strips
		// SymFlagGoType/SymFlagTypelink, breaking typelinks build).
		relocs := ldr.Relocs(s)
		for ri := 0; ri < relocs.Count(); ri++ {
			r := relocs.At(ri)
			if r.Type() != objabi.R_PEIMPORT {
				continue
			}
			rs := r.Sym()
			if rs == 0 {
				continue
			}
			if _, isShared := classify(rs); isShared {
				addSlot(rs)
			}
		}
	}

	if len(slotOrder) == 0 {
		return
	}

	// Create the slot array. For data slots we pre-fill with an R_ADDR
	// reloc to the local sym so even before the runtime resolver runs,
	// reads through the slot return a sensible value (the plugin's own
	// local copy). Text slots are left zero; the resolver MUST fill
	// them or load fails.
	slotsSu := ldr.CreateSymForUpdate(pluginResolveSlotsSym, 0)
	slotsSu.SetType(sym.SNOPTRDATA)
	slotsSu.SetReachable(true)
	slotData := make([]byte, len(slotOrder)*8)
	slotsSu.SetData(slotData)
	slotsSu.SetSize(int64(len(slotOrder) * 8))
	for i, target := range slotOrder {
		text, _ := classify(target)
		if text {
			continue
		}
		rel, _ := slotsSu.AddRel(objabi.R_ADDR)
		rel.SetSym(target)
		rel.SetOff(int32(i * 8))
		rel.SetSiz(8)
	}
	slotsSym := slotsSu.Sym()
	ldr.SetSymExtname(slotsSym, "go_plugin_resolveslots")
	ldr.SetAttrCgoExportDynamic(slotsSym, true)
	ldr.SetAttrLocal(slotsSym, false)
	ctxt.dynexp = append(ctxt.dynexp, slotsSym)

	// Build meta blob.
	const metaMagic = "GORESV01"
	meta := make([]byte, 0, 16+len(slotOrder)*24)
	meta = append(meta, metaMagic...)
	var u32 [4]byte
	binary.LittleEndian.PutUint32(u32[:], uint32(len(slotOrder)))
	meta = append(meta, u32[:]...)
	// pad header to 16 bytes
	meta = append(meta, 0, 0, 0, 0)
	for _, target := range slotOrder {
		name := ldr.SymName(target)
		text, _ := classify(target)
		var flags uint32
		if text {
			flags |= 1
		}
		binary.LittleEndian.PutUint32(u32[:], flags)
		meta = append(meta, u32[:]...)
		binary.LittleEndian.PutUint32(u32[:], uint32(len(name)))
		meta = append(meta, u32[:]...)
		meta = append(meta, name...)
		for len(meta)%4 != 0 {
			meta = append(meta, 0)
		}
	}
	metaSu := ldr.CreateSymForUpdate(pluginResolveMetaSym, 0)
	metaSu.SetType(sym.SRODATA)
	metaSu.SetReachable(true)
	metaSu.SetData(meta)
	metaSu.SetSize(int64(len(meta)))
	metaSym := metaSu.Sym()
	ldr.SetSymExtname(metaSym, "go_plugin_resolvemeta")
	ldr.SetAttrCgoExportDynamic(metaSym, true)
	ldr.SetAttrLocal(metaSym, false)
	ctxt.dynexp = append(ctxt.dynexp, metaSym)

	// Pass 2: rewrite R_PEIMPORT relocs.
	//   - target has a slot: rewrite to R_PCREL into slotsSym with
	//     addend = idx*8. Same encoding as before (4-byte PC-rel disp
	//     to the slot), just no SDYNIMPORT/IAT round trip.
	//   - target is local (no slot): the compiler emitted MOVQ-from-GOT
	//     because it was -dynlink. Since the symbol lives in this
	//     image we optimize MOVQ -> LEAQ + R_PCREL (same trick the
	//     Linux/amd64 linker does for R_X86_64_GOTPCRELX).
	var redirected, optimized int
	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if !ldr.AttrReachable(s) {
			continue
		}
		relocs := ldr.Relocs(s)
		hasPEImport := false
		for ri := 0; ri < relocs.Count(); ri++ {
			if relocs.At(ri).Type() == objabi.R_PEIMPORT {
				hasPEImport = true
				break
			}
		}
		if !hasPEImport {
			continue
		}
		su := ldr.MakeSymbolUpdater(s)
		data := su.Data()
		relocs = su.Relocs()
		writableEnsured := false
		for ri := 0; ri < relocs.Count(); ri++ {
			r := relocs.At(ri)
			if r.Type() != objabi.R_PEIMPORT {
				continue
			}
			rs := r.Sym()
			if rs == 0 {
				continue
			}
			if idx, ok := slotIdx[rs]; ok {
				su.SetRelocType(ri, objabi.R_PCREL)
				su.SetRelocSym(ri, slotsSym)
				su.SetRelocAdd(ri, int64(idx*8))
				redirected++
				continue
			}
			off := int(r.Off())
			if off < 2 || data[off-2] != 0x8b {
				// Not a plain MOVQ-from-GOT: leave it alone (linker
				// resolves the R_PEIMPORT against the local sym via
				// the usual PC-rel math — same as a stray R_PCREL).
				continue
			}
			if !writableEnsured {
				su.MakeWritable()
				data = su.Data()
				writableEnsured = true
			}
			data[off-2] = 0x8d // MOVQ -> LEAQ
			su.SetRelocType(ri, objabi.R_PCREL)
			optimized++
		}
	}

	// Pass 3: replace each shared text body with a JMP-via-slot thunk.
	// Layout: FF 25 dd dd dd dd  ; JMP qword ptr [rip+disp32]
	// R_PCREL at off=2 size=4 → disp resolves to (slot - ripAfter).
	// Once the runtime resolver writes the host address into the slot,
	// the indirect JMP lands in the host's text section.
	var thunked int
	for i, target := range slotOrder {
		text, _ := classify(target)
		if !text {
			continue
		}
		su := ldr.MakeSymbolUpdater(target)
		su.SetData([]byte{0xFF, 0x25, 0x00, 0x00, 0x00, 0x00})
		su.SetSize(6)
		su.ResetRelocs()
		rel, _ := su.AddRel(objabi.R_PCREL)
		rel.SetSym(slotsSym)
		rel.SetOff(2)
		rel.SetSiz(4)
		rel.SetAdd(int64(i * 8))
		thunked++
	}

	if ctxt.Debugvlog > 0 {
		ctxt.Logf("plugin: %d resolve slots, %d redirected relocs, %d local optimized, %d text thunks\n",
			len(slotOrder), redirected, optimized, thunked)
	}
}

// pluginShouldKeepLocal reports whether a symbol must be defined in the
// plugin DLL itself rather than imported from the host, even when a
// same-named symbol exists in the host. These are typically per-module
// metadata that the runtime expects to have its own copy of per plugin.
func pluginShouldKeepLocal(name string, t sym.SymKind) bool {
	switch name {
	case "runtime.firstmoduledata",
		"local.pluginmoduledata",
		"go:link.thispluginpath",
		"go:link.pkghashes",
		"go:link.addmoduledata",
		"go:link.addmoduledatainit":
		return true
	}
	// All link-internal symbols must remain local.
	if len(name) > 8 && name[:8] == "go:link." {
		return true
	}
	if len(name) > 6 && name[:6] == "local." {
		return true
	}
	return false
}

// peAppendHostExports runs from inside addexports(), after deadcode
// and all reachability passes. For windows host EXEs that import
// the "plugin" package, it appends every reachable Go symbol to the
// dexport list (deduping against syms already exported via
// cgo_export_dynamic). The PE export table emitted afterwards then
// covers every symbol a plugin could try to import via GetProcAddress.
func peAppendHostExports(ctxt *Link) {
	if ctxt.HeadType != objabi.Hwindows {
		return
	}
	if ctxt.BuildMode == BuildModePlugin {
		return
	}
	if ctxt.LibraryByPkg["plugin"] == nil {
		return
	}
	if buildcfg.GOARCH != "amd64" {
		return
	}
	ldr := ctxt.loader
	already := make(map[loader.Sym]bool, len(dexport))
	for _, s := range dexport {
		already[s] = true
	}
	added := 0
	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if already[s] {
			continue
		}
		if !ldr.AttrReachable(s) {
			continue
		}
		name := ldr.SymName(s)
		if name == "" {
			continue
		}
		if pluginShouldKeepLocal(name, ldr.SymType(s)) {
			continue
		}
		t := ldr.SymType(s)
		switch {
		case t.IsText():
			// Skip non-ABIInternal text symbols (e.g. ABI0 wrappers
			// generated for asm-callable entry points) when an
			// ABIInternal sibling with the same name exists. Plugins
			// compiled as Go code call runtime functions through the
			// ABIInternal calling convention (args in registers); if
			// we exported both, the PE export table would contain two
			// entries with identical names and the plugin's slot could
			// resolve to the wrong (stack-based) wrapper, producing a
			// silent ABI mismatch and a corrupted first argument.
			if ldr.SymVersion(s) != sym.SymVerABIInternal &&
				ldr.SymVersion(s) < sym.SymVerStatic {
				if s2 := ldr.Lookup(name, sym.SymVerABIInternal); s2 != 0 &&
					ldr.SymType(s2).IsText() && ldr.AttrReachable(s2) {
					continue
				}
			}
		case t == sym.SRODATA, t == sym.STYPE, t == sym.SSTRING, t == sym.SGOSTRING,
			t == sym.SGOFUNC, t == sym.SFUNCTAB, t == sym.STYPELINK, t == sym.SITABLINK:
		case t == sym.SDATA, t == sym.SNOPTRDATA, t == sym.SINITARR:
		case t == sym.SBSS, t == sym.SNOPTRBSS:
		default:
			continue
		}
		if ldr.SymExtname(s) != name {
			// mangleTypeSym may have rewritten Extname to a short hash;
			// plugin DLLs import the full Go name, so force the export
			// name back. Safe because we never feed these to an external
			// linker — internal linker only.
			ldr.SetSymExtname(s, name)
		}
		dexport = append(dexport, s)
		already[s] = true
		added++
	}
	if ctxt.Debugvlog > 0 {
		ctxt.Logf("plugin host: appended %d Go symbols to PE export table\n", added)
	}
	// PE name pointer table must be lexicographically sorted by export
	// name for GetProcAddress's binary search to work.
	sort.Slice(dexport, func(i, j int) bool { return ldr.SymExtname(dexport[i]) < ldr.SymExtname(dexport[j]) })
}
