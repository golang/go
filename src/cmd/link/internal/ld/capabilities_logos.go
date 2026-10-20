// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/goobj"
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmp"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path"
	"runtime"
	"slices"
	"strings"
)

// flagLogosCap names the JSON capabilities policy file cmd/go writes for
// logos builds. Empty (plain go, or go tool link by hand) disables the
// audit, so plain go keeps its upstream link behavior.
var flagLogosCap = flag.String("logos.cap", "", "read logos capabilities policy from `file` (logos builds only)")

// CapabilityKind classifies a protected sink.
type CapabilityKind uint8

// Sink capability kinds.
const (
	CapReflect CapabilityKind = iota + 1
	CapExec
	CapNet
	CapCgo
)

// capabilitySinks maps sink symbol names to their capability kind. The
// PLAN name runtime.logosFastCgoCall names no symbol in this tree; the
// tailored sink is runtime.logosCgocallFast, the M20 fast path.
var capabilitySinks = map[string]CapabilityKind{
	"reflect.ValueOf":          CapReflect,
	"reflect.TypeOf":           CapReflect,
	"reflect.New":              CapReflect,
	"syscall.Exec":             CapExec,
	"syscall.ForkExec":         CapExec,
	"os.StartProcess":          CapExec,
	"os.startProcess":          CapExec,
	"net.sysSocket":            CapNet,
	"runtime.cgocall":          CapCgo,
	"runtime.logosCgocallFast": CapCgo,
}

// Logos severity words shared with the cmd/go policy. The linker mirrors
// the policy JSON strings; it cannot import the cmd/go module.
const (
	logosDeny  = "deny"
	logosWarn  = "warn"
	logosAllow = "allow"
)

// logosCapPolicy mirrors the cmd/go LinkPolicy JSON (M12-S2): the two
// structs must carry the same keys. Unsafe sinks need no linker entry;
// M12-S1 covers them over the package graph.
type logosCapPolicy struct {
	TransitiveReflect       string   `json:"transitive_reflect"`
	TransitiveExec          string   `json:"transitive_exec"`
	TransitiveNet           string   `json:"transitive_net"`
	ReflectAllowlist        []string `json:"reflect_allowlist"`
	TransitiveExecAllowlist []string `json:"transitive_exec_allowlist"`
	TransitiveNetAllowlist  []string `json:"transitive_net_allowlist"`
	MainModules             []string `json:"main_modules"`
	CgoAllowed              []string `json:"cgo_allowed"`
}

// logosCapImpl is one callable method implementation: its name, its
// function type symbol, and the text symbol holding its code.
type logosCapImpl struct {
	name string
	typ  loader.Sym
	fn   loader.Sym
}

// logosCapHit is one reached sink with its kind and display name.
type logosCapHit struct {
	sink loader.Sym
	kind CapabilityKind
	name string
}

// capIsCgoTrampoline reports whether name is a cgo C-function wrapper:
// package-qualified _Cfunc_ symbols the cgo tool generates per C call
// site. External cgo calls leave the Go loader graph at the host object
// boundary (the wrapper points at an external C trampoline with no
// outgoing loader edges), so the wrapper, not runtime.cgocall, is the
// observable cgo-use point. _Ctype_ and _Cvar_ symbols carry no calls.
func capIsCgoTrampoline(name string) bool {
	_, after, ok := strings.Cut(name, "._Cfunc_")
	return ok && after != ""
}

// logosCapAuditor is a BFS over the live symbol graph from the link roots,
// recording the parent of every visit for shortest-chain backtraces. It
// runs after the deadcode fixpoint, so AttrReachable already excludes dead
// code: a dependency that imports a sink package without calling it
// contributes no path.
type logosCapAuditor struct {
	ctxt   *Link
	ldr    *loader.Loader
	policy *logosCapPolicy
	goroot string
	parent map[loader.Sym]loader.Sym
	queue  []loader.Sym
	impls  []logosCapImpl
	hits   []logosCapHit
}

// logosCapAudit runs the capabilities reachability audit after deadcode.
// The hook line lives in deadcode.go; every other line lives here.
func logosCapAudit(ctxt *Link, d *deadcodePass) {
	if *flagLogosCap == "" {
		return
	}
	data, err := os.ReadFile(*flagLogosCap)
	if err != nil {
		Errorf("logos capabilities: read policy %s: %v", *flagLogosCap, err)
		return
	}
	var policy logosCapPolicy
	if err := json.Unmarshal(data, &policy); err != nil {
		Errorf("logos capabilities: parse policy %s: %v", *flagLogosCap, err)
		return
	}
	goroot := cmp.Or(os.Getenv("GOROOT"), runtime.GOROOT())
	a := &logosCapAuditor{
		ctxt:   ctxt,
		ldr:    ctxt.loader,
		policy: &policy,
		goroot: goroot,
		parent: make(map[loader.Sym]loader.Sym),
	}
	a.collectImpls(d)
	a.seedRoots(d)
	a.flood(d)
	a.report()
}

// collectImpls indexes the callable implementations of the reachable
// types' methods. It scans the loader instead of reusing
// d.markableMethods: the deadcode fixpoint removes every matched method
// from that list, so only unmatched leftovers remain by audit time.
func (a *logosCapAuditor) collectImpls(d *deadcodePass) {
	for s := loader.Sym(1); int(s) < a.ldr.NDef(); s++ {
		if !a.ldr.AttrReachable(s) || !a.ldr.IsGoType(s) {
			continue
		}
		if len(a.ldr.Data(s)) == 0 || !decodetypeHasUncommon(a.ctxt.Arch, a.ldr.Data(s)) {
			continue
		}
		relocs := a.ldr.Relocs(s)
		var triples []int
		for i := 0; i < relocs.Count(); i++ {
			r := relocs.At(i)
			if r.Type() != objabi.R_METHODOFF {
				continue
			}
			if r.Weak() {
				convert := a.ctxt.linkShared && a.ldr.IsItab(s) ||
					a.ctxt.canUsePlugins && r.Type().IsDirectCall()
				if !convert {
					i += 2
					continue
				}
			}
			triples = append(triples, i)
			i += 2
		}
		sigs := d.decodetypeMethods(a.ldr, a.ctxt.Arch, s, &relocs)
		for j, r := range triples {
			if j >= len(sigs) {
				break
			}
			fn := relocs.At(r + 2).Sym()
			if fn == 0 || !a.ldr.AttrReachable(fn) {
				continue
			}
			a.impls = append(a.impls, logosCapImpl{name: sigs[j].name, typ: sigs[j].typ, fn: fn})
		}
	}
}

// seedRoots enqueues the link roots: the entry symbol, dynamic exports,
// the main inittasks, and the package init functions deadcode recorded.
func (a *logosCapAuditor) seedRoots(d *deadcodePass) {
	var names []string
	if a.ctxt.linkShared && (a.ctxt.BuildMode == BuildModeExe || a.ctxt.BuildMode == BuildModePIE) {
		names = append(names, "main.main", "main..inittask")
	} else {
		names = append(names, *flagEntrySymbol)
	}
	for _, name := range names {
		a.enqueue(a.ldr.Lookup(name, 0), 0)
		if abiInternalVer != 0 {
			a.enqueue(a.ldr.Lookup(name, abiInternalVer), 0)
		}
	}
	for _, s := range a.ctxt.dynexp {
		a.enqueue(s, 0)
	}
	for _, s := range a.ldr.WasmExports {
		a.enqueue(s, 0)
	}
	if a.ctxt.mainInittasks != 0 {
		a.enqueue(a.ctxt.mainInittasks, 0)
	}
	for _, s := range d.pkginits {
		a.enqueue(s, 0)
	}
}

// enqueue visits s with parent p, skipping dull, visited, and dead
// symbols. Parent presence marks visited; roots carry parent 0.
func (a *logosCapAuditor) enqueue(s, p loader.Sym) {
	if s == 0 || s == p {
		return
	}
	if _, ok := a.parent[s]; ok {
		return
	}
	if !a.ldr.AttrReachable(s) {
		return
	}
	a.parent[s] = p
	a.queue = append(a.queue, s)
}

// flood walks relocations from the roots, recording parents. Marker
// relocations (R_USETYPE, R_USEIFACE, R_INITORDER) and method-descriptor
// triples (R_METHODOFF) carry no calls and are skipped like in deadcode;
// R_USEIFACEMETHOD and R_USENAMEDMETHOD grow synthetic edges to the
// matching implementations with the call site as parent.
func (a *logosCapAuditor) flood(d *deadcodePass) {
	for len(a.queue) > 0 {
		s := a.queue[0]
		a.queue = a.queue[1:]
		name := a.ldr.SymName(s)
		if kind, ok := capabilitySinks[capSinkName(name)]; ok {
			a.hits = append(a.hits, logosCapHit{sink: s, kind: kind, name: capSinkName(name)})
		} else if capIsCgoTrampoline(name) && a.ldr.SymType(s).IsText() {
			a.hits = append(a.hits, logosCapHit{sink: s, kind: CapCgo, name: "runtime.cgocall"})
		}
		if a.ldr.IsReflectMethod(s) || a.callsReflectLookup(s) {
			for _, impl := range a.impls {
				if capIsExportedMethod(impl.name) {
					a.enqueue(impl.fn, s)
				}
			}
		}
		a.followRelocs(d, s)
		a.followAux(s)
		if a.ldr.IsExternal(s) {
			a.enqueue(a.ldr.OuterSym(s), s)
			a.enqueue(a.ldr.SubSym(s), s)
		}
	}
}

// reflectLookupSyms names the method-lookup functions whose use makes a
// function a reflection site, mirroring the compiler's REFLECTMETHOD flag
// (which covers only lookups the compiler cannot resolve statically).
var reflectLookupSyms = []string{
	"reflect.Value.Method",
	"reflect.Value.MethodByName",
	"reflect.Type.Method",
	"reflect.Type.MethodByName",
}

// callsReflectLookup reports whether s references a reflective method
// lookup, catching constant-argument lookups the compiler resolves
// without flagging REFLECTMETHOD.
func (a *logosCapAuditor) callsReflectLookup(s loader.Sym) bool {
	relocs := a.ldr.Relocs(s)
	for i := range relocs.Count() {
		r := relocs.At(i)
		t := r.Sym()
		if t == 0 || r.Type() == objabi.R_METHODOFF {
			continue
		}
		if slices.Contains(reflectLookupSyms, a.ldr.SymName(t)) {
			return true
		}
	}
	return false
}

// followRelocs enqueues the call and data targets of s.
func (a *logosCapAuditor) followRelocs(d *deadcodePass, s loader.Sym) {
	relocs := a.ldr.Relocs(s)
	for i := 0; i < relocs.Count(); i++ {
		r := relocs.At(i)
		if r.Weak() {
			convert := false
			if a.ctxt.linkShared && a.ldr.IsItab(s) {
				convert = true
			}
			if a.ctxt.canUsePlugins && r.Type().IsDirectCall() {
				convert = true
			}
			if !convert {
				continue
			}
		}
		switch r.Type() {
		case objabi.R_METHODOFF:
			i += 2
			continue
		case objabi.R_USETYPE, objabi.R_INITORDER:
			continue
		case objabi.R_USEIFACE:
			if rs := r.Sym(); a.ldr.IsItab(rs) {
				a.enqueue(rs, s)
			}
			continue
		case objabi.R_USEIFACEMETHOD:
			m := d.decodeIfaceMethod(a.ldr, a.ctxt.Arch, r.Sym(), r.Add())
			a.followIfaceMethod(s, m.name, m.typ)
			continue
		case objabi.R_USENAMEDMETHOD:
			a.followIfaceMethodName(s, d.decodeGenericIfaceMethod(a.ldr, r.Sym()))
			continue
		}
		a.enqueue(r.Sym(), s)
	}
}

// followIfaceMethod adds synthetic edges from a precise interface use
// site to the implementations matching the used method name and type.
func (a *logosCapAuditor) followIfaceMethod(from loader.Sym, name string, typ loader.Sym) {
	a.followIfaceImpls(from, name, typ)
}

// followIfaceMethodName adds synthetic edges from a generic interface use
// site to every implementation carrying the used method name.
func (a *logosCapAuditor) followIfaceMethodName(from loader.Sym, name string) {
	a.followIfaceImpls(from, name, 0)
}

// followIfaceImpls adds synthetic edges from an interface use site to the
// matching implementations. The nil type symbol matches any type.
func (a *logosCapAuditor) followIfaceImpls(from loader.Sym, name string, typ loader.Sym) {
	for _, impl := range a.impls {
		if impl.name != name {
			continue
		}
		if typ != 0 && impl.typ != typ {
			continue
		}
		a.enqueue(impl.fn, from)
	}
}

// followAux enqueues the auxiliary symbols of s, except type descriptors.
func (a *logosCapAuditor) followAux(s loader.Sym) {
	for i, n := 0, a.ldr.NAux(s); i < n; i++ {
		if aux := a.ldr.Aux(s, i); aux.Type() != goobj.AuxGotype {
			a.enqueue(aux.Sym(), s)
		}
	}
}

// chainTo rebuilds the root-to-sink chain through the parent map.
func (a *logosCapAuditor) chainTo(sink loader.Sym) []loader.Sym {
	var rev []loader.Sym
	for cur := sink; cur != 0; cur = a.parent[cur] {
		rev = append(rev, cur)
		if _, ok := a.parent[cur]; !ok {
			break
		}
	}
	slices.Reverse(rev)
	return rev
}

// report checks every reached sink against the policy and emits one
// diagnostic per violating chain.
func (a *logosCapAuditor) report() {
	for _, hit := range a.hits {
		chain := a.chainTo(hit.sink)
		dep := a.blame(chain)
		if dep == "" {
			continue
		}
		sinkName := hit.name
		if hit.kind == CapCgo {
			if slices.Contains(a.policy.CgoAllowed, dep) {
				continue
			}
			a.emit("ERROR", "unauthorized_cgo", dep, sinkName, chain,
				`Remove the dependency or allow cgo for its module with [build] cgo = "allow" in logos.toml.`)
			continue
		}
		rule := a.ruleFor(hit.kind)
		if rule.sev == logosAllow || capAllowlisted(rule.allow, dep) {
			continue
		}
		word := "ERROR"
		if rule.sev == logosWarn {
			word = "WARNING"
		}
		a.emit(word, rule.id, dep, sinkName, chain,
			fmt.Sprintf("Remove the dependency or explicitly authorize %q under '%s' in logos.toml.", dep+"/**", rule.allowKey))
	}
}

// capRule is the policy row for one sink kind.
type capRule struct {
	id       string
	sev      string
	allowKey string
	allow    []string
}

// ruleFor maps a sink kind to its policy row. Unknown severities fail
// closed to deny.
func (a *logosCapAuditor) ruleFor(kind CapabilityKind) capRule {
	switch kind {
	case CapReflect:
		return capRule{"unauthorized_reflect", capSeverityDeny(a.policy.TransitiveReflect), "reflect_allowlist", a.policy.ReflectAllowlist}
	case CapNet:
		return capRule{"unauthorized_net", capSeverityDeny(a.policy.TransitiveNet), "transitive_net_allowlist", a.policy.TransitiveNetAllowlist}
	default:
		return capRule{"unauthorized_exec", capSeverityDeny(a.policy.TransitiveExec), "transitive_exec_allowlist", a.policy.TransitiveExecAllowlist}
	}
}

// emit renders one chain and prints it: deny aborts the link through
// Errorf, warn prints and continues.
func (a *logosCapAuditor) emit(word, ruleID, dep, sink string, chain []loader.Sym, help string) {
	var frames []capFrame
	for _, s := range chain {
		if a.ldr.SymType(s).IsText() {
			frames = append(frames, a.frameOf(s))
		}
	}
	report := renderCapReport(word, ruleID, dep, sink, frames, help)
	if word == "ERROR" {
		Errorf("%s", report)
		return
	}
	fmt.Fprintln(os.Stderr, report)
}

// blame walks the chain from the sink back to the roots and returns the
// first package outside GOROOT that is neither standard library nor under
// a main module. Empty means no violation: pure standard-library
// reachability and first-party calls never blame.
func (a *logosCapAuditor) blame(chain []loader.Sym) string {
	for i := len(chain) - 1; i >= 0; i-- {
		f := a.frameOf(chain[i])
		if f.file != "unknown" && capPathUnder(f.file, a.goroot) {
			continue
		}
		pkg := capPkgOfSym(f.name)
		if capIsStdPkg(pkg) || a.underMainModules(pkg) {
			continue
		}
		return pkg
	}
	return ""
}

// underMainModules reports whether pkg sits under a main module.
func (a *logosCapAuditor) underMainModules(pkg string) bool {
	for _, m := range a.policy.MainModules {
		if m != "" && (pkg == m || strings.HasPrefix(pkg, m+"/")) {
			return true
		}
	}
	return false
}

// frameOf renders one symbol as a report frame, falling back to
// unknown:0 when the object carries no function info.
func (a *logosCapAuditor) frameOf(s loader.Sym) capFrame {
	name := a.ldr.SymName(s)
	file, line := "unknown", 0
	if fi := a.ldr.FuncInfo(s); fi.Valid() {
		fi.Preload()
		if f := a.frameFileOf(s, fi); f != "" {
			file = f
		}
		line = int(fi.StartLine())
	}
	return capFrame{name: name, file: file, line: line}
}

// frameFileOf returns the defining file of s: the single used file when
// unambiguous, else the file in effect at the function entry decoded from
// the pcfile table, else "". File Table index 0 is never used: the table
// order follows map iteration in the compiler and carries no meaning.
func (a *logosCapAuditor) frameFileOf(s loader.Sym, fi loader.FuncInfo) string {
	if fi.NumFile() == 1 {
		if cu := a.ldr.SymUnit(s); cu != nil && int(fi.File(0)) < len(cu.FileTable) {
			return expandFile(cu.FileTable[fi.File(0)])
		}
		return ""
	}
	_, pcfile, _, _, _ := a.ldr.PcdataAuxs(s, nil)
	if pcfile == 0 {
		return ""
	}
	idx, ok := capEntryFile(a.ldr.Data(pcfile))
	if !ok {
		return ""
	}
	if cu := a.ldr.SymUnit(s); cu != nil && int(idx) < len(cu.FileTable) {
		return expandFile(cu.FileTable[idx])
	}
	return ""
}

// capEntryFile decodes the file index in effect at a function's entry
// from its pcfile table. The table is a (value, pc) pair sequence where
// the first value applies from the entry, values are varint deltas from
// -1, and pcs are uvarint deltas. The first non-negative value wins;
// anything else means no position info.
func capEntryFile(data []byte) (int32, bool) {
	val := int32(-1)
	rest := data
	for len(rest) > 0 {
		d, n := binary.Varint(rest)
		if n <= 0 {
			return 0, false
		}
		rest = rest[n:]
		val += int32(d)
		if val >= 0 {
			return val, true
		}
		if _, n := binary.Uvarint(rest); n <= 0 {
			return 0, false
		} else {
			rest = rest[n:]
		}
	}
	return 0, false
}

// capFrame is one rendered call-chain frame.
type capFrame struct {
	name string
	file string
	line int
}

// renderCapReport renders one violation in the PLAN Sec 13.2 format.
func renderCapReport(word, ruleID, dep, sink string, frames []capFrame, help string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s [capabilities::%s]: Transitive dependency '%s' reaches '%s'\n", word, ruleID, dep, sink)
	b.WriteString("  --> Reachability Call Chain (cmd/link deadcode audit):\n")
	for i, f := range frames {
		marker := ""
		if i > 0 {
			marker = "  -> "
		}
		fmt.Fprintf(&b, "   %d. %s%-40s at %s:%d\n", i+1, marker, f.name+"()", f.file, f.line)
	}
	fmt.Fprintf(&b, " = Help: %s\n", help)
	return b.String()
}

// capSinkName normalizes a linker symbol name for sink matching by
// stripping ABI wrapper suffixes: cgo trampolines and other ABI0 callers
// reach runtime.cgocall through its .abi0 wrapper, not the bare symbol.
func capSinkName(name string) string {
	if base, ok := strings.CutSuffix(name, ".abi0"); ok {
		return base
	}
	if base, ok := strings.CutSuffix(name, ".abiinternal"); ok {
		return base
	}
	return name
}

// capOwnerName strips linker metadata suffixes (stack objects, argument
// maps) to the owning function symbol name. Metadata frames attribute to
// their owner's package, never to a dotted pseudo-package.
func capOwnerName(name string) string {
	base, suffix, ok := strings.CutLast(name, ".")
	if !ok {
		return name
	}
	if suffix == "stkobj" || suffix == "args_stackmap" {
		return base
	}
	if rest, ok := strings.CutPrefix(suffix, "arginfo"); ok && capIsDigits(rest) {
		return base
	}
	return name
}

// capIsDigits reports whether s is a non-empty run of ASCII digits.
func capIsDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := range len(s) {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// capPkgOfSym returns the package path of a linker symbol name. ABI
// wrapper suffixes are stripped first so versioned symbols attribute to
// their package; package init symbols (pkg.init, pkg.init.N,
// pkg..inittask) attribute to their package.
func capPkgOfSym(name string) string {
	base, _, _ := strings.Cut(capSinkName(capOwnerName(name)), "[")
	if pkg, _, ok := strings.Cut(base, ".("); ok {
		return pkg
	}
	if pkg, ok := strings.CutSuffix(base, "..inittask"); ok {
		return pkg
	}
	parts := strings.Split(base, ".")
	n := len(parts)
	switch {
	case n >= 2 && parts[n-1] == "init":
		parts = parts[:n-1]
	case n >= 3 && parts[n-2] == "init" && capIsDigits(parts[n-1]):
		parts = parts[:n-2]
	}
	if pkg, _, ok := strings.CutLast(strings.Join(parts, "."), "."); ok {
		return pkg
	}
	return strings.Join(parts, ".")
}

// capIsStdPkg reports whether pkg is standard library, main, or another
// dotless first element (command-line-arguments, GOPATH _-paths): those
// frames never blame.
func capIsStdPkg(pkg string) bool {
	first, _, _ := strings.Cut(pkg, "/")
	return !strings.Contains(first, ".")
}

// capIsExportedMethod reports whether a method name is exported.
func capIsExportedMethod(name string) bool {
	for _, r := range name {
		return r >= 'A' && r <= 'Z'
	}
	return false
}

// capPathUnder reports whether file sits under dir.
func capPathUnder(file, dir string) bool {
	return dir != "" && (file == dir || strings.HasPrefix(file, dir+string(os.PathSeparator)))
}

// capAllowlisted reports whether dep matches one of the allowlist globs.
// A trailing "/**" matches the prefix itself and everything under it; any
// other "**" segment spans zero or more segments; remaining segments match
// without crossing slashes.
func capAllowlisted(allowlist []string, dep string) bool {
	for _, pattern := range allowlist {
		if capMatchGlob(pattern, dep) {
			return true
		}
	}
	return false
}

// capMatchGlob matches one allowlist glob against one import path.
func capMatchGlob(pattern, dep string) bool {
	if prefix, ok := strings.CutSuffix(pattern, "/**"); ok {
		if dep == prefix || strings.HasPrefix(dep, prefix+"/") {
			return true
		}
	}
	if !strings.Contains(pattern, "**") {
		ok, err := path.Match(pattern, dep)
		return err == nil && ok
	}
	return capMatchSegs(strings.Split(pattern, "/"), strings.Split(dep, "/"))
}

// capMatchSegs matches glob segments against path segments, where a "**"
// segment spans zero or more segments and every other segment matches
// exactly one path segment.
func capMatchSegs(pats, segs []string) bool {
	if len(pats) == 0 {
		return len(segs) == 0
	}
	if pats[0] == "**" {
		for i := range len(segs) + 1 {
			if capMatchSegs(pats[1:], segs[i:]) {
				return true
			}
		}
		return false
	}
	if len(segs) == 0 {
		return false
	}
	ok, err := path.Match(pats[0], segs[0])
	if err != nil || !ok {
		return false
	}
	return capMatchSegs(pats[1:], segs[1:])
}

// capSeverityDeny normalizes a severity string, failing closed to deny.
func capSeverityDeny(sev string) string {
	if sev == logosAllow || sev == logosWarn {
		return sev
	}
	return logosDeny
}
