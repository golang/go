// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/goobj"
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path"
	"slices"
	"strings"
)

var flagLogosCap = flag.String("logos.cap", "", "read logos capabilities policy from `file` (logos builds only)")

type CapabilityKind uint8

const (
	CapReflect CapabilityKind = iota + 1
	CapExec
	CapNet
	CapCgo
	CapSyscall
)

var capabilitySinks = map[string]CapabilityKind{
	"reflect.ValueOf":                           CapReflect,
	"reflect.TypeOf":                            CapReflect,
	"reflect.New":                               CapReflect,
	"syscall.Exec":                              CapExec,
	"syscall.ForkExec":                          CapExec,
	"syscall.StartProcess":                      CapExec,
	"os.StartProcess":                           CapExec,
	"os.startProcess":                           CapExec,
	"net.sysSocket":                             CapNet,
	"syscall.Socket":                            CapNet,
	"syscall.Connect":                           CapNet,
	"runtime.cgocall":                           CapCgo,
	"runtime.logosCgocallFast":                  CapCgo,
	"syscall.Syscall":                           CapSyscall,
	"syscall.Syscall6":                          CapSyscall,
	"syscall.Syscall9":                          CapSyscall,
	"syscall.Syscall12":                         CapSyscall,
	"syscall.Syscall15":                         CapSyscall,
	"syscall.Syscall18":                         CapSyscall,
	"syscall.RawSyscall":                        CapSyscall,
	"syscall.RawSyscall6":                       CapSyscall,
	"syscall.SyscallN":                          CapSyscall,
	"syscall.(*Proc).Call":                      CapSyscall,
	"syscall.(*LazyProc).Call":                  CapSyscall,
	"syscall.syscall":                           CapSyscall,
	"syscall.syscallX":                          CapSyscall,
	"syscall.syscall6":                          CapSyscall,
	"syscall.syscall6X":                         CapSyscall,
	"syscall.syscall9":                          CapSyscall,
	"syscall.syscallPtr":                        CapSyscall,
	"syscall.rawSyscall":                        CapSyscall,
	"syscall.rawSyscall6":                       CapSyscall,
	"syscall.rawSyscall9":                       CapSyscall,
	"syscall.syscall10":                         CapSyscall,
	"syscall.syscall10X":                        CapSyscall,
	"syscall.rawSyscall6X":                      CapSyscall,
	"syscall.rawSyscall10X":                     CapSyscall,
	"syscall.sysvicall6":                        CapSyscall,
	"syscall.rawSysvicall6":                     CapSyscall,
	"syscall.AllThreadsSyscall":                 CapSyscall,
	"syscall.AllThreadsSyscall6":                CapSyscall,
	"syscall.runtime_doAllThreadsSyscall":       CapSyscall,
	"golang.org/x/sys/unix.Syscall":             CapSyscall,
	"golang.org/x/sys/unix.Syscall6":            CapSyscall,
	"golang.org/x/sys/unix.Syscall9":            CapSyscall,
	"golang.org/x/sys/unix.RawSyscall":          CapSyscall,
	"golang.org/x/sys/unix.RawSyscall6":         CapSyscall,
	"golang.org/x/sys/unix.SyscallNoError":      CapSyscall,
	"golang.org/x/sys/unix.RawSyscallNoError":   CapSyscall,
	"golang.org/x/sys/windows.(*Proc).Call":     CapSyscall,
	"golang.org/x/sys/windows.(*LazyProc).Call": CapSyscall,
	"golang.org/x/sys/plan9.Syscall":            CapSyscall,
	"golang.org/x/sys/plan9.Syscall6":           CapSyscall,
	"golang.org/x/sys/plan9.RawSyscall":         CapSyscall,
	"golang.org/x/sys/plan9.RawSyscall6":        CapSyscall,
}

const (
	logosDeny  = "deny"
	logosWarn  = "warn"
	logosAllow = "allow"
)

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

type logosCapImpl struct {
	name string
	typ  loader.Sym
	fn   loader.Sym
}

type logosCapHit struct {
	sink loader.Sym
	kind CapabilityKind
	name string
}

var capCgoWrapperMarks = []string{"._Cfunc_", "._C2func_"}

func capIsCgoTrampoline(name string) bool {
	for _, mark := range capCgoWrapperMarks {
		if _, after, ok := strings.Cut(name, mark); ok && after != "" {
			return true
		}
	}
	return false
}

type logosCapAuditor struct {
	ctxt   *Link
	ldr    *loader.Loader
	policy *logosCapPolicy
	std    map[string]bool
	graph  *capGraph
	roles  map[loader.Sym]capRole
	ifaced map[loader.Sym]bool
	queue  []loader.Sym
	impls  []logosCapImpl
	hits   []logosCapHit
	fanned bool
	deps   []string
}

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
	a := &logosCapAuditor{
		ctxt:   ctxt,
		ldr:    ctxt.loader,
		policy: &policy,
		std:    ctxt.loader.StdPackages(),
		graph: &capGraph{
			parent: make(map[loader.Sym]loader.Sym),
			preds:  make(map[loader.Sym][]loader.Sym),
		},
		roles:  make(map[loader.Sym]capRole),
		ifaced: make(map[loader.Sym]bool),
	}
	a.deps = parseLogosModinfoDeps(strdata["runtime.modinfo"])
	a.collectImpls(d)
	a.seedRoots(d)
	a.flood(d)
	for _, f := range a.findings() {
		a.judge(f)
	}
}

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

func (a *logosCapAuditor) seedRoots(d *deadcodePass) {
	if a.ctxt.BuildMode == BuildModeShared {
		for i, n := 1, a.ldr.NDef(); i < n; i++ {
			s := loader.Sym(i)
			if a.ldr.SymType(s).IsText() && a.ldr.SymSize(s) == 0 {
				continue
			}
			a.enqueue(s, 0)
		}
		a.enqueue(a.ctxt.mainInittasks, 0)
		return
	}
	var names []string
	if a.ctxt.linkShared && (a.ctxt.BuildMode == BuildModeExe || a.ctxt.BuildMode == BuildModePIE) {
		names = append(names, "main.main", "main..inittask")
	} else {
		names = append(names, *flagEntrySymbol)
	}
	if a.ctxt.BuildMode == BuildModePlugin {
		prefix := objabi.PathToPrefix(*flagPluginPath)
		names = append(names, prefix+"..inittask", prefix+".main", "go:plugin.tabs")
		if exports := a.ldr.Lookup("go:plugin.exports", 0); exports != 0 {
			relocs := a.ldr.Relocs(exports)
			for i := range relocs.Count() {
				a.enqueue(relocs.At(i).Sym(), 0)
			}
		}
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

func (a *logosCapAuditor) enqueue(s, p loader.Sym) {
	if s == 0 || s == p || !a.ldr.AttrReachable(s) {
		return
	}
	if a.graph.visit(s, p) {
		a.queue = append(a.queue, s)
	}
}

func (a *logosCapAuditor) flood(d *deadcodePass) {
	for len(a.queue) > 0 {
		s := a.queue[0]
		a.queue = a.queue[1:]
		name := a.ldr.SymName(s)
		base := capSinkName(name)
		if kind, ok := capabilitySinks[base]; ok {
			a.hits = append(a.hits, logosCapHit{sink: s, kind: kind, name: base})
		} else if capIsCgoTrampoline(name) && a.ldr.SymType(s).IsText() {
			a.hits = append(a.hits, logosCapHit{sink: s, kind: CapCgo, name: "runtime.cgocall"})
		}
		if a.isReflectSite(s) {
			for _, impl := range a.impls {
				if capIsExportedMethod(impl.name) {
					a.enqueue(impl.fn, s)
				}
			}
		}
		a.followRelocs(d, s)
		a.followAux(s)
		a.followInlTree(d, s)
		if a.ldr.IsExternal(s) {
			a.enqueue(a.ldr.OuterSym(s), s)
			a.enqueue(a.ldr.SubSym(s), s)
		}
	}
}

var reflectLookupSyms = []string{
	"reflect.Value.Method",
	"reflect.Value.MethodByName",
	"reflect.Type.Method",
	"reflect.Type.MethodByName",
}

func (a *logosCapAuditor) isReflectSite(s loader.Sym) bool {
	if !a.ldr.SymType(s).IsText() {
		return false
	}
	if a.ldr.IsReflectMethod(s) || a.callsReflectLookup(s) {
		return true
	}
	fi := a.ldr.FuncInfo(s)
	if fi.Valid() {
		fi.Preload()
		for i := range int(fi.NumInlTree()) {
			inlined := fi.InlTree(i).Func
			if inlined != 0 && (a.ldr.IsReflectMethod(inlined) || a.callsReflectLookup(inlined) || capIsReflectLookupName(a.ldr.SymName(inlined))) {
				return true
			}
		}
	}
	return false
}

func (a *logosCapAuditor) followInlTree(d *deadcodePass, s loader.Sym) {
	fi := a.ldr.FuncInfo(s)
	if !fi.Valid() {
		return
	}
	fi.Preload()
	ninl := int(fi.NumInlTree())
	for i := range ninl {
		node := fi.InlTree(i)
		inlined := node.Func
		if inlined == 0 {
			continue
		}
		var p loader.Sym
		if node.Parent >= 0 && int(node.Parent) < ninl {
			p = fi.InlTree(int(node.Parent)).Func
		}
		if p == 0 {
			p = s
		}
		if inlined == p {
			continue
		}
		a.graph.visit(inlined, p)
		name := a.ldr.SymName(inlined)
		base := capSinkName(name)
		if kind, ok := capabilitySinks[base]; ok {
			a.hits = append(a.hits, logosCapHit{sink: inlined, kind: kind, name: base})
		} else if capIsCgoTrampoline(name) {
			a.hits = append(a.hits, logosCapHit{sink: inlined, kind: CapCgo, name: "runtime.cgocall"})
		}
		a.followInlinedIface(d, inlined)
		if capIsReflectLookupName(name) || a.ldr.IsReflectMethod(inlined) || a.callsReflectLookup(inlined) {
			for _, impl := range a.impls {
				if capIsExportedMethod(impl.name) {
					a.enqueue(impl.fn, inlined)
				}
			}
		}
	}
}

func (a *logosCapAuditor) callsReflectLookup(s loader.Sym) bool {
	relocs := a.ldr.Relocs(s)
	for i := range relocs.Count() {
		r := relocs.At(i)
		t := r.Sym()
		if t == 0 || r.Type() == objabi.R_METHODOFF {
			continue
		}
		if capIsReflectLookupName(a.ldr.SymName(t)) {
			return true
		}
	}
	return false
}

func (a *logosCapAuditor) followRelocs(d *deadcodePass, s loader.Sym) {
	relocs := a.ldr.Relocs(s)
	site := a.inlineSite(s)
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
		case objabi.R_USEIFACEMETHOD, objabi.R_USENAMEDMETHOD:
			a.followIface(d, s, r)
			continue
		}
		a.enqueue(r.Sym(), site.at(int64(r.Off())))
	}
}

func (a *logosCapAuditor) followIface(d *deadcodePass, from loader.Sym, r loader.Reloc) {
	rs := r.Sym()
	if r.Type() == objabi.R_USENAMEDMETHOD {
		a.followIfaceImpls(from, d.decodeGenericIfaceMethod(a.ldr, rs), 0)
		return
	}
	if a.ctxt.linkShared && (a.ldr.SymType(rs) == sym.SDYNIMPORT || a.ldr.SymType(rs) == sym.Sxxx) {
		a.followExportedImpls(from)
		return
	}
	m := d.decodeIfaceMethod(a.ldr, a.ctxt.Arch, rs, r.Add())
	a.followIfaceImpls(from, m.name, m.typ)
}

func (a *logosCapAuditor) followInlinedIface(d *deadcodePass, inlined loader.Sym) {
	if a.ifaced[inlined] {
		return
	}
	a.ifaced[inlined] = true
	relocs := a.ldr.Relocs(inlined)
	for i := range relocs.Count() {
		if r := relocs.At(i); r.Type() == objabi.R_USEIFACEMETHOD || r.Type() == objabi.R_USENAMEDMETHOD {
			a.followIface(d, inlined, r)
		}
	}
}

type capInlineSite struct {
	owner   loader.Sym
	fi      loader.FuncInfo
	table   []byte
	quantum int
}

func (a *logosCapAuditor) inlineSite(s loader.Sym) capInlineSite {
	site := capInlineSite{owner: s, quantum: a.ctxt.Arch.MinLC}
	fi := a.ldr.FuncInfo(s)
	if !fi.Valid() {
		return site
	}
	fi.Preload()
	if fi.NumInlTree() == 0 {
		return site
	}
	if _, _, _, pcinline, _ := a.ldr.PcdataAuxs(s, nil); pcinline != 0 {
		site.fi = fi
		site.table = a.ldr.Data(pcinline)
	}
	return site
}

func (c *capInlineSite) at(off int64) loader.Sym {
	idx := capInlineAt(c.table, off, c.quantum)
	if idx < 0 || int(idx) >= int(c.fi.NumInlTree()) {
		return c.owner
	}
	if inlined := c.fi.InlTree(int(idx)).Func; inlined != 0 {
		return inlined
	}
	return c.owner
}

func capInlineAt(table []byte, off int64, quantum int) int32 {
	val, pc := int32(-1), int64(0)
	for len(table) > 0 {
		d, n := binary.Varint(table)
		if n <= 0 {
			return -1
		}
		table = table[n:]
		span, n := binary.Uvarint(table)
		if n <= 0 {
			return -1
		}
		table = table[n:]
		val += int32(d)
		pc += int64(span) * int64(quantum)
		if off < pc {
			return val
		}
	}
	return -1
}

func (a *logosCapAuditor) followExportedImpls(from loader.Sym) {
	if a.fanned {
		return
	}
	a.fanned = true
	for _, impl := range a.impls {
		if capIsExportedMethod(impl.name) {
			a.enqueue(impl.fn, from)
		}
	}
}

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

func (a *logosCapAuditor) followAux(s loader.Sym) {
	for i, n := 0, a.ldr.NAux(s); i < n; i++ {
		if aux := a.ldr.Aux(s, i); aux.Type() != goobj.AuxGotype {
			a.enqueue(aux.Sym(), s)
		}
	}
}

type capFinding struct {
	kind  CapabilityKind
	sink  string
	dep   string
	chain []loader.Sym
}

func (a *logosCapAuditor) findings() []capFinding {
	var out []capFinding
	index := make(map[string]int)
	for _, hit := range a.hits {
		for _, b := range a.graph.blamedChains(hit.sink, a.blameRole(hit)) {
			out = addFinding(out, index, capFinding{kind: hit.kind, sink: hit.report(), dep: b.dep, chain: b.chain})
		}
	}
	return out
}

func (h logosCapHit) report() string {
	if h.kind == CapCgo {
		return "runtime.cgocall"
	}
	return h.name
}

func (a *logosCapAuditor) directUse(hit logosCapHit) bool {
	return hit.kind == CapSyscall || hit.kind == CapCgo && a.role(hit.sink).dep == ""
}

func (a *logosCapAuditor) blameRole(hit logosCapHit) func(loader.Sym) capRole {
	if !a.directUse(hit) {
		return a.role
	}
	return func(s loader.Sym) capRole {
		r := a.role(s)
		switch {
		case s == hit.sink, a.isForwarder(s):
			return capRole{}
		case r.dep == "" && a.ldr.SymType(s).IsText():
			return capRole{wall: true}
		}
		return r
	}
}

func (a *logosCapAuditor) isForwarder(s loader.Sym) bool {
	return strings.HasSuffix(a.ldr.SymName(s), "-fm")
}

func addFinding(out []capFinding, index map[string]int, f capFinding) []capFinding {
	key := f.sink + "\x00" + f.dep
	i, seen := index[key]
	switch {
	case !seen:
		index[key] = len(out)
		return append(out, f)
	case len(f.chain) < len(out[i].chain):
		out[i] = f
	}
	return out
}

func (a *logosCapAuditor) report() {
	for _, f := range a.findings() {
		a.judge(f)
	}
}

func (a *logosCapAuditor) judge(f capFinding) {
	if f.kind == CapSyscall {
		a.judge(capFinding{kind: CapExec, sink: f.sink, dep: f.dep, chain: f.chain})
		a.judge(capFinding{kind: CapNet, sink: f.sink, dep: f.dep, chain: f.chain})
		return
	}
	if f.kind == CapCgo {
		if !slices.Contains(a.policy.CgoAllowed, f.dep) {
			a.emit("ERROR", "unauthorized_cgo", f.dep, f.sink, f.chain,
				`Remove the dependency or allow cgo for its module with [build] cgo = "allow" in logos.toml.`)
		}
		return
	}
	rule := a.ruleFor(f.kind)
	if rule.sev == logosAllow || capAllowlisted(rule.allow, f.dep) {
		return
	}
	word := "ERROR"
	if rule.sev == logosWarn {
		word = "WARNING"
	}
	if word == "WARNING" && isLogosTestBinary() {
		return
	}
	a.emit(word, rule.id, f.dep, f.sink, f.chain,
		fmt.Sprintf("Remove the dependency or explicitly authorize %q under '%s' in logos.toml.", f.dep+"/**", rule.allowKey))
}

type capRule struct {
	id       string
	sev      string
	allowKey string
	allow    []string
}

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

func (a *logosCapAuditor) role(s loader.Sym) capRole {
	if r, ok := a.roles[s]; ok {
		return r
	}
	r := a.roleOf(s)
	a.roles[s] = r
	return r
}

func (a *logosCapAuditor) roleOf(s loader.Sym) capRole {
	if !a.ldr.SymType(s).IsText() {
		return capRole{}
	}
	pkg := a.textPkg(s)
	switch {
	case pkg == "runtime":
		return capRole{wall: true}
	case pkg == "" || a.exemptPkg(pkg):
		return capRole{}
	}
	return capRole{dep: pkg}
}

func (a *logosCapAuditor) textPkg(s loader.Sym) string {
	name := a.ldr.SymName(s)
	if pkg := a.ldr.SymPkg(s); pkg != "" && a.ctxt.LibraryByPkg[pkg] != nil && strings.HasPrefix(name, objabi.PathToPrefix(pkg)+".") {
		return pkg
	}
	return a.namedPkg(name)
}

func (a *logosCapAuditor) namedPkg(name string) string {
	base, _, _ := strings.Cut(name, "[")
	elem := base[strings.LastIndexByte(base, '/')+1:]
	dot := strings.IndexByte(elem, '.')
	if dot < 0 {
		return ""
	}
	pkg, err := objabi.PrefixToPath(base[:len(base)-len(elem)+dot])
	if err != nil || a.ctxt.LibraryByPkg[pkg] == nil {
		return ""
	}
	return pkg
}

func (a *logosCapAuditor) exemptPkg(pkg string) bool {
	if capIsLocalPkg(pkg) || a.std[pkg] {
		return true
	}
	return a.mainOwns(pkg)
}

func (a *logosCapAuditor) mainOwns(pkg string) bool {
	owner := ""
	for _, m := range a.policy.MainModules {
		if m != "" && (pkg == m || strings.HasPrefix(pkg, m+"/")) && len(m) > len(owner) {
			owner = m
		}
	}
	if owner == "" {
		return false
	}
	for _, d := range a.deps {
		if d != "" && (pkg == d || strings.HasPrefix(pkg, d+"/")) && len(d) > len(owner) {
			return false
		}
	}
	return true
}

func parseLogosModinfoDeps(blob string) []string {
	var out []string
	for _, line := range strings.Split(blob, "\n") {
		f := strings.Split(line, "\t")
		if len(f) < 2 || f[0] != "dep" || f[1] == "" {
			continue
		}
		out = append(out, f[1])
	}
	return out
}

func isLogosTestBinary() bool {
	return strdata["testing.testBinary"] == "1"
}

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

type capFrame struct {
	name string
	file string
	line int
}

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

func capSinkName(name string) string {
	for {
		next := logosCloneBaseName(name)
		if base, ok := strings.CutSuffix(next, ".abi0"); ok {
			next = base
		} else if base, ok := strings.CutSuffix(next, ".abiinternal"); ok {
			next = base
		}
		if next == name {
			return name
		}
		name = next
	}
}

func capIsReflectLookupName(name string) bool {
	return slices.Contains(reflectLookupSyms, capSinkName(name))
}

func capIsLocalPkg(pkg string) bool {
	return pkg == "main" || pkg == "command-line-arguments" || strings.HasPrefix(pkg, "_/")
}

func capIsExportedMethod(name string) bool {
	for _, r := range name {
		return r >= 'A' && r <= 'Z'
	}
	return false
}

func capAllowlisted(allowlist []string, dep string) bool {
	for _, pattern := range allowlist {
		if capMatchGlob(pattern, dep) {
			return true
		}
	}
	return false
}

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

func capSeverityDeny(sev string) string {
	if sev == logosAllow || sev == logosWarn {
		return sev
	}
	return logosDeny
}
