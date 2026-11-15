// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"encoding/json"
	"strings"
	"testing"
)

func TestLogosCapSinks(t *testing.T) {
	for _, sink := range []string{
		"reflect.ValueOf", "reflect.TypeOf", "reflect.New",
		"syscall.Exec", "syscall.ForkExec", "syscall.StartProcess",
		"os.StartProcess", "os.startProcess",
		"net.sysSocket", "syscall.Socket", "syscall.Connect",
		"runtime.cgocall", "runtime.logosCgocallFast",
	} {
		if _, ok := capabilitySinks[sink]; !ok {
			t.Errorf("capabilitySinks missing %q", sink)
		}
	}
	for _, sink := range []string{
		"syscall.Exec", "syscall.ForkExec", "syscall.StartProcess",
		"os.StartProcess", "os.startProcess",
	} {
		if capabilitySinks[sink] != CapExec {
			t.Errorf("process sink %q must map to CapExec", sink)
		}
	}
	if capabilitySinks["syscall.Socket"] != CapNet || capabilitySinks["syscall.Connect"] != CapNet {
		t.Error("net sinks must map to CapNet")
	}
	if capabilitySinks["runtime.cgocall"] != CapCgo {
		t.Error("runtime.cgocall must map to CapCgo")
	}
	for _, sink := range []string{
		"syscall.Syscall", "syscall.Syscall6", "syscall.Syscall9",
		"syscall.Syscall12", "syscall.Syscall15", "syscall.Syscall18",
		"syscall.RawSyscall", "syscall.RawSyscall6", "syscall.SyscallN",
		"syscall.(*Proc).Call", "syscall.(*LazyProc).Call",
		"syscall.syscall", "syscall.syscallX", "syscall.syscall6", "syscall.syscall6X",
		"syscall.syscall9", "syscall.syscallPtr", "syscall.rawSyscall",
		"syscall.rawSyscall6", "syscall.rawSyscall9",
		"golang.org/x/sys/unix.Syscall", "golang.org/x/sys/unix.Syscall6",
		"golang.org/x/sys/unix.Syscall9", "golang.org/x/sys/unix.RawSyscall",
		"golang.org/x/sys/unix.RawSyscall6", "golang.org/x/sys/unix.SyscallNoError",
		"golang.org/x/sys/unix.RawSyscallNoError",
		"golang.org/x/sys/windows.(*Proc).Call", "golang.org/x/sys/windows.(*LazyProc).Call",
		"golang.org/x/sys/plan9.Syscall", "golang.org/x/sys/plan9.Syscall6",
		"golang.org/x/sys/plan9.RawSyscall", "golang.org/x/sys/plan9.RawSyscall6",
	} {
		if capabilitySinks[sink] != CapSyscall {
			t.Errorf("raw syscall entry %q must map to CapSyscall", sink)
		}
	}
}

func TestLogosCapDenyDecision(t *testing.T) {
	a := &logosCapAuditor{
		policy: &logosCapPolicy{
			TransitiveReflect: "warn",
			TransitiveExec:    "deny",
			TransitiveNet:     "deny",
			MainModules:       []string{"example.com/main"},
		},
	}
	for sink, wantID := range map[string]string{
		"syscall.Connect":      "unauthorized_net",
		"syscall.StartProcess": "unauthorized_exec",
	} {
		kind, ok := capabilitySinks[sink]
		if !ok {
			t.Fatalf("capabilitySinks missing %q", sink)
		}
		rule := a.ruleFor(kind)
		if rule.sev != logosDeny || rule.id != wantID {
			t.Errorf("ruleFor(%v) = %+v, want deny %q", kind, rule, wantID)
		}
		if dep := "example.com/ndep"; a.exemptPkg(dep) {
			t.Errorf("exemptPkg(%q) = true, want false", dep)
		} else if capAllowlisted(rule.allow, dep) {
			t.Errorf("capAllowlisted(%q) = true, want false", dep)
		}
	}
}

func TestLogosCapPolicyJSON(t *testing.T) {
	const doc = `{"transitive_reflect":"warn","transitive_exec":"deny","transitive_net":"allow",` +
		`"reflect_allowlist":["example.com/rt/**"],"transitive_exec_allowlist":[],` +
		`"transitive_net_allowlist":null,"main_modules":["example.com/main"],` +
		`"cgo_allowed":["example.com/cdep"]}`
	var p logosCapPolicy
	if err := json.Unmarshal([]byte(doc), &p); err != nil {
		t.Fatalf("Unmarshal policy: %v", err)
	}
	if p.TransitiveExec != "deny" || p.TransitiveReflect != "warn" || p.TransitiveNet != "allow" {
		t.Errorf("severities = %+v", p)
	}
	if len(p.ReflectAllowlist) != 1 || len(p.MainModules) != 1 || len(p.CgoAllowed) != 1 {
		t.Errorf("lists = %+v", p)
	}
}

func TestLogosCapTrampoline(t *testing.T) {
	for _, sym := range []string{
		"example.com/cdep._Cfunc_fortytwo",
		"example.com/cdep._Cfunc_fortytwo.abi0",
		"main._Cfunc_f",
	} {
		if !capIsCgoTrampoline(sym) {
			t.Errorf("capIsCgoTrampoline(%q) = false, want true", sym)
		}
	}
	for _, sym := range []string{
		"example.com/cdep._Ctype_int",
		"example.com/cdep._Cvar_x",
		"example.com/cdep.Ping",
		"runtime.cgocall",
	} {
		if capIsCgoTrampoline(sym) {
			t.Errorf("capIsCgoTrampoline(%q) = true, want false", sym)
		}
	}
}

func TestLogosCapSinkName(t *testing.T) {
	tests := []struct{ sym, want string }{
		{"runtime.cgocall", "runtime.cgocall"},
		{"runtime.cgocall.abi0", "runtime.cgocall"},
		{"os.StartProcess", "os.StartProcess"},
		{"example.com/dep.Init", "example.com/dep.Init"},
		{"syscall.Connect·logosct12", "syscall.Connect"},
		{"syscall.Connect·logosct1.abi0", "syscall.Connect"},
		{"example.com/dep.Foo_logosct1", "example.com/dep.Foo"},
		{"example.com/dep.Bar_logosct1.abi0", "example.com/dep.Bar"},
		{"reflect.Value.Method·logosct1", "reflect.Value.Method"},
	}
	for _, tc := range tests {
		if got := capSinkName(tc.sym); got != tc.want {
			t.Errorf("capSinkName(%q) = %q, want %q", tc.sym, got, tc.want)
		}
	}
}

func TestLogosCapReflectLookupClone(t *testing.T) {
	for _, sym := range []string{
		"reflect.Value.Method",
		"reflect.Value.Method·logosct1",
		"reflect.Value.MethodByName·logosct2",
		"reflect.Type.Method.abi0",
	} {
		if !capIsReflectLookupName(sym) {
			t.Errorf("capIsReflectLookupName(%q) = false, want true", sym)
		}
	}
	if capIsReflectLookupName("example.com/dep.Foo·logosct1") {
		t.Error("capIsReflectLookupName(dep clone) = true, want false")
	}
}

func TestLogosCapLocalPkg(t *testing.T) {
	for _, pkg := range []string{"main", "command-line-arguments", "_/home/dev/proj"} {
		if !capIsLocalPkg(pkg) {
			t.Errorf("capIsLocalPkg(%q) = false, want true", pkg)
		}
	}
	for _, pkg := range []string{"os/exec", "mylib", "example.com/dep", "maintenance", "_dep"} {
		if capIsLocalPkg(pkg) {
			t.Errorf("capIsLocalPkg(%q) = true, want false", pkg)
		}
	}
}

func TestLogosCapExemptPkg(t *testing.T) {
	a := &logosCapAuditor{
		std:    map[string]bool{"os/exec": true, "runtime": true, "vendor/golang.org/x/net": true},
		policy: &logosCapPolicy{MainModules: []string{"example.com/app"}},
	}
	for _, pkg := range []string{
		"os/exec", "runtime", "vendor/golang.org/x/net",
		"main", "command-line-arguments",
		"example.com/app", "example.com/app/internal/util",
	} {
		if !a.exemptPkg(pkg) {
			t.Errorf("exemptPkg(%q) = false, want true", pkg)
		}
	}
	for _, pkg := range []string{
		"mylib", "mylib/sub", "example.com/dep", "example.com/application",
		"github.com/evil/logger", "os/execx", "golang.org/x/net",
	} {
		if a.exemptPkg(pkg) {
			t.Errorf("exemptPkg(%q) = true, want false", pkg)
		}
	}
}

func TestLogosCapExemptPkgNestedModule(t *testing.T) {
	const blob = "path\tcommand-line-arguments\n" +
		"mod\texample.com/app\tv0.0.0\n" +
		"dep\texample.com/app/nested\tv1.2.3\th1:abc=\n"
	a := &logosCapAuditor{
		policy: &logosCapPolicy{
			MainModules: []string{"example.com/app"},
		},
		deps: parseLogosModinfoDeps(blob),
	}
	for _, pkg := range []string{
		"example.com/app",
		"example.com/app/internal/util",
	} {
		if !a.exemptPkg(pkg) {
			t.Errorf("exemptPkg(%q) = false, want true (ordinary subpackage stays exempt)", pkg)
		}
	}
	for _, pkg := range []string{
		"example.com/app/nested",
		"example.com/app/nested/sub",
	} {
		if a.exemptPkg(pkg) {
			t.Errorf("exemptPkg(%q) = true, want false (nested module is a dependency)", pkg)
		}
	}
	w := &logosCapAuditor{
		policy: &logosCapPolicy{
			MainModules: []string{"example.com/app", "example.com/app/nested"},
		},
		deps: parseLogosModinfoDeps(blob),
	}
	if !w.exemptPkg("example.com/app/nested") {
		t.Error(`exemptPkg("example.com/app/nested") = false, want true (nested main module stays exempt)`)
	}
	s := &logosCapAuditor{
		policy: &logosCapPolicy{
			MainModules: []string{"example.com/app", "example.com/app/nested/deep"},
		},
		deps: parseLogosModinfoDeps(blob),
	}
	if !s.exemptPkg("example.com/app/nested/deep") {
		t.Error(`exemptPkg("example.com/app/nested/deep") = false, want true (longer main prefix wins)`)
	}
	if s.exemptPkg("example.com/app/nested/other") {
		t.Error(`exemptPkg("example.com/app/nested/other") = true, want false (nested module is a dependency)`)
	}
}

func TestLogosCapModinfoDeps(t *testing.T) {
	blob := "\x30\x77\xaf\x0c\x92\x74\x08\x02\x41\xe1\xc1\x07\xe6\xd6\x18\xe6" +
		"path\tcommand-line-arguments\n" +
		"mod\texample.com/app\tv0.0.0\n" +
		"dep\texample.com/app/nested\tv1.2.3\th1:abc=\n" +
		"dep\tgithub.com/other/lib\tv0.9.0\n" +
		"=>\tgithub.com/other/lib\tv0.9.1\n" +
		"build\tbuildmode=exe\n" +
		"\xf9\x32\x43\x31\x86\x18\x20\x72\x00\x82\x42\x10\x41\x16\xd8\xf2"
	got := parseLogosModinfoDeps(blob)
	want := []string{"example.com/app/nested", "github.com/other/lib"}
	if len(got) != len(want) {
		t.Fatalf("parseLogosModinfoDeps = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("parseLogosModinfoDeps()[%d] = %q, want %q", i, got[i], want[i])
		}
	}
	if len(parseLogosModinfoDeps("")) != 0 {
		t.Error("empty blob must yield no dep modules")
	}
}

func TestLogosCapTextPkg(t *testing.T) {
	libs := make(map[string]*sym.Library)
	for _, pkg := range []string{"main", "bufio.test", "net", "os/exec"} {
		libs[pkg] = &sym.Library{Pkg: pkg}
	}
	er := loader.ErrorReporter{}
	a := &logosCapAuditor{
		ctxt: &Link{LibraryByPkg: libs},
		ldr:  loader.NewLoader(0, &er),
	}
	tests := []struct {
		symPkg, symName, want string
	}{
		{"bufio.test", "main.main", "main"},
		{"bufio.test", "main..stmp_1", "main"},
		{"main.test", "main.main", "main"},
		{"net", "net.sysSocket", "net"},
		{"os/exec", "os/exec.(*Cmd).Start", "os/exec"},
		{"type:noalg.struct { F uintptr; R reflect.NewType", "type:noalg.struct { F uintptr; R reflect.Type }", ""},
		{"flag.test", "flag.isZeroValue", ""},
		{"unlinked/pkg", "unlinked/pkg.Func", ""},
		{"", "type:.eq.[2]string", ""},
		{"net", "os/exec.Command", "os/exec"},
	}
	for _, tc := range tests {
		s := a.ldr.LookupOrCreateSym(tc.symName, 0)
		a.ldr.SetSymPkg(s, tc.symPkg)
		if got := a.textPkg(s); got != tc.want {
			t.Errorf("textPkg(SymPkg=%q, SymName=%q) = %q, want %q", tc.symPkg, tc.symName, got, tc.want)
		}
	}
}

func TestLogosCapNamedPkg(t *testing.T) {
	libs := make(map[string]*sym.Library)
	for _, pkg := range []string{"main", "encoding", "os/exec", "mylib", "example.com/dep", "gopkg.in/yaml.v2"} {
		libs[pkg] = &sym.Library{Pkg: pkg}
	}
	a := &logosCapAuditor{ctxt: &Link{LibraryByPkg: libs}}
	tests := []struct{ name, want string }{
		{"main.main", "main"},
		{"os/exec.Command", "os/exec"},
		{"os/exec.(*Cmd).Start", "os/exec"},
		{"mylib.Init", "mylib"},
		{"example.com/dep.Init", "example.com/dep"},
		{"example.com/dep.Each[go.shape.int]", "example.com/dep"},
		{"example.com/dep.Each[example.com/other.T]", "example.com/dep"},
		{"example.com/dep.T.Run", "example.com/dep"},
		{"gopkg.in/yaml%2ev2.Unmarshal", "gopkg.in/yaml.v2"},
		{"encoding.TextAppender.AppendText-fm", "encoding"},
		{"go:(*struct { encoding.TextAppender }).AppendText", ""},
		{"go:struct { encoding.TextMarshaler }", ""},
		{"type:.eq.[2]string", ""},
		{"unlinked.Func", ""},
		{"_rt0_amd64_linux", ""},
		{"example.com/dep%zz.Init", ""},
	}
	for _, tc := range tests {
		if got := a.namedPkg(tc.name); got != tc.want {
			t.Errorf("namedPkg(%q) = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestLogosCapAllowlisted(t *testing.T) {
	if !capAllowlisted([]string{"example.com/dep/**"}, "example.com/dep/sub") {
		t.Error("suffix glob must match child")
	}
	if capAllowlisted([]string{"example.com/dep/**"}, "example.com/other") {
		t.Error("suffix glob must reject other")
	}
	if !capAllowlisted([]string{"example.com/**/tail"}, "example.com/a/tail") {
		t.Error("middle doublestar must span")
	}
}

func TestLogosCapRender(t *testing.T) {
	frames := []capFrame{
		{name: "main.main", file: "example.com/main/cmd/main.lgo", line: 9},
		{name: "example.com/dep.Init", file: "example.com/dep/dep.go", line: 7},
		{name: "os.startProcess", file: "GOROOT/src/os/exec_posix.go", line: 26},
	}
	got := renderCapReport("ERROR", "unauthorized_exec", "example.com/dep", "os.startProcess", frames, "help text")
	for _, want := range []string{
		"ERROR [capabilities::unauthorized_exec]: Transitive dependency 'example.com/dep' reaches 'os.startProcess'",
		"1. main.main()",
		"2.   -> example.com/dep.Init()",
		"3.   -> os.startProcess()",
		"example.com/main/cmd/main.lgo:9",
		"GOROOT/src/os/exec_posix.go:26",
		"Reachability Call Chain (cmd/link deadcode audit)",
		" = Help: help text",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("render missing %q:\n%s", want, got)
		}
	}
}

func TestLogosCapEntryFile(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want int32
		ok   bool
	}{
		{"atext then file 2", []byte{0x00, 0x01, 0x06}, 2, true},
		{"immediate file 0", []byte{0x02}, 0, true},
		{"immediate file 1", []byte{0x04}, 1, true},
		{"empty", nil, 0, false},
		{"value without pc", []byte{0x00}, 0, false},
		{"truncated varint", []byte{0xff}, 0, false},
		{"all negative", []byte{0x00, 0x01, 0x00, 0x01, 0x00}, 0, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := capEntryFile(tc.data)
			if ok != tc.ok || (ok && got != tc.want) {
				t.Errorf("capEntryFile(%x) = (%d, %v), want (%d, %v)", tc.data, got, ok, tc.want, tc.ok)
			}
		})
	}
}

func TestLogosCapInlineAt(t *testing.T) {
	table := []byte{0x02, 0x04, 0x01, 0x02, 0x04, 0x03, 0x00}
	tests := []struct {
		name    string
		table   []byte
		off     int64
		quantum int
		want    int32
	}{
		{"first range start", table, 0, 1, 0},
		{"first range end", table, 3, 1, 0},
		{"outside any inlined call", table, 4, 1, -1},
		{"second inlined call", table, 6, 1, 1},
		{"last byte of table", table, 8, 1, 1},
		{"past the table", table, 9, 1, -1},
		{"scaled by the pc quantum", table, 15, 4, 0},
		{"scaled boundary", table, 16, 4, -1},
		{"empty table", nil, 0, 1, -1},
		{"truncated span", []byte{0x02}, 0, 1, -1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := capInlineAt(tc.table, tc.off, tc.quantum); got != tc.want {
				t.Errorf("capInlineAt(%x, %d, %d) = %d, want %d", tc.table, tc.off, tc.quantum, got, tc.want)
			}
		})
	}
}

func TestLogosCapSeverityDeny(t *testing.T) {
	if capSeverityDeny("bogus") != "deny" || capSeverityDeny("") != "deny" {
		t.Error("unknown severity must fail closed to deny")
	}
	if capSeverityDeny("warn") != "warn" || capSeverityDeny("allow") != "allow" {
		t.Error("known severities must pass through")
	}
}
