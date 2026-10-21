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
		"syscall.Exec", "syscall.ForkExec",
		"os.StartProcess", "os.startProcess",
		"net.sysSocket",
		"runtime.cgocall", "runtime.logosCgocallFast",
	} {
		if _, ok := capabilitySinks[sink]; !ok {
			t.Errorf("capabilitySinks missing %q", sink)
		}
	}
	if capabilitySinks["os.StartProcess"] != CapExec || capabilitySinks["os.startProcess"] != CapExec {
		t.Error("exec sinks must map to CapExec")
	}
	if capabilitySinks["runtime.cgocall"] != CapCgo {
		t.Error("runtime.cgocall must map to CapCgo")
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
	}
	for _, tc := range tests {
		if got := capSinkName(tc.sym); got != tc.want {
			t.Errorf("capSinkName(%q) = %q, want %q", tc.sym, got, tc.want)
		}
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

// TestLogosCapTextPkg pins the ownership decision for symbols whose
// loader metadata and whose name disagree. The generated test main is
// compiled as package main but recorded under the test binary's import
// path ("bufio.test"), and type-descriptor glue carries a name no linked
// package owns. Only metadata naming a package the link actually loaded,
// with a matching name prefix, may own the symbol; everything else falls
// back to the name parser.
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
		// The generated test-main glue: metadata names the test binary,
		// which does not match this symbol's name prefix, so the name's
		// package decides.
		{"bufio.test", "main.main", "main"},
		{"bufio.test", "main..stmp_1", "main"},
		{"main.test", "main.main", "main"},
		// Linked metadata with a matching prefix still wins.
		{"net", "net.sysSocket", "net"},
		{"os/exec", "os/exec.(*Cmd).Start", "os/exec"},
		// Metadata naming no loaded library falls back to the name.
		{"type:noalg.struct { F uintptr; R reflect.NewType", "type:noalg.struct { F uintptr; R reflect.Type }", ""},
		{"flag.test", "flag.isZeroValue", ""},
		{"unlinked/pkg", "unlinked/pkg.Func", ""},
		{"", "type:.eq.[2]string", ""},
		// Metadata that names a linked package but not this symbol's name
		// prefix falls back to the name, which names a linked package.
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

func TestLogosCapSeverityDeny(t *testing.T) {
	if capSeverityDeny("bogus") != "deny" || capSeverityDeny("") != "deny" {
		t.Error("unknown severity must fail closed to deny")
	}
	if capSeverityDeny("warn") != "warn" || capSeverityDeny("allow") != "allow" {
		t.Error("known severities must pass through")
	}
}
