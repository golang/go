// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

import (
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// M0-S4 attack: table tests for valid logos.toml documents.
// Every error reads <name>:<line>:<col>: <message> where <name> is the
// name argument passed to Parse.

var fourExts = []string{".lgo", ".📜", ".⚡", ".ʕ◔ϖ◔ʔ"}

type validCase struct {
	name string
	data string
	want File
}

var validDocs = []validCase{
	{
		name: "empty file yields defaults",
		data: "",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "basic string",
		data: "[project]\nname = \"myapp\"\n",
		want: File{
			Project: Project{Name: "myapp", Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "literal string",
		data: "[project]\nname = 'myapp'\n",
		want: File{
			Project: Project{Name: "myapp", Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "comments are ignored",
		data: "# leading comment\n[project] # table comment\n# full line\nname = \"m\" # trailing\n",
		want: File{
			Project: Project{Name: "m", Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "crlf line endings",
		data: "[project]\r\nname = \"m\"\r\n",
		want: File{
			Project: Project{Name: "m", Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "string arrays",
		data: "[project]\nstrict_files = [\".lgo\", \".⚡\"]\n[build]\ncgo_allowlist = [\"example.com/m/ok/**\"]\n",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: []string{".lgo", ".⚡"}},
			Build:   Build{Cgo: "forbid", CgoAllowlist: []string{"example.com/m/ok/**"}, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "trailing comma in array",
		data: "[project]\nstrict_files = [\n\".lgo\",\n\".⚡\",\n]\n",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: []string{".lgo", ".⚡"}},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "absent project table yields project defaults",
		data: "[build]\ncgo = \"allow\"\n",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "allow", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "absent build table yields build defaults",
		data: "[project]\nname = \"m\"\n",
		want: File{
			Project: Project{Name: "m", Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "explicit cgo forbid tracks presence",
		data: "[build]\ncgo = \"forbid\"\n",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "explicit cgo restricted tracks presence",
		data: "[build]\ncgo = \"restricted\"\n",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "restricted", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "explicit cgo allow tracks presence",
		data: "[build]\ncgo = \"allow\"\n",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "allow", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "opt level, synthetic pgo, and target cpu",
		data: "[build]\nopt_level = \"O3\"\nsynthetic_pgo = \"aggressive\"\ntarget_cpu = \"native\"\n",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O3", SyntheticPGO: "aggressive", SynthPGOSet: true, TargetCPU: "native"},
		},
	},
	{
		name: "synthetic pgo off tracks presence",
		data: "[build]\nsynthetic_pgo = \"off\"\n",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", SyntheticPGO: "off", SynthPGOSet: true, TargetCPU: "deterministic"},
		},
	},
	{
		name: "full document",
		data: "[project]\nname = \"m\"\nedition = \"2026\"\nstrict_files = [\".lgo\", \".📜\", \".⚡\", \".ʕ◔ϖ◔ʔ\"]\n[build]\ncgo = \"restricted\"\ncgo_allowlist = [\"example.com/m/ok/**\"]\n",
		want: File{
			Project: Project{Name: "m", Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "restricted", CgoSet: true, CgoAllowlist: []string{"example.com/m/ok/**"}, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "capabilities severities",
		data: "[capabilities]\ndirect_reflect = \"allow\"\ntransitive_reflect = \"deny\"\ntransitive_exec = \"warn\"\ntransitive_net = \"allow\"\ntransitive_unsafe = \"warn\"\n",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
			Capabilities: Capabilities{
				DirectReflect:     "allow",
				TransitiveReflect: "deny",
				TransitiveExec:    "warn",
				TransitiveNet:     "allow",
				TransitiveUnsafe:  "warn",
			},
		},
	},
	{
		name: "capabilities allowlists",
		data: "[capabilities]\nreflect_allowlist = [\"example.com/rt/**\"]\ntransitive_exec_allowlist = [\"example.com/exec\"]\ntransitive_net_allowlist = [\"example.com/net/**\"]\ntransitive_unsafe_allowlist = [\"example.com/dep/**\", \"example.com/other\"]\n",
		want: File{
			Project: Project{Edition: "2026", StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
			Capabilities: Capabilities{
				DirectReflect:             "deny",
				TransitiveReflect:         "warn",
				TransitiveExec:            "deny",
				TransitiveNet:             "deny",
				TransitiveUnsafe:          "deny",
				ReflectAllowlist:          []string{"example.com/rt/**"},
				TransitiveExecAllowlist:   []string{"example.com/exec"},
				TransitiveNetAllowlist:    []string{"example.com/net/**"},
				TransitiveUnsafeAllowlist: []string{"example.com/dep/**", "example.com/other"},
			},
		},
	},
}

func checkCapabilities(t *testing.T, got, want Capabilities) {
	t.Helper()
	if got.DirectReflect != want.DirectReflect {
		t.Errorf("DirectReflect = %q, want %q", got.DirectReflect, want.DirectReflect)
	}
	if got.TransitiveReflect != want.TransitiveReflect {
		t.Errorf("TransitiveReflect = %q, want %q", got.TransitiveReflect, want.TransitiveReflect)
	}
	if got.TransitiveExec != want.TransitiveExec {
		t.Errorf("TransitiveExec = %q, want %q", got.TransitiveExec, want.TransitiveExec)
	}
	if got.TransitiveNet != want.TransitiveNet {
		t.Errorf("TransitiveNet = %q, want %q", got.TransitiveNet, want.TransitiveNet)
	}
	if got.TransitiveUnsafe != want.TransitiveUnsafe {
		t.Errorf("TransitiveUnsafe = %q, want %q", got.TransitiveUnsafe, want.TransitiveUnsafe)
	}
	if !slices.Equal(got.ReflectAllowlist, want.ReflectAllowlist) {
		t.Errorf("ReflectAllowlist = %q, want %q", got.ReflectAllowlist, want.ReflectAllowlist)
	}
	if !slices.Equal(got.TransitiveExecAllowlist, want.TransitiveExecAllowlist) {
		t.Errorf("TransitiveExecAllowlist = %q, want %q", got.TransitiveExecAllowlist, want.TransitiveExecAllowlist)
	}
	if !slices.Equal(got.TransitiveNetAllowlist, want.TransitiveNetAllowlist) {
		t.Errorf("TransitiveNetAllowlist = %q, want %q", got.TransitiveNetAllowlist, want.TransitiveNetAllowlist)
	}
	if !slices.Equal(got.TransitiveUnsafeAllowlist, want.TransitiveUnsafeAllowlist) {
		t.Errorf("TransitiveUnsafeAllowlist = %q, want %q", got.TransitiveUnsafeAllowlist, want.TransitiveUnsafeAllowlist)
	}
}

func defaultCapabilities() Capabilities {
	return Capabilities{
		DirectReflect:     "deny",
		TransitiveReflect: "warn",
		TransitiveExec:    "deny",
		TransitiveNet:     "deny",
		TransitiveUnsafe:  "deny",
	}
}

// withCapDefaults fills unset severity fields with their defaults, so
// cases that only set allowlists still expect the default severities.
func withCapDefaults(c Capabilities) Capabilities {
	d := defaultCapabilities()
	if c.DirectReflect == "" {
		c.DirectReflect = d.DirectReflect
	}
	if c.TransitiveReflect == "" {
		c.TransitiveReflect = d.TransitiveReflect
	}
	if c.TransitiveExec == "" {
		c.TransitiveExec = d.TransitiveExec
	}
	if c.TransitiveNet == "" {
		c.TransitiveNet = d.TransitiveNet
	}
	if c.TransitiveUnsafe == "" {
		c.TransitiveUnsafe = d.TransitiveUnsafe
	}
	return c
}

func checkFile(t *testing.T, got *File, want File) {
	t.Helper()
	want.Capabilities = withCapDefaults(want.Capabilities)
	checkCapabilities(t, got.Capabilities, want.Capabilities)
	if got.Project.Name != want.Project.Name {
		t.Errorf("Project.Name = %q, want %q", got.Project.Name, want.Project.Name)
	}
	if got.Project.Edition != want.Project.Edition {
		t.Errorf("Project.Edition = %q, want %q", got.Project.Edition, want.Project.Edition)
	}
	if !reflect.DeepEqual(got.Project.StrictFiles, want.Project.StrictFiles) {
		t.Errorf("Project.StrictFiles = %q, want %q", got.Project.StrictFiles, want.Project.StrictFiles)
	}
	if got.Build.Cgo != want.Build.Cgo {
		t.Errorf("Build.Cgo = %q, want %q", got.Build.Cgo, want.Build.Cgo)
	}
	if got.Build.CgoSet != want.Build.CgoSet {
		t.Errorf("Build.CgoSet = %v, want %v", got.Build.CgoSet, want.Build.CgoSet)
	}
	if !slices.Equal(got.Build.CgoAllowlist, want.Build.CgoAllowlist) {
		t.Errorf("Build.CgoAllowlist = %q, want %q", got.Build.CgoAllowlist, want.Build.CgoAllowlist)
	}
	if got.Build.OptLevel != want.Build.OptLevel {
		t.Errorf("Build.OptLevel = %q, want %q", got.Build.OptLevel, want.Build.OptLevel)
	}
	if got.Build.SyntheticPGO != want.Build.SyntheticPGO {
		t.Errorf("Build.SyntheticPGO = %q, want %q", got.Build.SyntheticPGO, want.Build.SyntheticPGO)
	}
	if got.Build.SynthPGOSet != want.Build.SynthPGOSet {
		t.Errorf("Build.SynthPGOSet = %v, want %v", got.Build.SynthPGOSet, want.Build.SynthPGOSet)
	}
	if got.Build.TargetCPU != want.Build.TargetCPU {
		t.Errorf("Build.TargetCPU = %q, want %q", got.Build.TargetCPU, want.Build.TargetCPU)
	}
}

func TestParseValid(t *testing.T) {
	for _, tc := range validDocs {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Parse("logos.toml", []byte(tc.data))
			if err != nil {
				t.Fatalf("Parse(%q) error = %v", tc.data, err)
			}
			checkFile(t, got, tc.want)
		})
	}
}

func TestParseViolations(t *testing.T) {
	tests := []struct {
		name    string
		data    string
		wantErr string
	}{
		{"duplicate key", "[project]\nname = \"a\"\nname = \"b\"\n", `logos.toml:3:1: duplicate key "name"`},
		{"root key rejected", "name = \"x\"\n", `logos.toml:1:1: unknown key "name"`},
		{"root build key rejected", "cgo = \"allow\"\n", `logos.toml:1:1: unknown key "cgo"`},
		{"misplaced project key", "[build]\nname = \"x\"\n", `logos.toml:2:1: unknown key "name"`},
		{"duplicate table", "[project]\nname = \"a\"\n[project]\n", `logos.toml:3:1: duplicate table "project"`},
		{"unknown table", "[frobnicate]\n", `logos.toml:1:1: unknown table "frobnicate"`},
		{"unknown key", "[project]\nbogus = \"x\"\n", `logos.toml:2:1: unknown key "bogus"`},
		{"wrong value type", "[project]\nname = 42\n", `logos.toml:2:8: want string for key "name"`},
		{"wrong value type boolean", "[project]\nname = true\n", `logos.toml:2:8: want string for key "name"`},
		{"unterminated string", "[project]\nname = \"abc\n", `logos.toml:2:8: unterminated string`},
		{"invalid escape", "[project]\nname = \"a\\qb\"\n", `logos.toml:2:10: invalid escape 'q'`},
		{"nul in comment", "[project]\n# hi\x00\nname = \"m\"\n", `logos.toml:2:5: invalid control character U+0000`},
		{"nul in string", "[project]\nname = \"a\x00b\"\n", `logos.toml:2:10: invalid control character U+0000`},
		{"del rejected", "[project]\nname = \"m\"\x7f\n", `logos.toml:2:11: invalid control character U+007F`},
		{"lone cr rejected", "[project]\rname = \"m\"\n", `logos.toml:1:10: invalid control character U+000D`},
		{"non-utf8 byte", "[project]\nname = \"a\xff\"\n", `logos.toml:2:10: invalid utf-8`},
		{"inline table", "[project]\nname = {first = \"a\"}\n", `logos.toml:2:8: inline table not allowed`},
		{"float", "[project]\nname = 1.5\n", `logos.toml:2:8: float not allowed`},
		{"date", "[project]\nname = 1979-05-27\n", `logos.toml:2:8: date not allowed`},
		{"invalid cgo value", "[build]\ncgo = \"sometimes\"\n", `logos.toml:2:7: invalid cgo value "sometimes" (want forbid, restricted, or allow)`},
		{"invalid capabilities severity", "[capabilities]\ntransitive_unsafe = \"forbid\"\n", `logos.toml:2:21: invalid transitive_unsafe value "forbid" (want deny, warn, or allow)`},
		{"capabilities key in build table", "[build]\ntransitive_unsafe = \"deny\"\n", `logos.toml:2:1: unknown key "transitive_unsafe"`},
		{"build key in capabilities table", "[capabilities]\ncgo = \"allow\"\n", `logos.toml:2:1: unknown key "cgo"`},
		{"invalid capabilities allowlist pattern", "[capabilities]\ntransitive_unsafe_allowlist = [\"[\"]\n", `logos.toml:2:32: [configuration::invalid_capabilities_allowlist] transitive_unsafe_allowlist pattern "[": syntax error in pattern`},
		{"invalid strict_files entry", "[project]\nstrict_files = [\".go\"]\n", `logos.toml:2:17: invalid strict_files entry ".go" (want ".lgo", ".📜", ".⚡", or ".ʕ◔ϖ◔ʔ")`},
		{"empty project name", "[project]\nname = \"\"\n", `logos.toml:2:8: project name must not be empty`},
		{"unsupported edition", "[project]\nedition = \"2027\"\n", `logos.toml:2:11: unsupported edition "2027" (want "2026")`},
		{"bare opt level rejected", "[build]\nopt_level = \"3\"\n", `logos.toml:2:13: opt_level must be one of O0, O1, O2, O3, Os, Oz`},
		{"unknown opt level rejected", "[build]\nopt_level = \"O4\"\n", `logos.toml:2:13: opt_level must be one of O0, O1, O2, O3, Os, Oz`},
		{"invalid synthetic pgo", "[build]\nsynthetic_pgo = \"turbo\"\n", `logos.toml:2:17: synthetic_pgo must be one of off, standard, aggressive`},
		{"invalid target cpu", "[build]\ntarget_cpu = \"laptop\"\n", `logos.toml:2:14: target_cpu must be one of deterministic, baseline, native`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Parse("logos.toml", []byte(tc.data))
			if err == nil {
				t.Fatalf("Parse(%q) succeeded, want error %q (got %+v)", tc.data, tc.wantErr, got)
			}
			if err.Error() != tc.wantErr {
				t.Errorf("Parse(%q) error = %q, want %q", tc.data, err.Error(), tc.wantErr)
			}
		})
	}
}

func TestParseSizeLimit(t *testing.T) {
	const limit = 1 << 20
	comment := func(size int) []byte {
		return []byte(strings.Repeat("#", size-1) + "\n")
	}
	tests := []struct {
		size    int
		wantErr string
	}{
		{limit - 1, ""},
		{limit, ""},
		{limit + 1, `logos.toml:1:1: file too large`},
	}
	for _, tc := range tests {
		got, err := Parse("logos.toml", comment(tc.size))
		if tc.wantErr == "" {
			if err != nil {
				t.Fatalf("Parse(%d bytes) error = %v, want defaults", tc.size, err)
			}
			if got.Project.Edition != "2026" {
				t.Errorf("Parse(%d bytes) Edition = %q, want %q", tc.size, got.Project.Edition, "2026")
			}
			continue
		}
		if err == nil {
			t.Fatalf("Parse(%d bytes) succeeded, want error %q (got %+v)", tc.size, tc.wantErr, got)
		}
		if err.Error() != tc.wantErr {
			t.Errorf("Parse(%d bytes) error = %q, want %q", tc.size, err.Error(), tc.wantErr)
		}
	}
}

func TestParseErrorUsesNameArgument(t *testing.T) {
	_, err := Parse("custom.toml", []byte("[project]\nname = \"\"\n"))
	if err == nil {
		t.Fatal("Parse succeeded, want error")
	}
	if want := `custom.toml:2:8: project name must not be empty`; err.Error() != want {
		t.Errorf("error = %q, want %q", err.Error(), want)
	}
}

func TestLoadAbsentReturnsDefaults(t *testing.T) {
	got, err := Load(t.TempDir())
	if err != nil {
		t.Fatalf("Load(empty dir) error = %v", err)
	}
	checkFile(t, got, File{
		Project: Project{Edition: "2026", StrictFiles: fourExts},
		Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
	})
}

func TestLoadReadsFile(t *testing.T) {
	dir := t.TempDir()
	data := "[project]\nname = \"m\"\n[build]\ncgo = \"allow\"\n"
	if err := os.WriteFile(filepath.Join(dir, "logos.toml"), []byte(data), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}
	got, err := Load(dir)
	if err != nil {
		t.Fatalf("Load error = %v", err)
	}
	checkFile(t, got, File{
		Project: Project{Name: "m", Edition: "2026", StrictFiles: fourExts},
		Build:   Build{Cgo: "allow", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
	})
}

func FuzzParse(f *testing.F) {
	for _, tc := range validDocs {
		f.Add([]byte(tc.data))
	}
	f.Add([]byte("[project]\r\nname = \"m\"\r\n"))
	f.Add([]byte("[project]\nname = \"\\u00E9\\U0001F4DC\"\n"))
	f.Add([]byte("[project]\n# hi\x00\n"))
	f.Add([]byte("name = \"x\"\n"))
	f.Fuzz(func(t *testing.T, data []byte) {
		got, err := Parse("logos.toml", data)
		if err != nil {
			return
		}
		out := got.Format()
		re, err := Parse("logos.toml", out)
		if err != nil {
			t.Fatalf("re-parse of Format output failed: %v\nformatted: %q", err, out)
		}
		if !reflect.DeepEqual(got, re) {
			t.Fatalf("round trip mismatch:\nfirst:  %#v\nsecond: %#v\nformatted: %q", got, re, out)
		}
		out2 := re.Format()
		if string(out) != string(out2) {
			t.Fatalf("Format not idempotent:\nfirst:  %q\nsecond: %q", out, out2)
		}
	})
}
