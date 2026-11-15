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
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "basic string",
		data: "[build]\ncgo = \"allow\"\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "allow", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "literal string",
		data: "[build]\ncgo = 'restricted'\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "restricted", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "comments are ignored",
		data: "# leading comment\n[build] # table comment\n# full line\ncgo = \"allow\" # trailing\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "allow", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "crlf line endings",
		data: "[build]\r\ncgo = \"allow\"\r\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "allow", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "string arrays",
		data: "[project]\nstrict_files = [\".lgo\", \".⚡\"]\n[build]\ncgo_allowlist = [\"example.com/m/ok/**\"]\n",
		want: File{
			Project: Project{StrictFiles: []string{".lgo", ".⚡"}},
			Build:   Build{Cgo: "forbid", CgoAllowlist: []string{"example.com/m/ok/**"}, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "trailing comma in array",
		data: "[project]\nstrict_files = [\n\".lgo\",\n\".⚡\",\n]\n",
		want: File{
			Project: Project{StrictFiles: []string{".lgo", ".⚡"}},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "absent project table yields project defaults",
		data: "[build]\ncgo = \"allow\"\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "allow", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "absent build table yields build defaults",
		data: "[project]\nstrict_files = [\".lgo\"]\n",
		want: File{
			Project: Project{StrictFiles: []string{".lgo"}},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "explicit cgo forbid tracks presence",
		data: "[build]\ncgo = \"forbid\"\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "explicit cgo restricted tracks presence",
		data: "[build]\ncgo = \"restricted\"\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "restricted", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "explicit cgo allow tracks presence",
		data: "[build]\ncgo = \"allow\"\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "allow", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "opt level and target cpu",
		data: "[build]\nopt_level = \"O3\"\ntarget_cpu = \"native\"\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O3", TargetCPU: "native"},
		},
	},
	{
		name: "full document",
		data: "[project]\nstrict_files = [\".lgo\", \".📜\", \".⚡\", \".ʕ◔ϖ◔ʔ\"]\n[build]\ncgo = \"restricted\"\ncgo_allowlist = [\"example.com/m/ok/**\"]\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "restricted", CgoSet: true, CgoAllowlist: []string{"example.com/m/ok/**"}, OptLevel: "O2", TargetCPU: "deterministic"},
		},
	},
	{
		name: "capabilities severities",
		data: "[capabilities]\ntransitive_reflect = \"deny\"\ntransitive_exec = \"warn\"\ntransitive_net = \"allow\"\ntransitive_unsafe = \"warn\"\n",
		want: File{
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
			Capabilities: Capabilities{
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
			Project: Project{StrictFiles: fourExts},
			Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
			Capabilities: Capabilities{
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
		TransitiveReflect: "warn",
		TransitiveExec:    "deny",
		TransitiveNet:     "deny",
		TransitiveUnsafe:  "deny",
	}
}

func withCapDefaults(c Capabilities) Capabilities {
	d := defaultCapabilities()
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
		{"duplicate key", "[build]\ncgo = \"allow\"\ncgo = \"forbid\"\n", `logos.toml:3:1: duplicate key "cgo"`},
		{"root key rejected", "name = \"x\"\n", `logos.toml:1:1: unknown key "name"`},
		{"root build key rejected", "cgo = \"allow\"\n", `logos.toml:1:1: unknown key "cgo"`},
		{"misplaced project key", "[build]\nname = \"x\"\n", `logos.toml:2:1: unknown key "name"`},
		{"duplicate table", "[project]\nstrict_files = [\".lgo\"]\n[project]\n", `logos.toml:3:1: duplicate table "project"`},
		{"unknown table", "[frobnicate]\n", `logos.toml:1:1: unknown table "frobnicate"`},
		{"unknown key", "[project]\nbogus = \"x\"\n", `logos.toml:2:1: unknown key "bogus"`},
		{"project name rejected", "[project]\nname = \"m\"\n", `logos.toml:2:1: unknown key "name"`},
		{"project edition rejected", "[project]\nedition = \"2026\"\n", `logos.toml:2:1: unknown key "edition"`},
		{"direct reflect rejected", "[capabilities]\ndirect_reflect = \"deny\"\n", `logos.toml:2:1: unknown key "direct_reflect"`},
		{"wrong value type", "[build]\ncgo = 42\n", `logos.toml:2:7: want string for key "cgo"`},
		{"wrong value type boolean", "[build]\ncgo = true\n", `logos.toml:2:7: want string for key "cgo"`},
		{"unterminated string", "[build]\ncgo = \"abc\n", `logos.toml:2:7: unterminated string`},
		{"invalid escape", "[build]\ncgo = \"a\\qb\"\n", `logos.toml:2:9: invalid escape 'q'`},
		{"nul in comment", "[build]\n# hi\x00\ncgo = \"allow\"\n", `logos.toml:2:5: invalid control character U+0000`},
		{"nul in string", "[build]\ncgo = \"a\x00b\"\n", `logos.toml:2:9: invalid control character U+0000`},
		{"del rejected", "[build]\ncgo = \"allow\"\x7f\n", `logos.toml:2:14: invalid control character U+007F`},
		{"lone cr rejected", "[project]\rcgo = \"allow\"\n", `logos.toml:1:10: invalid control character U+000D`},
		{"non-utf8 byte", "[build]\ncgo = \"a\xff\"\n", `logos.toml:2:9: invalid utf-8`},
		{"inline table", "[build]\ncgo = {first = \"a\"}\n", `logos.toml:2:7: inline table not allowed`},
		{"float", "[build]\ncgo = 1.5\n", `logos.toml:2:7: float not allowed`},
		{"date", "[build]\ncgo = 1979-05-27\n", `logos.toml:2:7: date not allowed`},
		{"invalid cgo value", "[build]\ncgo = \"sometimes\"\n", `logos.toml:2:7: invalid cgo value "sometimes" (want forbid, restricted, or allow)`},
		{"invalid capabilities severity", "[capabilities]\ntransitive_unsafe = \"forbid\"\n", `logos.toml:2:21: invalid transitive_unsafe value "forbid" (want deny, warn, or allow)`},
		{"capabilities key in build table", "[build]\ntransitive_unsafe = \"deny\"\n", `logos.toml:2:1: unknown key "transitive_unsafe"`},
		{"build key in capabilities table", "[capabilities]\ncgo = \"allow\"\n", `logos.toml:2:1: unknown key "cgo"`},
		{"build gc key rejected", "[build]\ngc = \"greentea\"\n", `logos.toml:2:1: unknown key "gc"`},
		{"invalid capabilities allowlist pattern", "[capabilities]\ntransitive_unsafe_allowlist = [\"[\"]\n", `logos.toml:2:32: [configuration::invalid_capabilities_allowlist] transitive_unsafe_allowlist pattern "[": syntax error in pattern`},
		{"strict_files scalar rejected", "[project]\nstrict_files = \".lgo\"\n", `logos.toml:2:16: want string array for key "strict_files"`},
		{"invalid strict_files entry", "[project]\nstrict_files = [\".go\"]\n", `logos.toml:2:17: invalid strict_files entry ".go" (want ".lgo", ".📜", ".⚡", or ".ʕ◔ϖ◔ʔ")`},
		{"bare opt level rejected", "[build]\nopt_level = \"3\"\n", `logos.toml:2:13: opt_level must be one of O0, O1, O2, O3, Os`},
		{"unknown opt level rejected", "[build]\nopt_level = \"O4\"\n", `logos.toml:2:13: opt_level must be one of O0, O1, O2, O3, Os`},
		{"lowercase opt level rejected", "[build]\nopt_level = \"o3\"\n", `logos.toml:2:13: opt_level must be one of O0, O1, O2, O3, Os`},
		{"uppercase size level rejected", "[build]\nopt_level = \"OS\"\n", `logos.toml:2:13: opt_level must be one of O0, O1, O2, O3, Os`},
		{"digit zero size level rejected", "[build]\nopt_level = \"0s\"\n", `logos.toml:2:13: opt_level must be one of O0, O1, O2, O3, Os`},
		{"letter only opt level rejected", "[build]\nopt_level = \"O\"\n", `logos.toml:2:13: opt_level must be one of O0, O1, O2, O3, Os`},
		{"padded opt level rejected", "[build]\nopt_level = \"O03\"\n", `logos.toml:2:13: opt_level must be one of O0, O1, O2, O3, Os`},
		{"invalid target cpu", "[build]\ntarget_cpu = \"laptop\"\n", `logos.toml:2:14: target_cpu must be one of deterministic, baseline, native`},
		{"capitalized target cpu rejected", "[build]\ntarget_cpu = \"Native\"\n", `logos.toml:2:14: target_cpu must be one of deterministic, baseline, native`},
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

func TestParseAllowlistScalarsReportPositionAndKey(t *testing.T) {
	tests := []struct {
		name    string
		data    string
		wantPos string
		wantKey string
	}{
		{"reflect_allowlist", "[capabilities]\nreflect_allowlist = \"x\"\n", "logos.toml:2:21", `"reflect_allowlist"`},
		{"transitive_exec_allowlist", "[capabilities]\ntransitive_exec_allowlist = \"x\"\n", "logos.toml:2:29", `"transitive_exec_allowlist"`},
		{"transitive_net_allowlist", "[capabilities]\ntransitive_net_allowlist = \"x\"\n", "logos.toml:2:28", `"transitive_net_allowlist"`},
		{"transitive_unsafe_allowlist", "[capabilities]\ntransitive_unsafe_allowlist = \"x\"\n", "logos.toml:2:31", `"transitive_unsafe_allowlist"`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Parse("logos.toml", []byte(tc.data))
			if err == nil {
				t.Fatalf("Parse(%q) succeeded, want an error (got %+v)", tc.data, got)
			}
			if !strings.Contains(err.Error(), tc.wantPos) || !strings.Contains(err.Error(), tc.wantKey) {
				t.Errorf("Parse(%q) error = %q, want position %s and key %s", tc.data, err.Error(), tc.wantPos, tc.wantKey)
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
			if !slices.Equal(got.Project.StrictFiles, fourExts) {
				t.Errorf("Parse(%d bytes) StrictFiles = %q, want %q", tc.size, got.Project.StrictFiles, fourExts)
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
	_, err := Parse("custom.toml", []byte("[build]\ncgo = \"sometimes\"\n"))
	if err == nil {
		t.Fatal("Parse succeeded, want error")
	}
	if want := `custom.toml:2:7: invalid cgo value "sometimes" (want forbid, restricted, or allow)`; err.Error() != want {
		t.Errorf("error = %q, want %q", err.Error(), want)
	}
}

func TestLoadAbsentReturnsDefaults(t *testing.T) {
	got, err := Load(t.TempDir())
	if err != nil {
		t.Fatalf("Load(empty dir) error = %v", err)
	}
	checkFile(t, got, File{
		Project: Project{StrictFiles: fourExts},
		Build:   Build{Cgo: "forbid", OptLevel: "O2", TargetCPU: "deterministic"},
	})
	if got.Build.Arena != nil {
		t.Errorf("Build.Arena = %v, want nil (default true)", *got.Build.Arena)
	}
	if got.Build.CgoAllowlist != nil {
		t.Errorf("Build.CgoAllowlist = %q, want nil", got.Build.CgoAllowlist)
	}
	checkCapabilities(t, got.Capabilities, defaultCapabilities())
	if got.Lints.Groups != nil {
		t.Errorf("Lints.Groups = %v, want nil: lint defaults come from the compiler policy, not the file", got.Lints.Groups)
	}
	if got.Lints.Thresholds != nil {
		t.Errorf("Lints.Thresholds = %v, want nil: threshold defaults come from the compiler policy, not the file", got.Lints.Thresholds)
	}
	empty, err := Load(mustDirWithEmptyLintsFile(t))
	if err != nil {
		t.Fatalf("Load(dir with empty logos.toml) error = %v", err)
	}
	if !reflect.DeepEqual(got, empty) {
		t.Errorf("absent file = %#v, empty file = %#v, want equal", got, empty)
	}
}

func mustDirWithEmptyLintsFile(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "logos.toml"), nil, 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}
	return dir
}

func TestLoadReadsFile(t *testing.T) {
	dir := t.TempDir()
	data := "[project]\nstrict_files = [\".lgo\"]\n[build]\ncgo = \"allow\"\n"
	if err := os.WriteFile(filepath.Join(dir, "logos.toml"), []byte(data), 0644); err != nil {
		t.Fatalf("WriteFile error = %v", err)
	}
	got, err := Load(dir)
	if err != nil {
		t.Fatalf("Load error = %v", err)
	}
	checkFile(t, got, File{
		Project: Project{StrictFiles: []string{".lgo"}},
		Build:   Build{Cgo: "allow", CgoSet: true, OptLevel: "O2", TargetCPU: "deterministic"},
	})
}

func FuzzParse(f *testing.F) {
	for _, tc := range validDocs {
		f.Add([]byte(tc.data))
	}
	f.Add([]byte("[build]\r\ncgo = \"allow\"\r\n"))
	f.Add([]byte("[capabilities]\nreflect_allowlist = [\"\\u00E9\\U0001F4DC\"]\n"))
	f.Add([]byte("[project]\n# hi\x00\n"))
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
