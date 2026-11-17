// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestStructDefaultLexicalScope(t *testing.T) {
	for _, tc := range []struct {
		name string
		src  string
	}{
		{
			name: "file import",
			src: `package main
import "time"
type Cfg struct { Delay time.Duration = time.Second }
func main() {
	if (Cfg{}).Delay != 1000000000 { panic("wrong imported constant") }
	println("ok lexical scope")
}
`,
		},
		{
			name: "local constant",
			src: `package main
func main() {
	const local = 17
	type Cfg struct { Value int = local }
	if (Cfg{}).Value != 17 { panic("wrong local constant") }
	println("ok lexical scope")
}
`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runStructDefaultLowerings(t, tc.src, "ok lexical scope\n")
		})
	}
}

func TestStructDefaultRejectsInvalidDeclaration(t *testing.T) {
	for _, tc := range []struct {
		name string
		decl string
		want string
	}{
		{"byte below minimum", "Value uint8 = -1", "overflows"},
		{"byte minimum", "Value uint8 = 0", ""},
		{"byte maximum", "Value uint8 = 255", ""},
		{"byte above maximum", "Value uint8 = 256", "overflows"},
		{"wrong scalar type", `Value int = "bad"`, "cannot use"},
		{"wrong composite type", `Value []int = []string{"bad"}`, "cannot use"},
		{"dynamic map key", `Value map[int]int = map[int]int{key(): 1}`, "default_not_constant"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := "package p\nfunc key() int { return 1 }\ntype Unused struct { " + tc.decl + " }\n"
			file := ejectParse(t, syntax.NewFileBase("defaultcheck.lgo"), []byte(src), syntax.LogosMode)
			var errors []string
			conf := types2.Config{
				Logos: true,
				Error: func(err error) { errors = append(errors, err.Error()) },
			}
			_, err := conf.Check("p", []*syntax.File{file}, &types2.Info{})
			if tc.want == "" {
				if err != nil {
					t.Fatalf("valid field default rejected: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(strings.Join(errors, "\n"), tc.want) {
				t.Fatalf("field default errors = %v, want %q without constructing Unused", errors, tc.want)
			}
		})
	}
}

func runStructDefaultLowerings(t *testing.T, src, want string) {
	t.Helper()
	data := []byte(src)
	base := syntax.NewFileBase("defaultbinding.lgo")
	file := ejectParse(t, base, data, syntax.LogosMode|syntax.CheckBranches)
	pkg, info := ejectCheck(t, file)
	edits := ejectLower(t, pkg, info, file, data)
	ejected, err := ApplyEdits(data, base, edits)
	if err != nil {
		t.Fatalf("eject field defaults: %v", err)
	}
	var native bytes.Buffer
	if _, err := syntax.Fprint(&native, file, 0); err != nil {
		t.Fatalf("print lowered field defaults: %v", err)
	}
	t.Run("native tree", func(t *testing.T) {
		runStructDefaultGo(t, native.Bytes(), want)
	})
	t.Run("ejected text", func(t *testing.T) {
		runStructDefaultGo(t, ejected, want)
	})
}

func runStructDefaultGo(t *testing.T, src []byte, want string) {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module defaultbinding\n\ngo 1.27\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), src, 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "run", ".")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOTOOLCHAIN=local", "GOWORK=off")
	out, err := cmd.CombinedOutput()
	if err != nil || string(out) != want {
		t.Fatalf("field default behavior: %v\n%s\nwant %s", err, out, want)
	}
}
