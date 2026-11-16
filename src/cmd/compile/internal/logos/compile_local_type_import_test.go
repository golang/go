// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"internal/testenv"
)

// The graphfmt.Dot.Sprint method in x/tools v0.52.0 contains a local struct
// depending on its receiver's type parameter. Importing its facts must not
// resolve that local type through the public type importer.
func TestCompileImportedLocalType(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	goTool := testenv.GoToolPath(t)
	logosTool := wasmLogos(t)
	for _, tc := range []struct{ name, library, call string }{
		{"generic method", `type Box[T any] struct{}
func (Box[T]) Pass(value T) T {
	type local struct{ value T }
	return local{value}.value
}`, `(lib.Box[string]{}).Pass("ok")`},
		{"generic function", `func Pass[T any](value T) T {
	type local struct{ value T }
	return local{value}.value
}`, `lib.Pass("ok")`},
		{"nested two parameters", `func Pass[K comparable, V any](key K, value V) V {
	type local struct { key K; value V }
	type wrapper struct { local }
	return wrapper{local{key, value}}.value
}`, `lib.Pass(7, "ok")`},
		{"nongeneric local control", `func Pass(value string) string {
	type local struct{ value string }
	return local{value}.value
}`, `lib.Pass("ok")`},
		{"package generic control", `type Local[T any] struct{ value T }
func Pass[T any](value T) T { return Local[T]{value}.value }`, `lib.Pass("ok")`},
	} {
		for _, mode := range []struct{ name, tool, mainExt, libExt string }{
			{"go", goTool, "go", "go"},
			{"logos go", logosTool, "go", "go"},
			{"logos interop", logosTool, "lgo", "go"},
			{"logos overlay", logosTool, "lgo", "lgo"},
		} {
			t.Run(tc.name+"/"+mode.name, func(t *testing.T) {
				dir := t.TempDir()
				mainDir, libDir := filepath.Join(dir, "main"), filepath.Join(dir, "lib")
				for _, path := range []string{mainDir, libDir} {
					if err := os.Mkdir(path, 0o755); err != nil {
						t.Fatal(err)
					}
				}
				library := "package lib\n" + tc.library + "\n"
				body := "if " + tc.call + " != \"ok\" { panic(\"wrong value\") }\n"
				if mode.libExt == "lgo" {
					library += "func Option() int? { return 7 }\n"
					body += "if (lib.Option() ?? 0) != 7 { panic(\"lost option overlay\") }\n"
				}
				main := "package main\nimport \"example.com/localtype/lib\"\nfunc main() {\n" + body + "println(\"generic local ok\")\n}\n"
				for path, text := range map[string]string{
					filepath.Join(mainDir, "go.mod"):             "module example.com/localtype/main\n\ngo 1.27\n\nrequire example.com/localtype/lib v0.0.0\nreplace example.com/localtype/lib => ../lib\n",
					filepath.Join(libDir, "go.mod"):              "module example.com/localtype/lib\n\ngo 1.27\n",
					filepath.Join(mainDir, "main."+mode.mainExt): main,
					filepath.Join(libDir, "lib."+mode.libExt):    library,
				} {
					if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
						t.Fatal(err)
					}
				}
				cmd := testenv.Command(t, mode.tool, "run", "-p=1", ".")
				cmd.Dir = mainDir
				cmd.Env = append(archDiffEnv(cmd, runtime.GOOS, runtime.GOARCH), "GOPROXY=off", "GOSUMDB=off", "GOTOOLCHAIN=local")
				output, err := cmd.CombinedOutput()
				if err != nil || strings.TrimSpace(string(output)) != "generic local ok" {
					t.Fatalf("run: %v\n%s", err, output)
				}
			})
		}
	}
}
