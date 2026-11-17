// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// A host's unrelated environment must not consume the guest's fixed startup
// area. Build each guest once so only its execution environment varies.
func TestWasmNodeIgnoresAmbientEnvironment(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	wasmNode(t)
	for _, guest := range []struct {
		name, tool, suffix, source string
	}{
		{"go", testenv.GoToolPath(t), "go", `package main
func value() (int, error) { return 7, nil }
func main() {
	v, err := value()
	if err != nil { panic(err) }
	println(v)
}
`},
		{"logos", wasmLogos(t), "lgo", `package main
func value() (int, error) { return 7, nil }
func emit() error {
	v := value()?
	println(v)
	return nil
}
func main() {
	if err := emit(); err != nil { panic(err) }
}
`},
	} {
		t.Run(guest.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module example.com/wasmenv\n\ngo 1.27\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, "main."+guest.suffix), []byte(guest.source), 0o600); err != nil {
				t.Fatal(err)
			}
			wasm := filepath.Join(dir, "app.wasm")
			cmd := testenv.Command(t, guest.tool, "build", "-o", wasm, ".")
			cmd.Dir = dir
			cmd.Env = wasmGoEnv(cmd, "js")
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("build %s guest: %v\n%s", guest.name, err, out)
			}
			for _, size := range []int{0, 64, 8191, 8192, 8193, 16384} {
				t.Run(strconv.Itoa(size), func(t *testing.T) {
					t.Setenv("LOGOS_WASM_TEST_AMBIENT", strings.Repeat("x", size))
					got, err := runWasmNode(t, wasm)
					if err != nil || got != "7\n" {
						t.Fatalf("ambient value %d bytes: %v\ngot %q\nwant %q", size, err, got, "7\n")
					}
				})
			}
		})
	}
}
