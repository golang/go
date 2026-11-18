// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Keep the real producer unchanged. Every consumer is ordinary Go, and each
// driver must supply compiler-generated imports without exposing internal APIs.
func TestCompileSIMDTransitiveImports(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	goTool := testenv.GoToolPath(t)
	logosTool := filepath.Join(filepath.Dir(goTool), "logos")
	if runtime.GOOS == "windows" {
		logosTool += ".exe"
	}
	if _, err := os.Stat(logosTool); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(testenv.GOROOT(t), "src/simd/testdata/iface/iface.go")
	data, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != taggedMethodSourceSHA256 {
		t.Fatalf("real SIMD source changed: SHA256=%s, frozen=%s", got, taggedMethodSourceSHA256)
	}
	const imported = `import "example.com/simdimports/iface"`
	const read = `func Read(v iface.VectorC) iface.VectorC { return v.Foo(v) }`
	cases := []struct {
		name string
		body string
		deny bool
	}{
		{"transitive", imported + "\n" + read, false},
		{"type_only", imported + `
type Value struct { V iface.VectorC }
`, false},
		{"package_shadows", imported + `
var simd = 17
var bridge = 23
func Values() int { return simd + bridge }
` + read, false},
		{"parameter_shadows", imported + `
func Boundary(simd, bridge int) int {
	var v iface.VectorC
	_ = v.Foo(v)
	return simd + bridge
}
`, false},
		{"aliased_import", `import vector "simd"
func Read(v vector.Float32s) vector.Float32s { return v.Add(v) }
`, false},
		{"dot_import", `import . "simd"
func Read(v Float32s) Float32s { return v.Add(v) }
`, false},
		{"blank_import", imported + `
import _ "simd"
` + read, false},
		{"internal_denied", imported + `
import "simd/internal/bridge"
var _ bridge.ZeroSized
` + read, true},
	}
	dir := t.TempDir()
	files := map[string][]byte{
		"go.mod":         []byte("module example.com/simdimports\n\ngo 1.27.4\n"),
		"iface/iface.go": data,
	}
	for _, tc := range cases {
		files[tc.name+"/consumer.go"] = []byte("package consumer\n" + tc.body + "\n")
	}
	for name, content := range files {
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, content, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, profile := range []struct{ name, tool string }{{"fork-go", goTool}, {"logos", logosTool}} {
		t.Run(profile.name, func(t *testing.T) {
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					cmd := taggedMethodCommand(t, profile.tool, dir, "build", "-p=1", "./"+tc.name)
					output, err := cmd.CombinedOutput()
					if tc.deny {
						if err == nil || !strings.Contains(string(output), "use of internal package simd/internal/bridge not allowed") {
							t.Fatalf("explicit internal import: err=%v output=%s", err, output)
						}
					} else if err != nil {
						t.Fatalf("SIMD consumer failed: %v\n%s", err, output)
					}
				})
			}
		})
	}
	copied, err := os.ReadFile(filepath.Join(dir, "iface/iface.go"))
	if err != nil || !bytes.Equal(copied, data) {
		t.Fatalf("real SIMD source changed in the private module: %v", err)
	}
}
