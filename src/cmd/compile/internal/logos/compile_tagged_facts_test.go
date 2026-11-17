// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	gcimporter "cmd/compile/internal/importer"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types2"
	"internal/testenv"
)

// Exercise the real SIMD producer, archive codec, public type importer and
// metadata consumer. This compiles linux/amd64 archives without executing them.
func TestCompileSIMDTaggedFuncFacts(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	cmd := testenv.Command(t, testenv.GoToolPath(t), "list", "-p=1", "-export", "-deps", "-f",
		"{{if .Export}}{{.ImportPath}} {{.Export}}{{end}}", "simd")
	cmd.Env = append(archDiffEnv(cmd, "linux", "amd64"), "GOEXPERIMENT=none,simd", "CGO_ENABLED=0",
		"GOPROXY=off", "GOSUMDB=off", "GOTOOLCHAIN=local", "GOENV=off")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("compile SIMD export archives: %v\n%s", err, output)
	}
	archives := make(map[string]string)
	for line := range strings.Lines(string(output)) {
		path, archive, ok := strings.Cut(strings.TrimSpace(line), " ")
		if !ok || path == "" || archive == "" {
			continue
		}
		if _, exists := archives[path]; exists {
			t.Fatalf("duplicate export archive for %q", path)
		}
		archives[path] = archive
	}
	archive, ok := archives["simd"]
	if !ok {
		t.Fatalf("compiler did not report the simd export archive: %s", output)
	}
	stored := readFacts(t, archive)
	if stored == nil {
		t.Fatal("simd export archive has no __.LOGOS facts")
	}
	pkg, err := gcimporter.Import(make(map[string]*types2.Package), "simd", "", func(path string) (io.ReadCloser, error) {
		archive, ok := archives[path]
		if !ok {
			return nil, fmt.Errorf("missing export archive for %q", path)
		}
		return os.Open(archive)
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"Float32sFromArch@simd0", "Float32sFromArch@simd128"} {
		t.Run(name, func(t *testing.T) {
			var record facts.FnFact
			count := 0
			for _, candidate := range stored.Fns {
				if candidate.Object == name {
					record = candidate
					count++
				}
			}
			if count != 1 || !record.HasAssert {
				t.Fatalf("real SIMD producer emitted %d records for %q, want one known assertion record: %+v", count, name, record)
			}
			if record.AssertTP != "0" || record.AssertRTP != "0" {
				t.Fatalf("SIMD conversion assertion masks = %q/%q, want known 0/0", record.AssertTP, record.AssertRTP)
			}
			fn, ok := pkg.Scope().Lookup(name).(*types2.Func)
			if !ok {
				t.Fatalf("real export data omitted tagged function %q", name)
			}
			if fn.Signature().TypeParams().Len() != 1 {
				t.Fatalf("tagged conversion has %d type parameters, want 1", fn.Signature().TypeParams().Len())
			}
			if _, _, known := fn.LogosAssertEffects(); known {
				t.Fatal("raw import unexpectedly has assertion metadata before fact application")
			}
			if err := ApplyFuncFacts(pkg, []facts.FnFact{record}); err != nil {
				t.Fatal(err)
			}
			checkTaggedFactMasks(t, fn, record.AssertTP, record.AssertRTP)
		})
	}
}
