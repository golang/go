// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"cmd/compile/internal/logos/facts"
	"internal/testenv"
)

const taggedMethodSourceSHA256 = "00adc9cf8b53e4e78ef162d7675b847b11f688b0cdab9ce18739b9a79a5571dd"

// The fixture is the current SIMD interface example, copied byte-for-byte into
// a non-standard Go module. Its methods are not generic. The Go baseline and
// logos profile must both compile its consumer; logos must additionally emit
// decodable method facts with explicit, known-zero parameter mutation masks.
// Target programs are compiled but never executed.
func TestCompileSIMDTaggedMethodFacts(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	goTool := testenv.GoToolPath(t)
	logosTool := filepath.Join(filepath.Dir(goTool), "logos")
	if runtime.GOOS == "windows" {
		logosTool += ".exe"
	}
	if _, err := os.Stat(logosTool); err != nil {
		t.Fatalf("required logos binary beside the installed go tool: %v", err)
	}
	source := filepath.Join(testenv.GOROOT(t), "src", "simd", "testdata", "iface", "iface.go")
	data, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != taggedMethodSourceSHA256 {
		t.Fatalf("real SIMD source changed: SHA256=%s, frozen=%s", got, taggedMethodSourceSHA256)
	}
	dir := t.TempDir()
	for _, subdir := range []string{"iface", "consumer"} {
		if err := os.Mkdir(filepath.Join(dir, subdir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	files := map[string][]byte{
		"go.mod":         []byte("module example.com/taggedmethodreal\n\ngo 1.27.4\n"),
		"iface/iface.go": data,
		"consumer/consumer.go": []byte(`package consumer

import "example.com/taggedmethodreal/iface"

func Read(v iface.VectorC) iface.VectorC {
	return v.Foo(v)
}
`),
	}
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), content, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, profile := range []struct {
		name string
		tool string
	}{
		{"fork-go", goTool},
		{"logos", logosTool},
	} {
		t.Run(profile.name, func(t *testing.T) {
			archives := taggedMethodExports(t, profile.tool, dir)
			archive, ok := archives["example.com/taggedmethodreal/iface"]
			if !ok {
				t.Fatal("compiler omitted the real-source fixture export archive")
			}
			stored := readFacts(t, archive)
			if profile.name == "logos" {
				checkRealTaggedMethodFacts(t, stored)
			} else if stored != nil {
				for _, record := range stored.Fns {
					if record.ParamMutated != "" {
						t.Fatalf("plain fork-Go baseline unexpectedly enabled mutation facts: %+v", record)
					}
				}
			}
			cmd := taggedMethodCommand(t, profile.tool, dir, "build", "-p=1", "./consumer")
			output, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("%s consumer import failed: %v\n%s", profile.name, err, output)
			}
			copied, err := os.ReadFile(filepath.Join(dir, "iface", "iface.go"))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(copied, data) {
				t.Fatal("real SIMD source changed in the private module")
			}
		})
	}
}

func taggedMethodCommand(t *testing.T, tool, dir string, args ...string) *exec.Cmd {
	t.Helper()
	cmd := testenv.Command(t, tool, args...)
	cmd.Dir = dir
	cmd.Env = append(archDiffEnv(cmd, "linux", "amd64"),
		"GOAMD64=v1", "GOEXPERIMENT=none,simd", "CGO_ENABLED=0", "GOTOOLCHAIN=local", "GOENV=off",
		"GOFLAGS=-mod=readonly -buildvcs=false", "GOPROXY=off", "GOSUMDB=off")
	return cmd
}

func taggedMethodExports(t *testing.T, tool, dir string) map[string]string {
	t.Helper()
	cmd := taggedMethodCommand(t, tool, dir, "list", "-p=1", "-export", "-deps", "-f",
		"{{if .Export}}{{.ImportPath}} {{.Export}}{{end}}", "./iface")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("%s real-source export: %v\n%s", tool, err, stderr.Bytes())
	}
	archives := make(map[string]string)
	for line := range strings.Lines(string(output)) {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		path, archive, ok := strings.Cut(line, " ")
		if !ok || path == "" || archive == "" {
			t.Fatalf("malformed export receipt: %q", line)
		}
		if _, exists := archives[path]; exists {
			t.Fatalf("duplicate export archive for %q", path)
		}
		archives[path] = archive
	}
	return archives
}

func checkRealTaggedMethodFacts(t *testing.T, stored *facts.Facts) {
	t.Helper()
	if stored == nil || !stored.GoPackage {
		t.Fatal("logos profile did not emit a Go-package fact member")
	}
	for _, tag := range []string{"simd0", "simd128", "simd256", "simd512"} {
		for _, method := range []struct {
			object          string
			receiverMutated bool
		}{
			{"VectorC@%s.Foo", true},
			{"VectorC@%s.Bar", true},
			{"(*Q@%s).Qux", false},
		} {
			object := fmt.Sprintf(method.object, tag)
			var found facts.FnFact
			count := 0
			for _, record := range stored.Fns {
				if record.Object == object {
					found = record
					count++
				}
			}
			if count != 1 {
				t.Errorf("real SIMD producer emitted %d records for %q, want exactly one", count, object)
				continue
			}
			if found.ParamMutated != "0" || found.ReceiverMutated != method.receiverMutated {
				t.Errorf("%s mutation metadata = %+v, want explicit ParamMutated=0 and ReceiverMutated=%v", object, found, method.receiverMutated)
			}
		}
	}
}
