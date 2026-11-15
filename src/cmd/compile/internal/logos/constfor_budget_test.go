// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"context"
	"errors"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func writeNestedProbe(t *testing.T, dir string, fields, depth int) {
	t.Helper()
	var b strings.Builder
	b.WriteString("module nest\n\ngo 1.27\n")
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(b.String()), 0666); err != nil {
		t.Fatal(err)
	}
	b.Reset()
	b.WriteString("package main\n\nimport (\n\t\"comptime\"\n\t\"fmt\"\n)\n\ntype S struct {\n")
	for i := range fields {
		b.WriteString("\tF" + strconv.Itoa(i) + " int\n")
	}
	b.WriteString("}\n\nfunc show(v S) int {\n\tn := 0\n")
	for d := range depth {
		b.WriteString(strings.Repeat("\t", d+1) + "const for _, f" + strconv.Itoa(d) + " := range comptime.Fields[S]() {\n")
	}
	b.WriteString(strings.Repeat("\t", depth+1) + "n +=")
	for d := range depth {
		if d > 0 {
			b.WriteString(" +")
		}
		b.WriteString(" v.[f" + strconv.Itoa(d) + "]")
	}
	b.WriteString("\n")
	for d := range depth {
		b.WriteString(strings.Repeat("\t", depth-d) + "}\n")
	}
	b.WriteString("\treturn n\n}\n\nfunc main() {\n\tfmt.Println(show(S{}))\n}\n")
	if err := os.WriteFile(filepath.Join(dir, "main.lgo"), []byte(b.String()), 0666); err != nil {
		t.Fatal(err)
	}
}

func TestConstForNestedExpansionTripsFast(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	writeNestedProbe(t, dir, 10, 9)
	logos := filepath.Join(testenv.GOROOT(t), "bin", "logos")
	backstop := 180 * time.Second
	backstopErr := errors.New("nested expansion backstop elapsed")
	ctx, cancel := context.WithTimeoutCause(t.Context(), backstop, backstopErr)
	defer cancel()
	cmd := exec.CommandContext(ctx, logos, "build", "./...")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOTOOLCHAIN=local", "GOPROXY=off", "GOFLAGS=-buildvcs=false", "GOCACHE="+filepath.Join(dir, "gocache"))
	start := time.Now()
	out, err := cmd.CombinedOutput()
	if cause := context.Cause(ctx); cause != nil {
		t.Fatalf("nested expansion ran %v without tripping the node limit: %v", time.Since(start), cause)
	}
	if err == nil {
		t.Fatalf("nested build succeeded, want node_limit failure")
	}
	if !strings.Contains(string(out), "[comptime::node_limit]") {
		t.Fatalf("output lacks node_limit rule: %q", out)
	}
}
