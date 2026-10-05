// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"bytes"
	"crypto/internal/fips140"
	"internal/diff"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestMLKEM1024UpToDate checks that mlkem1024.go matches the output of
// generate1024.go, so that changes to mlkem768.go are not left out of the
// generated file.
func TestMLKEM1024UpToDate(t *testing.T) {
	testenv.MustHaveSource(t)
	testenv.MustHaveGoRun(t)

	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	if fips140.Version() != "latest" {
		t.Skip("skipping for frozen FIPS 140-3 module snapshot")
	}

	// Ask "go list" for the location of the package, as it might be the
	// unpacked frozen tree selected with GOFIPS140.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "list", "-f", `{{.Dir}}`, "crypto/internal/fips140/mlkem")
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("go list: %v\n%s", err, out)
	}
	dir := strings.TrimSpace(string(out))

	generated := filepath.Join(t.TempDir(), "mlkem1024.go")
	cmd = testenv.Command(t, testenv.GoToolPath(t), "run", "generate1024.go", "-input", "mlkem768.go", "-output", generated)
	cmd.Dir = dir
	if out, err := testenv.CleanCmdEnv(cmd).CombinedOutput(); err != nil {
		t.Fatalf("go run generate1024.go: %v\n%s", err, out)
	}

	want, err := os.ReadFile(generated)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(dir, "mlkem1024.go"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("mlkem1024.go is out of date; run go generate in %s\n%s", dir,
			diff.Diff("mlkem1024.go", got, "generate1024.go", want))
	}
}
