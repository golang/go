// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmac

import (
	"crypto/internal/cryptotest"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

const linkerTestProgram = `
package main
import (
	"crypto/hmac"
	"crypto/sha256"
)
func main() {
	h := hmac.New(sha256.New, []byte("key"))
	h.Write([]byte("hello"))
	if len(h.Sum(nil)) != 32 {
		panic("bad length")
	}
	println("OK")
}
`

// TestLinker ensures that using crypto/hmac with one hash does not bring the
// other hashes into the binary. In particular, the approved hash check must
// not pull in SHA-3.
func TestLinker(t *testing.T) {
	if testing.Short() {
		t.Skip("test requires running 'go build'")
	}
	testenv.MustHaveGoBuild(t)
	// Older modules check the hash type with a type switch that imports SHA-3.
	cryptotest.MustMinimumFIPS140ModuleVersion(t, "v1.28.0")

	dir := t.TempDir()
	hello := filepath.Join(dir, "hello.go")
	if err := os.WriteFile(hello, []byte(linkerTestProgram), 0664); err != nil {
		t.Fatal(err)
	}

	run := func(args ...string) string {
		cmd := testenv.Command(t, args[0], args[1:]...)
		cmd.Dir = dir
		out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
		if err != nil {
			t.Fatalf("%v: %v\n%s", args, err, string(out))
		}
		return string(out)
	}

	run(testenv.GoToolPath(t), "build", "-o", "hello.exe", "hello.go")
	if out := run("./hello.exe"); out != "OK\n" {
		t.Error("unexpected output:", out)
	}

	// In a snapshot, the module packages are crypto/internal/fips140/v1.2.3/...
	symbolRE := regexp.MustCompile(`(?m)T (crypto/(?:internal/fips140/(?:v[^/]+/)?)?([a-z0-9]+)\..*)$`)

	var consistent bool
	nm := run(testenv.GoToolPath(t), "tool", "nm", "hello.exe")
	for _, match := range symbolRE.FindAllStringSubmatch(nm, -1) {
		symbol, pkg := match[1], match[2]
		if pkg == "hmac" {
			consistent = true
		}
		if pkg == "sha3" || pkg == "sha512" {
			t.Errorf("unexpected symbol in program using only HMAC-SHA-256: %s", symbol)
		}
	}
	if !consistent {
		t.Error("no HMAC symbols found in program using crypto/hmac, test is broken")
	}
}
