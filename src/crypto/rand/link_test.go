// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

const linkerTestProgram = `
package main
import "crypto/rand"
func main() {
	b := make([]byte, 32)
	rand.Read(b)
	println("OK")
}
`

// TestLinker ensures that using crypto/rand does not bring unrelated
// algorithms into the binary. In particular, the DRBG uses AES-CTR, but that
// must not pull in GCM or the crypto/cipher package.
func TestLinker(t *testing.T) {
	if testing.Short() {
		t.Skip("test requires running 'go build'")
	}
	testenv.MustHaveGoBuild(t)

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

	var consistent bool
	nm := run(testenv.GoToolPath(t), "tool", "nm", "hello.exe")
	for _, match := range regexp.MustCompile(`(?m)T (crypto/.*)$`).FindAllStringSubmatch(nm, -1) {
		symbol := match[1]
		if strings.HasPrefix(symbol, "crypto/internal/fips140/drbg.") {
			consistent = true
		}
		if strings.HasPrefix(symbol, "crypto/internal/fips140/aes/gcm.") ||
			strings.HasPrefix(symbol, "crypto/cipher.") {
			t.Errorf("unexpected symbol in program using only crypto/rand: %s", symbol)
		}
	}
	if !consistent {
		t.Error("no DRBG symbols found in program using crypto/rand, test is broken")
	}
}
