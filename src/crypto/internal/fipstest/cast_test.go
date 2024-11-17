// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"fmt"
	"internal/testenv"
	"io/fs"
	"os"
	"regexp"
	"strings"
	"testing"

	// Import packages that define CASTs to test them.
	_ "crypto/internal/fips/aes"
	_ "crypto/internal/fips/aes/gcm"
	_ "crypto/internal/fips/drbg"
	"crypto/internal/fips/ecdh"
	"crypto/internal/fips/ecdsa"
	_ "crypto/internal/fips/hkdf"
	_ "crypto/internal/fips/hmac"
	"crypto/internal/fips/mlkem"
	"crypto/internal/fips/sha256"
	_ "crypto/internal/fips/sha3"
	_ "crypto/internal/fips/sha512"
	_ "crypto/internal/fips/tls12"
	_ "crypto/internal/fips/tls13"
	"crypto/rand"
)

func findAllCASTs(t *testing.T) map[string]struct{} {
	testenv.MustHaveSource(t)

	// Ask "go list" for the location of the crypto/internal/fips tree, as it
	// might be the unpacked frozen tree selected with GOFIPS140.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "list", "-f", `{{.Dir}}`, "crypto/internal/fips")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go list: %v\n%s", err, out)
	}
	fipsDir := strings.TrimSpace(string(out))
	t.Logf("FIPS module directory: %s", fipsDir)

	// Find all invocations of fips.CAST or fips.PCT.
	allCASTs := make(map[string]struct{})
	castRe := regexp.MustCompile(`fips\.(CAST|PCT)\("([^"]+)"`)
	if err := fs.WalkDir(os.DirFS(fipsDir), ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		data, err := os.ReadFile(fipsDir + "/" + path)
		if err != nil {
			return err
		}
		for _, m := range castRe.FindAllSubmatch(data, -1) {
			allCASTs[string(m[2])] = struct{}{}
		}
		return nil
	}); err != nil {
		t.Fatalf("WalkDir: %v", err)
	}

	return allCASTs
}

// TestConditionals causes the conditional CASTs and PCTs to be invoked.
func TestConditionals(t *testing.T) {
	mlkem.GenerateKey768()
	ecdh.GenerateKeyP256(rand.Reader)
	k, err := ecdsa.GenerateKey(ecdsa.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	ecdsa.SignDeterministic(ecdsa.P256(), sha256.New, k, make([]byte, 32))
	t.Log("completed successfully")
}

func TestCASTFailures(t *testing.T) {
	testenv.MustHaveExec(t)

	allCASTs := findAllCASTs(t)
	if len(allCASTs) == 0 {
		t.Fatal("no CASTs found")
	}

	for name := range allCASTs {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			cmd := testenv.Command(t, testenv.Executable(t), "-test.run=TestConditionals", "-test.v")
			cmd = testenv.CleanCmdEnv(cmd)
			cmd.Env = append(cmd.Env, fmt.Sprintf("GODEBUG=failfipscast=%s,fips140=on", name))
			out, err := cmd.CombinedOutput()
			if err == nil {
				t.Error(err)
			} else {
				t.Logf("CAST/PCT %s failed and caused the program to exit or the test to fail", name)
				t.Logf("%s", out)
			}
			if strings.Contains(string(out), "completed successfully") {
				t.Errorf("CAST/PCT %s failure did not stop the program", name)
			}
		})
	}
}
