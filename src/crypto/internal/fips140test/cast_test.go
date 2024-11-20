// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"crypto/rand"
	"fmt"
	"internal/testenv"
	"io/fs"
	"os"
	"regexp"
	"strings"
	"testing"

	// Import packages that define CASTs to test them.
	_ "crypto/internal/fips140/aes"
	_ "crypto/internal/fips140/aes/gcm"
	_ "crypto/internal/fips140/drbg"
	"crypto/internal/fips140/ecdh"
	"crypto/internal/fips140/ecdsa"
	"crypto/internal/fips140/ed25519"
	_ "crypto/internal/fips140/hkdf"
	_ "crypto/internal/fips140/hmac"
	"crypto/internal/fips140/mlkem"
	"crypto/internal/fips140/rsa"
	"crypto/internal/fips140/sha256"
	_ "crypto/internal/fips140/sha3"
	_ "crypto/internal/fips140/sha512"
	_ "crypto/internal/fips140/tls12"
	_ "crypto/internal/fips140/tls13"
)

func findAllCASTs(t *testing.T) map[string]struct{} {
	testenv.MustHaveSource(t)

	// Ask "go list" for the location of the crypto/internal/fips140 tree, as it
	// might be the unpacked frozen tree selected with GOFIPS140.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "list", "-f", `{{.Dir}}`, "crypto/internal/fips140")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go list: %v\n%s", err, out)
	}
	fipsDir := strings.TrimSpace(string(out))
	t.Logf("FIPS module directory: %s", fipsDir)

	// Find all invocations of fips140.CAST or fips140.PCT.
	allCASTs := make(map[string]struct{})
	castRe := regexp.MustCompile(`fips140\.(CAST|PCT)\("([^"]+)"`)
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
	k, err := ecdh.GenerateKey(ecdh.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	ecdh.ECDH(ecdh.P256(), k, k.PublicKey())
	kDSA, err := ecdsa.GenerateKey(ecdsa.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	ecdsa.SignDeterministic(ecdsa.P256(), sha256.New, kDSA, make([]byte, 32))
	k25519, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	ed25519.Sign(k25519, make([]byte, 32))
	rsa.VerifyPKCS1v15(&rsa.PublicKey{}, "", nil, nil)
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
