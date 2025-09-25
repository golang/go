// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"crypto"
	"crypto/rand"
	"fmt"
	"internal/testenv"
	"io/fs"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"

	"crypto/internal/fips140"
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

var allCASTs = []string{
	"AES-CBC",
	"CTR_DRBG",
	"CounterKDF",
	"DetECDSA P-256 SHA2-512 sign",
	"ECDH PCT",
	"ECDSA P-256 SHA2-512 sign and verify",
	"ECDSA PCT",
	"Ed25519 sign and verify",
	"Ed25519 sign and verify PCT",
	"HKDF-SHA2-256",
	"HMAC-SHA2-256",
	"KAS-ECC-SSC P-256",
	"ML-KEM PCT",
	"ML-KEM PCT",
	"ML-KEM-768",
	"PBKDF2",
	"RSA sign and verify PCT",
	"RSASSA-PKCS-v1.5 2048-bit sign and verify",
	"SHA2-256",
	"SHA2-512",
	"TLSv1.2-SHA2-256",
	"TLSv1.3-SHA2-256",
	"cSHAKE128",
}

func TestAllCASTs(t *testing.T) {
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
	var foundCASTs []string
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
			foundCASTs = append(foundCASTs, string(m[2]))
		}
		return nil
	}); err != nil {
		t.Fatalf("WalkDir: %v", err)
	}

	slices.Sort(foundCASTs)
	if !slices.Equal(foundCASTs, allCASTs) {
		t.Errorf("AllCASTs is out of date. Found CASTs: %#v", foundCASTs)
	}
}

// TestConditionals causes the conditional CASTs and PCTs to be invoked.
func TestConditionals(t *testing.T) {
	mlkem.GenerateKey768()
	kDH, err := ecdh.GenerateKey(ecdh.P256(), rand.Reader)
	if err != nil {
		t.Error(err)
	} else {
		ecdh.ECDH(ecdh.P256(), kDH, kDH.PublicKey())
	}
	kDSA, err := ecdsa.GenerateKey(ecdsa.P256(), rand.Reader)
	if err != nil {
		t.Error(err)
	} else {
		ecdsa.SignDeterministic(ecdsa.P256(), sha256.New, kDSA, make([]byte, 32))
	}
	k25519, err := ed25519.GenerateKey()
	if err != nil {
		t.Error(err)
	} else {
		ed25519.Sign(k25519, make([]byte, 32))
	}
	kRSA, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Error(err)
	} else {
		rsa.SignPKCS1v15(kRSA, crypto.SHA256.String(), make([]byte, 32))
	}
	t.Log("completed successfully")
}

func TestCASTPasses(t *testing.T) {
	testenv.MustHaveExec(t)
	if err := fips140.Supported(); err != nil {
		t.Skipf("FIPS140 not supported: %v", err)
	}

	cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^TestConditionals$", "-test.v")
	cmd.Env = append(cmd.Env, "GODEBUG=fips140=debug")
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	if err != nil || !strings.Contains(string(out), "completed successfully") {
		t.Errorf("TestConditionals did not complete successfully")
	}

	for _, name := range allCASTs {
		t.Run(name, func(t *testing.T) {
			if !strings.Contains(string(out), fmt.Sprintf("passed: %s\n", name)) {
				t.Errorf("CAST/PCT %s success was not logged", name)
			} else {
				t.Logf("CAST/PCT succeeded: %s", name)
			}
		})
	}
}

func TestCASTFailures(t *testing.T) {
	testenv.MustHaveExec(t)
	if err := fips140.Supported(); err != nil {
		t.Skipf("FIPS140 not supported: %v", err)
	}

	for _, name := range allCASTs {
		t.Run(name, func(t *testing.T) {
			// Don't parallelize if running in verbose mode, to produce a less
			// confusing recoding for the validation lab.
			if !testing.Verbose() {
				t.Parallel()
			}
			t.Logf("Testing CAST/PCT failure...")
			cmd := testenv.Command(t, testenv.Executable(t), "-test.run=TestConditionals", "-test.v")
			cmd.Env = append(cmd.Env, fmt.Sprintf("GODEBUG=failfipscast=%s,fips140=on", name))
			out, err := cmd.CombinedOutput()
			t.Logf("%s", out)
			if err == nil {
				t.Fatal("Test did not fail as expected")
			}
			if strings.Contains(string(out), "completed successfully") {
				t.Errorf("CAST/PCT %s failure did not stop the program", name)
			} else if !strings.Contains(string(out), "self-test failed: "+name) {
				t.Errorf("CAST/PCT %s failure did not log the expected message", name)
			} else {
				t.Logf("CAST/PCT %s failed as expected and caused the program to exit", name)
			}
		})
	}
}
