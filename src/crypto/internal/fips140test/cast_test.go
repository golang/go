// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
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
	// Parse an RSA key to hit the PCT rather than generating one (which is slow).
	block, _ := pem.Decode([]byte(strings.ReplaceAll(
		`-----BEGIN RSA TESTING KEY-----
MIIEowIBAAKCAQEAsPnoGUOnrpiSqt4XynxA+HRP7S+BSObI6qJ7fQAVSPtRkqso
tWxQYLEYzNEx5ZSHTGypibVsJylvCfuToDTfMul8b/CZjP2Ob0LdpYrNH6l5hvFE
89FU1nZQF15oVLOpUgA7wGiHuEVawrGfey92UE68mOyUVXGweJIVDdxqdMoPvNNU
l86BU02vlBiESxOuox+dWmuVV7vfYZ79Toh/LUK43YvJh+rhv4nKuF7iHjVjBd9s
B6iDjj70HFldzOQ9r8SRI+9NirupPTkF5AKNe6kUhKJ1luB7S27ZkvB3tSTT3P59
3VVJvnzOjaA1z6Cz+4+eRvcysqhrRgFlwI9TEwIDAQABAoIBAEEYiyDP29vCzx/+
dS3LqnI5BjUuJhXUnc6AWX/PCgVAO+8A+gZRgvct7PtZb0sM6P9ZcLrweomlGezI
FrL0/6xQaa8bBr/ve/a8155OgcjFo6fZEw3Dz7ra5fbSiPmu4/b/kvrg+Br1l77J
aun6uUAs1f5B9wW+vbR7tzbT/mxaUeDiBzKpe15GwcvbJtdIVMa2YErtRjc1/5B2
BGVXyvlJv0SIlcIEMsHgnAFOp1ZgQ08aDzvilLq8XVMOahAhP1O2A3X8hKdXPyrx
IVWE9bS9ptTo+eF6eNl+d7htpKGEZHUxinoQpWEBTv+iOoHsVunkEJ3vjLP3lyI/
fY0NQ1ECgYEA3RBXAjgvIys2gfU3keImF8e/TprLge1I2vbWmV2j6rZCg5r/AS0u
pii5CvJ5/T5vfJPNgPBy8B/yRDs+6PJO1GmnlhOkG9JAIPkv0RBZvR0PMBtbp6nT
Y3yo1lwamBVBfY6rc0sLTzosZh2aGoLzrHNMQFMGaauORzBFpY5lU50CgYEAzPHl
u5DI6Xgep1vr8QvCUuEesCOgJg8Yh1UqVoY/SmQh6MYAv1I9bLGwrb3WW/7kqIoD
fj0aQV5buVZI2loMomtU9KY5SFIsPV+JuUpy7/+VE01ZQM5FdY8wiYCQiVZYju9X
Wz5LxMNoz+gT7pwlLCsC4N+R8aoBk404aF1gum8CgYAJ7VTq7Zj4TFV7Soa/T1eE
k9y8a+kdoYk3BASpCHJ29M5R2KEA7YV9wrBklHTz8VzSTFTbKHEQ5W5csAhoL5Fo
qoHzFFi3Qx7MHESQb9qHyolHEMNx6QdsHUn7rlEnaTTyrXh3ifQtD6C0yTmFXUIS
CW9wKApOrnyKJ9nI0HcuZQKBgQCMtoV6e9VGX4AEfpuHvAAnMYQFgeBiYTkBKltQ
XwozhH63uMMomUmtSG87Sz1TmrXadjAhy8gsG6I0pWaN7QgBuFnzQ/HOkwTm+qKw
AsrZt4zeXNwsH7QXHEJCFnCmqw9QzEoZTrNtHJHpNboBuVnYcoueZEJrP8OnUG3r
UjmopwKBgAqB2KYYMUqAOvYcBnEfLDmyZv9BTVNHbR2lKkMYqv5LlvDaBxVfilE0
2riO4p6BaAdvzXjKeRrGNEKoHNBpOSfYCOM16NjL8hIZB1CaV3WbT5oY+jp7Mzd5
7d56RZOE+ERK2uz/7JX9VSsM/LbH9pJibd4e8mikDS9ntciqOH/3
-----END RSA TESTING KEY-----`, "TESTING KEY", "PRIVATE KEY")))
	if _, err := x509.ParsePKCS1PrivateKey(block.Bytes); err != nil {
		t.Fatal(err)
	}
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
