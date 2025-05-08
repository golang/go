// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crypto_test

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"internal/testenv"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

type messageSignerOnly struct {
	k *rsa.PrivateKey
}

func (s *messageSignerOnly) Public() crypto.PublicKey {
	return s.k.Public()
}

func (s *messageSignerOnly) Sign(_ io.Reader, _ []byte, _ crypto.SignerOpts) ([]byte, error) {
	return nil, errors.New("unimplemented")
}

func (s *messageSignerOnly) SignMessage(rand io.Reader, msg []byte, opts crypto.SignerOpts) ([]byte, error) {
	h := opts.HashFunc().New()
	h.Write(msg)
	digest := h.Sum(nil)
	return s.k.Sign(rand, digest, opts)
}

func TestSignMessage(t *testing.T) {
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
	k, _ := x509.ParsePKCS1PrivateKey(block.Bytes)

	msg := []byte("hello :)")

	h := crypto.SHA256.New()
	h.Write(msg)
	digest := h.Sum(nil)

	sig, err := crypto.SignMessage(k, rand.Reader, msg, &rsa.PSSOptions{Hash: crypto.SHA256})
	if err != nil {
		t.Fatalf("SignMessage failed with Signer: %s", err)
	}
	if err := rsa.VerifyPSS(&k.PublicKey, crypto.SHA256, digest, sig, nil); err != nil {
		t.Errorf("VerifyPSS failed for Signer signature: %s", err)
	}

	sig, err = crypto.SignMessage(&messageSignerOnly{k}, rand.Reader, msg, &rsa.PSSOptions{Hash: crypto.SHA256})
	if err != nil {
		t.Fatalf("SignMessage failed with MessageSigner: %s", err)
	}
	if err := rsa.VerifyPSS(&k.PublicKey, crypto.SHA256, digest, sig, nil); err != nil {
		t.Errorf("VerifyPSS failed for MessageSigner signature: %s", err)
	}
}

func TestDisallowedAssemblyInstructions(t *testing.T) {
	// This test enforces the cryptography assembly policy rule that we do not
	// use BYTE or WORD instructions, since these instructions can obscure what
	// the assembly is actually doing. If we do not support specific
	// instructions in the assembler, we should not be using them until we do.
	//
	// Instead of using the output of the 'go tool asm' tool, we take the simple
	// approach and just search the text of .s files for usage of BYTE and WORD.
	// We do this because the assembler itself will sometimes insert WORD
	// instructions for things like function preambles etc.

	boringSigPath := filepath.Join("internal", "boring", "sig")

	matcher, err := regexp.Compile(`(^|;)\s(BYTE|WORD)\s`)
	if err != nil {
		t.Fatal(err)
	}
	if err := filepath.WalkDir(filepath.Join(testenv.GOROOT(t), "src/crypto"), func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			t.Fatal(err)
		}
		if d.IsDir() || !strings.HasSuffix(path, ".s") {
			return nil
		}
		if strings.Contains(path, boringSigPath) {
			return nil
		}

		f, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}

		i := 1
		for line := range bytes.Lines(f) {
			if matcher.Match(line) {
				t.Errorf("%s:%d assembly contains BYTE or WORD instruction (%q)", path, i, string(line))
			}
			i++
		}

		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
