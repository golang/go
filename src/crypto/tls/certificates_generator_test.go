// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

//go:generate go test -run ^TestGenerateCertificates$ crypto/tls -generate

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/fips140"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"flag"
	"fmt"
	"internal/testenv"
	"math/big"
	"os"
	"strings"
	"testing"
	"testing/cryptotest"
	"time"
)

var generate = flag.Bool("generate", false, "regenerate certificates_test.go")

func TestGenerateCertificates(t *testing.T) {
	testenv.MustHaveSource(t)
	if testing.Short() && !*generate {
		t.Skip("set -generate to regenerate certificates_test.go, or run without -short to check")
	}
	if fips140.Version() == "v1.0.0" {
		t.Skip("FIPS 140-3 module v1.0.0 doesn't support SetGlobalRandom")
	}

	// Allow RSA keys below 1024 bits for testRSA512.
	t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",rsa1024min=0")
	// Unset cryptocustomrand to avoid MaybeReadByte non-determinism.
	t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",cryptocustomrand=0")
	cryptotest.SetGlobalRandom(t, 0)

	notBefore := time.Unix(1476984729, 0).Add(-100 * 24 * time.Hour)
	notAfter := time.Unix(1476984729, 0).Add(100 * 24 * time.Hour)
	serial := int64(0)
	nextSerial := func() *big.Int {
		serial++
		return big.NewInt(serial)
	}

	// Root CA key and cert.
	rootKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	rootTemplate := &x509.Certificate{
		SerialNumber:          nextSerial(),
		Subject:               pkix.Name{CommonName: "Root"},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	rootDER, err := x509.CreateCertificate(rand.Reader, rootTemplate, rootTemplate, &rootKey.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	rootCert, err := x509.ParseCertificate(rootDER)
	if err != nil {
		t.Fatal(err)
	}

	// Client Root CA key and cert.
	clientRootKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	clientRootTemplate := &x509.Certificate{
		SerialNumber:          nextSerial(),
		Subject:               pkix.Name{CommonName: "Client Root"},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	clientRootDER, err := x509.CreateCertificate(rand.Reader, clientRootTemplate, clientRootTemplate, &clientRootKey.PublicKey, clientRootKey)
	if err != nil {
		t.Fatal(err)
	}
	clientRootCert, err := x509.ParseCertificate(clientRootDER)
	if err != nil {
		t.Fatal(err)
	}

	// Helper to create a leaf template.
	serverLeaf := func(cn string, san string) *x509.Certificate {
		return &x509.Certificate{
			SerialNumber:          nextSerial(),
			Subject:               pkix.Name{CommonName: cn},
			NotBefore:             notBefore,
			NotAfter:              notAfter,
			KeyUsage:              x509.KeyUsageDigitalSignature,
			ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
			BasicConstraintsValid: true,
			DNSNames:              []string{san},
		}
	}
	clientLeaf := func(cn string, san string) *x509.Certificate {
		return &x509.Certificate{
			SerialNumber:          nextSerial(),
			Subject:               pkix.Name{CommonName: cn},
			NotBefore:             notBefore,
			NotAfter:              notAfter,
			KeyUsage:              x509.KeyUsageDigitalSignature,
			ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			BasicConstraintsValid: true,
			DNSNames:              []string{san},
		}
	}

	type certKeyPair struct {
		name    string
		comment string
		certPEM string
		keyPEM  string
	}
	var pairs []certKeyPair

	emit := func(name, comment string, certDER []byte, key any) {
		keyDER, err := x509.MarshalPKCS8PrivateKey(key)
		if err != nil {
			t.Fatal(err)
		}
		certPEM := string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER}))
		keyPEM := string(pem.EncodeToMemory(&pem.Block{Type: "TESTING KEY", Bytes: keyDER}))
		pairs = append(pairs, certKeyPair{name, comment, strings.TrimSpace(certPEM), strings.TrimSpace(keyPEM)})
	}

	// Roots.
	emit("testRoot", "Self-signed RSA 2048 root CA, CN=Root.", rootDER, rootKey)
	emit("testClientRoot", "Self-signed RSA 2048 root CA, CN=Client Root.", clientRootDER, clientRootKey)

	// Server certs issued by root.

	// ECDSA P-256 (default).
	ecdsaP256Key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := serverLeaf("ECDSA P-256", "test.golang.example")
	der, err := x509.CreateCertificate(rand.Reader, tmpl, rootCert, &ecdsaP256Key.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testECDSAP256", "ECDSA P-256 server leaf, SAN=test.golang.example, issued by Root.", der, ecdsaP256Key)

	// RSA 2048.
	rsa2048Key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = serverLeaf("RSA 2048", "test.golang.example")
	der, err = x509.CreateCertificate(rand.Reader, tmpl, rootCert, &rsa2048Key.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testRSA2048", "RSA 2048 server leaf, SAN=test.golang.example, issued by Root.", der, rsa2048Key)

	// ECDSA P-384.
	ecdsaP384Key, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = serverLeaf("ECDSA P-384", "test.golang.example")
	der, err = x509.CreateCertificate(rand.Reader, tmpl, rootCert, &ecdsaP384Key.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testECDSAP384", "ECDSA P-384 server leaf, SAN=test.golang.example, issued by Root.", der, ecdsaP384Key)

	// ECDSA P-521.
	ecdsaP521Key, err := ecdsa.GenerateKey(elliptic.P521(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = serverLeaf("ECDSA P-521", "test.golang.example")
	der, err = x509.CreateCertificate(rand.Reader, tmpl, rootCert, &ecdsaP521Key.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testECDSAP521", "ECDSA P-521 server leaf, SAN=test.golang.example, issued by Root.", der, ecdsaP521Key)

	// Ed25519.
	ed25519Pub, ed25519Key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = serverLeaf("Ed25519", "test.golang.example")
	der, err = x509.CreateCertificate(rand.Reader, tmpl, rootCert, ed25519Pub, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testEd25519", "Ed25519 server leaf, SAN=test.golang.example, issued by Root.", der, ed25519Key)

	// RSA-PSS: signed by root with SHA512WithRSAPSS. The leaf SPKI is
	// rsaEncryption while the signatureAlgorithm is rsassaPss, for use
	// with the rsa_pss_rsae_* SignatureSchemes.
	rsaPSSKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = serverLeaf("RSA-PSS", "test.golang.example")
	tmpl.SignatureAlgorithm = x509.SHA512WithRSAPSS
	der, err = x509.CreateCertificate(rand.Reader, tmpl, rootCert, &rsaPSSKey.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testRSAPSS", "RSA 2048 server leaf, SAN=test.golang.example, issued by Root.\n\t// Signature algorithm is SHA512WithRSAPSS (rsaEncryption SPKI, rsassaPss signature).", der, rsaPSSKey)

	// RSA 1024: key is intentionally too small for rsa_pss_rsae_sha512
	// (which requires at least 1040 bits), but large enough for
	// rsa_pss_rsae_sha256. Used by TestHandshakeServerRSAPSS.
	rsa1024Key, err := rsa.GenerateKey(rand.Reader, 1024)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = serverLeaf("RSA 1024", "test.golang.example")
	der, err = x509.CreateCertificate(rand.Reader, tmpl, rootCert, &rsa1024Key.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testRSA1024", "RSA 1024 server leaf, SAN=test.golang.example, issued by Root.\n\t// Key is too small for rsa_pss_rsae_sha512; used by TestHandshakeServerRSAPSS.", der, rsa1024Key)

	// RSA 512: key is too small for any rsa_pss_rsae_* SignatureScheme
	// (the smallest, SHA-256, requires at least 528 bits). Used by
	// TestKeyTooSmallForRSAPSS.
	rsa512Key, err := rsa.GenerateKey(rand.Reader, 512)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = serverLeaf("RSA 512", "test.golang.example")
	der, err = x509.CreateCertificate(rand.Reader, tmpl, rootCert, &rsa512Key.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testRSA512", "RSA 512 server leaf, SAN=test.golang.example, issued by Root.\n\t// Key is too small for any rsa_pss_rsae_*; used by TestKeyTooSmallForRSAPSS.", der, rsa512Key)

	// SNI cert (different SAN for SNI mismatch testing).
	sniKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = serverLeaf("different.example.com", "different.example.com")
	der, err = x509.CreateCertificate(rand.Reader, tmpl, rootCert, &sniKey.PublicKey, rootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testSNI", "ECDSA P-256 server leaf, SAN=different.example.com, issued by Root.", der, sniKey)

	// Client certs issued by client root.

	clientRSAKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = clientLeaf("clientAuth RSA 2048", "test.golang.example")
	der, err = x509.CreateCertificate(rand.Reader, tmpl, clientRootCert, &clientRSAKey.PublicKey, clientRootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testClientRSA2048", "RSA 2048 client leaf, SAN=test.golang.example, issued by Client Root.", der, clientRSAKey)

	clientECDSAKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = clientLeaf("clientAuth ECDSA P-256", "test.golang.example")
	der, err = x509.CreateCertificate(rand.Reader, tmpl, clientRootCert, &clientECDSAKey.PublicKey, clientRootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testClientECDSAP256", "ECDSA P-256 client leaf, SAN=test.golang.example, issued by Client Root.", der, clientECDSAKey)

	clientEd25519Pub, clientEd25519Key, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = clientLeaf("clientAuth Ed25519", "test.golang.example")
	der, err = x509.CreateCertificate(rand.Reader, tmpl, clientRootCert, clientEd25519Pub, clientRootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testClientEd25519", "Ed25519 client leaf, SAN=test.golang.example, issued by Client Root.", der, clientEd25519Key)

	// Client RSA-PSS: signed by client root with SHA512WithRSAPSS. The leaf
	// SPKI is rsaEncryption while the signatureAlgorithm is rsassaPss.
	clientRSAPSSKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	tmpl = clientLeaf("clientAuth RSA-PSS", "test.golang.example")
	tmpl.SignatureAlgorithm = x509.SHA512WithRSAPSS
	der, err = x509.CreateCertificate(rand.Reader, tmpl, clientRootCert, &clientRSAPSSKey.PublicKey, clientRootKey)
	if err != nil {
		t.Fatal(err)
	}
	emit("testClientRSAPSS", "RSA 2048 client leaf, SAN=test.golang.example, issued by Client Root.\n\t// Signature algorithm is SHA512WithRSAPSS (rsaEncryption SPKI, rsassaPss signature).", der, clientRSAPSSKey)

	// Generate certificates_test.go.
	var buf bytes.Buffer
	fmt.Fprint(&buf, `// Code generated by certificates_generator_test.go; DO NOT EDIT.
// To regenerate, run: go generate

package tls

import "crypto/x509"

`)

	fmt.Fprint(&buf, `var (
`)
	for _, p := range pairs {
		fmt.Fprintf(&buf, "\t// %s\n", p.comment)
		fmt.Fprintf(&buf, "\t%sCert = parseTestCert(%sCertPEM, %sKeyPEM)\n\n",
			p.name, p.name, p.name)
	}
	fmt.Fprint(&buf, `	// x509.CertPool containing testRootCert.
	testRootCertPool = newTestCertPool(testRootCertPEM)
	// x509.CertPool containing testClientRootCert.
	testClientRootCertPool = newTestCertPool(testClientRootCertPEM)
)

`)

	for _, p := range pairs {
		fmt.Fprintf(&buf, "const %sCertPEM = `\n%s`\n\n", p.name, p.certPEM)
		fmt.Fprintf(&buf, "const %sKeyPEM = `\n%s`\n\n", p.name, p.keyPEM)
	}

	fmt.Fprint(&buf, `func parseTestCert(certPEM, keyPEM string) Certificate {
	tlsCert, err := X509KeyPair([]byte(certPEM), []byte(testingKey(keyPEM)))
	if err != nil {
		panic(err)
	}
	return tlsCert
}

func newTestCertPool(certPEM string) *x509.CertPool {
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM([]byte(certPEM)) {
		panic("failed to parse certificate for pool")
	}
	return pool
}
`)

	if *generate {
		if err := os.WriteFile("certificates_test.go", buf.Bytes(), 0644); err != nil {
			t.Fatal(err)
		}
		t.Log("wrote certificates_test.go")
	} else {
		// Check that the generated content matches the existing file.
		existing, err := os.ReadFile("certificates_test.go")
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(existing, buf.Bytes()) {
			t.Fatal("certificates_test.go is out of date; run go generate to update it")
		}
	}
}
