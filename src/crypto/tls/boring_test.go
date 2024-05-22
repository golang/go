// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto

package tls

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/internal/boring/fipstls"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"internal/obscuretestdata"
	"math/big"
	"net"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestBoringServerProtocolVersion(t *testing.T) {
	test := func(t *testing.T, name string, v uint16, msg string) {
		t.Run(name, func(t *testing.T) {
			serverConfig := testConfig.Clone()
			serverConfig.MinVersion = VersionSSL30
			clientConfig := testConfig.Clone()
			clientConfig.MinVersion = v
			clientConfig.MaxVersion = v
			_, _, err := testHandshake(t, clientConfig, serverConfig)
			if msg == "" {
				if err != nil {
					t.Fatalf("got error: %v, expected success", err)
				}
			} else {
				if err == nil {
					t.Fatalf("got success, expected error")
				}
				if !strings.Contains(err.Error(), msg) {
					t.Fatalf("got error %v, expected %q", err, msg)
				}
			}
		})
	}

	test(t, "VersionTLS10", VersionTLS10, "")
	test(t, "VersionTLS11", VersionTLS11, "")
	test(t, "VersionTLS12", VersionTLS12, "")
	test(t, "VersionTLS13", VersionTLS13, "")

	t.Run("fipstls", func(t *testing.T) {
		fipstls.Force()
		defer fipstls.Abandon()
		test(t, "VersionTLS10", VersionTLS10, "supported versions")
		test(t, "VersionTLS11", VersionTLS11, "supported versions")
		test(t, "VersionTLS12", VersionTLS12, "")
		test(t, "VersionTLS13", VersionTLS13, "supported versions")
	})
}

func isBoringVersion(v uint16) bool {
	return v == VersionTLS12
}

func isBoringCipherSuite(id uint16) bool {
	switch id {
	case TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
		TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
		TLS_RSA_WITH_AES_128_GCM_SHA256,
		TLS_RSA_WITH_AES_256_GCM_SHA384:
		return true
	}
	return false
}

func isBoringCurve(id CurveID) bool {
	switch id {
	case CurveP256, CurveP384, CurveP521:
		return true
	}
	return false
}

func isECDSA(id uint16) bool {
	for _, suite := range cipherSuites {
		if suite.id == id {
			return suite.flags&suiteECSign == suiteECSign
		}
	}
	panic(fmt.Sprintf("unknown cipher suite %#x", id))
}

func isBoringSignatureScheme(alg SignatureScheme) bool {
	switch alg {
	default:
		return false
	case PKCS1WithSHA256,
		ECDSAWithP256AndSHA256,
		PKCS1WithSHA384,
		ECDSAWithP384AndSHA384,
		PKCS1WithSHA512,
		ECDSAWithP521AndSHA512,
		PSSWithSHA256,
		PSSWithSHA384,
		PSSWithSHA512:
		// ok
	}
	return true
}

func TestBoringServerCipherSuites(t *testing.T) {
	serverConfig := testConfig.Clone()
	serverConfig.CipherSuites = allCipherSuites()
	serverConfig.Certificates = make([]Certificate, 1)

	for _, id := range allCipherSuites() {
		if isECDSA(id) {
			serverConfig.Certificates[0].Certificate = [][]byte{testECDSACertificate}
			serverConfig.Certificates[0].PrivateKey = testECDSAPrivateKey
		} else {
			serverConfig.Certificates[0].Certificate = [][]byte{testRSACertificate}
			serverConfig.Certificates[0].PrivateKey = testRSAPrivateKey
		}
		serverConfig.BuildNameToCertificate()
		t.Run(fmt.Sprintf("suite=%#x", id), func(t *testing.T) {
			clientHello := &clientHelloMsg{
				vers:               VersionTLS12,
				random:             make([]byte, 32),
				cipherSuites:       []uint16{id},
				compressionMethods: []uint8{compressionNone},
				supportedCurves:    defaultCurvePreferences(),
				supportedPoints:    []uint8{pointFormatUncompressed},
			}

			testClientHello(t, serverConfig, clientHello)
			t.Run("fipstls", func(t *testing.T) {
				fipstls.Force()
				defer fipstls.Abandon()
				msg := ""
				if !isBoringCipherSuite(id) {
					msg = "no cipher suite supported by both client and server"
				}
				testClientHelloFailure(t, serverConfig, clientHello, msg)
			})
		})
	}
}

func TestBoringServerCurves(t *testing.T) {
	serverConfig := testConfig.Clone()
	serverConfig.Certificates = make([]Certificate, 1)
	serverConfig.Certificates[0].Certificate = [][]byte{testECDSACertificate}
	serverConfig.Certificates[0].PrivateKey = testECDSAPrivateKey
	serverConfig.BuildNameToCertificate()

	for _, curveid := range defaultCurvePreferences() {
		t.Run(fmt.Sprintf("curve=%d", curveid), func(t *testing.T) {
			clientConfig := testConfig.Clone()
			clientConfig.CurvePreferences = []CurveID{curveid}
			if curveid == x25519Kyber768Draft00 {
				// x25519Kyber768Draft00 is not supported standalone.
				clientConfig.CurvePreferences = append(clientConfig.CurvePreferences, X25519)
			}
			if _, _, err := testHandshake(t, clientConfig, serverConfig); err != nil {
				t.Fatalf("got error: %v, expected success", err)
			}

			// With fipstls forced, bad curves should be rejected.
			t.Run("fipstls", func(t *testing.T) {
				fipstls.Force()
				defer fipstls.Abandon()
				_, _, err := testHandshake(t, clientConfig, serverConfig)
				if err != nil && isBoringCurve(curveid) {
					t.Fatalf("got error: %v, expected success", err)
				} else if err == nil && !isBoringCurve(curveid) {
					t.Fatalf("got success, expected error")
				}
			})
		})
	}
}

func boringHandshake(t *testing.T, clientConfig, serverConfig *Config) (clientErr, serverErr error) {
	c, s := localPipe(t)
	client := Client(c, clientConfig)
	server := Server(s, serverConfig)
	done := make(chan error, 1)
	go func() {
		done <- client.Handshake()
		c.Close()
	}()
	serverErr = server.Handshake()
	s.Close()
	clientErr = <-done
	return
}

func TestBoringServerSignatureAndHash(t *testing.T) {
	defer func() {
		testingOnlyForceClientHelloSignatureAlgorithms = nil
	}()

	for _, sigHash := range defaultSupportedSignatureAlgorithms {
		t.Run(fmt.Sprintf("%v", sigHash), func(t *testing.T) {
			serverConfig := testConfig.Clone()
			serverConfig.Certificates = make([]Certificate, 1)

			testingOnlyForceClientHelloSignatureAlgorithms = []SignatureScheme{sigHash}

			sigType, _, _ := typeAndHashFromSignatureScheme(sigHash)
			switch sigType {
			case signaturePKCS1v15, signatureRSAPSS:
				serverConfig.CipherSuites = []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}
				serverConfig.Certificates[0].Certificate = [][]byte{testRSA2048Certificate}
				serverConfig.Certificates[0].PrivateKey = testRSA2048PrivateKey
			case signatureEd25519:
				serverConfig.CipherSuites = []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256}
				serverConfig.Certificates[0].Certificate = [][]byte{testEd25519Certificate}
				serverConfig.Certificates[0].PrivateKey = testEd25519PrivateKey
			case signatureECDSA:
				serverConfig.CipherSuites = []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256}
				serverConfig.Certificates[0].Certificate = [][]byte{testECDSACertificate}
				serverConfig.Certificates[0].PrivateKey = testECDSAPrivateKey
			}
			serverConfig.BuildNameToCertificate()
			// PKCS#1 v1.5 signature algorithms can't be used standalone in TLS
			// 1.3, and the ECDSA ones bind to the curve used.
			serverConfig.MaxVersion = VersionTLS12

			clientErr, serverErr := boringHandshake(t, testConfig, serverConfig)
			if clientErr != nil {
				t.Fatalf("expected handshake with %#x to succeed; client error: %v; server error: %v", sigHash, clientErr, serverErr)
			}

			// With fipstls forced, bad curves should be rejected.
			t.Run("fipstls", func(t *testing.T) {
				fipstls.Force()
				defer fipstls.Abandon()
				clientErr, _ := boringHandshake(t, testConfig, serverConfig)
				if isBoringSignatureScheme(sigHash) {
					if clientErr != nil {
						t.Fatalf("expected handshake with %#x to succeed; err=%v", sigHash, clientErr)
					}
				} else {
					if clientErr == nil {
						t.Fatalf("expected handshake with %#x to fail, but it succeeded", sigHash)
					}
				}
			})
		})
	}
}

func TestBoringClientHello(t *testing.T) {
	// Test that no matter what we put in the client config,
	// the client does not offer non-FIPS configurations.
	fipstls.Force()
	defer fipstls.Abandon()

	c, s := net.Pipe()
	defer c.Close()
	defer s.Close()

	clientConfig := testConfig.Clone()
	// All sorts of traps for the client to avoid.
	clientConfig.MinVersion = VersionSSL30
	clientConfig.MaxVersion = VersionTLS13
	clientConfig.CipherSuites = allCipherSuites()
	clientConfig.CurvePreferences = defaultCurvePreferences()

	go Client(c, clientConfig).Handshake()
	srv := Server(s, testConfig)
	msg, err := srv.readHandshake(nil)
	if err != nil {
		t.Fatal(err)
	}
	hello, ok := msg.(*clientHelloMsg)
	if !ok {
		t.Fatalf("unexpected message type %T", msg)
	}

	if !isBoringVersion(hello.vers) {
		t.Errorf("client vers=%#x, want %#x (TLS 1.2)", hello.vers, VersionTLS12)
	}
	for _, v := range hello.supportedVersions {
		if !isBoringVersion(v) {
			t.Errorf("client offered disallowed version %#x", v)
		}
	}
	for _, id := range hello.cipherSuites {
		if !isBoringCipherSuite(id) {
			t.Errorf("client offered disallowed suite %#x", id)
		}
	}
	for _, id := range hello.supportedCurves {
		if !isBoringCurve(id) {
			t.Errorf("client offered disallowed curve %d", id)
		}
	}
	for _, sigHash := range hello.supportedSignatureAlgorithms {
		if !isBoringSignatureScheme(sigHash) {
			t.Errorf("client offered disallowed signature-and-hash %v", sigHash)
		}
	}
}

func TestBoringCertAlgs(t *testing.T) {
	// NaCl, arm and wasm time out generating keys. Nothing in this test is architecture-specific, so just don't bother on those.
	if runtime.GOOS == "nacl" || runtime.GOARCH == "arm" || runtime.GOOS == "js" {
		t.Skipf("skipping on %s/%s because key generation takes too long", runtime.GOOS, runtime.GOARCH)
	}

	// Set up some roots, intermediate CAs, and leaf certs with various algorithms.
	// X_Y is X signed by Y.
	R1 := boringCert(t, "R1", boringRSAKey(t, 2048), nil, boringCertCA|boringCertFIPSOK)
	R2 := boringCert(t, "R2", boringRSAKey(t, 512), nil, boringCertCA)

	M1_R1 := boringCert(t, "M1_R1", boringECDSAKey(t, elliptic.P256()), R1, boringCertCA|boringCertFIPSOK)
	M2_R1 := boringCert(t, "M2_R1", boringECDSAKey(t, elliptic.P224()), R1, boringCertCA)

	I_R1 := boringCert(t, "I_R1", boringRSAKey(t, 3072), R1, boringCertCA|boringCertFIPSOK)
	I_R2 := boringCert(t, "I_R2", I_R1.key, R2, boringCertCA|boringCertFIPSOK)
	I_M1 := boringCert(t, "I_M1", I_R1.key, M1_R1, boringCertCA|boringCertFIPSOK)
	I_M2 := boringCert(t, "I_M2", I_R1.key, M2_R1, boringCertCA|boringCertFIPSOK)

	L1_I := boringCert(t, "L1_I", boringECDSAKey(t, elliptic.P384()), I_R1, boringCertLeaf|boringCertFIPSOK)
	L2_I := boringCert(t, "L2_I", boringRSAKey(t, 1024), I_R1, boringCertLeaf)

	// client verifying server cert
	testServerCert := func(t *testing.T, desc string, pool *x509.CertPool, key interface{}, list [][]byte, ok bool) {
		clientConfig := testConfig.Clone()
		clientConfig.RootCAs = pool
		clientConfig.InsecureSkipVerify = false
		clientConfig.ServerName = "example.com"

		serverConfig := testConfig.Clone()
		serverConfig.Certificates = []Certificate{{Certificate: list, PrivateKey: key}}
		serverConfig.BuildNameToCertificate()

		clientErr, _ := boringHandshake(t, clientConfig, serverConfig)

		if (clientErr == nil) == ok {
			if ok {
				t.Logf("%s: accept", desc)
			} else {
				t.Logf("%s: reject", desc)
			}
		} else {
			if ok {
				t.Errorf("%s: BAD reject (%v)", desc, clientErr)
			} else {
				t.Errorf("%s: BAD accept", desc)
			}
		}
	}

	// server verifying client cert
	testClientCert := func(t *testing.T, desc string, pool *x509.CertPool, key interface{}, list [][]byte, ok bool) {
		clientConfig := testConfig.Clone()
		clientConfig.ServerName = "example.com"
		clientConfig.Certificates = []Certificate{{Certificate: list, PrivateKey: key}}

		serverConfig := testConfig.Clone()
		serverConfig.ClientCAs = pool
		serverConfig.ClientAuth = RequireAndVerifyClientCert

		_, serverErr := boringHandshake(t, clientConfig, serverConfig)

		if (serverErr == nil) == ok {
			if ok {
				t.Logf("%s: accept", desc)
			} else {
				t.Logf("%s: reject", desc)
			}
		} else {
			if ok {
				t.Errorf("%s: BAD reject (%v)", desc, serverErr)
			} else {
				t.Errorf("%s: BAD accept", desc)
			}
		}
	}

	// Run simple basic test with known answers before proceeding to
	// exhaustive test with computed answers.
	r1pool := x509.NewCertPool()
	r1pool.AddCert(R1.cert)
	testServerCert(t, "basic", r1pool, L2_I.key, [][]byte{L2_I.der, I_R1.der}, true)
	testClientCert(t, "basic (client cert)", r1pool, L2_I.key, [][]byte{L2_I.der, I_R1.der}, true)
	fipstls.Force()
	testServerCert(t, "basic (fips)", r1pool, L2_I.key, [][]byte{L2_I.der, I_R1.der}, false)
	testClientCert(t, "basic (fips, client cert)", r1pool, L2_I.key, [][]byte{L2_I.der, I_R1.der}, false)
	fipstls.Abandon()

	if t.Failed() {
		t.Fatal("basic test failed, skipping exhaustive test")
	}

	if testing.Short() {
		t.Logf("basic test passed; skipping exhaustive test in -short mode")
		return
	}

	for l := 1; l <= 2; l++ {
		leaf := L1_I
		if l == 2 {
			leaf = L2_I
		}
		for i := 0; i < 64; i++ {
			reachable := map[string]bool{leaf.parentOrg: true}
			reachableFIPS := map[string]bool{leaf.parentOrg: leaf.fipsOK}
			list := [][]byte{leaf.der}
			listName := leaf.name
			addList := func(cond int, c *boringCertificate) {
				if cond != 0 {
					list = append(list, c.der)
					listName += "," + c.name
					if reachable[c.org] {
						reachable[c.parentOrg] = true
					}
					if reachableFIPS[c.org] && c.fipsOK {
						reachableFIPS[c.parentOrg] = true
					}
				}
			}
			addList(i&1, I_R1)
			addList(i&2, I_R2)
			addList(i&4, I_M1)
			addList(i&8, I_M2)
			addList(i&16, M1_R1)
			addList(i&32, M2_R1)

			for r := 1; r <= 3; r++ {
				pool := x509.NewCertPool()
				rootName := ","
				shouldVerify := false
				shouldVerifyFIPS := false
				addRoot := func(cond int, c *boringCertificate) {
					if cond != 0 {
						rootName += "," + c.name
						pool.AddCert(c.cert)
						if reachable[c.org] {
							shouldVerify = true
						}
						if reachableFIPS[c.org] && c.fipsOK {
							shouldVerifyFIPS = true
						}
					}
				}
				addRoot(r&1, R1)
				addRoot(r&2, R2)
				rootName = rootName[1:] // strip leading comma
				testServerCert(t, listName+"->"+rootName[1:], pool, leaf.key, list, shouldVerify)
				testClientCert(t, listName+"->"+rootName[1:]+"(client cert)", pool, leaf.key, list, shouldVerify)
				fipstls.Force()
				testServerCert(t, listName+"->"+rootName[1:]+" (fips)", pool, leaf.key, list, shouldVerifyFIPS)
				testClientCert(t, listName+"->"+rootName[1:]+" (fips, client cert)", pool, leaf.key, list, shouldVerifyFIPS)
				fipstls.Abandon()
			}
		}
	}
}

const (
	boringCertCA = iota
	boringCertLeaf
	boringCertFIPSOK = 0x80
)

func boringRSAKey(t *testing.T, size int) *rsa.PrivateKey {
	k, err := rsa.GenerateKey(rand.Reader, size)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

func boringECDSAKey(t *testing.T, curve elliptic.Curve) *ecdsa.PrivateKey {
	k, err := ecdsa.GenerateKey(curve, rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

type boringCertificate struct {
	name      string
	org       string
	parentOrg string
	der       []byte
	cert      *x509.Certificate
	key       interface{}
	fipsOK    bool
}

func boringCert(t *testing.T, name string, key interface{}, parent *boringCertificate, mode int) *boringCertificate {
	org := name
	parentOrg := ""
	if i := strings.Index(org, "_"); i >= 0 {
		org = org[:i]
		parentOrg = name[i+1:]
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Organization: []string{org},
		},
		NotBefore: time.Unix(0, 0),
		NotAfter:  time.Unix(0, 0),

		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
	}
	if mode&^boringCertFIPSOK == boringCertLeaf {
		tmpl.DNSNames = []string{"example.com"}
	} else {
		tmpl.IsCA = true
		tmpl.KeyUsage |= x509.KeyUsageCertSign
	}

	var pcert *x509.Certificate
	var pkey interface{}
	if parent != nil {
		pcert = parent.cert
		pkey = parent.key
	} else {
		pcert = tmpl
		pkey = key
	}

	var pub interface{}
	switch k := key.(type) {
	case *rsa.PrivateKey:
		pub = &k.PublicKey
	case *ecdsa.PrivateKey:
		pub = &k.PublicKey
	default:
		t.Fatalf("invalid key %T", key)
	}

	der, err := x509.CreateCertificate(rand.Reader, tmpl, pcert, pub, pkey)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatal(err)
	}

	fipsOK := mode&boringCertFIPSOK != 0
	return &boringCertificate{name, org, parentOrg, der, cert, key, fipsOK}
}

// A self-signed test certificate with an RSA key of size 2048, for testing
// RSA-PSS with SHA512. SAN of example.golang.
var (
	testRSA2048Certificate []byte
	testRSA2048PrivateKey  *rsa.PrivateKey
)

func init() {
	block, _ := pem.Decode(obscuretestdata.Rot13([]byte(`
-----ORTVA PREGVSVPNGR-----
ZVVP/mPPNrrtNjVONtVENYUUK/xu4+4mZH9QnemORpDjQDLWXbMVuipANDRYODNj
RwRDZN4TN1HRPuZUDJAgMFOQomNrSj0kZGNkZQRkAGN0ZQInSj0lZQRlZwxkAGN0
ZQInZOVkRQNBOtAIONbGO0SwoJHtD28jttRvZN0TPFdTFVo3QDRONDHNN4VOQjNj
ttRXNbVONDPs8sx0A6vrPOK4VBIVsXvgg4xTpBDYrvzPsfwddUplfZVITRgSFZ6R
4Nl141s/7VdqJ0HgVdAo4CKuEBVQ7lQkE284kY6KoPhi/g5uC3HpruLp3uzYvlIq
ZxMDvMJgsHHWs/1dBgZ+buAt59YEJc4q+6vK0yn1WY3RjPVpxxAwW9uDoS7Co2PF
+RF9Lb55XNnc8XBoycpE8ZOFA38odajwsDqPKiBRBwnz2UHkXmRSK5ZN+sN0zr4P
vbPpPEYJXy+TbA9S8sNOsbM+G+2rny4QYhB95eKE8FeBVIOu3KSBe/EIuwgKpAIS
MXpiQg6q68I6wNXNLXz5ayw9TCcq4i+eNtZONNTwHQOBZN4TN1HqQjRO/jDRNjVS
bQNGOtAIUFHRQQNXOtteOtRSODpQNGNZOtAIUEZONs8RNwNNZOxTN1HqRDDFZOPP
QzI4LJ1joTHhM29fLJ5aZN0TPFdTFVo3QDROPjHNN4VONDPBbLfIpSPOuobdr3JU
qP6I7KKKRPzawu01e8u80li0AE379aFQ3pj2Z+UXinKlfJdey5uwTIXj0igjQ81e
I4WmQh7VsVbt5z8+DAP+7YdQMfm88iQXBefblFIBzHPtzPXSKrj+YN+rB/vDRWGe
7rafqqBrKWRc27Rq5iJ+xzJJ3Dztyp2Tjl8jSeZQVdaeaBmON4bPaQRtgKWg0mbt
aEjosRZNJv1nDEl5qG9XN3FC9zb5FrGSFmTTUvR4f4tUHr7wifNSS2dtgQ6+jU6f
m9o6fukaP7t5VyOXuV7FIO/Hdg2lqW+xU1LowZpVd6ANZ5rAZXtMhWe3+mjfFtju
TAnR
-----RAQ PREGVSVPNGR-----`)))
	testRSA2048Certificate = block.Bytes

	block, _ = pem.Decode(obscuretestdata.Rot13([]byte(`
-----ORTVA EFN CEVINGR XRL-----
ZVVRcNVONNXPNDRNa/U5AQrbattI+PQyFUlbeorWOaQxP3bcta7V6du3ZeQPSEuY
EHwBuBNZgrAK/+lXaIgSYFXwJ+Q14HGvN+8t8HqiBZF+y2jee/7rLG91UUbJUA4M
v4fyKGWTHVzIeK1SPK/9nweGCdVGLBsF0IdrUshby9WJgFF9kZNvUWWQLlsLHTkr
m29txiuRiJXBrFtTdsPwz5nKRsQNHwq/T6c8V30UDy7muQb2cgu1ZFfkOI+GNCaj
AWahNbdNaNxF1vcsudQsEsUjNK6Tsx/gazcrNl7wirn10sRdmvSDLq1kGd/0ILL7
I3QIEJFaYj7rariSrbjPtTPchM5L/Ew6KrY/djVQNDNONbVONDPAcZMvsq/it42u
UqPiYhMnLF0E7FhaSycbKRfygTqYSfac0VsbWM/htSDOFNVVsYjZhzH6bKN1m7Hi
98nVLI61QrCeGPQIQSOfUoAzC8WNb8JgohfRojq5mlbO7YLT2+pyxWxyJR73XdHd
ezV+HWrlFpy2Tva7MGkOKm1JCOx9IjpajxrnKctNFVOJ23suRPZ9taLRRjnOrm5G
6Zr8q1gUgLDi7ifXr7eb9j9/UXeEKrwdLXX1YkxusSevlI+z8YMWMa2aKBn6T3tS
Ao8Dx1Hx5CHORAOzlZSWuG4Z/hhFd4LgZeeB2tv8D+sCuhTmp5FfuLXEOc0J4C5e
zgIPgRSENbTONZRAOVSYeI2+UfTw0kLSnfXbi/DCr6UFGE1Uu2VMBAc+bX4bfmJR
wOG4IpaVGzcy6gP1Jl4TpekwAtXVSMNw+1k1YHHYqbeKxhT8le0gNuT9mAlsJfFl
CeFbiP0HIome8Wkkyn+xDIkRDDdJDkCyRIhY8xKnVQN6Ylg1Uchn2YiCNbTONADM
p6Yd2G7+OkYkAqv2z8xMmrw5xtmOc/KqIfoSJEyroVK2XeSUfeUmG9CHx3QR1iMX
Z6cmGg94aDuJFxQtPnj1FbuRyW3USVSjphfS1FWNp3cDrcq8ht6VLqycQZYgOw/C
/5C6OIHgtb05R4+V/G3vLngztyDkGgyM0ExFI2yyNbTONYBKxXSK7nuCis0JxfQu
hGshSBGCbbjtDT0RctJ0jEqPkrt/WYvp3yFQ0tfggDI2JfErpelJpknryEt10EzB
38OobtzunS4kitfFihwBsvMGR8bX1G43Z+6AXfVyZY3LVYocH/9nWkCJl0f2QdQe
pDWuMeyx+cmwON7Oas/HEqjkNbTNXE/PAj14Q+zeY3LYoovPKvlqdkIjki5cqMqm
8guv3GApfJP4vTHEqpIdosHvaICqWvKr/Xnp3JTPrEWnSItoXNBkYgv1EO5ZxVut
Q8rlhcOdx4J1Y1txekdfqw4GSykxjZljwy2R2F4LlD8COg6I04QbIEMfVXmdm+CS
HvbaCd0PtLOPLKidvbWuCrjxBd/L5jeQOrMJ1SDX5DQ9J5Z8/5mkq4eqiWgwuoWc
bBegiZqey6hcl9Um4OWQ3SKjISvCSR7wdrAdv0S21ivYkOCZZQ3HBQS6YY5RlYvE
9I4kIZF8XKkit7ekfhdmZCfpIvnJHY6JAIOufQ2+92qUkFKmm5RWXD==
-----RAQ EFN CEVINGR XRL-----`)))
	var err error
	testRSA2048PrivateKey, err = x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		panic(err)
	}
}
