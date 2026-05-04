// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/fips140"
	"crypto/internal/boring"
	"crypto/internal/cryptotest"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"internal/testenv"
	"math/big"
	"net"
	"runtime"
	"strings"
	"testing"
	"time"
)

var testConfigFIPS140 = &Config{
	Time:         testTime,
	Certificates: []Certificate{testECDSAP256Cert, testRSAPSSCert, testEd25519Cert},
	RootCAs:      testRootCertPool,
	ServerName:   "test.golang.example",
}

func allCipherSuitesIncludingTLS13() []uint16 {
	s := allCipherSuites()
	for _, suite := range cipherSuitesTLS13 {
		s = append(s, suite.id)
	}
	return s
}

func isTLS13CipherSuite(id uint16) bool {
	for _, suite := range cipherSuitesTLS13 {
		if id == suite.id {
			return true
		}
	}
	return false
}

func generateKeyShare(group CurveID) keyShare {
	ke, err := keyExchangeForCurveID(group)
	if err != nil {
		panic(err)
	}
	_, shares, err := ke.keyShares(rand.Reader)
	if err != nil {
		panic(err)
	}
	return shares[0]
}

func TestFIPSServerProtocolVersion(t *testing.T) {
	test := func(t *testing.T, name string, v uint16, msg string) {
		t.Run(name, func(t *testing.T) {
			serverConfig := testConfigFIPS140.Clone()
			serverConfig.MinVersion = VersionSSL30
			serverConfig.MaxVersion = VersionTLS13
			clientConfig := testConfigFIPS140.Clone()
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

	runWithFIPSDisabled(t, func(t *testing.T) {
		test(t, "VersionTLS10", VersionTLS10, "")
		test(t, "VersionTLS11", VersionTLS11, "")
		test(t, "VersionTLS12", VersionTLS12, "")
		test(t, "VersionTLS13", VersionTLS13, "")
	})

	runWithFIPSEnabled(t, func(t *testing.T) {
		test(t, "VersionTLS10", VersionTLS10, "supported versions")
		test(t, "VersionTLS11", VersionTLS11, "supported versions")
		test(t, "VersionTLS12", VersionTLS12, "")
		test(t, "VersionTLS13", VersionTLS13, "")
	})

	if !fips140.Enforced() {
		cryptotest.RerunWithFIPS140Enforced(t)
	}
}

func isFIPSVersion(v uint16) bool {
	return v == VersionTLS12 || v == VersionTLS13
}

func isFIPSCipherSuite(id uint16) bool {
	name := CipherSuiteName(id)
	if isTLS13CipherSuite(id) {
		switch id {
		case TLS_AES_128_GCM_SHA256, TLS_AES_256_GCM_SHA384:
			return true
		case TLS_CHACHA20_POLY1305_SHA256:
			return false
		default:
			panic("unknown TLS 1.3 cipher suite: " + name)
		}
	}
	switch id {
	case TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
		TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
		TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384:
		return true
	case TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256,
		TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256:
		// Only for the native module.
		return !boring.Enabled
	}
	switch {
	case strings.Contains(name, "CHACHA20"):
		return false
	case strings.HasSuffix(name, "_SHA"): // SHA-1
		return false
	case strings.HasPrefix(name, "TLS_RSA"): // RSA kex
		return false
	default:
		panic("unknown cipher suite: " + name)
	}
}

func isFIPSCurve(id CurveID) bool {
	switch id {
	case CurveP256, CurveP384, CurveP521:
		return true
	case X25519MLKEM768, SecP256r1MLKEM768, SecP384r1MLKEM1024:
		// Only for the native module.
		return !boring.Enabled
	case X25519:
		return false
	default:
		panic("unknown curve: " + id.String())
	}
}

func isECDSA(id uint16) bool {
	for _, suite := range cipherSuites {
		if suite.id == id {
			return suite.flags&suiteECSign == suiteECSign
		}
	}
	return false // TLS 1.3 cipher suites are not tied to the signature algorithm.
}

func isFIPSSignatureScheme(alg SignatureScheme) bool {
	switch alg {
	case PKCS1WithSHA256,
		ECDSAWithP256AndSHA256,
		PKCS1WithSHA384,
		ECDSAWithP384AndSHA384,
		PKCS1WithSHA512,
		ECDSAWithP521AndSHA512,
		PSSWithSHA256,
		PSSWithSHA384,
		PSSWithSHA512:
		return true
	case Ed25519:
		// Only for the native module.
		return !boring.Enabled
	case PKCS1WithSHA1, ECDSAWithSHA1:
		return false
	default:
		panic("unknown signature scheme: " + alg.String())
	}
}

func TestFIPSServerCipherSuites(t *testing.T) {
	for _, id := range allCipherSuitesIncludingTLS13() {
		t.Run(fmt.Sprintf("suite=%s", CipherSuiteName(id)), func(t *testing.T) {
			serverConfig := testConfigFIPS140.Clone()
			clientHello := &clientHelloMsg{
				vers:                         VersionTLS12,
				random:                       make([]byte, 32),
				cipherSuites:                 []uint16{id},
				compressionMethods:           []uint8{compressionNone},
				supportedCurves:              []CurveID{CurveP256},
				keyShares:                    []keyShare{generateKeyShare(CurveP256)},
				supportedPoints:              []uint8{pointFormatUncompressed},
				supportedVersions:            []uint16{VersionTLS12},
				supportedSignatureAlgorithms: allowedSignatureAlgorithmsFIPS,
			}
			if isTLS13CipherSuite(id) {
				clientHello.supportedVersions = []uint16{VersionTLS13}
			} else {
				serverConfig.CipherSuites = []uint16{id}
			}

			runWithFIPSDisabled(t, func(t *testing.T) {
				testClientHello(t, serverConfig, clientHello)
			})

			runWithFIPSEnabled(t, func(t *testing.T) {
				msg := ""
				if !isFIPSCipherSuite(id) {
					msg = "no cipher suite supported by both client and server"
				}
				testClientHelloFailure(t, serverConfig, clientHello, msg)
			})
		})
	}

	if !fips140.Enforced() {
		cryptotest.RerunWithFIPS140Enforced(t)
	}
}

func TestFIPSServerCurves(t *testing.T) {
	for _, curveid := range defaultCurvePreferences() {
		t.Run(fmt.Sprintf("curve=%v", curveid), func(t *testing.T) {
			testConfig := testConfigFIPS140.Clone()
			testConfig.CurvePreferences = []CurveID{curveid}

			runWithFIPSDisabled(t, func(t *testing.T) {
				if _, _, err := testHandshake(t, testConfig, testConfig); err != nil {
					t.Fatalf("got error: %v, expected success", err)
				}
			})

			// With fipstls forced, bad curves should be rejected.
			runWithFIPSEnabled(t, func(t *testing.T) {
				_, _, err := testHandshake(t, testConfig, testConfig)
				if err != nil && isFIPSCurve(curveid) {
					t.Fatalf("got error: %v, expected success", err)
				} else if err == nil && !isFIPSCurve(curveid) {
					t.Fatalf("got success, expected error")
				}
			})
		})
	}

	if !fips140.Enforced() {
		cryptotest.RerunWithFIPS140Enforced(t)
	}
}

func fipsHandshake(t *testing.T, clientConfig, serverConfig *Config) (clientErr, serverErr error) {
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

func TestFIPSServerSignatureAndHash(t *testing.T) {
	defer func() {
		testingOnlySupportedSignatureAlgorithms = nil
	}()
	testenv.SetGODEBUG(t, "tlssha1=1")

	for _, sigHash := range defaultSupportedSignatureAlgorithms() {
		t.Run(fmt.Sprintf("%v", sigHash), func(t *testing.T) {
			serverConfig := testConfigFIPS140.Clone()
			testingOnlySupportedSignatureAlgorithms = []SignatureScheme{sigHash}
			// PKCS#1 v1.5 signature algorithms can't be used standalone in TLS
			// 1.3, and the ECDSA ones bind to the curve used.
			serverConfig.MaxVersion = VersionTLS12

			runWithFIPSDisabled(t, func(t *testing.T) {
				clientErr, serverErr := fipsHandshake(t, testConfigFIPS140, serverConfig)
				if clientErr != nil {
					t.Fatalf("expected handshake with %v to succeed; client error: %v; server error: %v", sigHash, clientErr, serverErr)
				}
			})

			// With fipstls forced, bad curves should be rejected.
			runWithFIPSEnabled(t, func(t *testing.T) {
				clientErr, _ := fipsHandshake(t, testConfigFIPS140, serverConfig)
				if isFIPSSignatureScheme(sigHash) {
					if clientErr != nil {
						t.Fatalf("expected handshake with %v to succeed; err=%v", sigHash, clientErr)
					}
				} else {
					if clientErr == nil {
						t.Fatalf("expected handshake with %v to fail, but it succeeded", sigHash)
					}
				}
			})
		})
	}

	if !fips140.Enforced() {
		cryptotest.RerunWithFIPS140Enforced(t)
	}
}

func TestFIPSClientHello(t *testing.T) {
	runWithFIPSEnabled(t, testFIPSClientHello)
}

func testFIPSClientHello(t *testing.T) {
	// Test that no matter what we put in the client config,
	// the client does not offer non-FIPS configurations.

	c, s := net.Pipe()
	defer c.Close()
	defer s.Close()

	clientConfig := testConfigFIPS140.Clone()
	// All sorts of traps for the client to avoid.
	clientConfig.MinVersion = VersionSSL30
	clientConfig.MaxVersion = VersionTLS13
	clientConfig.CipherSuites = allCipherSuites()
	clientConfig.CurvePreferences = defaultCurvePreferences()

	go Client(c, clientConfig).Handshake()
	srv := Server(s, testConfigFIPS140)
	msg, err := srv.readHandshake(nil)
	if err != nil {
		t.Fatal(err)
	}
	hello, ok := msg.(*clientHelloMsg)
	if !ok {
		t.Fatalf("unexpected message type %T", msg)
	}

	if !isFIPSVersion(hello.vers) {
		t.Errorf("client vers=%#x", hello.vers)
	}
	for _, v := range hello.supportedVersions {
		if !isFIPSVersion(v) {
			t.Errorf("client offered disallowed version %#x", v)
		}
	}
	for _, id := range hello.cipherSuites {
		if !isFIPSCipherSuite(id) {
			t.Errorf("client offered disallowed suite %v", CipherSuiteName(id))
		}
	}
	for _, id := range hello.supportedCurves {
		if !isFIPSCurve(id) {
			t.Errorf("client offered disallowed curve %v", id)
		}
	}
	for _, sigHash := range hello.supportedSignatureAlgorithms {
		if !isFIPSSignatureScheme(sigHash) {
			t.Errorf("client offered disallowed signature-and-hash %v", sigHash)
		}
	}
}

func TestFIPSCertAlgs(t *testing.T) {
	// arm and wasm time out generating keys. Nothing in this test is
	// architecture-specific, so just don't bother on those.
	if testenv.CPUIsSlow() {
		t.Skipf("skipping on %s/%s because key generation takes too long", runtime.GOOS, runtime.GOARCH)
	}

	// Set up some roots, intermediate CAs, and leaf certs with various algorithms.
	// X_Y is X signed by Y.
	R1 := fipsCert(t, "R1", fipsRSAKey(t, 2048), nil, fipsCertCA|fipsCertFIPSOK)
	R2 := fipsCert(t, "R2", fipsRSAKey(t, 1024), nil, fipsCertCA)
	R3 := fipsCert(t, "R3", fipsRSAKey(t, 4096), nil, fipsCertCA|fipsCertFIPSOK)

	M1_R1 := fipsCert(t, "M1_R1", fipsECDSAKey(t, elliptic.P256()), R1, fipsCertCA|fipsCertFIPSOK)
	M2_R1 := fipsCert(t, "M2_R1", fipsECDSAKey(t, elliptic.P224()), R1, fipsCertCA)

	I_R1 := fipsCert(t, "I_R1", fipsRSAKey(t, 3072), R1, fipsCertCA|fipsCertFIPSOK)
	I_R2 := fipsCert(t, "I_R2", I_R1.key, R2, fipsCertCA|fipsCertFIPSOK)
	I_M1 := fipsCert(t, "I_M1", I_R1.key, M1_R1, fipsCertCA|fipsCertFIPSOK)
	I_M2 := fipsCert(t, "I_M2", I_R1.key, M2_R1, fipsCertCA|fipsCertFIPSOK)

	I_R3 := fipsCert(t, "I_R3", fipsRSAKey(t, 3072), R3, fipsCertCA|fipsCertFIPSOK)
	fipsCert(t, "I_R3", I_R3.key, R3, fipsCertCA|fipsCertFIPSOK)

	L1_I := fipsCert(t, "L1_I", fipsECDSAKey(t, elliptic.P384()), I_R1, fipsCertLeaf|fipsCertFIPSOK)
	L2_I := fipsCert(t, "L2_I", fipsRSAKey(t, 1024), I_R1, fipsCertLeaf)

	// client verifying server cert
	testServerCert := func(t *testing.T, desc string, pool *x509.CertPool, key any, list [][]byte, ok bool) {
		clientConfig := testConfig.Clone()
		clientConfig.RootCAs = pool
		clientConfig.InsecureSkipVerify = false
		clientConfig.ServerName = "example.com"

		serverConfig := testConfig.Clone()
		serverConfig.Certificates = []Certificate{{Certificate: list, PrivateKey: key}}

		clientErr, _ := fipsHandshake(t, clientConfig, serverConfig)

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
	testClientCert := func(t *testing.T, desc string, pool *x509.CertPool, key any, list [][]byte, ok bool) {
		clientConfig := testConfig.Clone()
		clientConfig.ServerName = "example.com"
		clientConfig.Certificates = []Certificate{{Certificate: list, PrivateKey: key}}

		serverConfig := testConfig.Clone()
		serverConfig.ClientCAs = pool
		serverConfig.ClientAuth = RequireAndVerifyClientCert

		_, serverErr := fipsHandshake(t, clientConfig, serverConfig)

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

	runWithFIPSDisabled(t, func(t *testing.T) {
		testServerCert(t, "basic", r1pool, L2_I.key, [][]byte{L2_I.der, I_R1.der}, true)
		testClientCert(t, "basic (client cert)", r1pool, L2_I.key, [][]byte{L2_I.der, I_R1.der}, true)
	})

	runWithFIPSEnabled(t, func(t *testing.T) {
		testServerCert(t, "basic (fips)", r1pool, L2_I.key, [][]byte{L2_I.der, I_R1.der}, false)
		testClientCert(t, "basic (fips, client cert)", r1pool, L2_I.key, [][]byte{L2_I.der, I_R1.der}, false)
	})

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
			addList := func(cond int, c *fipsCertificate) {
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
				addRoot := func(cond int, c *fipsCertificate) {
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

				runWithFIPSDisabled(t, func(t *testing.T) {
					testServerCert(t, listName+"->"+rootName[1:], pool, leaf.key, list, shouldVerify)
					testClientCert(t, listName+"->"+rootName[1:]+"(client cert)", pool, leaf.key, list, shouldVerify)
				})

				runWithFIPSEnabled(t, func(t *testing.T) {
					testServerCert(t, listName+"->"+rootName[1:]+" (fips)", pool, leaf.key, list, shouldVerifyFIPS)
					testClientCert(t, listName+"->"+rootName[1:]+" (fips, client cert)", pool, leaf.key, list, shouldVerifyFIPS)
				})
			}
		}
	}
}

const (
	fipsCertCA = iota
	fipsCertLeaf
	fipsCertFIPSOK = 0x80
)

func fipsRSAKey(t *testing.T, size int) *rsa.PrivateKey {
	k, err := rsa.GenerateKey(rand.Reader, size)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

func fipsECDSAKey(t *testing.T, curve elliptic.Curve) *ecdsa.PrivateKey {
	k, err := ecdsa.GenerateKey(curve, rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

type fipsCertificate struct {
	name      string
	org       string
	parentOrg string
	der       []byte
	cert      *x509.Certificate
	key       any
	fipsOK    bool
}

func fipsCert(t *testing.T, name string, key any, parent *fipsCertificate, mode int) *fipsCertificate {
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
	if mode&^fipsCertFIPSOK == fipsCertLeaf {
		tmpl.DNSNames = []string{"example.com"}
	} else {
		tmpl.IsCA = true
		tmpl.KeyUsage |= x509.KeyUsageCertSign
	}

	var pcert *x509.Certificate
	var pkey any
	if parent != nil {
		pcert = parent.cert
		pkey = parent.key
	} else {
		pcert = tmpl
		pkey = key
	}

	var pub any
	var desc string
	switch k := key.(type) {
	case *rsa.PrivateKey:
		pub = &k.PublicKey
		desc = fmt.Sprintf("RSA-%d", k.N.BitLen())
	case *ecdsa.PrivateKey:
		pub = &k.PublicKey
		desc = "ECDSA-" + k.Curve.Params().Name
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

	fipsOK := mode&fipsCertFIPSOK != 0
	runWithFIPSEnabled(t, func(t *testing.T) {
		if isCertificateAllowedFIPS(cert) != fipsOK {
			t.Errorf("fipsAllowCert(cert with %s key) = %v, want %v", desc, !fipsOK, fipsOK)
		}
	})

	return &fipsCertificate{name, org, parentOrg, der, cert, key, fipsOK}
}
