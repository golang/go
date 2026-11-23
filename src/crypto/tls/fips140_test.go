// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/fips140"
	"crypto/hpke"
	"crypto/internal/boring"
	"crypto/internal/cryptotest"
	"crypto/mldsa"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"internal/testenv"
	"math/big"
	"net"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

func testConfigFIPS140() *Config {
	return &Config{
		Time:         testTime,
		Certificates: []Certificate{testECDSAP256Cert, testRSAPSSCert, testEd25519Cert, testMLDSA44Cert, testMLDSA65Cert, testMLDSA87Cert},
		RootCAs:      testRootCertPool,
		ServerName:   "test.golang.example",
		KeyLogWriter: testKeyLogWriter,
	}
}

func allCipherSuitesIncludingTLS13() []uint16 {
	s := make([]uint16, 0, len(cipherSuites))
	for _, suite := range cipherSuites {
		s = append(s, suite.id)
	}
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
			serverConfig := testConfigFIPS140()
			serverConfig.MinVersion = VersionSSL30
			serverConfig.MaxVersion = VersionTLS13
			clientConfig := testConfigFIPS140()
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
	case X25519MLKEM768, SecP256r1MLKEM768, SecP384r1MLKEM1024, MLKEM1024:
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
	case Ed25519, MLDSA44, MLDSA65, MLDSA87:
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
			serverConfig := testConfigFIPS140()
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
	for _, curveid := range curvePreferenceOrder() {
		t.Run(fmt.Sprintf("curve=%v", curveid), func(t *testing.T) {
			testConfig := testConfigFIPS140()
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
			isMLDSA := sigHash == MLDSA44 || sigHash == MLDSA65 || sigHash == MLDSA87
			if isMLDSA {
				cryptotest.MustMinimumFIPS140ModuleVersion(t, "v1.26.0")
			}
			serverConfig := testConfigFIPS140()
			testingOnlySupportedSignatureAlgorithms = []SignatureScheme{sigHash}
			// PKCS#1 v1.5 signature algorithms can't be used standalone in TLS
			// 1.3, and the ECDSA ones bind to the curve used. However, ML-DSA
			// requires TLS 1.3.
			if !isMLDSA {
				serverConfig.MaxVersion = VersionTLS12
			}

			runWithFIPSDisabled(t, func(t *testing.T) {
				clientErr, serverErr := fipsHandshake(t, testConfigFIPS140(), serverConfig)
				if clientErr != nil {
					t.Fatalf("expected handshake with %v to succeed; client error: %v; server error: %v", sigHash, clientErr, serverErr)
				}
			})

			// With fipstls forced, bad curves should be rejected.
			runWithFIPSEnabled(t, func(t *testing.T) {
				clientErr, _ := fipsHandshake(t, testConfigFIPS140(), serverConfig)
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

	clientConfig := testConfigFIPS140()
	// All sorts of traps for the client to avoid.
	clientConfig.MinVersion = VersionSSL30
	clientConfig.MaxVersion = VersionTLS13
	clientConfig.CipherSuites = allCipherSuitesIncludingTLS13()
	clientConfig.CurvePreferences = curvePreferenceOrder()

	go Client(c, clientConfig).Handshake()
	srv := Server(s, testConfigFIPS140())
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
	var L3_I *fipsCertificate
	if fips140.Version() != "v1.0.0" {
		// ML-DSA is not implemented by the Go+BoringCrypto FIPS 140 module.
		mldsaFlags := fipsCertLeaf | fipsCertFIPSOK
		if boring.Enabled {
			mldsaFlags = fipsCertLeaf
		}
		L3_I = fipsCert(t, "L3_I", fipsMLDSAKey(t, mldsa.MLDSA44()), I_R1, mldsaFlags)
	}

	// client verifying server cert
	testServerCert := func(t *testing.T, desc string, pool *x509.CertPool, key any, list [][]byte, ok bool) {
		clientConfig := testConfigFIPS140()
		clientConfig.RootCAs = pool
		clientConfig.InsecureSkipVerify = false
		clientConfig.ServerName = "example.com"
		clientConfig.Time = func() time.Time { return time.Unix(0, 0) }

		serverConfig := testConfigFIPS140()
		serverConfig.Certificates = []Certificate{{Certificate: list, PrivateKey: key}}
		serverConfig.Time = func() time.Time { return time.Unix(0, 0) }

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
		clientConfig := testConfigFIPS140()
		clientConfig.InsecureSkipVerify = true
		clientConfig.Certificates = []Certificate{{Certificate: list, PrivateKey: key}}
		clientConfig.Time = func() time.Time { return time.Unix(0, 0) }

		serverConfig := testConfigFIPS140()
		serverConfig.ClientCAs = pool
		serverConfig.ClientAuth = RequireAndVerifyClientCert
		serverConfig.Time = func() time.Time { return time.Unix(0, 0) }

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
		if L3_I != nil {
			testServerCert(t, "basic ML-DSA", r1pool, L3_I.key, [][]byte{L3_I.der, I_R1.der}, true)
			testClientCert(t, "basic ML-DSA (client cert)", r1pool, L3_I.key, [][]byte{L3_I.der, I_R1.der}, true)
		}
	})

	runWithFIPSEnabled(t, func(t *testing.T) {
		testServerCert(t, "basic (fips)", r1pool, L2_I.key, [][]byte{L2_I.der, I_R1.der}, false)
		testClientCert(t, "basic (fips, client cert)", r1pool, L2_I.key, [][]byte{L2_I.der, I_R1.der}, false)
		if L3_I != nil {
			testServerCert(t, "basic ML-DSA (fips)", r1pool, L3_I.key, [][]byte{L3_I.der, I_R1.der}, L3_I.fipsOK)
			testClientCert(t, "basic ML-DSA (fips, client cert)", r1pool, L3_I.key, [][]byte{L3_I.der, I_R1.der}, L3_I.fipsOK)
		}
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

func TestFIPSCertificateWithoutVerification(t *testing.T) {
	badCert := fipsCert(t, "L", fipsRSAKey(t, 1024), nil, fipsCertLeaf)

	serverCert := func(t *testing.T, version uint16) (clientErr, serverErr error) {
		clientConfig := testConfigFIPS140()
		clientConfig.InsecureSkipVerify = true
		clientConfig.MinVersion = version
		clientConfig.MaxVersion = version
		serverConfig := testConfigFIPS140()
		serverConfig.Certificates = []Certificate{{Certificate: [][]byte{badCert.der}, PrivateKey: badCert.key}}
		serverConfig.MinVersion = version
		serverConfig.MaxVersion = version
		return fipsHandshake(t, clientConfig, serverConfig)
	}

	clientCert := func(t *testing.T, version uint16) (clientErr, serverErr error) {
		clientConfig := testConfigFIPS140()
		clientConfig.InsecureSkipVerify = true
		clientConfig.Certificates = []Certificate{{Certificate: [][]byte{badCert.der}, PrivateKey: badCert.key}}
		clientConfig.MinVersion = version
		clientConfig.MaxVersion = version
		serverConfig := testConfigFIPS140()
		serverConfig.ClientAuth = RequireAnyClientCert
		serverConfig.MinVersion = version
		serverConfig.MaxVersion = version
		return fipsHandshake(t, clientConfig, serverConfig)
	}

	for _, version := range []uint16{VersionTLS12, VersionTLS13} {
		t.Run(VersionName(version), func(t *testing.T) {
			runWithFIPSDisabled(t, func(t *testing.T) {
				if clientErr, serverErr := serverCert(t, version); clientErr != nil {
					t.Errorf("server cert: expected success; client error: %v; server error: %v", clientErr, serverErr)
				}
				if clientErr, serverErr := clientCert(t, version); serverErr != nil {
					t.Errorf("client cert: expected success; client error: %v; server error: %v", clientErr, serverErr)
				}
			})

			runWithFIPSEnabled(t, func(t *testing.T) {
				const want = "not allowed in FIPS 140-3 mode"
				if clientErr, _ := serverCert(t, version); clientErr == nil || !strings.Contains(clientErr.Error(), want) {
					t.Errorf("server cert: got client error %v, want error containing %q", clientErr, want)
				}
				if _, serverErr := clientCert(t, version); serverErr == nil || !strings.Contains(serverErr.Error(), want) {
					t.Errorf("client cert: got server error %v, want error containing %q", serverErr, want)
				}
			})
		})
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

func fipsMLDSAKey(t *testing.T, params mldsa.Parameters) *mldsa.PrivateKey {
	k, err := mldsa.GenerateKey(params)
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
	case *mldsa.PrivateKey:
		pub = k.PublicKey()
		desc = k.PublicKey().Parameters().String()
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

func TestFIPS140NoRestrictions(t *testing.T) {
	if fips140.Enforced() {
		t.Skip("noFIPS140Restrictions is not supported in fips140=only mode")
	}

	// Generate a 1024-bit RSA cert chain once for the certificate subtests.
	r1 := fipsCert(t, "R1", fipsRSAKey(t, 2048), nil, fipsCertCA|fipsCertFIPSOK)
	l1024 := fipsCert(t, "L1024", fipsRSAKey(t, 1024), r1, fipsCertLeaf)
	r1Pool := x509.NewCertPool()
	r1Pool.AddCert(r1.cert)

	runCases := func(t *testing.T, wantOptOutSuccess bool) {
		t.Run("ProtocolVersion", func(t *testing.T) {
			for _, v := range []uint16{VersionTLS10, VersionTLS11} {
				t.Run(VersionName(v), func(t *testing.T) {
					clientConfig := testConfigFIPS140()
					clientConfig.MinVersion = v
					clientConfig.MaxVersion = v
					clientConfig.noFIPS140Restrictions = true

					serverConfig := testConfigFIPS140()
					serverConfig.MinVersion = v
					serverConfig.MaxVersion = v
					serverConfig.noFIPS140Restrictions = true

					_, _, err := testHandshake(t, clientConfig, serverConfig)
					if wantOptOutSuccess && err != nil {
						t.Fatalf("got error %v, expected success", err)
					}
					if !wantOptOutSuccess && err == nil {
						t.Fatalf("got success, expected error when FIPS 140 is forced")
					}

					// When only one side opts out, the other side must still reject.
					if wantOptOutSuccess {
						restrictedServer := serverConfig.Clone()
						restrictedServer.noFIPS140Restrictions = false
						if _, _, err := testHandshake(t, clientConfig, restrictedServer); err == nil {
							t.Fatalf("expected error when server does not opt out")
						}
					}
				})
			}
		})

		t.Run("CipherSuites", func(t *testing.T) {
			// TLS 1.2 non-FIPS 140 cipher suites (ChaCha20-Poly1305 and static RSA).
			for _, suite := range []uint16{
				TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
				TLS_RSA_WITH_AES_128_GCM_SHA256,
			} {
				t.Run(CipherSuiteName(suite), func(t *testing.T) {
					clientConfig := testConfigFIPS140()
					clientConfig.MinVersion = VersionTLS12
					clientConfig.MaxVersion = VersionTLS12
					clientConfig.CipherSuites = []uint16{suite}
					clientConfig.noFIPS140Restrictions = true

					serverConfig := testConfigFIPS140()
					serverConfig.MinVersion = VersionTLS12
					serverConfig.MaxVersion = VersionTLS12
					serverConfig.CipherSuites = []uint16{suite}
					serverConfig.noFIPS140Restrictions = true

					_, _, err := testHandshake(t, clientConfig, serverConfig)
					if wantOptOutSuccess && err != nil {
						t.Fatalf("got error %v, expected success", err)
					}
					if !wantOptOutSuccess && err == nil {
						t.Fatalf("got success, expected error when FIPS 140 is forced")
					}
				})
			}

			// TLS 1.3 non-FIPS 140 cipher suite (TLS_CHACHA20_POLY1305_SHA256):
			// verify client offers it and server accepts it.
			t.Run("TLS13ChaCha20ClientOffer", func(t *testing.T) {
				c, s := net.Pipe()
				defer c.Close()
				defer s.Close()

				clientConfig := testConfigFIPS140()
				clientConfig.MinVersion = VersionTLS13
				clientConfig.MaxVersion = VersionTLS13
				clientConfig.noFIPS140Restrictions = true

				go Client(c, clientConfig).Handshake()
				srv := Server(s, testConfigFIPS140())
				msg, err := srv.readHandshake(nil)
				if err != nil {
					t.Fatal(err)
				}
				hello := msg.(*clientHelloMsg)
				hasChaCha := slices.Contains(hello.cipherSuites, TLS_CHACHA20_POLY1305_SHA256)
				if hasChaCha != wantOptOutSuccess {
					t.Fatalf("ClientHello offered TLS_CHACHA20_POLY1305_SHA256 = %v, want %v", hasChaCha, wantOptOutSuccess)
				}
			})

			t.Run("TLS13ChaCha20ServerSelect", func(t *testing.T) {
				serverConfig := testConfigFIPS140()
				serverConfig.noFIPS140Restrictions = true
				clientHello := &clientHelloMsg{
					vers:                         VersionTLS12,
					random:                       make([]byte, 32),
					cipherSuites:                 []uint16{TLS_CHACHA20_POLY1305_SHA256},
					compressionMethods:           []uint8{compressionNone},
					supportedCurves:              []CurveID{CurveP256},
					keyShares:                    []keyShare{generateKeyShare(CurveP256)},
					supportedPoints:              []uint8{pointFormatUncompressed},
					supportedVersions:            []uint16{VersionTLS13},
					supportedSignatureAlgorithms: allowedSignatureAlgorithmsFIPS,
				}
				if wantOptOutSuccess {
					testClientHello(t, serverConfig, clientHello)
				} else {
					testClientHelloFailure(t, serverConfig, clientHello, "no cipher suite supported by both client and server")
				}
			})
		})

		t.Run("Curves", func(t *testing.T) {
			for _, v := range []uint16{VersionTLS12, VersionTLS13} {
				t.Run(VersionName(v)+"/X25519", func(t *testing.T) {
					cfg := testConfigFIPS140()
					cfg.MinVersion = v
					cfg.MaxVersion = v
					cfg.CurvePreferences = []CurveID{X25519}
					cfg.noFIPS140Restrictions = true

					_, _, err := testHandshake(t, cfg, cfg)
					if wantOptOutSuccess && err != nil {
						t.Fatalf("got error %v, expected success", err)
					}
					if !wantOptOutSuccess && err == nil {
						t.Fatalf("got success, expected error when FIPS 140 is forced")
					}
				})
			}
		})

		t.Run("SignatureAlgorithms", func(t *testing.T) {
			testenv.SetGODEBUG(t, "tlssha1=1")
			defer func() {
				testingOnlySupportedSignatureAlgorithms = nil
			}()
			testingOnlySupportedSignatureAlgorithms = []SignatureScheme{PKCS1WithSHA1}

			clientConfig := testConfigFIPS140()
			clientConfig.MaxVersion = VersionTLS12
			clientConfig.Certificates = []Certificate{testRSAPSSCert}
			clientConfig.noFIPS140Restrictions = true

			serverConfig := testConfigFIPS140()
			serverConfig.MaxVersion = VersionTLS12
			serverConfig.ClientAuth = RequireAnyClientCert
			serverConfig.noFIPS140Restrictions = true

			clientErr, serverErr := fipsHandshake(t, clientConfig, serverConfig)
			if wantOptOutSuccess {
				if clientErr != nil || serverErr != nil {
					t.Fatalf("expected SHA-1 handshake with client auth to succeed; clientErr=%v serverErr=%v", clientErr, serverErr)
				}
			} else {
				if clientErr == nil && serverErr == nil {
					t.Fatalf("expected SHA-1 handshake to fail when FIPS 140 is forced")
				}
			}

			// Verify ClientHelloInfo.SupportsCertificate and CertificateRequestInfo.SupportsCertificate.
			chi := &ClientHelloInfo{
				SupportedVersions: []uint16{VersionTLS12},
				CipherSuites:      []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				SupportedCurves:   []CurveID{CurveP256},
				SupportedPoints:   []uint8{pointFormatUncompressed},
				SignatureSchemes:  []SignatureScheme{PKCS1WithSHA1},
				config:            serverConfig,
			}
			err := chi.SupportsCertificate(&testRSAPSSCert)
			if wantOptOutSuccess && err != nil {
				t.Fatalf("ClientHelloInfo.SupportsCertificate(PKCS1WithSHA1) = %v, want nil", err)
			}
			if !wantOptOutSuccess && err == nil {
				t.Fatalf("ClientHelloInfo.SupportsCertificate(PKCS1WithSHA1) = nil, want error when FIPS 140 is forced")
			}

			cri := &CertificateRequestInfo{
				Version:               VersionTLS12,
				SignatureSchemes:      []SignatureScheme{PKCS1WithSHA1},
				noFIPS140Restrictions: clientConfig.noFIPS140Restrictions,
			}
			err = cri.SupportsCertificate(&testRSAPSSCert)
			if wantOptOutSuccess && err != nil {
				t.Fatalf("CertificateRequestInfo.SupportsCertificate(PKCS1WithSHA1) = %v, want nil", err)
			}
			if !wantOptOutSuccess && err == nil {
				t.Fatalf("CertificateRequestInfo.SupportsCertificate(PKCS1WithSHA1) = nil, want error when FIPS 140 is forced")
			}

			// A zero-value CertificateRequestInfo must fail closed under FIPS 140 restrictions.
			zeroCRI := &CertificateRequestInfo{
				Version:          VersionTLS12,
				SignatureSchemes: []SignatureScheme{PKCS1WithSHA1},
			}
			if err := zeroCRI.SupportsCertificate(&testRSAPSSCert); err == nil {
				t.Fatalf("zero-value CertificateRequestInfo.SupportsCertificate(PKCS1WithSHA1) = nil, want error")
			}
		})

		t.Run("Certificates", func(t *testing.T) {
			for _, v := range []uint16{VersionTLS12, VersionTLS13} {
				t.Run(VersionName(v), func(t *testing.T) {
					// Verified chain (fipsAllowedChains) for server cert.
					clientConfig := testConfigFIPS140()
					clientConfig.MinVersion = v
					clientConfig.MaxVersion = v
					clientConfig.RootCAs = r1Pool
					clientConfig.ServerName = "example.com"
					clientConfig.Time = func() time.Time { return time.Unix(0, 0) }
					clientConfig.noFIPS140Restrictions = true

					serverConfig := testConfigFIPS140()
					serverConfig.MinVersion = v
					serverConfig.MaxVersion = v
					serverConfig.Certificates = []Certificate{{Certificate: [][]byte{l1024.der}, PrivateKey: l1024.key}}
					serverConfig.Time = func() time.Time { return time.Unix(0, 0) }
					serverConfig.noFIPS140Restrictions = true

					clientErr, _ := fipsHandshake(t, clientConfig, serverConfig)
					if wantOptOutSuccess && clientErr != nil {
						t.Fatalf("server cert (verified): got %v, want nil", clientErr)
					}
					if !wantOptOutSuccess && clientErr == nil {
						t.Fatalf("server cert (verified): got nil, want error when FIPS 140 is forced")
					}

					// Unverified leaf (InsecureSkipVerify) for server cert.
					clientConfigSkip := clientConfig.Clone()
					clientConfigSkip.InsecureSkipVerify = true
					clientErr, _ = fipsHandshake(t, clientConfigSkip, serverConfig)
					if wantOptOutSuccess && clientErr != nil {
						t.Fatalf("server cert (InsecureSkipVerify): got %v, want nil", clientErr)
					}
					if !wantOptOutSuccess && clientErr == nil {
						t.Fatalf("server cert (InsecureSkipVerify): got nil, want error when FIPS 140 is forced")
					}

					// Verified chain (fipsAllowedChains) for client cert.
					mClient := testConfigFIPS140()
					mClient.MinVersion = v
					mClient.MaxVersion = v
					mClient.InsecureSkipVerify = true
					mClient.Certificates = []Certificate{{Certificate: [][]byte{l1024.der}, PrivateKey: l1024.key}}
					mClient.Time = func() time.Time { return time.Unix(0, 0) }
					mClient.noFIPS140Restrictions = true

					mServer := testConfigFIPS140()
					mServer.MinVersion = v
					mServer.MaxVersion = v
					mServer.ClientCAs = r1Pool
					mServer.ClientAuth = RequireAndVerifyClientCert
					mServer.Time = func() time.Time { return time.Unix(0, 0) }
					mServer.noFIPS140Restrictions = true

					_, serverErr := fipsHandshake(t, mClient, mServer)
					if wantOptOutSuccess && serverErr != nil {
						t.Fatalf("client cert (verified): got %v, want nil", serverErr)
					}
					if !wantOptOutSuccess && serverErr == nil {
						t.Fatalf("client cert (verified): got nil, want error when FIPS 140 is forced")
					}

					// Unverified leaf (RequireAnyClientCert) for client cert.
					mServerAny := mServer.Clone()
					mServerAny.ClientAuth = RequireAnyClientCert
					_, serverErr = fipsHandshake(t, mClient, mServerAny)
					if wantOptOutSuccess && serverErr != nil {
						t.Fatalf("client cert (RequireAnyClientCert): got %v, want nil", serverErr)
					}
					if !wantOptOutSuccess && serverErr == nil {
						t.Fatalf("client cert (RequireAnyClientCert): got nil, want error when FIPS 140 is forced")
					}
				})
			}
		})

		t.Run("ECHGrease", func(t *testing.T) {
			hello := &clientHelloMsg{
				vers:               VersionTLS12,
				random:             make([]byte, 32),
				cipherSuites:       []uint16{TLS_AES_128_GCM_SHA256},
				compressionMethods: []uint8{compressionNone},
				supportedVersions:  []uint16{VersionTLS13},
			}
			// Byte 0 == 2 selects ChaCha20Poly1305 when len(aeads) == 3,
			// or AES128GCM when len(aeads) == 2 (FIPS 140).
			// X25519 produces a 32-byte encap key, P-256 produces 65 bytes.
			cfg := &Config{
				Rand:                  bytes.NewReader(bytes.Repeat([]byte{2}, 512)),
				noFIPS140Restrictions: true,
			}
			ext, err := newGreaseECHExtension(cfg.rand(), hello, fips140Restrictions(cfg.noFIPS140Restrictions))
			if err != nil {
				t.Fatal(err)
			}
			_, cs, _, encap, _, err := parseECHExt(ext)
			if err != nil {
				t.Fatal(err)
			}
			if wantOptOutSuccess {
				if cs.AEADID != hpke.ChaCha20Poly1305().ID() {
					t.Errorf("ECH GREASE AEAD = %#x, want ChaCha20Poly1305 (%#x)", cs.AEADID, hpke.ChaCha20Poly1305().ID())
				}
				if len(encap) != 32 {
					t.Errorf("ECH GREASE encap len = %d, want 32 (X25519)", len(encap))
				}
			} else {
				if cs.AEADID == hpke.ChaCha20Poly1305().ID() {
					t.Errorf("ECH GREASE used ChaCha20Poly1305 when FIPS 140 is forced")
				}
				if len(encap) != 65 {
					t.Errorf("ECH GREASE encap len = %d, want 65 (P-256)", len(encap))
				}
			}
		})

		t.Run("EMSResumption", func(t *testing.T) {
			// Check server-side TLS 1.2 resumption without EMS.
			serverConfig := testConfigFIPS140()
			serverConfig.MinVersion = VersionTLS12
			serverConfig.MaxVersion = VersionTLS12
			serverConfig.noFIPS140Restrictions = true

			sess := &SessionState{
				version:         VersionTLS12,
				cipherSuite:     TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				createdAt:       uint64(testTime().Unix()),
				secret:          make([]byte, 48),
				extMasterSecret: false,
			}
			c, s := localPipe(t)
			defer c.Close()
			defer s.Close()
			srv := Server(s, serverConfig)
			srv.vers = VersionTLS12
			srv.ticketKeys = serverConfig.ticketKeys(nil)
			ticketBytes, err := sess.Bytes()
			if err != nil {
				t.Fatal(err)
			}
			encryptedTicket, err := serverConfig.encryptTicket(ticketBytes, srv.ticketKeys)
			if err != nil {
				t.Fatal(err)
			}
			hs := &serverHandshakeState{
				c: srv,
				clientHello: &clientHelloMsg{
					vers:                 VersionTLS12,
					cipherSuites:         []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
					sessionTicket:        encryptedTicket,
					ticketSupported:      true,
					extendedMasterSecret: false,
				},
				hello:     new(serverHelloMsg),
				ecdheOk:   true,
				rsaSignOk: true,
			}
			if err := hs.checkForResumption(); err != nil {
				t.Fatal(err)
			}
			if srv.didResume != wantOptOutSuccess {
				t.Fatalf("server TLS 1.2 non-EMS didResume = %v, want %v", srv.didResume, wantOptOutSuccess)
			}

			// Check client-side TLS 1.2 resumption without EMS.
			clientConfig := testConfigFIPS140()
			clientConfig.MinVersion = VersionTLS12
			clientConfig.MaxVersion = VersionTLS12
			clientConfig.InsecureSkipVerify = true
			clientConfig.ClientSessionCache = NewLRUClientSessionCache(1)
			clientConfig.noFIPS140Restrictions = true
			cli := Client(c, clientConfig)
			cacheKey := cli.clientSessionCacheKey()
			leaf, err := testRSAPSSCert.leaf()
			if err != nil {
				t.Fatal(err)
			}
			clientConfig.ClientSessionCache.Put(cacheKey, &ClientSessionState{
				session: &SessionState{
					version:          VersionTLS12,
					isClient:         true,
					cipherSuite:      TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
					createdAt:        uint64(testTime().Unix()),
					secret:           make([]byte, 48),
					extMasterSecret:  false,
					peerCertificates: []*x509.Certificate{leaf},
					ticket:           []byte("ticket"),
				},
			})
			hello := &clientHelloMsg{
				vers:              VersionTLS12,
				supportedVersions: []uint16{VersionTLS12},
				cipherSuites:      []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
				ticketSupported:   true,
			}
			resumed, _, _, err := cli.loadSession(hello)
			if err != nil {
				t.Fatal(err)
			}
			if (resumed != nil) != wantOptOutSuccess {
				t.Fatalf("client TLS 1.2 non-EMS loadSession resumed = %v, want %v", resumed != nil, wantOptOutSuccess)
			}
		})
	}

	if fips140.Enabled() {
		t.Run("real-fips140=on", func(t *testing.T) {
			runCases(t, true)
		})
	} else {
		runWithFIPSEnabled(t, func(t *testing.T) {
			runCases(t, true)
		})
		runWithFIPS140Forced(t, func(t *testing.T) {
			runCases(t, false)
		})
		cryptotest.RerunWithFIPS140Enabled(t)
	}
}
