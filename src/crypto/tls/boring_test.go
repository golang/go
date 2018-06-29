// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/internal/boring/fipstls"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"math/big"
	"net"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestBoringServerProtocolVersion(t *testing.T) {
	test := func(name string, v uint16, msg string) {
		t.Run(name, func(t *testing.T) {
			serverConfig := testConfig.Clone()
			serverConfig.MinVersion = VersionSSL30
			clientHello := &clientHelloMsg{
				vers:               v,
				cipherSuites:       allCipherSuites(),
				compressionMethods: []uint8{compressionNone},
			}
			testClientHelloFailure(t, serverConfig, clientHello, msg)
		})
	}

	test("VersionSSL30", VersionSSL30, "")
	test("VersionTLS10", VersionTLS10, "")
	test("VersionTLS11", VersionTLS11, "")
	test("VersionTLS12", VersionTLS12, "")

	fipstls.Force()
	defer fipstls.Abandon()
	test("VersionSSL30", VersionSSL30, "unsupported, maximum protocol version")
	test("VersionTLS10", VersionTLS10, "unsupported, maximum protocol version")
	test("VersionTLS11", VersionTLS11, "unsupported, maximum protocol version")
	test("VersionTLS12", VersionTLS12, "")
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
			return suite.flags&suiteECDSA == suiteECDSA
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
		ECDSAWithP521AndSHA512:
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
				cipherSuites:       []uint16{id},
				compressionMethods: []uint8{compressionNone},
				supportedCurves:    defaultCurvePreferences,
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

	for _, curveid := range defaultCurvePreferences {
		t.Run(fmt.Sprintf("curve=%d", curveid), func(t *testing.T) {
			clientHello := &clientHelloMsg{
				vers:               VersionTLS12,
				cipherSuites:       []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
				compressionMethods: []uint8{compressionNone},
				supportedCurves:    []CurveID{curveid},
				supportedPoints:    []uint8{pointFormatUncompressed},
			}

			testClientHello(t, serverConfig, clientHello)

			// With fipstls forced, bad curves should be rejected.
			t.Run("fipstls", func(t *testing.T) {
				fipstls.Force()
				defer fipstls.Abandon()
				msg := ""
				if !isBoringCurve(curveid) {
					msg = "no cipher suite supported by both client and server"
				}
				testClientHelloFailure(t, serverConfig, clientHello, msg)
			})
		})
	}
}

func boringHandshake(t *testing.T, clientConfig, serverConfig *Config) (clientErr, serverErr error) {
	c, s := realNetPipe(t)
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
	serverConfig := testConfig.Clone()
	serverConfig.Certificates = make([]Certificate, 1)

	defer func() {
		testingOnlyForceClientHelloSignatureAlgorithms = nil
	}()

	for _, sigHash := range defaultSupportedSignatureAlgorithms {
		testingOnlyForceClientHelloSignatureAlgorithms = []SignatureScheme{sigHash}

		t.Run(fmt.Sprintf("%v", sigHash), func(t *testing.T) {
			if sigHash == PKCS1WithSHA1 || sigHash == PKCS1WithSHA256 || sigHash == PKCS1WithSHA384 || sigHash == PKCS1WithSHA512 {
				serverConfig.CipherSuites = []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}
				serverConfig.Certificates[0].Certificate = [][]byte{testRSACertificate}
				serverConfig.Certificates[0].PrivateKey = testRSAPrivateKey
			} else {
				serverConfig.CipherSuites = []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256}
				serverConfig.Certificates = make([]Certificate, 1)
				serverConfig.Certificates[0].Certificate = [][]byte{testECDSACertificate}
				serverConfig.Certificates[0].PrivateKey = testECDSAPrivateKey
			}
			serverConfig.BuildNameToCertificate()

			clientErr, _ := boringHandshake(t, testConfig, serverConfig)
			if clientErr != nil {
				t.Fatalf("expected handshake with %v to succeed; err=%v", sigHash, clientErr)
			}

			// With fipstls forced, bad curves should be rejected.
			t.Run("fipstls", func(t *testing.T) {
				fipstls.Force()
				defer fipstls.Abandon()
				clientErr, _ := boringHandshake(t, testConfig, serverConfig)
				if isBoringSignatureScheme(sigHash) {
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
	clientConfig.CipherSuites = allCipherSuites()
	clientConfig.CurvePreferences = defaultCurvePreferences

	go Client(c, testConfig).Handshake()
	srv := Server(s, testConfig)
	msg, err := srv.readHandshake()
	if err != nil {
		t.Fatal(err)
	}
	hello, ok := msg.(*clientHelloMsg)
	if !ok {
		t.Fatalf("unexpected message type %T", msg)
	}

	if hello.vers != VersionTLS12 {
		t.Errorf("client vers=%#x, want %#x (TLS 1.2)", hello.vers, VersionTLS12)
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
	R2 := boringCert(t, "R2", boringRSAKey(t, 4096), nil, boringCertCA)

	M1_R1 := boringCert(t, "M1_R1", boringECDSAKey(t, elliptic.P256()), R1, boringCertCA|boringCertFIPSOK)
	M2_R1 := boringCert(t, "M2_R1", boringECDSAKey(t, elliptic.P224()), R1, boringCertCA)

	I_R1 := boringCert(t, "I_R1", boringRSAKey(t, 3072), R1, boringCertCA|boringCertFIPSOK)
	I_R2 := boringCert(t, "I_R2", I_R1.key, R2, boringCertCA|boringCertFIPSOK)
	I_M1 := boringCert(t, "I_M1", I_R1.key, M1_R1, boringCertCA|boringCertFIPSOK)
	I_M2 := boringCert(t, "I_M2", I_R1.key, M2_R1, boringCertCA|boringCertFIPSOK)

	L1_I := boringCert(t, "L1_I", boringECDSAKey(t, elliptic.P384()), I_R1, boringCertLeaf|boringCertFIPSOK)
	L2_I := boringCert(t, "L2_I", boringRSAKey(t, 1024), I_R1, boringCertLeaf)

	// boringCert checked that isBoringCertificate matches the caller's boringCertFIPSOK bit.
	// If not, no point in building bigger end-to-end tests.
	if t.Failed() {
		t.Fatalf("isBoringCertificate failures; not continuing")
	}

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

	// Tell isBoringCertificate to enforce FIPS restrictions for this check.
	fipstls.Force()
	defer fipstls.Abandon()

	fipsOK := mode&boringCertFIPSOK != 0
	if isBoringCertificate(cert) != fipsOK {
		t.Errorf("isBoringCertificate(cert with %s key) = %v, want %v", desc, !fipsOK, fipsOK)
	}
	return &boringCertificate{name, org, parentOrg, der, cert, key, fipsOK}
}

func boringPool(t *testing.T, list ...*boringCertificate) *x509.CertPool {
	pool := x509.NewCertPool()
	for _, c := range list {
		cert, err := x509.ParseCertificate(c.der)
		if err != nil {
			t.Fatal(err)
		}
		pool.AddCert(cert)
	}
	return pool
}

func boringList(t *testing.T, list ...*boringCertificate) [][]byte {
	var all [][]byte
	for _, c := range list {
		all = append(all, c.der)
	}
	return all
}

// realNetPipe is like net.Pipe but returns an actual network socket pair,
// which has buffering that avoids various deadlocks if both sides
// try to speak at the same time.
func realNetPipe(t *testing.T) (net.Conn, net.Conn) {
	l := newLocalListener(t)
	defer l.Close()
	c, err := net.Dial("tcp", l.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	s, err := l.Accept()
	if err != nil {
		c.Close()
		t.Fatal(err)
	}
	return c, s
}
