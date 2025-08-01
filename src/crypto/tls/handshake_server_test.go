// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ecdh"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls/internal/fips140tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func testClientHello(t *testing.T, serverConfig *Config, m handshakeMessage) {
	t.Helper()
	testClientHelloFailure(t, serverConfig, m, "")
}

// testFatal is a hack to prevent the compiler from complaining that there is a
// call to t.Fatal from a non-test goroutine
func testFatal(t *testing.T, err error) {
	t.Helper()
	t.Fatal(err)
}

func testClientHelloFailure(t *testing.T, serverConfig *Config, m handshakeMessage, expectedSubStr string) {
	c, s := localPipe(t)
	go func() {
		cli := Client(c, testConfig)
		if ch, ok := m.(*clientHelloMsg); ok {
			cli.vers = ch.vers
		}
		if _, err := cli.writeHandshakeRecord(m, nil); err != nil {
			testFatal(t, err)
		}
		c.Close()
	}()
	ctx := context.Background()
	conn := Server(s, serverConfig)
	ch, ech, err := conn.readClientHello(ctx)
	if conn.vers == VersionTLS13 {
		hs := serverHandshakeStateTLS13{
			c:           conn,
			ctx:         ctx,
			clientHello: ch,
			echContext:  ech,
		}
		if err == nil {
			err = hs.processClientHello()
		}
		if err == nil {
			err = hs.checkForResumption()
		}
		if err == nil {
			err = hs.pickCertificate()
		}
	} else {
		hs := serverHandshakeState{
			c:           conn,
			ctx:         ctx,
			clientHello: ch,
		}
		if err == nil {
			err = hs.processClientHello()
		}
		if err == nil {
			err = hs.pickCipherSuite()
		}
	}
	s.Close()
	t.Helper()
	if len(expectedSubStr) == 0 {
		if err != nil && err != io.EOF {
			t.Errorf("Got error: %s; expected to succeed", err)
		}
	} else if err == nil || !strings.Contains(err.Error(), expectedSubStr) {
		t.Errorf("Got error: %v; expected to match substring '%s'", err, expectedSubStr)
	}
}

func TestSimpleError(t *testing.T) {
	testClientHelloFailure(t, testConfig, &serverHelloDoneMsg{}, "unexpected handshake message")
}

var badProtocolVersions = []uint16{0x0000, 0x0005, 0x0100, 0x0105, 0x0200, 0x0205, VersionSSL30}

func TestRejectBadProtocolVersion(t *testing.T) {
	config := testConfig.Clone()
	config.MinVersion = VersionSSL30
	for _, v := range badProtocolVersions {
		testClientHelloFailure(t, config, &clientHelloMsg{
			vers:   v,
			random: make([]byte, 32),
		}, "unsupported versions")
	}
	testClientHelloFailure(t, config, &clientHelloMsg{
		vers:              VersionTLS12,
		supportedVersions: badProtocolVersions,
		random:            make([]byte, 32),
	}, "unsupported versions")
}

func TestNoSuiteOverlap(t *testing.T) {
	clientHello := &clientHelloMsg{
		vers:               VersionTLS12,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{0xff00},
		compressionMethods: []uint8{compressionNone},
	}
	testClientHelloFailure(t, testConfig, clientHello, "no cipher suite supported by both client and server")
}

func TestNoCompressionOverlap(t *testing.T) {
	clientHello := &clientHelloMsg{
		vers:               VersionTLS12,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		compressionMethods: []uint8{0xff},
	}
	testClientHelloFailure(t, testConfig, clientHello, "client does not support uncompressed connections")
}

func TestNoRC4ByDefault(t *testing.T) {
	clientHello := &clientHelloMsg{
		vers:               VersionTLS12,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_RSA_WITH_RC4_128_SHA},
		compressionMethods: []uint8{compressionNone},
	}
	serverConfig := testConfig.Clone()
	// Reset the enabled cipher suites to nil in order to test the
	// defaults.
	serverConfig.CipherSuites = nil
	testClientHelloFailure(t, serverConfig, clientHello, "no cipher suite supported by both client and server")
}

func TestRejectSNIWithTrailingDot(t *testing.T) {
	testClientHelloFailure(t, testConfig, &clientHelloMsg{
		vers:       VersionTLS12,
		random:     make([]byte, 32),
		serverName: "foo.com.",
	}, "decoding message")
}

func TestDontSelectECDSAWithRSAKey(t *testing.T) {
	// Test that, even when both sides support an ECDSA cipher suite, it
	// won't be selected if the server's private key doesn't support it.
	clientHello := &clientHelloMsg{
		vers:               VersionTLS12,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384},
		compressionMethods: []uint8{compressionNone},
		supportedCurves:    []CurveID{CurveP256},
		supportedPoints:    []uint8{pointFormatUncompressed},
	}
	serverConfig := testConfig.Clone()
	serverConfig.CipherSuites = clientHello.cipherSuites
	serverConfig.Certificates = make([]Certificate, 1)
	serverConfig.Certificates[0].Certificate = [][]byte{testECDSACertificate}
	serverConfig.Certificates[0].PrivateKey = testECDSAPrivateKey
	serverConfig.BuildNameToCertificate()
	// First test that it *does* work when the server's key is ECDSA.
	testClientHello(t, serverConfig, clientHello)

	// Now test that switching to an RSA key causes the expected error (and
	// not an internal error about a signing failure).
	serverConfig.Certificates = testConfig.Certificates
	testClientHelloFailure(t, serverConfig, clientHello, "no cipher suite supported by both client and server")
}

func TestDontSelectRSAWithECDSAKey(t *testing.T) {
	// Test that, even when both sides support an RSA cipher suite, it
	// won't be selected if the server's private key doesn't support it.
	clientHello := &clientHelloMsg{
		vers:               VersionTLS12,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		compressionMethods: []uint8{compressionNone},
		supportedCurves:    []CurveID{CurveP256},
		supportedPoints:    []uint8{pointFormatUncompressed},
	}
	serverConfig := testConfig.Clone()
	serverConfig.CipherSuites = clientHello.cipherSuites
	// First test that it *does* work when the server's key is RSA.
	testClientHello(t, serverConfig, clientHello)

	// Now test that switching to an ECDSA key causes the expected error
	// (and not an internal error about a signing failure).
	serverConfig.Certificates = make([]Certificate, 1)
	serverConfig.Certificates[0].Certificate = [][]byte{testECDSACertificate}
	serverConfig.Certificates[0].PrivateKey = testECDSAPrivateKey
	serverConfig.BuildNameToCertificate()
	testClientHelloFailure(t, serverConfig, clientHello, "no cipher suite supported by both client and server")
}

func TestRenegotiationExtension(t *testing.T) {
	skipFIPS(t) // #70505

	clientHello := &clientHelloMsg{
		vers:                         VersionTLS12,
		compressionMethods:           []uint8{compressionNone},
		random:                       make([]byte, 32),
		secureRenegotiationSupported: true,
		cipherSuites:                 []uint16{TLS_RSA_WITH_RC4_128_SHA},
	}

	bufChan := make(chan []byte, 1)
	c, s := localPipe(t)

	go func() {
		cli := Client(c, testConfig)
		cli.vers = clientHello.vers
		if _, err := cli.writeHandshakeRecord(clientHello, nil); err != nil {
			testFatal(t, err)
		}

		buf := make([]byte, 1024)
		n, err := c.Read(buf)
		if err != nil {
			t.Errorf("Server read returned error: %s", err)
		}
		c.Close()
		bufChan <- buf[:n]
	}()

	Server(s, testConfig).Handshake()
	buf := <-bufChan

	if len(buf) < 5+4 {
		t.Fatalf("Server returned short message of length %d", len(buf))
	}
	// buf contains a TLS record, with a 5 byte record header and a 4 byte
	// handshake header. The length of the ServerHello is taken from the
	// handshake header.
	serverHelloLen := int(buf[6])<<16 | int(buf[7])<<8 | int(buf[8])

	var serverHello serverHelloMsg
	// unmarshal expects to be given the handshake header, but
	// serverHelloLen doesn't include it.
	if !serverHello.unmarshal(buf[5 : 9+serverHelloLen]) {
		t.Fatalf("Failed to parse ServerHello")
	}

	if !serverHello.secureRenegotiationSupported {
		t.Errorf("Secure renegotiation extension was not echoed.")
	}
}

func TestTLS12OnlyCipherSuites(t *testing.T) {
	skipFIPS(t) // No TLS 1.1 in FIPS mode.

	// Test that a Server doesn't select a TLS 1.2-only cipher suite when
	// the client negotiates TLS 1.1.
	clientHello := &clientHelloMsg{
		vers:   VersionTLS11,
		random: make([]byte, 32),
		cipherSuites: []uint16{
			// The Server, by default, will use the client's
			// preference order. So the GCM cipher suite
			// will be selected unless it's excluded because
			// of the version in this ClientHello.
			TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			TLS_RSA_WITH_RC4_128_SHA,
		},
		compressionMethods: []uint8{compressionNone},
		supportedCurves:    []CurveID{CurveP256, CurveP384, CurveP521},
		supportedPoints:    []uint8{pointFormatUncompressed},
	}

	c, s := localPipe(t)
	replyChan := make(chan any)
	go func() {
		cli := Client(c, testConfig)
		cli.vers = clientHello.vers
		if _, err := cli.writeHandshakeRecord(clientHello, nil); err != nil {
			testFatal(t, err)
		}
		reply, err := cli.readHandshake(nil)
		c.Close()
		if err != nil {
			replyChan <- err
		} else {
			replyChan <- reply
		}
	}()
	config := testConfig.Clone()
	config.CipherSuites = clientHello.cipherSuites
	Server(s, config).Handshake()
	s.Close()
	reply := <-replyChan
	if err, ok := reply.(error); ok {
		t.Fatal(err)
	}
	serverHello, ok := reply.(*serverHelloMsg)
	if !ok {
		t.Fatalf("didn't get ServerHello message in reply. Got %v\n", reply)
	}
	if s := serverHello.cipherSuite; s != TLS_RSA_WITH_RC4_128_SHA {
		t.Fatalf("bad cipher suite from server: %x", s)
	}
}

func TestTLSPointFormats(t *testing.T) {
	// Test that a Server returns the ec_point_format extension when ECC is
	// negotiated, and not on a RSA handshake or if ec_point_format is missing.
	tests := []struct {
		name                string
		cipherSuites        []uint16
		supportedCurves     []CurveID
		supportedPoints     []uint8
		wantSupportedPoints bool
	}{
		{"ECC", []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}, []CurveID{CurveP256}, []uint8{pointFormatUncompressed}, true},
		{"ECC without ec_point_format", []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}, []CurveID{CurveP256}, nil, false},
		{"ECC with extra values", []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}, []CurveID{CurveP256}, []uint8{13, 37, pointFormatUncompressed, 42}, true},
		{"RSA", []uint16{TLS_RSA_WITH_AES_256_GCM_SHA384}, nil, nil, false},
		{"RSA with ec_point_format", []uint16{TLS_RSA_WITH_AES_256_GCM_SHA384}, nil, []uint8{pointFormatUncompressed}, false},
	}
	for _, tt := range tests {
		// The RSA subtests should be enabled for FIPS 140 required mode: #70505
		if strings.HasPrefix(tt.name, "RSA") && fips140tls.Required() {
			t.Logf("skipping in FIPS mode.")
			continue
		}
		t.Run(tt.name, func(t *testing.T) {
			clientHello := &clientHelloMsg{
				vers:               VersionTLS12,
				random:             make([]byte, 32),
				cipherSuites:       tt.cipherSuites,
				compressionMethods: []uint8{compressionNone},
				supportedCurves:    tt.supportedCurves,
				supportedPoints:    tt.supportedPoints,
			}

			c, s := localPipe(t)
			replyChan := make(chan any)
			go func() {
				clientConfig := testConfig.Clone()
				clientConfig.Certificates = []Certificate{{Certificate: [][]byte{testRSA2048Certificate}, PrivateKey: testRSA2048PrivateKey}}
				cli := Client(c, clientConfig)
				cli.vers = clientHello.vers
				if _, err := cli.writeHandshakeRecord(clientHello, nil); err != nil {
					testFatal(t, err)
				}
				reply, err := cli.readHandshake(nil)
				c.Close()
				if err != nil {
					replyChan <- err
				} else {
					replyChan <- reply
				}
			}()
			serverConfig := testConfig.Clone()
			serverConfig.Certificates = []Certificate{{Certificate: [][]byte{testRSA2048Certificate}, PrivateKey: testRSA2048PrivateKey}}
			serverConfig.CipherSuites = clientHello.cipherSuites
			Server(s, serverConfig).Handshake()
			s.Close()
			reply := <-replyChan
			if err, ok := reply.(error); ok {
				t.Fatal(err)
			}
			serverHello, ok := reply.(*serverHelloMsg)
			if !ok {
				t.Fatalf("didn't get ServerHello message in reply. Got %v\n", reply)
			}
			if tt.wantSupportedPoints {
				if !bytes.Equal(serverHello.supportedPoints, []uint8{pointFormatUncompressed}) {
					t.Fatal("incorrect ec_point_format extension from server")
				}
			} else {
				if len(serverHello.supportedPoints) != 0 {
					t.Fatalf("unexpected ec_point_format extension from server: %v", serverHello.supportedPoints)
				}
			}
		})
	}
}

func TestAlertForwarding(t *testing.T) {
	c, s := localPipe(t)
	go func() {
		Client(c, testConfig).sendAlert(alertUnknownCA)
		c.Close()
	}()

	err := Server(s, testConfig).Handshake()
	s.Close()
	var opErr *net.OpError
	if !errors.As(err, &opErr) || opErr.Err != error(alertUnknownCA) {
		t.Errorf("Got error: %s; expected: %s", err, error(alertUnknownCA))
	}
}

func TestClose(t *testing.T) {
	c, s := localPipe(t)
	go c.Close()

	err := Server(s, testConfig).Handshake()
	s.Close()
	if err != io.EOF {
		t.Errorf("Got error: %s; expected: %s", err, io.EOF)
	}
}

func TestVersion(t *testing.T) {
	serverConfig := &Config{
		Certificates: testConfig.Certificates,
		MaxVersion:   VersionTLS13,
	}
	clientConfig := &Config{
		InsecureSkipVerify: true,
		MinVersion:         VersionTLS12,
	}
	state, _, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if state.Version != VersionTLS13 {
		t.Fatalf("incorrect version %x, should be %x", state.Version, VersionTLS11)
	}

	clientConfig.MinVersion = 0
	serverConfig.MaxVersion = VersionTLS11
	_, _, err = testHandshake(t, clientConfig, serverConfig)
	if err == nil {
		t.Fatalf("expected failure to connect with TLS 1.0/1.1")
	}
}

func TestCipherSuitePreference(t *testing.T) {
	skipFIPS(t) // No RC4 or CHACHA20_POLY1305 in FIPS mode.

	serverConfig := &Config{
		CipherSuites: []uint16{TLS_RSA_WITH_RC4_128_SHA, TLS_AES_128_GCM_SHA256,
			TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256},
		Certificates: testConfig.Certificates,
		MaxVersion:   VersionTLS12,
		GetConfigForClient: func(chi *ClientHelloInfo) (*Config, error) {
			if chi.CipherSuites[0] != TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 {
				t.Error("the advertised order should not depend on Config.CipherSuites")
			}
			if len(chi.CipherSuites) != 2+len(defaultCipherSuitesTLS13) {
				t.Error("the advertised TLS 1.2 suites should be filtered by Config.CipherSuites")
			}
			return nil, nil
		},
	}
	clientConfig := &Config{
		CipherSuites:       []uint16{TLS_RSA_WITH_AES_128_CBC_SHA, TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256},
		InsecureSkipVerify: true,
	}
	state, _, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if state.CipherSuite != TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256 {
		t.Error("the preference order should not depend on Config.CipherSuites")
	}
}

func TestSCTHandshake(t *testing.T) {
	t.Run("TLSv12", func(t *testing.T) { testSCTHandshake(t, VersionTLS12) })
	t.Run("TLSv13", func(t *testing.T) { testSCTHandshake(t, VersionTLS13) })
}

func testSCTHandshake(t *testing.T, version uint16) {
	expected := [][]byte{[]byte("certificate"), []byte("transparency")}
	serverConfig := &Config{
		Certificates: []Certificate{{
			Certificate:                 [][]byte{testRSACertificate},
			PrivateKey:                  testRSAPrivateKey,
			SignedCertificateTimestamps: expected,
		}},
		MaxVersion: version,
	}
	clientConfig := &Config{
		InsecureSkipVerify: true,
	}
	_, state, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	actual := state.SignedCertificateTimestamps
	if len(actual) != len(expected) {
		t.Fatalf("got %d scts, want %d", len(actual), len(expected))
	}
	for i, sct := range expected {
		if !bytes.Equal(sct, actual[i]) {
			t.Fatalf("SCT #%d was %x, but expected %x", i, actual[i], sct)
		}
	}
}

func TestCrossVersionResume(t *testing.T) {
	t.Run("TLSv12", func(t *testing.T) { testCrossVersionResume(t, VersionTLS12) })
	t.Run("TLSv13", func(t *testing.T) { testCrossVersionResume(t, VersionTLS13) })
}

func testCrossVersionResume(t *testing.T, version uint16) {
	serverConfig := &Config{
		CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		Certificates: testConfig.Certificates,
		Time:         testTime,
	}
	clientConfig := &Config{
		CipherSuites:       []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		InsecureSkipVerify: true,
		ClientSessionCache: NewLRUClientSessionCache(1),
		ServerName:         "servername",
		MinVersion:         VersionTLS12,
		Time:               testTime,
	}

	// Establish a session at TLS 1.3.
	clientConfig.MaxVersion = VersionTLS13
	_, _, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}

	// The client session cache now contains a TLS 1.3 session.
	state, _, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if !state.DidResume {
		t.Fatalf("handshake did not resume at the same version")
	}

	// Test that the server will decline to resume at a lower version.
	clientConfig.MaxVersion = VersionTLS12
	state, _, err = testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if state.DidResume {
		t.Fatalf("handshake resumed at a lower version")
	}

	// The client session cache now contains a TLS 1.2 session.
	state, _, err = testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if !state.DidResume {
		t.Fatalf("handshake did not resume at the same version")
	}

	// Test that the server will decline to resume at a higher version.
	clientConfig.MaxVersion = VersionTLS13
	state, _, err = testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if state.DidResume {
		t.Fatalf("handshake resumed at a higher version")
	}
}

// Note: see comment in handshake_test.go for details of how the reference
// tests work.

// serverTest represents a test of the TLS server handshake against a reference
// implementation.
type serverTest struct {
	// name is a freeform string identifying the test and the file in which
	// the expected results will be stored.
	name string
	// command, if not empty, contains a series of arguments for the
	// command to run for the reference server.
	command []string
	// expectedPeerCerts contains a list of PEM blocks of expected
	// certificates from the client.
	expectedPeerCerts []string
	// config, if not nil, contains a custom Config to use for this test.
	config *Config
	// expectHandshakeErrorIncluding, when not empty, contains a string
	// that must be a substring of the error resulting from the handshake.
	expectHandshakeErrorIncluding string
	// validate, if not nil, is a function that will be called with the
	// ConnectionState of the resulting connection. It returns false if the
	// ConnectionState is unacceptable.
	validate func(ConnectionState) error
	// wait, if true, prevents this subtest from calling t.Parallel.
	// If false, runServerTest* returns immediately.
	wait bool
}

var defaultClientCommand = []string{"openssl", "s_client", "-no_ticket"}

// connFromCommand starts opens a listening socket and starts the reference
// client to connect to it. It returns a recordingConn that wraps the resulting
// connection.
func (test *serverTest) connFromCommand() (conn *recordingConn, child *exec.Cmd, err error) {
	l, err := net.ListenTCP("tcp", &net.TCPAddr{
		IP:   net.IPv4(127, 0, 0, 1),
		Port: 0,
	})
	if err != nil {
		return nil, nil, err
	}
	defer l.Close()

	port := l.Addr().(*net.TCPAddr).Port

	var command []string
	command = append(command, test.command...)
	if len(command) == 0 {
		command = defaultClientCommand
	}
	command = append(command, "-connect")
	command = append(command, fmt.Sprintf("127.0.0.1:%d", port))
	cmd := exec.Command(command[0], command[1:]...)
	cmd.Stdin = nil
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := cmd.Start(); err != nil {
		return nil, nil, err
	}

	connChan := make(chan any, 1)
	go func() {
		tcpConn, err := l.Accept()
		if err != nil {
			connChan <- err
			return
		}
		connChan <- tcpConn
	}()

	var tcpConn net.Conn
	select {
	case connOrError := <-connChan:
		if err, ok := connOrError.(error); ok {
			return nil, nil, err
		}
		tcpConn = connOrError.(net.Conn)
	case <-time.After(2 * time.Second):
		return nil, nil, errors.New("timed out waiting for connection from child process")
	}

	record := &recordingConn{
		Conn: tcpConn,
	}

	return record, cmd, nil
}

func (test *serverTest) dataPath() string {
	return filepath.Join("testdata", "Server-"+test.name)
}

func (test *serverTest) loadData() (flows [][]byte, err error) {
	in, err := os.Open(test.dataPath())
	if err != nil {
		return nil, err
	}
	defer in.Close()
	return parseTestData(in)
}

func (test *serverTest) run(t *testing.T, write bool) {
	var serverConn net.Conn
	var recordingConn *recordingConn
	var childProcess *exec.Cmd

	if write {
		var err error
		recordingConn, childProcess, err = test.connFromCommand()
		if err != nil {
			t.Fatalf("Failed to start subcommand: %s", err)
		}
		serverConn = recordingConn
		defer func() {
			if t.Failed() {
				t.Logf("OpenSSL output:\n\n%s", childProcess.Stdout)
			}
		}()
	} else {
		flows, err := test.loadData()
		if err != nil {
			t.Fatalf("Failed to load data from %s", test.dataPath())
		}
		serverConn = &replayingConn{t: t, flows: flows, reading: true}
	}
	config := test.config
	if config == nil {
		config = testConfig
	}
	server := Server(serverConn, config)

	_, err := server.Write([]byte("hello, world\n"))
	if len(test.expectHandshakeErrorIncluding) > 0 {
		if err == nil {
			t.Errorf("Error expected, but no error returned")
		} else if s := err.Error(); !strings.Contains(s, test.expectHandshakeErrorIncluding) {
			t.Errorf("Error expected containing '%s' but got '%s'", test.expectHandshakeErrorIncluding, s)
		}
	} else {
		if err != nil {
			t.Logf("Error from Server.Write: '%s'", err)
		}
	}
	server.Close()

	connState := server.ConnectionState()
	peerCerts := connState.PeerCertificates
	if len(peerCerts) == len(test.expectedPeerCerts) {
		for i, peerCert := range peerCerts {
			block, _ := pem.Decode([]byte(test.expectedPeerCerts[i]))
			if !bytes.Equal(block.Bytes, peerCert.Raw) {
				t.Fatalf("%s: mismatch on peer cert %d", test.name, i+1)
			}
		}
	} else {
		t.Fatalf("%s: mismatch on peer list length: %d (wanted) != %d (got)", test.name, len(test.expectedPeerCerts), len(peerCerts))
	}

	if test.validate != nil && !t.Failed() {
		if err := test.validate(connState); err != nil {
			t.Fatalf("validate callback returned error: %s", err)
		}
	}

	if write {
		serverConn.Close()
		path := test.dataPath()
		out, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			t.Fatalf("Failed to create output file: %s", err)
		}
		defer out.Close()
		recordingConn.Close()
		if len(recordingConn.flows) < 3 {
			if len(test.expectHandshakeErrorIncluding) == 0 {
				t.Fatalf("Handshake failed")
			}
		}
		recordingConn.WriteTo(out)
		t.Logf("Wrote %s\n", path)
		childProcess.Wait()
	}
}

func runServerTestForVersion(t *testing.T, template *serverTest, version, option string) {
	// Make a deep copy of the template before going parallel.
	test := *template
	if template.config != nil {
		test.config = template.config.Clone()
	}
	test.name = version + "-" + test.name
	if len(test.command) == 0 {
		test.command = defaultClientCommand
	}
	test.command = append([]string(nil), test.command...)
	test.command = append(test.command, option)

	runTestAndUpdateIfNeeded(t, version, test.run, test.wait)
}

func runServerTestTLS10(t *testing.T, template *serverTest) {
	runServerTestForVersion(t, template, "TLSv10", "-tls1")
}

func runServerTestTLS11(t *testing.T, template *serverTest) {
	runServerTestForVersion(t, template, "TLSv11", "-tls1_1")
}

func runServerTestTLS12(t *testing.T, template *serverTest) {
	runServerTestForVersion(t, template, "TLSv12", "-tls1_2")
}

func runServerTestTLS13(t *testing.T, template *serverTest) {
	runServerTestForVersion(t, template, "TLSv13", "-tls1_3")
}

func TestHandshakeServerRSARC4(t *testing.T) {
	test := &serverTest{
		name:    "RSA-RC4",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "RC4-SHA"},
	}
	runServerTestTLS10(t, test)
	runServerTestTLS11(t, test)
	runServerTestTLS12(t, test)
}

func TestHandshakeServerRSA3DES(t *testing.T) {
	test := &serverTest{
		name:    "RSA-3DES",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "DES-CBC3-SHA"},
	}
	runServerTestTLS10(t, test)
	runServerTestTLS12(t, test)
}

func TestHandshakeServerRSAAES(t *testing.T) {
	test := &serverTest{
		name:    "RSA-AES",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA"},
	}
	runServerTestTLS10(t, test)
	runServerTestTLS12(t, test)
}

func TestHandshakeServerAESGCM(t *testing.T) {
	test := &serverTest{
		name:    "RSA-AES-GCM",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-RSA-AES128-GCM-SHA256"},
	}
	runServerTestTLS12(t, test)
}

func TestHandshakeServerAES256GCMSHA384(t *testing.T) {
	test := &serverTest{
		name:    "RSA-AES256-GCM-SHA384",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-RSA-AES256-GCM-SHA384"},
	}
	runServerTestTLS12(t, test)
}

func TestHandshakeServerAES128SHA256(t *testing.T) {
	test := &serverTest{
		name:    "AES128-SHA256",
		command: []string{"openssl", "s_client", "-no_ticket", "-ciphersuites", "TLS_AES_128_GCM_SHA256"},
	}
	runServerTestTLS13(t, test)
}
func TestHandshakeServerAES256SHA384(t *testing.T) {
	test := &serverTest{
		name:    "AES256-SHA384",
		command: []string{"openssl", "s_client", "-no_ticket", "-ciphersuites", "TLS_AES_256_GCM_SHA384"},
	}
	runServerTestTLS13(t, test)
}
func TestHandshakeServerCHACHA20SHA256(t *testing.T) {
	test := &serverTest{
		name:    "CHACHA20-SHA256",
		command: []string{"openssl", "s_client", "-no_ticket", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256"},
	}
	runServerTestTLS13(t, test)
}

func TestHandshakeServerECDHEECDSAAES(t *testing.T) {
	config := testConfig.Clone()
	config.Certificates = make([]Certificate, 1)
	config.Certificates[0].Certificate = [][]byte{testECDSACertificate}
	config.Certificates[0].PrivateKey = testECDSAPrivateKey
	config.BuildNameToCertificate()

	test := &serverTest{
		name:    "ECDHE-ECDSA-AES",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-ECDSA-AES256-SHA", "-ciphersuites", "TLS_AES_128_GCM_SHA256"},
		config:  config,
	}
	runServerTestTLS10(t, test)
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)
}

func TestHandshakeServerX25519(t *testing.T) {
	config := testConfig.Clone()
	config.CurvePreferences = []CurveID{X25519}

	test := &serverTest{
		name:    "X25519",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-RSA-CHACHA20-POLY1305", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256", "-curves", "X25519"},
		config:  config,
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)
}

func TestHandshakeServerP256(t *testing.T) {
	config := testConfig.Clone()
	config.CurvePreferences = []CurveID{CurveP256}

	test := &serverTest{
		name:    "P256",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-RSA-CHACHA20-POLY1305", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256", "-curves", "P-256"},
		config:  config,
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)
}

func TestHandshakeServerHelloRetryRequest(t *testing.T) {
	config := testConfig.Clone()
	config.CurvePreferences = []CurveID{CurveP256}

	test := &serverTest{
		name:    "HelloRetryRequest",
		command: []string{"openssl", "s_client", "-no_ticket", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256", "-curves", "X25519:P-256"},
		config:  config,
		validate: func(cs ConnectionState) error {
			if !cs.testingOnlyDidHRR {
				return errors.New("expected HelloRetryRequest")
			}
			return nil
		},
	}
	runServerTestTLS13(t, test)
}

// TestHandshakeServerKeySharePreference checks that we prefer a key share even
// if it's later in the CurvePreferences order.
func TestHandshakeServerKeySharePreference(t *testing.T) {
	config := testConfig.Clone()
	config.CurvePreferences = []CurveID{X25519, CurveP256}

	test := &serverTest{
		name:    "KeySharePreference",
		command: []string{"openssl", "s_client", "-no_ticket", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256", "-curves", "P-256:X25519"},
		config:  config,
		validate: func(cs ConnectionState) error {
			if cs.testingOnlyDidHRR {
				return errors.New("unexpected HelloRetryRequest")
			}
			return nil
		},
	}
	runServerTestTLS13(t, test)
}

func TestHandshakeServerALPN(t *testing.T) {
	config := testConfig.Clone()
	config.NextProtos = []string{"proto1", "proto2"}

	test := &serverTest{
		name: "ALPN",
		// Note that this needs OpenSSL 1.0.2 because that is the first
		// version that supports the -alpn flag.
		command: []string{"openssl", "s_client", "-alpn", "proto2,proto1", "-cipher", "ECDHE-RSA-CHACHA20-POLY1305", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256"},
		config:  config,
		validate: func(state ConnectionState) error {
			// The server's preferences should override the client.
			if state.NegotiatedProtocol != "proto1" {
				return fmt.Errorf("Got protocol %q, wanted proto1", state.NegotiatedProtocol)
			}
			return nil
		},
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)
}

func TestHandshakeServerALPNNoMatch(t *testing.T) {
	config := testConfig.Clone()
	config.NextProtos = []string{"proto3"}

	test := &serverTest{
		name: "ALPN-NoMatch",
		// Note that this needs OpenSSL 1.0.2 because that is the first
		// version that supports the -alpn flag.
		command:                       []string{"openssl", "s_client", "-alpn", "proto2,proto1", "-cipher", "ECDHE-RSA-CHACHA20-POLY1305", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256"},
		config:                        config,
		expectHandshakeErrorIncluding: "client requested unsupported application protocol",
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)
}

func TestHandshakeServerALPNNotConfigured(t *testing.T) {
	config := testConfig.Clone()
	config.NextProtos = nil

	test := &serverTest{
		name: "ALPN-NotConfigured",
		// Note that this needs OpenSSL 1.0.2 because that is the first
		// version that supports the -alpn flag.
		command: []string{"openssl", "s_client", "-alpn", "proto2,proto1", "-cipher", "ECDHE-RSA-CHACHA20-POLY1305", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256"},
		config:  config,
		validate: func(state ConnectionState) error {
			if state.NegotiatedProtocol != "" {
				return fmt.Errorf("Got protocol %q, wanted nothing", state.NegotiatedProtocol)
			}
			return nil
		},
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)
}

func TestHandshakeServerALPNFallback(t *testing.T) {
	config := testConfig.Clone()
	config.NextProtos = []string{"proto1", "h2", "proto2"}

	test := &serverTest{
		name: "ALPN-Fallback",
		// Note that this needs OpenSSL 1.0.2 because that is the first
		// version that supports the -alpn flag.
		command: []string{"openssl", "s_client", "-alpn", "proto3,http/1.1,proto4", "-cipher", "ECDHE-RSA-CHACHA20-POLY1305", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256"},
		config:  config,
		validate: func(state ConnectionState) error {
			if state.NegotiatedProtocol != "" {
				return fmt.Errorf("Got protocol %q, wanted nothing", state.NegotiatedProtocol)
			}
			return nil
		},
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)
}

// TestHandshakeServerSNI involves a client sending an SNI extension of
// "snitest.com", which happens to match the CN of testSNICertificate. The test
// verifies that the server correctly selects that certificate.
func TestHandshakeServerSNI(t *testing.T) {
	test := &serverTest{
		name:    "SNI",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA", "-servername", "snitest.com"},
	}
	runServerTestTLS12(t, test)
}

// TestHandshakeServerSNIGetCertificate is similar to TestHandshakeServerSNI, but
// tests the dynamic GetCertificate method
func TestHandshakeServerSNIGetCertificate(t *testing.T) {
	config := testConfig.Clone()

	// Replace the NameToCertificate map with a GetCertificate function
	nameToCert := config.NameToCertificate
	config.NameToCertificate = nil
	config.GetCertificate = func(clientHello *ClientHelloInfo) (*Certificate, error) {
		cert := nameToCert[clientHello.ServerName]
		return cert, nil
	}
	test := &serverTest{
		name:    "SNI-GetCertificate",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA", "-servername", "snitest.com"},
		config:  config,
	}
	runServerTestTLS12(t, test)
}

// TestHandshakeServerSNIGetCertificateNotFound is similar to
// TestHandshakeServerSNICertForName, but tests to make sure that when the
// GetCertificate method doesn't return a cert, we fall back to what's in
// the NameToCertificate map.
func TestHandshakeServerSNIGetCertificateNotFound(t *testing.T) {
	config := testConfig.Clone()

	config.GetCertificate = func(clientHello *ClientHelloInfo) (*Certificate, error) {
		return nil, nil
	}
	test := &serverTest{
		name:    "SNI-GetCertificateNotFound",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA", "-servername", "snitest.com"},
		config:  config,
	}
	runServerTestTLS12(t, test)
}

// TestHandshakeServerGetCertificateExtensions tests to make sure that the
// Extensions passed to GetCertificate match what we expect based on the
// clientHelloMsg
func TestHandshakeServerGetCertificateExtensions(t *testing.T) {
	const errMsg = "TestHandshakeServerGetCertificateExtensions error"
	// ensure the test condition inside our GetCertificate callback
	// is actually invoked
	var called atomic.Int32

	testVersions := []uint16{VersionTLS12, VersionTLS13}
	for _, vers := range testVersions {
		t.Run(fmt.Sprintf("TLS version %04x", vers), func(t *testing.T) {
			pk, _ := ecdh.P256().GenerateKey(rand.Reader)
			clientHello := &clientHelloMsg{
				vers:                         vers,
				random:                       make([]byte, 32),
				cipherSuites:                 []uint16{TLS_AES_128_GCM_SHA256},
				compressionMethods:           []uint8{compressionNone},
				serverName:                   "test",
				keyShares:                    []keyShare{{group: CurveP256, data: pk.PublicKey().Bytes()}},
				supportedCurves:              []CurveID{CurveP256},
				supportedSignatureAlgorithms: []SignatureScheme{ECDSAWithP256AndSHA256},
			}

			// the clientHelloMsg initialized just above is serialized with
			// two extensions: server_name(0) and application_layer_protocol_negotiation(16)
			expectedExtensions := []uint16{
				extensionServerName,
				extensionSupportedCurves,
				extensionSignatureAlgorithms,
				extensionKeyShare,
			}

			if vers == VersionTLS13 {
				clientHello.supportedVersions = []uint16{VersionTLS13}
				expectedExtensions = append(expectedExtensions, extensionSupportedVersions)
			}

			// Go's TLS client presents extensions in the ClientHello sorted by extension ID
			slices.Sort(expectedExtensions)

			serverConfig := testConfig.Clone()
			serverConfig.GetCertificate = func(clientHello *ClientHelloInfo) (*Certificate, error) {
				if !slices.Equal(expectedExtensions, clientHello.Extensions) {
					t.Errorf("expected extensions on ClientHelloInfo (%v) to match clientHelloMsg (%v)", expectedExtensions, clientHello.Extensions)
				}
				called.Add(1)

				return nil, errors.New(errMsg)
			}
			testClientHelloFailure(t, serverConfig, clientHello, errMsg)
		})
	}

	if int(called.Load()) != len(testVersions) {
		t.Error("expected our GetCertificate test to be called twice")
	}
}

// TestHandshakeServerSNIGetCertificateError tests to make sure that errors in
// GetCertificate result in a tls alert.
func TestHandshakeServerSNIGetCertificateError(t *testing.T) {
	const errMsg = "TestHandshakeServerSNIGetCertificateError error"

	serverConfig := testConfig.Clone()
	serverConfig.GetCertificate = func(clientHello *ClientHelloInfo) (*Certificate, error) {
		return nil, errors.New(errMsg)
	}

	clientHello := &clientHelloMsg{
		vers:               VersionTLS12,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		compressionMethods: []uint8{compressionNone},
		serverName:         "test",
	}
	testClientHelloFailure(t, serverConfig, clientHello, errMsg)
}

// TestHandshakeServerEmptyCertificates tests that GetCertificates is called in
// the case that Certificates is empty, even without SNI.
func TestHandshakeServerEmptyCertificates(t *testing.T) {
	const errMsg = "TestHandshakeServerEmptyCertificates error"

	serverConfig := testConfig.Clone()
	serverConfig.GetCertificate = func(clientHello *ClientHelloInfo) (*Certificate, error) {
		return nil, errors.New(errMsg)
	}
	serverConfig.Certificates = nil

	clientHello := &clientHelloMsg{
		vers:               VersionTLS12,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		compressionMethods: []uint8{compressionNone},
	}
	testClientHelloFailure(t, serverConfig, clientHello, errMsg)

	// With an empty Certificates and a nil GetCertificate, the server
	// should always return a “no certificates” error.
	serverConfig.GetCertificate = nil

	clientHello = &clientHelloMsg{
		vers:               VersionTLS12,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		compressionMethods: []uint8{compressionNone},
	}
	testClientHelloFailure(t, serverConfig, clientHello, "no certificates")
}

func TestServerResumption(t *testing.T) {
	sessionFilePath := tempFile("")
	defer os.Remove(sessionFilePath)

	testIssue := &serverTest{
		name:    "IssueTicket",
		command: []string{"openssl", "s_client", "-cipher", "AES128-SHA", "-ciphersuites", "TLS_AES_128_GCM_SHA256", "-sess_out", sessionFilePath},
		wait:    true,
	}
	testResume := &serverTest{
		name:    "Resume",
		command: []string{"openssl", "s_client", "-cipher", "AES128-SHA", "-ciphersuites", "TLS_AES_128_GCM_SHA256", "-sess_in", sessionFilePath},
		validate: func(state ConnectionState) error {
			if !state.DidResume {
				return errors.New("did not resume")
			}
			return nil
		},
	}

	runServerTestTLS12(t, testIssue)
	runServerTestTLS12(t, testResume)

	runServerTestTLS13(t, testIssue)
	runServerTestTLS13(t, testResume)

	config := testConfig.Clone()
	config.CurvePreferences = []CurveID{CurveP256}

	testResumeHRR := &serverTest{
		name: "Resume-HelloRetryRequest",
		command: []string{"openssl", "s_client", "-curves", "X25519:P-256", "-cipher", "AES128-SHA", "-ciphersuites",
			"TLS_AES_128_GCM_SHA256", "-sess_in", sessionFilePath},
		config: config,
		validate: func(state ConnectionState) error {
			if !state.DidResume {
				return errors.New("did not resume")
			}
			return nil
		},
	}

	runServerTestTLS13(t, testResumeHRR)
}

func TestServerResumptionDisabled(t *testing.T) {
	sessionFilePath := tempFile("")
	defer os.Remove(sessionFilePath)

	config := testConfig.Clone()

	testIssue := &serverTest{
		name:    "IssueTicketPreDisable",
		command: []string{"openssl", "s_client", "-cipher", "AES128-SHA", "-ciphersuites", "TLS_AES_128_GCM_SHA256", "-sess_out", sessionFilePath},
		config:  config,
		wait:    true,
	}
	testResume := &serverTest{
		name:    "ResumeDisabled",
		command: []string{"openssl", "s_client", "-cipher", "AES128-SHA", "-ciphersuites", "TLS_AES_128_GCM_SHA256", "-sess_in", sessionFilePath},
		config:  config,
		validate: func(state ConnectionState) error {
			if state.DidResume {
				return errors.New("resumed with SessionTicketsDisabled")
			}
			return nil
		},
	}

	config.SessionTicketsDisabled = false
	runServerTestTLS12(t, testIssue)
	config.SessionTicketsDisabled = true
	runServerTestTLS12(t, testResume)

	config.SessionTicketsDisabled = false
	runServerTestTLS13(t, testIssue)
	config.SessionTicketsDisabled = true
	runServerTestTLS13(t, testResume)
}

func TestFallbackSCSV(t *testing.T) {
	serverConfig := Config{
		Certificates: testConfig.Certificates,
		MinVersion:   VersionTLS11,
	}
	test := &serverTest{
		name:   "FallbackSCSV",
		config: &serverConfig,
		// OpenSSL 1.0.1j is needed for the -fallback_scsv option.
		command:                       []string{"openssl", "s_client", "-fallback_scsv"},
		expectHandshakeErrorIncluding: "inappropriate protocol fallback",
	}
	runServerTestTLS11(t, test)
}

func TestHandshakeServerExportKeyingMaterial(t *testing.T) {
	test := &serverTest{
		name:    "ExportKeyingMaterial",
		command: []string{"openssl", "s_client", "-cipher", "ECDHE-RSA-AES256-SHA", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256"},
		config:  testConfig.Clone(),
		validate: func(state ConnectionState) error {
			if km, err := state.ExportKeyingMaterial("test", nil, 42); err != nil {
				return fmt.Errorf("ExportKeyingMaterial failed: %v", err)
			} else if len(km) != 42 {
				return fmt.Errorf("Got %d bytes from ExportKeyingMaterial, wanted %d", len(km), 42)
			}
			return nil
		},
	}
	runServerTestTLS10(t, test)
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)
}

func TestHandshakeServerRSAPKCS1v15(t *testing.T) {
	test := &serverTest{
		name:    "RSA-RSAPKCS1v15",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-RSA-CHACHA20-POLY1305", "-sigalgs", "rsa_pkcs1_sha256"},
	}
	runServerTestTLS12(t, test)
}

func TestHandshakeServerRSAPSS(t *testing.T) {
	// We send rsa_pss_rsae_sha512 first, as the test key won't fit, and we
	// verify the server implementation will disregard the client preference in
	// that case. See Issue 29793.
	test := &serverTest{
		name:    "RSA-RSAPSS",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-RSA-CHACHA20-POLY1305", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256", "-sigalgs", "rsa_pss_rsae_sha512:rsa_pss_rsae_sha256"},
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)

	test = &serverTest{
		name:                          "RSA-RSAPSS-TooSmall",
		command:                       []string{"openssl", "s_client", "-no_ticket", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256", "-sigalgs", "rsa_pss_rsae_sha512"},
		expectHandshakeErrorIncluding: "peer doesn't support any of the certificate's signature algorithms",
	}
	runServerTestTLS13(t, test)
}

func TestHandshakeServerEd25519(t *testing.T) {
	config := testConfig.Clone()
	config.Certificates = make([]Certificate, 1)
	config.Certificates[0].Certificate = [][]byte{testEd25519Certificate}
	config.Certificates[0].PrivateKey = testEd25519PrivateKey
	config.BuildNameToCertificate()

	test := &serverTest{
		name:    "Ed25519",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-ECDSA-CHACHA20-POLY1305", "-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256"},
		config:  config,
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)
}

func benchmarkHandshakeServer(b *testing.B, version uint16, cipherSuite uint16, curve CurveID, cert []byte, key crypto.PrivateKey) {
	config := testConfig.Clone()
	config.CipherSuites = []uint16{cipherSuite}
	config.CurvePreferences = []CurveID{curve}
	config.Certificates = make([]Certificate, 1)
	config.Certificates[0].Certificate = [][]byte{cert}
	config.Certificates[0].PrivateKey = key
	config.BuildNameToCertificate()

	clientConn, serverConn := localPipe(b)
	serverConn = &recordingConn{Conn: serverConn}
	go func() {
		config := testConfig.Clone()
		config.MaxVersion = version
		config.CurvePreferences = []CurveID{curve}
		client := Client(clientConn, config)
		client.Handshake()
	}()
	server := Server(serverConn, config)
	if err := server.Handshake(); err != nil {
		b.Fatalf("handshake failed: %v", err)
	}
	serverConn.Close()
	flows := serverConn.(*recordingConn).flows

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		replay := &replayingConn{t: b, flows: slices.Clone(flows), reading: true}
		server := Server(replay, config)
		if err := server.Handshake(); err != nil {
			b.Fatalf("handshake failed: %v", err)
		}
	}
}

func BenchmarkHandshakeServer(b *testing.B) {
	b.Run("RSA", func(b *testing.B) {
		benchmarkHandshakeServer(b, VersionTLS12, TLS_RSA_WITH_AES_128_GCM_SHA256,
			0, testRSACertificate, testRSAPrivateKey)
	})
	b.Run("ECDHE-P256-RSA", func(b *testing.B) {
		b.Run("TLSv13", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS13, TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
				CurveP256, testRSACertificate, testRSAPrivateKey)
		})
		b.Run("TLSv12", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS12, TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
				CurveP256, testRSACertificate, testRSAPrivateKey)
		})
	})
	b.Run("ECDHE-P256-ECDSA-P256", func(b *testing.B) {
		b.Run("TLSv13", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS13, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
				CurveP256, testP256Certificate, testP256PrivateKey)
		})
		b.Run("TLSv12", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS12, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
				CurveP256, testP256Certificate, testP256PrivateKey)
		})
	})
	b.Run("ECDHE-X25519-ECDSA-P256", func(b *testing.B) {
		b.Run("TLSv13", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS13, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
				X25519, testP256Certificate, testP256PrivateKey)
		})
		b.Run("TLSv12", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS12, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
				X25519, testP256Certificate, testP256PrivateKey)
		})
	})
	b.Run("ECDHE-P521-ECDSA-P521", func(b *testing.B) {
		if testECDSAPrivateKey.PublicKey.Curve != elliptic.P521() {
			b.Fatal("test ECDSA key doesn't use curve P-521")
		}
		b.Run("TLSv13", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS13, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
				CurveP521, testECDSACertificate, testECDSAPrivateKey)
		})
		b.Run("TLSv12", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS12, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
				CurveP521, testECDSACertificate, testECDSAPrivateKey)
		})
	})
}

func TestClientAuth(t *testing.T) {
	var certPath, keyPath, ecdsaCertPath, ecdsaKeyPath, ed25519CertPath, ed25519KeyPath string

	if *update {
		certPath = tempFile(clientCertificatePEM)
		defer os.Remove(certPath)
		keyPath = tempFile(clientKeyPEM)
		defer os.Remove(keyPath)
		ecdsaCertPath = tempFile(clientECDSACertificatePEM)
		defer os.Remove(ecdsaCertPath)
		ecdsaKeyPath = tempFile(clientECDSAKeyPEM)
		defer os.Remove(ecdsaKeyPath)
		ed25519CertPath = tempFile(clientEd25519CertificatePEM)
		defer os.Remove(ed25519CertPath)
		ed25519KeyPath = tempFile(clientEd25519KeyPEM)
		defer os.Remove(ed25519KeyPath)
	} else {
		t.Parallel()
	}

	config := testConfig.Clone()
	config.ClientAuth = RequestClientCert

	test := &serverTest{
		name:    "ClientAuthRequestedNotGiven",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA", "-ciphersuites", "TLS_AES_128_GCM_SHA256"},
		config:  config,
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)

	test = &serverTest{
		name: "ClientAuthRequestedAndGiven",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA", "-ciphersuites", "TLS_AES_128_GCM_SHA256",
			"-cert", certPath, "-key", keyPath, "-client_sigalgs", "rsa_pss_rsae_sha256"},
		config:            config,
		expectedPeerCerts: []string{clientCertificatePEM},
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)

	test = &serverTest{
		name: "ClientAuthRequestedAndECDSAGiven",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA", "-ciphersuites", "TLS_AES_128_GCM_SHA256",
			"-cert", ecdsaCertPath, "-key", ecdsaKeyPath},
		config:            config,
		expectedPeerCerts: []string{clientECDSACertificatePEM},
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)

	test = &serverTest{
		name: "ClientAuthRequestedAndEd25519Given",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA", "-ciphersuites", "TLS_AES_128_GCM_SHA256",
			"-cert", ed25519CertPath, "-key", ed25519KeyPath},
		config:            config,
		expectedPeerCerts: []string{clientEd25519CertificatePEM},
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)

	test = &serverTest{
		name: "ClientAuthRequestedAndPKCS1v15Given",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA",
			"-cert", certPath, "-key", keyPath, "-client_sigalgs", "rsa_pkcs1_sha256"},
		config:            config,
		expectedPeerCerts: []string{clientCertificatePEM},
	}
	runServerTestTLS12(t, test)
}

func TestSNIGivenOnFailure(t *testing.T) {
	const expectedServerName = "test.testing"

	clientHello := &clientHelloMsg{
		vers:               VersionTLS12,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		compressionMethods: []uint8{compressionNone},
		serverName:         expectedServerName,
	}

	serverConfig := testConfig.Clone()
	// Erase the server's cipher suites to ensure the handshake fails.
	serverConfig.CipherSuites = nil

	c, s := localPipe(t)
	go func() {
		cli := Client(c, testConfig)
		cli.vers = clientHello.vers
		if _, err := cli.writeHandshakeRecord(clientHello, nil); err != nil {
			testFatal(t, err)
		}
		c.Close()
	}()
	conn := Server(s, serverConfig)
	ctx := context.Background()
	ch, _, err := conn.readClientHello(ctx)
	hs := serverHandshakeState{
		c:           conn,
		ctx:         ctx,
		clientHello: ch,
	}
	if err == nil {
		err = hs.processClientHello()
	}
	if err == nil {
		err = hs.pickCipherSuite()
	}
	defer s.Close()

	if err == nil {
		t.Error("No error reported from server")
	}

	cs := hs.c.ConnectionState()
	if cs.HandshakeComplete {
		t.Error("Handshake registered as complete")
	}

	if cs.ServerName != expectedServerName {
		t.Errorf("Expected ServerName of %q, but got %q", expectedServerName, cs.ServerName)
	}
}

var getConfigForClientTests = []struct {
	setup          func(config *Config)
	callback       func(clientHello *ClientHelloInfo) (*Config, error)
	errorSubstring string
	verify         func(config *Config) error
}{
	{
		nil,
		func(clientHello *ClientHelloInfo) (*Config, error) {
			return nil, nil
		},
		"",
		nil,
	},
	{
		nil,
		func(clientHello *ClientHelloInfo) (*Config, error) {
			return nil, errors.New("should bubble up")
		},
		"should bubble up",
		nil,
	},
	{
		nil,
		func(clientHello *ClientHelloInfo) (*Config, error) {
			config := testConfig.Clone()
			// Setting a maximum version of TLS 1.1 should cause
			// the handshake to fail, as the client MinVersion is TLS 1.2.
			config.MaxVersion = VersionTLS11
			return config, nil
		},
		"client offered only unsupported versions",
		nil,
	},
	{
		func(config *Config) {
			for i := range config.SessionTicketKey {
				config.SessionTicketKey[i] = byte(i)
			}
			config.sessionTicketKeys = nil
		},
		func(clientHello *ClientHelloInfo) (*Config, error) {
			config := testConfig.Clone()
			for i := range config.SessionTicketKey {
				config.SessionTicketKey[i] = 0
			}
			config.sessionTicketKeys = nil
			return config, nil
		},
		"",
		func(config *Config) error {
			if config.SessionTicketKey == [32]byte{} {
				return fmt.Errorf("expected SessionTicketKey to be set")
			}
			return nil
		},
	},
	{
		func(config *Config) {
			var dummyKey [32]byte
			for i := range dummyKey {
				dummyKey[i] = byte(i)
			}

			config.SetSessionTicketKeys([][32]byte{dummyKey})
		},
		func(clientHello *ClientHelloInfo) (*Config, error) {
			config := testConfig.Clone()
			config.sessionTicketKeys = nil
			return config, nil
		},
		"",
		func(config *Config) error {
			if config.SessionTicketKey == [32]byte{} {
				return fmt.Errorf("expected SessionTicketKey to be set")
			}
			return nil
		},
	},
}

func TestGetConfigForClient(t *testing.T) {
	serverConfig := testConfig.Clone()
	clientConfig := testConfig.Clone()
	clientConfig.MinVersion = VersionTLS12

	for i, test := range getConfigForClientTests {
		if test.setup != nil {
			test.setup(serverConfig)
		}

		var configReturned *Config
		serverConfig.GetConfigForClient = func(clientHello *ClientHelloInfo) (*Config, error) {
			config, err := test.callback(clientHello)
			configReturned = config
			return config, err
		}
		c, s := localPipe(t)
		done := make(chan error)

		go func() {
			defer s.Close()
			done <- Server(s, serverConfig).Handshake()
		}()

		clientErr := Client(c, clientConfig).Handshake()
		c.Close()

		serverErr := <-done

		if len(test.errorSubstring) == 0 {
			if serverErr != nil || clientErr != nil {
				t.Errorf("test[%d]: expected no error but got serverErr: %q, clientErr: %q", i, serverErr, clientErr)
			}
			if test.verify != nil {
				if err := test.verify(configReturned); err != nil {
					t.Errorf("test[%d]: verify returned error: %v", i, err)
				}
			}
		} else {
			if serverErr == nil {
				t.Errorf("test[%d]: expected error containing %q but got no error", i, test.errorSubstring)
			} else if !strings.Contains(serverErr.Error(), test.errorSubstring) {
				t.Errorf("test[%d]: expected error to contain %q but it was %q", i, test.errorSubstring, serverErr)
			}
		}
	}
}

func TestCloseServerConnectionOnIdleClient(t *testing.T) {
	clientConn, serverConn := localPipe(t)
	server := Server(serverConn, testConfig.Clone())
	go func() {
		clientConn.Write([]byte{'0'})
		server.Close()
	}()
	server.SetReadDeadline(time.Now().Add(time.Minute))
	err := server.Handshake()
	if err != nil {
		if err, ok := err.(net.Error); ok && err.Timeout() {
			t.Errorf("Expected a closed network connection error but got '%s'", err.Error())
		}
	} else {
		t.Errorf("Error expected, but no error returned")
	}
}

func TestCloneHash(t *testing.T) {
	h1 := crypto.SHA256.New()
	h1.Write([]byte("test"))
	s1 := h1.Sum(nil)
	h2 := cloneHash(h1, crypto.SHA256)
	s2 := h2.Sum(nil)
	if !bytes.Equal(s1, s2) {
		t.Error("cloned hash generated a different sum")
	}
}

func expectError(t *testing.T, err error, sub string) {
	if err == nil {
		t.Errorf(`expected error %q, got nil`, sub)
	} else if !strings.Contains(err.Error(), sub) {
		t.Errorf(`expected error %q, got %q`, sub, err)
	}
}

func TestKeyTooSmallForRSAPSS(t *testing.T) {
	cert, err := X509KeyPair([]byte(`-----BEGIN CERTIFICATE-----
MIIBcTCCARugAwIBAgIQGjQnkCFlUqaFlt6ixyz/tDANBgkqhkiG9w0BAQsFADAS
MRAwDgYDVQQKEwdBY21lIENvMB4XDTE5MDExODIzMjMyOFoXDTIwMDExODIzMjMy
OFowEjEQMA4GA1UEChMHQWNtZSBDbzBcMA0GCSqGSIb3DQEBAQUAA0sAMEgCQQDd
ez1rFUDwax2HTxbcnFUP9AhcgEGMHVV2nn4VVEWFJB6I8C/Nkx0XyyQlrmFYBzEQ
nIPhKls4T0hFoLvjJnXpAgMBAAGjTTBLMA4GA1UdDwEB/wQEAwIFoDATBgNVHSUE
DDAKBggrBgEFBQcDATAMBgNVHRMBAf8EAjAAMBYGA1UdEQQPMA2CC2V4YW1wbGUu
Y29tMA0GCSqGSIb3DQEBCwUAA0EAxDuUS+BrrS3c+h+k+fQPOmOScy6yTX9mHw0Q
KbucGamXYEy0URIwOdO0tQ3LHPc1YGvYSPwkDjkjqECs2Vm/AA==
-----END CERTIFICATE-----`), []byte(testingKey(`-----BEGIN RSA TESTING KEY-----
MIIBOgIBAAJBAN17PWsVQPBrHYdPFtycVQ/0CFyAQYwdVXaefhVURYUkHojwL82T
HRfLJCWuYVgHMRCcg+EqWzhPSEWgu+MmdekCAwEAAQJBALjQYNTdXF4CFBbXwUz/
yt9QFDYT9B5WT/12jeGAe653gtYS6OOi/+eAkGmzg1GlRnw6fOfn+HYNFDORST7z
4j0CIQDn2xz9hVWQEu9ee3vecNT3f60huDGTNoRhtqgweQGX0wIhAPSLj1VcRZEz
nKpbtU22+PbIMSJ+e80fmY9LIPx5N4HTAiAthGSimMR9bloz0EY3GyuUEyqoDgMd
hXxjuno2WesoJQIgemilbcALXpxsLmZLgcQ2KSmaVr7jb5ECx9R+hYKTw1sCIG4s
T+E0J8wlH24pgwQHzy7Ko2qLwn1b5PW8ecrlvP1g
-----END RSA TESTING KEY-----`)))
	if err != nil {
		t.Fatal(err)
	}

	clientConn, serverConn := localPipe(t)
	client := Client(clientConn, testConfig)
	done := make(chan struct{})
	go func() {
		config := testConfig.Clone()
		config.Certificates = []Certificate{cert}
		config.MinVersion = VersionTLS13
		server := Server(serverConn, config)
		err := server.Handshake()
		expectError(t, err, "key size too small")
		close(done)
	}()
	err = client.Handshake()
	expectError(t, err, "handshake failure")
	<-done
}

func TestMultipleCertificates(t *testing.T) {
	clientConfig := testConfig.Clone()
	clientConfig.CipherSuites = []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256}
	clientConfig.MaxVersion = VersionTLS12

	serverConfig := testConfig.Clone()
	serverConfig.Certificates = []Certificate{{
		Certificate: [][]byte{testECDSACertificate},
		PrivateKey:  testECDSAPrivateKey,
	}, {
		Certificate: [][]byte{testRSACertificate},
		PrivateKey:  testRSAPrivateKey,
	}}

	_, clientState, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatal(err)
	}
	if got := clientState.PeerCertificates[0].PublicKeyAlgorithm; got != x509.RSA {
		t.Errorf("expected RSA certificate, got %v", got)
	}
}

func TestAESCipherReordering(t *testing.T) {
	skipFIPS(t) // No CHACHA20_POLY1305 for FIPS.

	currentAESSupport := hasAESGCMHardwareSupport
	defer func() { hasAESGCMHardwareSupport = currentAESSupport }()

	tests := []struct {
		name            string
		clientCiphers   []uint16
		serverHasAESGCM bool
		serverCiphers   []uint16
		expectedCipher  uint16
	}{
		{
			name: "server has hardware AES, client doesn't (pick ChaCha)",
			clientCiphers: []uint16{
				TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
				TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				TLS_RSA_WITH_AES_128_CBC_SHA,
			},
			serverHasAESGCM: true,
			expectedCipher:  TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
		},
		{
			name: "client prefers AES-GCM, server doesn't have hardware AES (pick ChaCha)",
			clientCiphers: []uint16{
				TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
				TLS_RSA_WITH_AES_128_CBC_SHA,
			},
			serverHasAESGCM: false,
			expectedCipher:  TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
		},
		{
			name: "client prefers AES-GCM, server has hardware AES (pick AES-GCM)",
			clientCiphers: []uint16{
				TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
				TLS_RSA_WITH_AES_128_CBC_SHA,
			},
			serverHasAESGCM: true,
			expectedCipher:  TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		},
		{
			name: "client prefers AES-GCM and sends GREASE, server has hardware AES (pick AES-GCM)",
			clientCiphers: []uint16{
				0x0A0A, // GREASE value
				TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
				TLS_RSA_WITH_AES_128_CBC_SHA,
			},
			serverHasAESGCM: true,
			expectedCipher:  TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		},
		{
			name: "client prefers AES-GCM and doesn't support ChaCha, server doesn't have hardware AES (pick AES-GCM)",
			clientCiphers: []uint16{
				TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
				TLS_RSA_WITH_AES_128_CBC_SHA,
			},
			serverHasAESGCM: false,
			expectedCipher:  TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		},
		{
			name: "client prefers AES-GCM and AES-CBC over ChaCha, server doesn't have hardware AES (pick ChaCha)",
			clientCiphers: []uint16{
				TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				TLS_RSA_WITH_AES_128_CBC_SHA,
				TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
			},
			serverHasAESGCM: false,
			expectedCipher:  TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
		},
		{
			name: "client prefers AES-GCM over ChaCha and sends GREASE, server doesn't have hardware AES (pick ChaCha)",
			clientCiphers: []uint16{
				0x0A0A, // GREASE value
				TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
				TLS_RSA_WITH_AES_128_CBC_SHA,
			},
			serverHasAESGCM: false,
			expectedCipher:  TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
		},
		{
			name: "client supports multiple AES-GCM, server doesn't have hardware AES and doesn't support ChaCha (AES-GCM)",
			clientCiphers: []uint16{
				TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
				TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
				TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			},
			serverHasAESGCM: false,
			serverCiphers: []uint16{
				TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
				TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			},
			expectedCipher: TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		},
		{
			name: "client prefers AES-GCM, server has hardware but doesn't support AES (pick ChaCha)",
			clientCiphers: []uint16{
				TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
				TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
				TLS_RSA_WITH_AES_128_CBC_SHA,
			},
			serverHasAESGCM: true,
			serverCiphers: []uint16{
				TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
			},
			expectedCipher: TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hasAESGCMHardwareSupport = tc.serverHasAESGCM
			hs := &serverHandshakeState{
				c: &Conn{
					config: &Config{
						CipherSuites: tc.serverCiphers,
					},
					vers: VersionTLS12,
				},
				clientHello: &clientHelloMsg{
					cipherSuites: tc.clientCiphers,
					vers:         VersionTLS12,
				},
				ecdheOk:      true,
				rsaSignOk:    true,
				rsaDecryptOk: true,
			}

			err := hs.pickCipherSuite()
			if err != nil {
				t.Errorf("pickCipherSuite failed: %s", err)
			}

			if tc.expectedCipher != hs.suite.id {
				t.Errorf("unexpected cipher chosen: want %d, got %d", tc.expectedCipher, hs.suite.id)
			}
		})
	}
}

func TestAESCipherReorderingTLS13(t *testing.T) {
	skipFIPS(t) // No CHACHA20_POLY1305 for FIPS.

	currentAESSupport := hasAESGCMHardwareSupport
	defer func() { hasAESGCMHardwareSupport = currentAESSupport }()

	tests := []struct {
		name            string
		clientCiphers   []uint16
		serverHasAESGCM bool
		expectedCipher  uint16
	}{
		{
			name: "server has hardware AES, client doesn't (pick ChaCha)",
			clientCiphers: []uint16{
				TLS_CHACHA20_POLY1305_SHA256,
				TLS_AES_128_GCM_SHA256,
			},
			serverHasAESGCM: true,
			expectedCipher:  TLS_CHACHA20_POLY1305_SHA256,
		},
		{
			name: "neither server nor client have hardware AES (pick ChaCha)",
			clientCiphers: []uint16{
				TLS_CHACHA20_POLY1305_SHA256,
				TLS_AES_128_GCM_SHA256,
			},
			serverHasAESGCM: false,
			expectedCipher:  TLS_CHACHA20_POLY1305_SHA256,
		},
		{
			name: "client prefers AES, server doesn't have hardware (pick ChaCha)",
			clientCiphers: []uint16{
				TLS_AES_128_GCM_SHA256,
				TLS_CHACHA20_POLY1305_SHA256,
			},
			serverHasAESGCM: false,
			expectedCipher:  TLS_CHACHA20_POLY1305_SHA256,
		},
		{
			name: "client prefers AES and sends GREASE, server doesn't have hardware (pick ChaCha)",
			clientCiphers: []uint16{
				0x0A0A, // GREASE value
				TLS_AES_128_GCM_SHA256,
				TLS_CHACHA20_POLY1305_SHA256,
			},
			serverHasAESGCM: false,
			expectedCipher:  TLS_CHACHA20_POLY1305_SHA256,
		},
		{
			name: "client prefers AES, server has hardware AES (pick AES)",
			clientCiphers: []uint16{
				TLS_AES_128_GCM_SHA256,
				TLS_CHACHA20_POLY1305_SHA256,
			},
			serverHasAESGCM: true,
			expectedCipher:  TLS_AES_128_GCM_SHA256,
		},
		{
			name: "client prefers AES and sends GREASE, server has hardware AES (pick AES)",
			clientCiphers: []uint16{
				0x0A0A, // GREASE value
				TLS_AES_128_GCM_SHA256,
				TLS_CHACHA20_POLY1305_SHA256,
			},
			serverHasAESGCM: true,
			expectedCipher:  TLS_AES_128_GCM_SHA256,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hasAESGCMHardwareSupport = tc.serverHasAESGCM
			pk, _ := ecdh.X25519().GenerateKey(rand.Reader)
			hs := &serverHandshakeStateTLS13{
				c: &Conn{
					config: &Config{},
					vers:   VersionTLS13,
				},
				clientHello: &clientHelloMsg{
					cipherSuites:       tc.clientCiphers,
					supportedVersions:  []uint16{VersionTLS13},
					compressionMethods: []uint8{compressionNone},
					keyShares:          []keyShare{{group: X25519, data: pk.PublicKey().Bytes()}},
					supportedCurves:    []CurveID{X25519},
				},
			}

			err := hs.processClientHello()
			if err != nil {
				t.Errorf("pickCipherSuite failed: %s", err)
			}

			if tc.expectedCipher != hs.suite.id {
				t.Errorf("unexpected cipher chosen: want %d, got %d", tc.expectedCipher, hs.suite.id)
			}
		})
	}
}

// TestServerHandshakeContextCancellation tests that canceling
// the context given to the server side conn.HandshakeContext
// interrupts the in-progress handshake.
func TestServerHandshakeContextCancellation(t *testing.T) {
	c, s := localPipe(t)
	ctx, cancel := context.WithCancel(context.Background())
	unblockClient := make(chan struct{})
	defer close(unblockClient)
	go func() {
		cancel()
		<-unblockClient
		_ = c.Close()
	}()
	conn := Server(s, testConfig)
	// Initiates server side handshake, which will block until a client hello is read
	// unless the cancellation works.
	err := conn.HandshakeContext(ctx)
	if err == nil {
		t.Fatal("Server handshake did not error when the context was canceled")
	}
	if err != context.Canceled {
		t.Errorf("Unexpected server handshake error: %v", err)
	}
	if runtime.GOOS == "js" || runtime.GOOS == "wasip1" {
		t.Skip("conn.Close does not error as expected when called multiple times on GOOS=js or GOOS=wasip1")
	}
	err = conn.Close()
	if err == nil {
		t.Error("Server connection was not closed when the context was canceled")
	}
}

// TestHandshakeContextHierarchy tests whether the contexts
// available to GetClientCertificate and GetCertificate are
// derived from the context provided to HandshakeContext, and
// that those contexts are canceled after HandshakeContext has
// returned.
func TestHandshakeContextHierarchy(t *testing.T) {
	c, s := localPipe(t)
	clientErr := make(chan error, 1)
	clientConfig := testConfig.Clone()
	serverConfig := testConfig.Clone()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	key := struct{}{}
	ctx = context.WithValue(ctx, key, true)
	go func() {
		defer close(clientErr)
		defer c.Close()
		var innerCtx context.Context
		clientConfig.Certificates = nil
		clientConfig.GetClientCertificate = func(certificateRequest *CertificateRequestInfo) (*Certificate, error) {
			if val, ok := certificateRequest.Context().Value(key).(bool); !ok || !val {
				t.Errorf("GetClientCertificate context was not child of HandshakeContext")
			}
			innerCtx = certificateRequest.Context()
			return &Certificate{
				Certificate: [][]byte{testRSACertificate},
				PrivateKey:  testRSAPrivateKey,
			}, nil
		}
		cli := Client(c, clientConfig)
		err := cli.HandshakeContext(ctx)
		if err != nil {
			clientErr <- err
			return
		}
		select {
		case <-innerCtx.Done():
		default:
			t.Errorf("GetClientCertificate context was not canceled after HandshakeContext returned.")
		}
	}()
	var innerCtx context.Context
	serverConfig.Certificates = nil
	serverConfig.ClientAuth = RequestClientCert
	serverConfig.GetCertificate = func(clientHello *ClientHelloInfo) (*Certificate, error) {
		if val, ok := clientHello.Context().Value(key).(bool); !ok || !val {
			t.Errorf("GetClientCertificate context was not child of HandshakeContext")
		}
		innerCtx = clientHello.Context()
		return &Certificate{
			Certificate: [][]byte{testRSACertificate},
			PrivateKey:  testRSAPrivateKey,
		}, nil
	}
	conn := Server(s, serverConfig)
	err := conn.HandshakeContext(ctx)
	if err != nil {
		t.Errorf("Unexpected server handshake error: %v", err)
	}
	select {
	case <-innerCtx.Done():
	default:
		t.Errorf("GetCertificate context was not canceled after HandshakeContext returned.")
	}
	if err := <-clientErr; err != nil {
		t.Errorf("Unexpected client error: %v", err)
	}
}
