// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/x509"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"math/big"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// zeroSource is an io.Reader that returns an unlimited number of zero bytes.
type zeroSource struct{}

func (zeroSource) Read(b []byte) (n int, err error) {
	for i := range b {
		b[i] = 0
	}

	return len(b), nil
}

var testConfig *Config

func allCipherSuites() []uint16 {
	ids := make([]uint16, len(cipherSuites))
	for i, suite := range cipherSuites {
		ids[i] = suite.id
	}

	return ids
}

func init() {
	testConfig = &Config{
		Time:               func() time.Time { return time.Unix(0, 0) },
		Rand:               zeroSource{},
		Certificates:       make([]Certificate, 2),
		InsecureSkipVerify: true,
		MinVersion:         VersionSSL30,
		MaxVersion:         VersionTLS13,
		CipherSuites:       allCipherSuites(),
	}
	testConfig.Certificates[0].Certificate = [][]byte{testRSACertificate}
	testConfig.Certificates[0].PrivateKey = testRSAPrivateKey
	testConfig.Certificates[1].Certificate = [][]byte{testSNICertificate}
	testConfig.Certificates[1].PrivateKey = testRSAPrivateKey
	testConfig.BuildNameToCertificate()
	if keyFile := os.Getenv("SSLKEYLOGFILE"); keyFile != "" {
		f, err := os.OpenFile(keyFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			panic("failed to open SSLKEYLOGFILE: " + err.Error())
		}
		testConfig.KeyLogWriter = f
	}
}

func testClientHello(t *testing.T, serverConfig *Config, m handshakeMessage) {
	testClientHelloFailure(t, serverConfig, m, "")
}

func testClientHelloFailure(t *testing.T, serverConfig *Config, m handshakeMessage, expectedSubStr string) {
	c, s := localPipe(t)
	go func() {
		cli := Client(c, testConfig)
		if ch, ok := m.(*clientHelloMsg); ok {
			cli.vers = ch.vers
		}
		cli.writeRecord(recordTypeHandshake, m.marshal())
		c.Close()
	}()
	conn := Server(s, serverConfig)
	ch, err := conn.readClientHello()
	hs := serverHandshakeState{
		c:           conn,
		clientHello: ch,
	}
	if err == nil {
		err = hs.processClientHello()
	}
	if err == nil {
		err = hs.pickCipherSuite()
	}
	s.Close()
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

var badProtocolVersions = []uint16{0x0000, 0x0005, 0x0100, 0x0105, 0x0200, 0x0205}

func TestRejectBadProtocolVersion(t *testing.T) {
	for _, v := range badProtocolVersions {
		testClientHelloFailure(t, testConfig, &clientHelloMsg{
			vers:   v,
			random: make([]byte, 32),
		}, "unsupported versions")
	}
	testClientHelloFailure(t, testConfig, &clientHelloMsg{
		vers:              VersionTLS12,
		supportedVersions: badProtocolVersions,
		random:            make([]byte, 32),
	}, "unsupported versions")
}

func TestNoSuiteOverlap(t *testing.T) {
	clientHello := &clientHelloMsg{
		vers:               VersionTLS10,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{0xff00},
		compressionMethods: []uint8{compressionNone},
	}
	testClientHelloFailure(t, testConfig, clientHello, "no cipher suite supported by both client and server")
}

func TestNoCompressionOverlap(t *testing.T) {
	clientHello := &clientHelloMsg{
		vers:               VersionTLS10,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_RSA_WITH_RC4_128_SHA},
		compressionMethods: []uint8{0xff},
	}
	testClientHelloFailure(t, testConfig, clientHello, "client does not support uncompressed connections")
}

func TestNoRC4ByDefault(t *testing.T) {
	clientHello := &clientHelloMsg{
		vers:               VersionTLS10,
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
	}, "unexpected message")
}

func TestDontSelectECDSAWithRSAKey(t *testing.T) {
	// Test that, even when both sides support an ECDSA cipher suite, it
	// won't be selected if the server's private key doesn't support it.
	clientHello := &clientHelloMsg{
		vers:               VersionTLS10,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA},
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
		vers:               VersionTLS10,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA},
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
	clientHello := &clientHelloMsg{
		vers:                         VersionTLS12,
		compressionMethods:           []uint8{compressionNone},
		random:                       make([]byte, 32),
		secureRenegotiationSupported: true,
		cipherSuites:                 []uint16{TLS_RSA_WITH_RC4_128_SHA},
	}

	bufChan := make(chan []byte)
	c, s := localPipe(t)

	go func() {
		cli := Client(c, testConfig)
		cli.vers = clientHello.vers
		cli.writeRecord(recordTypeHandshake, clientHello.marshal())

		buf := make([]byte, 1024)
		n, err := c.Read(buf)
		if err != nil {
			t.Errorf("Server read returned error: %s", err)
			return
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
	replyChan := make(chan interface{})
	go func() {
		cli := Client(c, testConfig)
		cli.vers = clientHello.vers
		cli.writeRecord(recordTypeHandshake, clientHello.marshal())
		reply, err := cli.readHandshake()
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

func TestAlertForwarding(t *testing.T) {
	c, s := localPipe(t)
	go func() {
		Client(c, testConfig).sendAlert(alertUnknownCA)
		c.Close()
	}()

	err := Server(s, testConfig).Handshake()
	s.Close()
	if e, ok := err.(*net.OpError); !ok || e.Err != error(alertUnknownCA) {
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

func testHandshake(t *testing.T, clientConfig, serverConfig *Config) (serverState, clientState ConnectionState, err error) {
	c, s := localPipe(t)
	errChan := make(chan error)
	go func() {
		cli := Client(c, clientConfig)
		err := cli.Handshake()
		if err != nil {
			errChan <- fmt.Errorf("client: %v", err)
			c.Close()
			return
		}
		defer cli.Close()
		clientState = cli.ConnectionState()
		buf, err := ioutil.ReadAll(cli)
		if err != nil {
			t.Errorf("failed to call cli.Read: %v", err)
		}
		if got := string(buf); got != opensslSentinel {
			t.Errorf("read %q from TLS connection, but expected %q", got, opensslSentinel)
		}
		errChan <- nil
	}()
	server := Server(s, serverConfig)
	err = server.Handshake()
	if err == nil {
		serverState = server.ConnectionState()
		if _, err := io.WriteString(server, opensslSentinel); err != nil {
			t.Errorf("failed to call server.Write: %v", err)
		}
		if err := server.Close(); err != nil {
			t.Errorf("failed to call server.Close: %v", err)
		}
		err = <-errChan
	} else {
		s.Close()
		<-errChan
	}
	return
}

func TestVersion(t *testing.T) {
	serverConfig := &Config{
		Certificates: testConfig.Certificates,
		MaxVersion:   VersionTLS11,
	}
	clientConfig := &Config{
		InsecureSkipVerify: true,
	}
	state, _, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if state.Version != VersionTLS11 {
		t.Fatalf("Incorrect version %x, should be %x", state.Version, VersionTLS11)
	}
}

func TestCipherSuitePreference(t *testing.T) {
	serverConfig := &Config{
		CipherSuites: []uint16{TLS_RSA_WITH_RC4_128_SHA, TLS_RSA_WITH_AES_128_CBC_SHA, TLS_ECDHE_RSA_WITH_RC4_128_SHA},
		Certificates: testConfig.Certificates,
		MaxVersion:   VersionTLS11,
	}
	clientConfig := &Config{
		CipherSuites:       []uint16{TLS_RSA_WITH_AES_128_CBC_SHA, TLS_RSA_WITH_RC4_128_SHA},
		InsecureSkipVerify: true,
	}
	state, _, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if state.CipherSuite != TLS_RSA_WITH_AES_128_CBC_SHA {
		// By default the server should use the client's preference.
		t.Fatalf("Client's preference was not used, got %x", state.CipherSuite)
	}

	serverConfig.PreferServerCipherSuites = true
	state, _, err = testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if state.CipherSuite != TLS_RSA_WITH_RC4_128_SHA {
		t.Fatalf("Server's preference was not used, got %x", state.CipherSuite)
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
		CipherSuites: []uint16{TLS_RSA_WITH_AES_128_CBC_SHA},
		Certificates: testConfig.Certificates,
	}
	clientConfig := &Config{
		CipherSuites:       []uint16{TLS_RSA_WITH_AES_128_CBC_SHA},
		InsecureSkipVerify: true,
		ClientSessionCache: NewLRUClientSessionCache(1),
		ServerName:         "servername",
	}

	// Establish a session at TLS 1.1.
	clientConfig.MaxVersion = VersionTLS11
	_, _, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}

	// The client session cache now contains a TLS 1.1 session.
	state, _, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if !state.DidResume {
		t.Fatalf("handshake did not resume at the same version")
	}

	// Test that the server will decline to resume at a lower version.
	clientConfig.MaxVersion = VersionTLS10
	state, _, err = testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if state.DidResume {
		t.Fatalf("handshake resumed at a lower version")
	}

	// The client session cache now contains a TLS 1.0 session.
	state, _, err = testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if !state.DidResume {
		t.Fatalf("handshake did not resume at the same version")
	}

	// Test that the server will decline to resume at a higher version.
	clientConfig.MaxVersion = VersionTLS11
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

	connChan := make(chan interface{})
	go func() {
		tcpConn, err := l.Accept()
		if err != nil {
			connChan <- err
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
		output.WriteTo(os.Stdout)
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
	checkOpenSSLVersion(t)

	var clientConn, serverConn net.Conn
	var recordingConn *recordingConn
	var childProcess *exec.Cmd

	if write {
		var err error
		recordingConn, childProcess, err = test.connFromCommand()
		if err != nil {
			t.Fatalf("Failed to start subcommand: %s", err)
		}
		serverConn = recordingConn
	} else {
		clientConn, serverConn = localPipe(t)
	}
	config := test.config
	if config == nil {
		config = testConfig
	}
	server := Server(serverConn, config)
	connStateChan := make(chan ConnectionState, 1)
	go func() {
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
		serverConn.Close()
		connStateChan <- server.ConnectionState()
	}()

	if !write {
		flows, err := test.loadData()
		if err != nil {
			t.Fatalf("%s: failed to load data from %s", test.name, test.dataPath())
		}
		for i, b := range flows {
			if i%2 == 0 {
				clientConn.SetWriteDeadline(time.Now().Add(1 * time.Minute))
				clientConn.Write(b)
				continue
			}
			bb := make([]byte, len(b))
			clientConn.SetReadDeadline(time.Now().Add(1 * time.Minute))
			n, err := io.ReadFull(clientConn, bb)
			if err != nil {
				t.Fatalf("%s #%d: %s\nRead %d, wanted %d, got %x, wanted %x\n", test.name, i+1, err, n, len(bb), bb[:n], b)
			}
			if !bytes.Equal(b, bb) {
				t.Fatalf("%s #%d: mismatch on read: got:%x want:%x", test.name, i+1, bb, b)
			}
		}
		clientConn.Close()
	}

	connState := <-connStateChan
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

	if test.validate != nil {
		if err := test.validate(connState); err != nil {
			t.Fatalf("validate callback returned error: %s", err)
		}
	}

	if write {
		path := test.dataPath()
		out, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			t.Fatalf("Failed to create output file: %s", err)
		}
		defer out.Close()
		recordingConn.Close()
		if len(recordingConn.flows) < 3 {
			childProcess.Stdout.(*bytes.Buffer).WriteTo(os.Stdout)
			if len(test.expectHandshakeErrorIncluding) == 0 {
				t.Fatalf("Handshake failed")
			}
		}
		recordingConn.WriteTo(out)
		fmt.Printf("Wrote %s\n", path)
		childProcess.Wait()
	}
}

func runServerTestForVersion(t *testing.T, template *serverTest, version, option string) {
	t.Run(version, func(t *testing.T) {
		// Make a deep copy of the template before going parallel.
		test := *template
		if template.config != nil {
			test.config = template.config.Clone()
		}

		if !*update && !template.wait {
			t.Parallel()
		}

		test.name = version + "-" + test.name
		if len(test.command) == 0 {
			test.command = defaultClientCommand
		}
		test.command = append([]string(nil), test.command...)
		test.command = append(test.command, option)
		test.run(t, *update)
	})
}

func runServerTestSSLv3(t *testing.T, template *serverTest) {
	runServerTestForVersion(t, template, "SSLv3", "-ssl3")
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
	runServerTestSSLv3(t, test)
	runServerTestTLS10(t, test)
	runServerTestTLS11(t, test)
	runServerTestTLS12(t, test)
}

func TestHandshakeServerRSA3DES(t *testing.T) {
	test := &serverTest{
		name:    "RSA-3DES",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "DES-CBC3-SHA"},
	}
	runServerTestSSLv3(t, test)
	runServerTestTLS10(t, test)
	runServerTestTLS12(t, test)
}

func TestHandshakeServerRSAAES(t *testing.T) {
	test := &serverTest{
		name:    "RSA-AES",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA"},
	}
	runServerTestSSLv3(t, test)
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
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-RSA-AES128-GCM-SHA256", "-curves", "X25519"},
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
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-RSA-AES128-GCM-SHA256", "-curves", "P-256"},
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
		command: []string{"openssl", "s_client", "-no_ticket", "-curves", "X25519:P-256"},
		config:  config,
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
		command: []string{"openssl", "s_client", "-alpn", "proto2,proto1"},
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
		command: []string{"openssl", "s_client", "-alpn", "proto2,proto1"},
		config:  config,
		validate: func(state ConnectionState) error {
			// Rather than reject the connection, Go doesn't select
			// a protocol when there is no overlap.
			if state.NegotiatedProtocol != "" {
				return fmt.Errorf("Got protocol %q, wanted ''", state.NegotiatedProtocol)
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

// TestHandshakeServerSNICertForName is similar to TestHandshakeServerSNI, but
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

// TestHandshakeServerSNICertForNameNotFound is similar to
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

// TestHandshakeServerSNICertForNameError tests to make sure that errors in
// GetCertificate result in a tls alert.
func TestHandshakeServerSNIGetCertificateError(t *testing.T) {
	const errMsg = "TestHandshakeServerSNIGetCertificateError error"

	serverConfig := testConfig.Clone()
	serverConfig.GetCertificate = func(clientHello *ClientHelloInfo) (*Certificate, error) {
		return nil, errors.New(errMsg)
	}

	clientHello := &clientHelloMsg{
		vers:               VersionTLS10,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_RSA_WITH_RC4_128_SHA},
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
		vers:               VersionTLS10,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_RSA_WITH_RC4_128_SHA},
		compressionMethods: []uint8{compressionNone},
	}
	testClientHelloFailure(t, serverConfig, clientHello, errMsg)

	// With an empty Certificates and a nil GetCertificate, the server
	// should always return a “no certificates” error.
	serverConfig.GetCertificate = nil

	clientHello = &clientHelloMsg{
		vers:               VersionTLS10,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_RSA_WITH_RC4_128_SHA},
		compressionMethods: []uint8{compressionNone},
	}
	testClientHelloFailure(t, serverConfig, clientHello, "no certificates")
}

// TestCipherSuiteCertPreferance ensures that we select an RSA ciphersuite with
// an RSA certificate and an ECDSA ciphersuite with an ECDSA certificate.
func TestCipherSuiteCertPreferenceECDSA(t *testing.T) {
	config := testConfig.Clone()
	config.CipherSuites = []uint16{TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA, TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA}
	config.PreferServerCipherSuites = true

	test := &serverTest{
		name:   "CipherSuiteCertPreferenceRSA",
		config: config,
	}
	runServerTestTLS12(t, test)

	config = testConfig.Clone()
	config.CipherSuites = []uint16{TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA, TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA}
	config.Certificates = []Certificate{
		{
			Certificate: [][]byte{testECDSACertificate},
			PrivateKey:  testECDSAPrivateKey,
		},
	}
	config.BuildNameToCertificate()
	config.PreferServerCipherSuites = true

	test = &serverTest{
		name:   "CipherSuiteCertPreferenceECDSA",
		config: config,
	}
	runServerTestTLS12(t, test)
}

func TestServerResumption(t *testing.T) {
	sessionFilePath := tempFile("")
	defer os.Remove(sessionFilePath)

	testIssue := &serverTest{
		name:    "IssueTicket",
		command: []string{"openssl", "s_client", "-cipher", "AES128-SHA", "-sess_out", sessionFilePath},
		wait:    true,
	}
	testResume := &serverTest{
		name:    "Resume",
		command: []string{"openssl", "s_client", "-cipher", "AES128-SHA", "-sess_in", sessionFilePath},
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
		name:    "Resume-HelloRetryRequest",
		command: []string{"openssl", "s_client", "-curves", "X25519:P-256", "-sess_in", sessionFilePath},
		config:  config,
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
		command: []string{"openssl", "s_client", "-cipher", "AES128-SHA", "-sess_out", sessionFilePath},
		config:  config,
		wait:    true,
	}
	testResume := &serverTest{
		name:    "ResumeDisabled",
		command: []string{"openssl", "s_client", "-cipher", "AES128-SHA", "-sess_in", sessionFilePath},
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
		command: []string{"openssl", "s_client"},
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
		command: []string{"openssl", "s_client", "-no_ticket", "-sigalgs", "rsa_pkcs1_sha256"},
	}
	runServerTestTLS12(t, test)
}

func TestHandshakeServerRSAPSS(t *testing.T) {
	test := &serverTest{
		name:    "RSA-RSAPSS",
		command: []string{"openssl", "s_client", "-no_ticket", "-sigalgs", "rsa_pss_rsae_sha256"},
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

	feeder := make(chan struct{})
	clientConn, serverConn = localPipe(b)

	go func() {
		for range feeder {
			for i, f := range flows {
				if i%2 == 0 {
					clientConn.Write(f)
					continue
				}
				ff := make([]byte, len(f))
				n, err := io.ReadFull(clientConn, ff)
				if err != nil {
					b.Errorf("#%d: %s\nRead %d, wanted %d, got %x, wanted %x\n", i+1, err, n, len(ff), ff[:n], f)
				}
				if !bytes.Equal(f, ff) {
					b.Errorf("#%d: mismatch on read: got:%x want:%x", i+1, ff, f)
				}
			}
		}
	}()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		feeder <- struct{}{}
		server := Server(serverConn, config)
		if err := server.Handshake(); err != nil {
			b.Fatalf("handshake failed: %v", err)
		}
	}
	close(feeder)
}

func BenchmarkHandshakeServer(b *testing.B) {
	b.Run("RSA", func(b *testing.B) {
		benchmarkHandshakeServer(b, VersionTLS12, TLS_RSA_WITH_AES_128_GCM_SHA256,
			0, testRSACertificate, testRSAPrivateKey)
	})
	b.Run("ECDHE-P256-RSA", func(b *testing.B) {
		b.Run("TLSv13", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS13, TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
				CurveP256, testRSACertificate, testRSAPrivateKey)
		})
		b.Run("TLSv12", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS12, TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
				CurveP256, testRSACertificate, testRSAPrivateKey)
		})
	})
	b.Run("ECDHE-P256-ECDSA-P256", func(b *testing.B) {
		b.Run("TLSv13", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS13, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
				CurveP256, testP256Certificate, testP256PrivateKey)
		})
		b.Run("TLSv12", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS12, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
				CurveP256, testP256Certificate, testP256PrivateKey)
		})
	})
	b.Run("ECDHE-X25519-ECDSA-P256", func(b *testing.B) {
		b.Run("TLSv13", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS13, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
				X25519, testP256Certificate, testP256PrivateKey)
		})
		b.Run("TLSv12", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS12, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
				X25519, testP256Certificate, testP256PrivateKey)
		})
	})
	b.Run("ECDHE-P521-ECDSA-P521", func(b *testing.B) {
		if testECDSAPrivateKey.PublicKey.Curve != elliptic.P521() {
			b.Fatal("test ECDSA key doesn't use curve P-521")
		}
		b.Run("TLSv13", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS13, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
				CurveP521, testECDSACertificate, testECDSAPrivateKey)
		})
		b.Run("TLSv12", func(b *testing.B) {
			benchmarkHandshakeServer(b, VersionTLS12, TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
				CurveP521, testECDSACertificate, testECDSAPrivateKey)
		})
	})
}

const clientCertificatePEM = `
-----BEGIN CERTIFICATE-----
MIIB7zCCAVigAwIBAgIQXBnBiWWDVW/cC8m5k5/pvDANBgkqhkiG9w0BAQsFADAS
MRAwDgYDVQQKEwdBY21lIENvMB4XDTE2MDgxNzIxNTIzMVoXDTE3MDgxNzIxNTIz
MVowEjEQMA4GA1UEChMHQWNtZSBDbzCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkC
gYEAum+qhr3Pv5/y71yUYHhv6BPy0ZZvzdkybiI3zkH5yl0prOEn2mGi7oHLEMff
NFiVhuk9GeZcJ3NgyI14AvQdpJgJoxlwaTwlYmYqqyIjxXuFOE8uCXMyp70+m63K
hAfmDzr/d8WdQYUAirab7rCkPy1MTOZCPrtRyN1IVPQMjkcCAwEAAaNGMEQwDgYD
VR0PAQH/BAQDAgWgMBMGA1UdJQQMMAoGCCsGAQUFBwMBMAwGA1UdEwEB/wQCMAAw
DwYDVR0RBAgwBocEfwAAATANBgkqhkiG9w0BAQsFAAOBgQBGq0Si+yhU+Fpn+GKU
8ZqyGJ7ysd4dfm92lam6512oFmyc9wnTN+RLKzZ8Aa1B0jLYw9KT+RBrjpW5LBeK
o0RIvFkTgxYEiKSBXCUNmAysEbEoVr4dzWFihAm/1oDGRY2CLLTYg5vbySK3KhIR
e/oCO8HJ/+rJnahJ05XX1Q7lNQ==
-----END CERTIFICATE-----`

const clientKeyPEM = `
-----BEGIN RSA PRIVATE KEY-----
MIICXQIBAAKBgQC6b6qGvc+/n/LvXJRgeG/oE/LRlm/N2TJuIjfOQfnKXSms4Sfa
YaLugcsQx980WJWG6T0Z5lwnc2DIjXgC9B2kmAmjGXBpPCViZiqrIiPFe4U4Ty4J
czKnvT6brcqEB+YPOv93xZ1BhQCKtpvusKQ/LUxM5kI+u1HI3UhU9AyORwIDAQAB
AoGAEJZ03q4uuMb7b26WSQsOMeDsftdatT747LGgs3pNRkMJvTb/O7/qJjxoG+Mc
qeSj0TAZXp+PXXc3ikCECAc+R8rVMfWdmp903XgO/qYtmZGCorxAHEmR80SrfMXv
PJnznLQWc8U9nphQErR+tTESg7xWEzmFcPKwnZd1xg8ERYkCQQDTGtrFczlB2b/Z
9TjNMqUlMnTLIk/a/rPE2fLLmAYhK5sHnJdvDURaH2mF4nso0EGtENnTsh6LATnY
dkrxXGm9AkEA4hXHG2q3MnhgK1Z5hjv+Fnqd+8bcbII9WW4flFs15EKoMgS1w/PJ
zbsySaSy5IVS8XeShmT9+3lrleed4sy+UwJBAJOOAbxhfXP5r4+5R6ql66jES75w
jUCVJzJA5ORJrn8g64u2eGK28z/LFQbv9wXgCwfc72R468BdawFSLa/m2EECQGbZ
rWiFla26IVXV0xcD98VWJsTBZMlgPnSOqoMdM1kSEd4fUmlAYI/dFzV1XYSkOmVr
FhdZnklmpVDeu27P4c0CQQCuCOup0FlJSBpWY1TTfun/KMBkBatMz0VMA3d7FKIU
csPezl677Yjo8u1r/KzeI6zLg87Z8E6r6ZWNc9wBSZK6
-----END RSA PRIVATE KEY-----`

const clientECDSACertificatePEM = `
-----BEGIN CERTIFICATE-----
MIIB/DCCAV4CCQCaMIRsJjXZFzAJBgcqhkjOPQQBMEUxCzAJBgNVBAYTAkFVMRMw
EQYDVQQIEwpTb21lLVN0YXRlMSEwHwYDVQQKExhJbnRlcm5ldCBXaWRnaXRzIFB0
eSBMdGQwHhcNMTIxMTE0MTMyNTUzWhcNMjIxMTEyMTMyNTUzWjBBMQswCQYDVQQG
EwJBVTEMMAoGA1UECBMDTlNXMRAwDgYDVQQHEwdQeXJtb250MRIwEAYDVQQDEwlK
b2VsIFNpbmcwgZswEAYHKoZIzj0CAQYFK4EEACMDgYYABACVjJF1FMBexFe01MNv
ja5oHt1vzobhfm6ySD6B5U7ixohLZNz1MLvT/2XMW/TdtWo+PtAd3kfDdq0Z9kUs
jLzYHQFMH3CQRnZIi4+DzEpcj0B22uCJ7B0rxE4wdihBsmKo+1vx+U56jb0JuK7q
ixgnTy5w/hOWusPTQBbNZU6sER7m8TAJBgcqhkjOPQQBA4GMADCBiAJCAOAUxGBg
C3JosDJdYUoCdFzCgbkWqD8pyDbHgf9stlvZcPE4O1BIKJTLCRpS8V3ujfK58PDa
2RU6+b0DeoeiIzXsAkIBo9SKeDUcSpoj0gq+KxAxnZxfvuiRs9oa9V2jI/Umi0Vw
jWVim34BmT0Y9hCaOGGbLlfk+syxis7iI6CH8OFnUes=
-----END CERTIFICATE-----`

const clientECDSAKeyPEM = `
-----BEGIN EC PARAMETERS-----
BgUrgQQAIw==
-----END EC PARAMETERS-----
-----BEGIN EC PRIVATE KEY-----
MIHcAgEBBEIBkJN9X4IqZIguiEVKMqeBUP5xtRsEv4HJEtOpOGLELwO53SD78Ew8
k+wLWoqizS3NpQyMtrU8JFdWfj+C57UNkOugBwYFK4EEACOhgYkDgYYABACVjJF1
FMBexFe01MNvja5oHt1vzobhfm6ySD6B5U7ixohLZNz1MLvT/2XMW/TdtWo+PtAd
3kfDdq0Z9kUsjLzYHQFMH3CQRnZIi4+DzEpcj0B22uCJ7B0rxE4wdihBsmKo+1vx
+U56jb0JuK7qixgnTy5w/hOWusPTQBbNZU6sER7m8Q==
-----END EC PRIVATE KEY-----`

func TestClientAuth(t *testing.T) {
	var certPath, keyPath, ecdsaCertPath, ecdsaKeyPath string

	if *update {
		certPath = tempFile(clientCertificatePEM)
		defer os.Remove(certPath)
		keyPath = tempFile(clientKeyPEM)
		defer os.Remove(keyPath)
		ecdsaCertPath = tempFile(clientECDSACertificatePEM)
		defer os.Remove(ecdsaCertPath)
		ecdsaKeyPath = tempFile(clientECDSAKeyPEM)
		defer os.Remove(ecdsaKeyPath)
	} else {
		t.Parallel()
	}

	config := testConfig.Clone()
	config.ClientAuth = RequestClientCert

	test := &serverTest{
		name:    "ClientAuthRequestedNotGiven",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA"},
		config:  config,
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)

	test = &serverTest{
		name: "ClientAuthRequestedAndGiven",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA",
			"-cert", certPath, "-key", keyPath, "-sigalgs", "rsa_pss_rsae_sha256"},
		config:            config,
		expectedPeerCerts: []string{clientCertificatePEM},
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)

	test = &serverTest{
		name: "ClientAuthRequestedAndECDSAGiven",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA",
			"-cert", ecdsaCertPath, "-key", ecdsaKeyPath},
		config:            config,
		expectedPeerCerts: []string{clientECDSACertificatePEM},
	}
	runServerTestTLS12(t, test)
	runServerTestTLS13(t, test)

	test = &serverTest{
		name: "ClientAuthRequestedAndPKCS1v15Given",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA",
			"-cert", certPath, "-key", keyPath, "-sigalgs", "rsa_pkcs1_sha256"},
		config:            config,
		expectedPeerCerts: []string{clientCertificatePEM},
	}
	runServerTestTLS12(t, test)
}

func TestSNIGivenOnFailure(t *testing.T) {
	const expectedServerName = "test.testing"

	clientHello := &clientHelloMsg{
		vers:               VersionTLS10,
		random:             make([]byte, 32),
		cipherSuites:       []uint16{TLS_RSA_WITH_RC4_128_SHA},
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
		cli.writeRecord(recordTypeHandshake, clientHello.marshal())
		c.Close()
	}()
	conn := Server(s, serverConfig)
	ch, err := conn.readClientHello()
	hs := serverHandshakeState{
		c:           conn,
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
			// The value of SessionTicketKey should have been
			// duplicated into the per-connection Config.
			for i := range config.SessionTicketKey {
				if b := config.SessionTicketKey[i]; b != byte(i) {
					return fmt.Errorf("SessionTicketKey was not duplicated from original Config: byte %d has value %d", i, b)
				}
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
			// The session ticket keys should have been duplicated
			// into the per-connection Config.
			if l := len(config.sessionTicketKeys); l != 1 {
				return fmt.Errorf("got len(sessionTicketKeys) == %d, wanted 1", l)
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

func bigFromString(s string) *big.Int {
	ret := new(big.Int)
	ret.SetString(s, 10)
	return ret
}

func fromHex(s string) []byte {
	b, _ := hex.DecodeString(s)
	return b
}

var testRSACertificate = fromHex("3082024b308201b4a003020102020900e8f09d3fe25beaa6300d06092a864886f70d01010b0500301f310b3009060355040a1302476f3110300e06035504031307476f20526f6f74301e170d3136303130313030303030305a170d3235303130313030303030305a301a310b3009060355040a1302476f310b300906035504031302476f30819f300d06092a864886f70d010101050003818d0030818902818100db467d932e12270648bc062821ab7ec4b6a25dfe1e5245887a3647a5080d92425bc281c0be97799840fb4f6d14fd2b138bc2a52e67d8d4099ed62238b74a0b74732bc234f1d193e596d9747bf3589f6c613cc0b041d4d92b2b2423775b1c3bbd755dce2054cfa163871d1e24c4f31d1a508baab61443ed97a77562f414c852d70203010001a38193308190300e0603551d0f0101ff0404030205a0301d0603551d250416301406082b0601050507030106082b06010505070302300c0603551d130101ff0402300030190603551d0e041204109f91161f43433e49a6de6db680d79f60301b0603551d230414301280104813494d137e1631bba301d5acab6e7b30190603551d1104123010820e6578616d706c652e676f6c616e67300d06092a864886f70d01010b0500038181009d30cc402b5b50a061cbbae55358e1ed8328a9581aa938a495a1ac315a1a84663d43d32dd90bf297dfd320643892243a00bccf9c7db74020015faad3166109a276fd13c3cce10c5ceeb18782f16c04ed73bbb343778d0c1cf10fa1d8408361c94c722b9daedb4606064df4c1b33ec0d1bd42d4dbfe3d1360845c21d33be9fae7")

var testRSACertificateIssuer = fromHex("3082021930820182a003020102020900ca5e4e811a965964300d06092a864886f70d01010b0500301f310b3009060355040a1302476f3110300e06035504031307476f20526f6f74301e170d3136303130313030303030305a170d3235303130313030303030305a301f310b3009060355040a1302476f3110300e06035504031307476f20526f6f7430819f300d06092a864886f70d010101050003818d0030818902818100d667b378bb22f34143b6cd2008236abefaf2852adf3ab05e01329e2c14834f5105df3f3073f99dab5442d45ee5f8f57b0111c8cb682fbb719a86944eebfffef3406206d898b8c1b1887797c9c5006547bb8f00e694b7a063f10839f269f2c34fff7a1f4b21fbcd6bfdfb13ac792d1d11f277b5c5b48600992203059f2a8f8cc50203010001a35d305b300e0603551d0f0101ff040403020204301d0603551d250416301406082b0601050507030106082b06010505070302300f0603551d130101ff040530030101ff30190603551d0e041204104813494d137e1631bba301d5acab6e7b300d06092a864886f70d01010b050003818100c1154b4bab5266221f293766ae4138899bd4c5e36b13cee670ceeaa4cbdf4f6679017e2fe649765af545749fe4249418a56bd38a04b81e261f5ce86b8d5c65413156a50d12449554748c59a30c515bc36a59d38bddf51173e899820b282e40aa78c806526fd184fb6b4cf186ec728edffa585440d2b3225325f7ab580e87dd76")

// testRSAPSSCertificate has signatureAlgorithm rsassaPss, and subjectPublicKeyInfo
// algorithm rsaEncryption, for use with the rsa_pss_rsae_* SignatureSchemes.
// See also TestRSAPSSKeyError. testRSAPSSCertificate is self-signed.
var testRSAPSSCertificate = fromHex("308202583082018da003020102021100f29926eb87ea8a0db9fcc247347c11b0304106092a864886f70d01010a3034a00f300d06096086480165030402010500a11c301a06092a864886f70d010108300d06096086480165030402010500a20302012030123110300e060355040a130741636d6520436f301e170d3137313132333136313631305a170d3138313132333136313631305a30123110300e060355040a130741636d6520436f30819f300d06092a864886f70d010101050003818d0030818902818100db467d932e12270648bc062821ab7ec4b6a25dfe1e5245887a3647a5080d92425bc281c0be97799840fb4f6d14fd2b138bc2a52e67d8d4099ed62238b74a0b74732bc234f1d193e596d9747bf3589f6c613cc0b041d4d92b2b2423775b1c3bbd755dce2054cfa163871d1e24c4f31d1a508baab61443ed97a77562f414c852d70203010001a3463044300e0603551d0f0101ff0404030205a030130603551d25040c300a06082b06010505070301300c0603551d130101ff04023000300f0603551d110408300687047f000001304106092a864886f70d01010a3034a00f300d06096086480165030402010500a11c301a06092a864886f70d010108300d06096086480165030402010500a20302012003818100cdac4ef2ce5f8d79881042707f7cbf1b5a8a00ef19154b40151771006cd41626e5496d56da0c1a139fd84695593cb67f87765e18aa03ea067522dd78d2a589b8c92364e12838ce346c6e067b51f1a7e6f4b37ffab13f1411896679d18e880e0ba09e302ac067efca460288e9538122692297ad8093d4f7dd701424d7700a46a1")

var testECDSACertificate = fromHex("3082020030820162020900b8bf2d47a0d2ebf4300906072a8648ce3d04013045310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c7464301e170d3132313132323135303633325a170d3232313132303135303633325a3045310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c746430819b301006072a8648ce3d020106052b81040023038186000400c4a1edbe98f90b4873367ec316561122f23d53c33b4d213dcd6b75e6f6b0dc9adf26c1bcb287f072327cb3642f1c90bcea6823107efee325c0483a69e0286dd33700ef0462dd0da09c706283d881d36431aa9e9731bd96b068c09b23de76643f1a5c7fe9120e5858b65f70dd9bd8ead5d7f5d5ccb9b69f30665b669a20e227e5bffe3b300906072a8648ce3d040103818c0030818802420188a24febe245c5487d1bacf5ed989dae4770c05e1bb62fbdf1b64db76140d311a2ceee0b7e927eff769dc33b7ea53fcefa10e259ec472d7cacda4e970e15a06fd00242014dfcbe67139c2d050ebd3fa38c25c13313830d9406bbd4377af6ec7ac9862eddd711697f857c56defb31782be4c7780daecbbe9e4e3624317b6a0f399512078f2a")

var testSNICertificate = fromHex("0441883421114c81480804c430820237308201a0a003020102020900e8f09d3fe25beaa6300d06092a864886f70d01010b0500301f310b3009060355040a1302476f3110300e06035504031307476f20526f6f74301e170d3136303130313030303030305a170d3235303130313030303030305a3023310b3009060355040a1302476f311430120603550403130b736e69746573742e636f6d30819f300d06092a864886f70d010101050003818d0030818902818100db467d932e12270648bc062821ab7ec4b6a25dfe1e5245887a3647a5080d92425bc281c0be97799840fb4f6d14fd2b138bc2a52e67d8d4099ed62238b74a0b74732bc234f1d193e596d9747bf3589f6c613cc0b041d4d92b2b2423775b1c3bbd755dce2054cfa163871d1e24c4f31d1a508baab61443ed97a77562f414c852d70203010001a3773075300e0603551d0f0101ff0404030205a0301d0603551d250416301406082b0601050507030106082b06010505070302300c0603551d130101ff0402300030190603551d0e041204109f91161f43433e49a6de6db680d79f60301b0603551d230414301280104813494d137e1631bba301d5acab6e7b300d06092a864886f70d01010b0500038181007beeecff0230dbb2e7a334af65430b7116e09f327c3bbf918107fc9c66cb497493207ae9b4dbb045cb63d605ec1b5dd485bb69124d68fa298dc776699b47632fd6d73cab57042acb26f083c4087459bc5a3bb3ca4d878d7fe31016b7bc9a627438666566e3389bfaeebe6becc9a0093ceed18d0f9ac79d56f3a73f18188988ed")

var testP256Certificate = fromHex("308201693082010ea00302010202105012dc24e1124ade4f3e153326ff27bf300a06082a8648ce3d04030230123110300e060355040a130741636d6520436f301e170d3137303533313232343934375a170d3138303533313232343934375a30123110300e060355040a130741636d6520436f3059301306072a8648ce3d020106082a8648ce3d03010703420004c02c61c9b16283bbcc14956d886d79b358aa614596975f78cece787146abf74c2d5dc578c0992b4f3c631373479ebf3892efe53d21c4f4f1cc9a11c3536b7f75a3463044300e0603551d0f0101ff0404030205a030130603551d25040c300a06082b06010505070301300c0603551d130101ff04023000300f0603551d1104083006820474657374300a06082a8648ce3d0403020349003046022100963712d6226c7b2bef41512d47e1434131aaca3ba585d666c924df71ac0448b3022100f4d05c725064741aef125f243cdbccaa2a5d485927831f221c43023bd5ae471a")

var testRSAPrivateKey = &rsa.PrivateKey{
	PublicKey: rsa.PublicKey{
		N: bigFromString("153980389784927331788354528594524332344709972855165340650588877572729725338415474372475094155672066328274535240275856844648695200875763869073572078279316458648124537905600131008790701752441155668003033945258023841165089852359980273279085783159654751552359397986180318708491098942831252291841441726305535546071"),
		E: 65537,
	},
	D: bigFromString("7746362285745539358014631136245887418412633787074173796862711588221766398229333338511838891484974940633857861775630560092874987828057333663969469797013996401149696897591265769095952887917296740109742927689053276850469671231961384712725169432413343763989564437170644270643461665184965150423819594083121075825"),
	Primes: []*big.Int{
		bigFromString("13299275414352936908236095374926261633419699590839189494995965049151460173257838079863316944311313904000258169883815802963543635820059341150014695560313417"),
		bigFromString("11578103692682951732111718237224894755352163854919244905974423810539077224889290605729035287537520656160688625383765857517518932447378594964220731750802463"),
	},
}

var testECDSAPrivateKey = &ecdsa.PrivateKey{
	PublicKey: ecdsa.PublicKey{
		Curve: elliptic.P521(),
		X:     bigFromString("2636411247892461147287360222306590634450676461695221912739908880441342231985950069527906976759812296359387337367668045707086543273113073382714101597903639351"),
		Y:     bigFromString("3204695818431246682253994090650952614555094516658732116404513121125038617915183037601737180082382202488628239201196033284060130040574800684774115478859677243"),
	},
	D: bigFromString("5477294338614160138026852784385529180817726002953041720191098180813046231640184669647735805135001309477695746518160084669446643325196003346204701381388769751"),
}

var testP256PrivateKey, _ = x509.ParseECPrivateKey(fromHex("30770201010420012f3b52bc54c36ba3577ad45034e2e8efe1e6999851284cb848725cfe029991a00a06082a8648ce3d030107a14403420004c02c61c9b16283bbcc14956d886d79b358aa614596975f78cece787146abf74c2d5dc578c0992b4f3c631373479ebf3892efe53d21c4f4f1cc9a11c3536b7f75"))

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
