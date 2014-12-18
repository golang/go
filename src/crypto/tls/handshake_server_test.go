// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
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

func init() {
	testConfig = &Config{
		Time:               func() time.Time { return time.Unix(0, 0) },
		Rand:               zeroSource{},
		Certificates:       make([]Certificate, 2),
		InsecureSkipVerify: true,
		MinVersion:         VersionSSL30,
		MaxVersion:         VersionTLS12,
	}
	testConfig.Certificates[0].Certificate = [][]byte{testRSACertificate}
	testConfig.Certificates[0].PrivateKey = testRSAPrivateKey
	testConfig.Certificates[1].Certificate = [][]byte{testSNICertificate}
	testConfig.Certificates[1].PrivateKey = testRSAPrivateKey
	testConfig.BuildNameToCertificate()
}

func testClientHelloFailure(t *testing.T, m handshakeMessage, expectedSubStr string) {
	// Create in-memory network connection,
	// send message to server.  Should return
	// expected error.
	c, s := net.Pipe()
	go func() {
		cli := Client(c, testConfig)
		if ch, ok := m.(*clientHelloMsg); ok {
			cli.vers = ch.vers
		}
		cli.writeRecord(recordTypeHandshake, m.marshal())
		c.Close()
	}()
	err := Server(s, testConfig).Handshake()
	s.Close()
	if err == nil || !strings.Contains(err.Error(), expectedSubStr) {
		t.Errorf("Got error: %s; expected to match substring '%s'", err, expectedSubStr)
	}
}

func TestSimpleError(t *testing.T) {
	testClientHelloFailure(t, &serverHelloDoneMsg{}, "unexpected handshake message")
}

var badProtocolVersions = []uint16{0x0000, 0x0005, 0x0100, 0x0105, 0x0200, 0x0205}

func TestRejectBadProtocolVersion(t *testing.T) {
	for _, v := range badProtocolVersions {
		testClientHelloFailure(t, &clientHelloMsg{vers: v}, "unsupported, maximum protocol version")
	}
}

func TestNoSuiteOverlap(t *testing.T) {
	clientHello := &clientHelloMsg{
		vers:               0x0301,
		cipherSuites:       []uint16{0xff00},
		compressionMethods: []uint8{0},
	}
	testClientHelloFailure(t, clientHello, "no cipher suite supported by both client and server")
}

func TestNoCompressionOverlap(t *testing.T) {
	clientHello := &clientHelloMsg{
		vers:               0x0301,
		cipherSuites:       []uint16{TLS_RSA_WITH_RC4_128_SHA},
		compressionMethods: []uint8{0xff},
	}
	testClientHelloFailure(t, clientHello, "client does not support uncompressed connections")
}

func TestTLS12OnlyCipherSuites(t *testing.T) {
	// Test that a Server doesn't select a TLS 1.2-only cipher suite when
	// the client negotiates TLS 1.1.
	var zeros [32]byte

	clientHello := &clientHelloMsg{
		vers:   VersionTLS11,
		random: zeros[:],
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

	c, s := net.Pipe()
	var reply interface{}
	var clientErr error
	go func() {
		cli := Client(c, testConfig)
		cli.vers = clientHello.vers
		cli.writeRecord(recordTypeHandshake, clientHello.marshal())
		reply, clientErr = cli.readHandshake()
		c.Close()
	}()
	config := *testConfig
	config.CipherSuites = clientHello.cipherSuites
	Server(s, &config).Handshake()
	s.Close()
	if clientErr != nil {
		t.Fatal(clientErr)
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
	c, s := net.Pipe()
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
	c, s := net.Pipe()
	go c.Close()

	err := Server(s, testConfig).Handshake()
	s.Close()
	if err != io.EOF {
		t.Errorf("Got error: %s; expected: %s", err, io.EOF)
	}
}

func testHandshake(clientConfig, serverConfig *Config) (state ConnectionState, err error) {
	c, s := net.Pipe()
	done := make(chan bool)
	go func() {
		cli := Client(c, clientConfig)
		cli.Handshake()
		c.Close()
		done <- true
	}()
	server := Server(s, serverConfig)
	err = server.Handshake()
	if err == nil {
		state = server.ConnectionState()
	}
	s.Close()
	<-done
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
	state, err := testHandshake(clientConfig, serverConfig)
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
	state, err := testHandshake(clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if state.CipherSuite != TLS_RSA_WITH_AES_128_CBC_SHA {
		// By default the server should use the client's preference.
		t.Fatalf("Client's preference was not used, got %x", state.CipherSuite)
	}

	serverConfig.PreferServerCipherSuites = true
	state, err = testHandshake(clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if state.CipherSuite != TLS_RSA_WITH_RC4_128_SHA {
		t.Fatalf("Server's preference was not used, got %x", state.CipherSuite)
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
	// expectAlert, if true, indicates that a fatal alert should be returned
	// when handshaking with the server.
	expectAlert bool
	// expectHandshakeErrorIncluding, when not empty, contains a string
	// that must be a substring of the error resulting from the handshake.
	expectHandshakeErrorIncluding string
	// validate, if not nil, is a function that will be called with the
	// ConnectionState of the resulting connection. It returns false if the
	// ConnectionState is unacceptable.
	validate func(ConnectionState) error
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
		clientConn, serverConn = net.Pipe()
	}
	config := test.config
	if config == nil {
		config = testConfig
	}
	server := Server(serverConn, config)
	connStateChan := make(chan ConnectionState, 1)
	go func() {
		var err error
		if _, err = server.Write([]byte("hello, world\n")); err != nil {
			t.Logf("Error from Server.Write: %s", err)
		}
		if len(test.expectHandshakeErrorIncluding) > 0 {
			if err == nil {
				t.Errorf("Error expected, but no error returned")
			} else if s := err.Error(); !strings.Contains(s, test.expectHandshakeErrorIncluding) {
				t.Errorf("Error expected containing '%s' but got '%s'", test.expectHandshakeErrorIncluding, s)
			}
		}
		server.Close()
		serverConn.Close()
		connStateChan <- server.ConnectionState()
	}()

	if !write {
		flows, err := test.loadData()
		if err != nil {
			if !test.expectAlert {
				t.Fatalf("%s: failed to load data from %s", test.name, test.dataPath())
			}
		}
		for i, b := range flows {
			if i%2 == 0 {
				clientConn.Write(b)
				continue
			}
			bb := make([]byte, len(b))
			n, err := io.ReadFull(clientConn, bb)
			if test.expectAlert {
				if err == nil {
					t.Fatal("Expected read failure but read succeeded")
				}
			} else {
				if err != nil {
					t.Fatalf("%s #%d: %s\nRead %d, wanted %d, got %x, wanted %x\n", test.name, i+1, err, n, len(bb), bb[:n], b)
				}
				if !bytes.Equal(b, bb) {
					t.Fatalf("%s #%d: mismatch on read: got:%x want:%x", test.name, i+1, bb, b)
				}
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

func runServerTestForVersion(t *testing.T, template *serverTest, prefix, option string) {
	test := *template
	test.name = prefix + test.name
	if len(test.command) == 0 {
		test.command = defaultClientCommand
	}
	test.command = append([]string(nil), test.command...)
	test.command = append(test.command, option)
	test.run(t, *update)
}

func runServerTestSSLv3(t *testing.T, template *serverTest) {
	runServerTestForVersion(t, template, "SSLv3-", "-ssl3")
}

func runServerTestTLS10(t *testing.T, template *serverTest) {
	runServerTestForVersion(t, template, "TLSv10-", "-tls1")
}

func runServerTestTLS11(t *testing.T, template *serverTest) {
	runServerTestForVersion(t, template, "TLSv11-", "-tls1_1")
}

func runServerTestTLS12(t *testing.T, template *serverTest) {
	runServerTestForVersion(t, template, "TLSv12-", "-tls1_2")
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

func TestHandshakeServerECDHEECDSAAES(t *testing.T) {
	config := *testConfig
	config.Certificates = make([]Certificate, 1)
	config.Certificates[0].Certificate = [][]byte{testECDSACertificate}
	config.Certificates[0].PrivateKey = testECDSAPrivateKey
	config.BuildNameToCertificate()

	test := &serverTest{
		name:    "ECDHE-ECDSA-AES",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "ECDHE-ECDSA-AES256-SHA"},
		config:  &config,
	}
	runServerTestTLS10(t, test)
	runServerTestTLS12(t, test)
}

func TestHandshakeServerALPN(t *testing.T) {
	config := *testConfig
	config.NextProtos = []string{"proto1", "proto2"}

	test := &serverTest{
		name: "ALPN",
		// Note that this needs OpenSSL 1.0.2 because that is the first
		// version that supports the -alpn flag.
		command: []string{"openssl", "s_client", "-alpn", "proto2,proto1"},
		config:  &config,
		validate: func(state ConnectionState) error {
			// The server's preferences should override the client.
			if state.NegotiatedProtocol != "proto1" {
				return fmt.Errorf("Got protocol %q, wanted proto1", state.NegotiatedProtocol)
			}
			return nil
		},
	}
	runServerTestTLS12(t, test)
}

func TestHandshakeServerALPNNoMatch(t *testing.T) {
	config := *testConfig
	config.NextProtos = []string{"proto3"}

	test := &serverTest{
		name: "ALPN-NoMatch",
		// Note that this needs OpenSSL 1.0.2 because that is the first
		// version that supports the -alpn flag.
		command: []string{"openssl", "s_client", "-alpn", "proto2,proto1"},
		config:  &config,
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
	config := *testConfig

	// Replace the NameToCertificate map with a GetCertificate function
	nameToCert := config.NameToCertificate
	config.NameToCertificate = nil
	config.GetCertificate = func(clientHello *ClientHelloInfo) (*Certificate, error) {
		cert, _ := nameToCert[clientHello.ServerName]
		return cert, nil
	}
	test := &serverTest{
		name:    "SNI",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA", "-servername", "snitest.com"},
		config:  &config,
	}
	runServerTestTLS12(t, test)
}

// TestHandshakeServerSNICertForNameNotFound is similar to
// TestHandshakeServerSNICertForName, but tests to make sure that when the
// GetCertificate method doesn't return a cert, we fall back to what's in
// the NameToCertificate map.
func TestHandshakeServerSNIGetCertificateNotFound(t *testing.T) {
	config := *testConfig

	config.GetCertificate = func(clientHello *ClientHelloInfo) (*Certificate, error) {
		return nil, nil
	}
	test := &serverTest{
		name:    "SNI",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA", "-servername", "snitest.com"},
		config:  &config,
	}
	runServerTestTLS12(t, test)
}

// TestHandshakeServerSNICertForNameError tests to make sure that errors in
// GetCertificate result in a tls alert.
func TestHandshakeServerSNIGetCertificateError(t *testing.T) {
	config := *testConfig

	config.GetCertificate = func(clientHello *ClientHelloInfo) (*Certificate, error) {
		return nil, fmt.Errorf("Test error in GetCertificate")
	}
	test := &serverTest{
		name:        "SNI",
		command:     []string{"openssl", "s_client", "-no_ticket", "-cipher", "AES128-SHA", "-servername", "snitest.com"},
		config:      &config,
		expectAlert: true,
	}
	runServerTestTLS12(t, test)
}

// TestCipherSuiteCertPreferance ensures that we select an RSA ciphersuite with
// an RSA certificate and an ECDSA ciphersuite with an ECDSA certificate.
func TestCipherSuiteCertPreferenceECDSA(t *testing.T) {
	config := *testConfig
	config.CipherSuites = []uint16{TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA, TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA}
	config.PreferServerCipherSuites = true

	test := &serverTest{
		name:   "CipherSuiteCertPreferenceRSA",
		config: &config,
	}
	runServerTestTLS12(t, test)

	config = *testConfig
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
		config: &config,
	}
	runServerTestTLS12(t, test)
}

func TestResumption(t *testing.T) {
	sessionFilePath := tempFile("")
	defer os.Remove(sessionFilePath)

	test := &serverTest{
		name:    "IssueTicket",
		command: []string{"openssl", "s_client", "-cipher", "RC4-SHA", "-sess_out", sessionFilePath},
	}
	runServerTestTLS12(t, test)

	test = &serverTest{
		name:    "Resume",
		command: []string{"openssl", "s_client", "-cipher", "RC4-SHA", "-sess_in", sessionFilePath},
	}
	runServerTestTLS12(t, test)
}

func TestResumptionDisabled(t *testing.T) {
	sessionFilePath := tempFile("")
	defer os.Remove(sessionFilePath)

	config := *testConfig

	test := &serverTest{
		name:    "IssueTicketPreDisable",
		command: []string{"openssl", "s_client", "-cipher", "RC4-SHA", "-sess_out", sessionFilePath},
		config:  &config,
	}
	runServerTestTLS12(t, test)

	config.SessionTicketsDisabled = true

	test = &serverTest{
		name:    "ResumeDisabled",
		command: []string{"openssl", "s_client", "-cipher", "RC4-SHA", "-sess_in", sessionFilePath},
		config:  &config,
	}
	runServerTestTLS12(t, test)

	// One needs to manually confirm that the handshake in the golden data
	// file for ResumeDisabled does not include a resumption handshake.
}

func TestFallbackSCSV(t *testing.T) {
	serverConfig := &Config{
		Certificates: testConfig.Certificates,
	}
	test := &serverTest{
		name:   "FallbackSCSV",
		config: serverConfig,
		// OpenSSL 1.0.1j is needed for the -fallback_scsv option.
		command: []string{"openssl", "s_client", "-fallback_scsv"},
		expectHandshakeErrorIncluding: "inppropriate protocol fallback",
	}
	runServerTestTLS11(t, test)
}

// cert.pem and key.pem were generated with generate_cert.go
// Thus, they have no ExtKeyUsage fields and trigger an error
// when verification is turned on.

const clientCertificatePEM = `
-----BEGIN CERTIFICATE-----
MIIB7TCCAVigAwIBAgIBADALBgkqhkiG9w0BAQUwJjEQMA4GA1UEChMHQWNtZSBD
bzESMBAGA1UEAxMJMTI3LjAuMC4xMB4XDTExMTIwODA3NTUxMloXDTEyMTIwNzA4
MDAxMlowJjEQMA4GA1UEChMHQWNtZSBDbzESMBAGA1UEAxMJMTI3LjAuMC4xMIGc
MAsGCSqGSIb3DQEBAQOBjAAwgYgCgYBO0Hsx44Jk2VnAwoekXh6LczPHY1PfZpIG
hPZk1Y/kNqcdK+izIDZFI7Xjla7t4PUgnI2V339aEu+H5Fto5OkOdOwEin/ekyfE
ARl6vfLcPRSr0FTKIQzQTW6HLlzF0rtNS0/Otiz3fojsfNcCkXSmHgwa2uNKWi7e
E5xMQIhZkwIDAQABozIwMDAOBgNVHQ8BAf8EBAMCAKAwDQYDVR0OBAYEBAECAwQw
DwYDVR0jBAgwBoAEAQIDBDALBgkqhkiG9w0BAQUDgYEANh+zegx1yW43RmEr1b3A
p0vMRpqBWHyFeSnIyMZn3TJWRSt1tukkqVCavh9a+hoV2cxVlXIWg7nCto/9iIw4
hB2rXZIxE0/9gzvGnfERYraL7KtnvshksBFQRlgXa5kc0x38BvEO5ZaoDPl4ILdE
GFGNEH5PlGffo05wc46QkYU=
-----END CERTIFICATE-----`

const clientKeyPEM = `
-----BEGIN RSA PRIVATE KEY-----
MIICWgIBAAKBgE7QezHjgmTZWcDCh6ReHotzM8djU99mkgaE9mTVj+Q2px0r6LMg
NkUjteOVru3g9SCcjZXff1oS74fkW2jk6Q507ASKf96TJ8QBGXq98tw9FKvQVMoh
DNBNbocuXMXSu01LT862LPd+iOx81wKRdKYeDBra40paLt4TnExAiFmTAgMBAAEC
gYBxvXd8yNteFTns8A/2yomEMC4yeosJJSpp1CsN3BJ7g8/qTnrVPxBy+RU+qr63
t2WquaOu/cr5P8iEsa6lk20tf8pjKLNXeX0b1RTzK8rJLbS7nGzP3tvOhL096VtQ
dAo4ROEaro0TzYpHmpciSvxVIeEIAAdFDObDJPKqcJAxyQJBAJizfYgK8Gzx9fsx
hxp+VteCbVPg2euASH5Yv3K5LukRdKoSzHE2grUVQgN/LafC0eZibRanxHegYSr7
7qaswKUCQQCEIWor/X4XTMdVj3Oj+vpiw75y/S9gh682+myZL+d/02IEkwnB098P
RkKVpenBHyrGg0oeN5La7URILWKj7CPXAkBKo6F+d+phNjwIFoN1Xb/RA32w/D1I
saG9sF+UEhRt9AxUfW/U/tIQ9V0ZHHcSg1XaCM5Nvp934brdKdvTOKnJAkBD5h/3
Rybatlvg/fzBEaJFyq09zhngkxlZOUtBVTqzl17RVvY2orgH02U4HbCHy4phxOn7
qTdQRYlHRftgnWK1AkANibn9PRYJ7mJyJ9Dyj2QeNcSkSTzrt0tPvUMf4+meJymN
1Ntu5+S1DLLzfxlaljWG6ylW6DNxujCyuXIV2rvA
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
	}

	config := *testConfig
	config.ClientAuth = RequestClientCert

	test := &serverTest{
		name:    "ClientAuthRequestedNotGiven",
		command: []string{"openssl", "s_client", "-no_ticket", "-cipher", "RC4-SHA"},
		config:  &config,
	}
	runServerTestTLS12(t, test)

	test = &serverTest{
		name:              "ClientAuthRequestedAndGiven",
		command:           []string{"openssl", "s_client", "-no_ticket", "-cipher", "RC4-SHA", "-cert", certPath, "-key", keyPath},
		config:            &config,
		expectedPeerCerts: []string{clientCertificatePEM},
	}
	runServerTestTLS12(t, test)

	test = &serverTest{
		name:              "ClientAuthRequestedAndECDSAGiven",
		command:           []string{"openssl", "s_client", "-no_ticket", "-cipher", "RC4-SHA", "-cert", ecdsaCertPath, "-key", ecdsaKeyPath},
		config:            &config,
		expectedPeerCerts: []string{clientECDSACertificatePEM},
	}
	runServerTestTLS12(t, test)
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

var testRSACertificate = fromHex("308202b030820219a00302010202090085b0bba48a7fb8ca300d06092a864886f70d01010505003045310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c7464301e170d3130303432343039303933385a170d3131303432343039303933385a3045310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c746430819f300d06092a864886f70d010101050003818d0030818902818100bb79d6f517b5e5bf4610d0dc69bee62b07435ad0032d8a7a4385b71452e7a5654c2c78b8238cb5b482e5de1f953b7e62a52ca533d6fe125c7a56fcf506bffa587b263fb5cd04d3d0c921964ac7f4549f5abfef427100fe1899077f7e887d7df10439c4a22edb51c97ce3c04c3b326601cfafb11db8719a1ddbdb896baeda2d790203010001a381a73081a4301d0603551d0e04160414b1ade2855acfcb28db69ce2369ded3268e18883930750603551d23046e306c8014b1ade2855acfcb28db69ce2369ded3268e188839a149a4473045310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c746482090085b0bba48a7fb8ca300c0603551d13040530030101ff300d06092a864886f70d010105050003818100086c4524c76bb159ab0c52ccf2b014d7879d7a6475b55a9566e4c52b8eae12661feb4f38b36e60d392fdf74108b52513b1187a24fb301dbaed98b917ece7d73159db95d31d78ea50565cd5825a2d5a5f33c4b6d8c97590968c0f5298b5cd981f89205ff2a01ca31b9694dda9fd57e970e8266d71999b266e3850296c90a7bdd9")

var testECDSACertificate = fromHex("3082020030820162020900b8bf2d47a0d2ebf4300906072a8648ce3d04013045310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c7464301e170d3132313132323135303633325a170d3232313132303135303633325a3045310b3009060355040613024155311330110603550408130a536f6d652d53746174653121301f060355040a1318496e7465726e6574205769646769747320507479204c746430819b301006072a8648ce3d020106052b81040023038186000400c4a1edbe98f90b4873367ec316561122f23d53c33b4d213dcd6b75e6f6b0dc9adf26c1bcb287f072327cb3642f1c90bcea6823107efee325c0483a69e0286dd33700ef0462dd0da09c706283d881d36431aa9e9731bd96b068c09b23de76643f1a5c7fe9120e5858b65f70dd9bd8ead5d7f5d5ccb9b69f30665b669a20e227e5bffe3b300906072a8648ce3d040103818c0030818802420188a24febe245c5487d1bacf5ed989dae4770c05e1bb62fbdf1b64db76140d311a2ceee0b7e927eff769dc33b7ea53fcefa10e259ec472d7cacda4e970e15a06fd00242014dfcbe67139c2d050ebd3fa38c25c13313830d9406bbd4377af6ec7ac9862eddd711697f857c56defb31782be4c7780daecbbe9e4e3624317b6a0f399512078f2a")

var testSNICertificate = fromHex("308201f23082015da003020102020100300b06092a864886f70d01010530283110300e060355040a130741636d6520436f311430120603550403130b736e69746573742e636f6d301e170d3132303431313137343033355a170d3133303431313137343533355a30283110300e060355040a130741636d6520436f311430120603550403130b736e69746573742e636f6d30819d300b06092a864886f70d01010103818d0030818902818100bb79d6f517b5e5bf4610d0dc69bee62b07435ad0032d8a7a4385b71452e7a5654c2c78b8238cb5b482e5de1f953b7e62a52ca533d6fe125c7a56fcf506bffa587b263fb5cd04d3d0c921964ac7f4549f5abfef427100fe1899077f7e887d7df10439c4a22edb51c97ce3c04c3b326601cfafb11db8719a1ddbdb896baeda2d790203010001a3323030300e0603551d0f0101ff0404030200a0300d0603551d0e0406040401020304300f0603551d2304083006800401020304300b06092a864886f70d0101050381810089c6455f1c1f5ef8eb1ab174ee2439059f5c4259bb1a8d86cdb1d056f56a717da40e95ab90f59e8deaf627c157995094db0802266eb34fc6842dea8a4b68d9c1389103ab84fb9e1f85d9b5d23ff2312c8670fbb540148245a4ebafe264d90c8a4cf4f85b0fac12ac2fc4a3154bad52462868af96c62c6525d652b6e31845bdcc")

var testRSAPrivateKey = &rsa.PrivateKey{
	PublicKey: rsa.PublicKey{
		N: bigFromString("131650079503776001033793877885499001334664249354723305978524647182322416328664556247316495448366990052837680518067798333412266673813370895702118944398081598789828837447552603077848001020611640547221687072142537202428102790818451901395596882588063427854225330436740647715202971973145151161964464812406232198521"),
		E: 65537,
	},
	D: bigFromString("29354450337804273969007277378287027274721892607543397931919078829901848876371746653677097639302788129485893852488285045793268732234230875671682624082413996177431586734171663258657462237320300610850244186316880055243099640544518318093544057213190320837094958164973959123058337475052510833916491060913053867729"),
	Primes: []*big.Int{
		bigFromString("11969277782311800166562047708379380720136961987713178380670422671426759650127150688426177829077494755200794297055316163155755835813760102405344560929062149"),
		bigFromString("10998999429884441391899182616418192492905073053684657075974935218461686523870125521822756579792315215543092255516093840728890783887287417039645833477273829"),
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
