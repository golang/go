// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls/internal/fips140tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"internal/byteorder"
	"io"
	"math/big"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Note: see comment in handshake_test.go for details of how the reference
// tests work.

// opensslInputEvent enumerates possible inputs that can be sent to an `openssl
// s_client` process.
type opensslInputEvent int

const (
	// opensslRenegotiate causes OpenSSL to request a renegotiation of the
	// connection.
	opensslRenegotiate opensslInputEvent = iota

	// opensslSendBanner causes OpenSSL to send the contents of
	// opensslSentinel on the connection.
	opensslSendSentinel

	// opensslKeyUpdate causes OpenSSL to send a key update message to the
	// client and request one back.
	opensslKeyUpdate
)

const opensslSentinel = "SENTINEL\n"

type opensslInput chan opensslInputEvent

func (i opensslInput) Read(buf []byte) (n int, err error) {
	for event := range i {
		switch event {
		case opensslRenegotiate:
			return copy(buf, []byte("R\n")), nil
		case opensslKeyUpdate:
			return copy(buf, []byte("K\n")), nil
		case opensslSendSentinel:
			return copy(buf, []byte(opensslSentinel)), nil
		default:
			panic("unknown event")
		}
	}

	return 0, io.EOF
}

// opensslOutputSink is an io.Writer that receives the stdout and stderr from an
// `openssl` process and sends a value to handshakeComplete or readKeyUpdate
// when certain messages are seen.
type opensslOutputSink struct {
	handshakeComplete chan struct{}
	readKeyUpdate     chan struct{}
	all               []byte
	line              []byte
}

func newOpensslOutputSink() *opensslOutputSink {
	return &opensslOutputSink{make(chan struct{}), make(chan struct{}), nil, nil}
}

// opensslEndOfHandshake is a message that the “openssl s_server” tool will
// print when a handshake completes if run with “-state”.
const opensslEndOfHandshake = "SSL_accept:SSLv3/TLS write finished"

// opensslReadKeyUpdate is a message that the “openssl s_server” tool will
// print when a KeyUpdate message is received if run with “-state”.
const opensslReadKeyUpdate = "SSL_accept:TLSv1.3 read client key update"

func (o *opensslOutputSink) Write(data []byte) (n int, err error) {
	o.line = append(o.line, data...)
	o.all = append(o.all, data...)

	for {
		line, next, ok := bytes.Cut(o.line, []byte("\n"))
		if !ok {
			break
		}

		if bytes.Equal([]byte(opensslEndOfHandshake), line) {
			o.handshakeComplete <- struct{}{}
		}
		if bytes.Equal([]byte(opensslReadKeyUpdate), line) {
			o.readKeyUpdate <- struct{}{}
		}
		o.line = next
	}

	return len(data), nil
}

func (o *opensslOutputSink) String() string {
	return string(o.all)
}

// clientTest represents a test of the TLS client handshake against a reference
// implementation.
type clientTest struct {
	// name is a freeform string identifying the test and the file in which
	// the expected results will be stored.
	name string
	// args, if not empty, contains a series of arguments for the
	// command to run for the reference server.
	args []string
	// config, if not nil, contains a custom Config to use for this test.
	config *Config
	// cert, if not empty, contains a DER-encoded certificate for the
	// reference server.
	cert []byte
	// key, if not nil, contains either a *rsa.PrivateKey, ed25519.PrivateKey or
	// *ecdsa.PrivateKey which is the private key for the reference server.
	key any
	// extensions, if not nil, contains a list of extension data to be returned
	// from the ServerHello. The data should be in standard TLS format with
	// a 2-byte uint16 type, 2-byte data length, followed by the extension data.
	extensions [][]byte
	// validate, if not nil, is a function that will be called with the
	// ConnectionState of the resulting connection. It returns a non-nil
	// error if the ConnectionState is unacceptable.
	validate func(ConnectionState) error
	// numRenegotiations is the number of times that the connection will be
	// renegotiated.
	numRenegotiations int
	// renegotiationExpectedToFail, if not zero, is the number of the
	// renegotiation attempt that is expected to fail.
	renegotiationExpectedToFail int
	// checkRenegotiationError, if not nil, is called with any error
	// arising from renegotiation. It can map expected errors to nil to
	// ignore them.
	checkRenegotiationError func(renegotiationNum int, err error) error
	// sendKeyUpdate will cause the server to send a KeyUpdate message.
	sendKeyUpdate bool
}

var serverCommand = []string{"openssl", "s_server", "-no_ticket", "-num_tickets", "0"}

// connFromCommand starts the reference server process, connects to it and
// returns a recordingConn for the connection. The stdin return value is an
// opensslInput for the stdin of the child process. It must be closed before
// Waiting for child.
func (test *clientTest) connFromCommand() (conn *recordingConn, child *exec.Cmd, stdin opensslInput, stdout *opensslOutputSink, err error) {
	cert := testRSACertificate
	if len(test.cert) > 0 {
		cert = test.cert
	}
	certPath := tempFile(string(cert))
	defer os.Remove(certPath)

	var key any = testRSAPrivateKey
	if test.key != nil {
		key = test.key
	}
	derBytes, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		panic(err)
	}

	var pemOut bytes.Buffer
	pem.Encode(&pemOut, &pem.Block{Type: "PRIVATE KEY", Bytes: derBytes})

	keyPath := tempFile(pemOut.String())
	defer os.Remove(keyPath)

	var command []string
	command = append(command, serverCommand...)
	command = append(command, test.args...)
	command = append(command, "-cert", certPath, "-certform", "DER", "-key", keyPath)
	// serverPort contains the port that OpenSSL will listen on. OpenSSL
	// can't take "0" as an argument here so we have to pick a number and
	// hope that it's not in use on the machine. Since this only occurs
	// when -update is given and thus when there's a human watching the
	// test, this isn't too bad.
	const serverPort = 24323
	command = append(command, "-accept", strconv.Itoa(serverPort))

	if len(test.extensions) > 0 {
		var serverInfo bytes.Buffer
		for _, ext := range test.extensions {
			pem.Encode(&serverInfo, &pem.Block{
				Type:  fmt.Sprintf("SERVERINFO FOR EXTENSION %d", byteorder.BEUint16(ext)),
				Bytes: ext,
			})
		}
		serverInfoPath := tempFile(serverInfo.String())
		defer os.Remove(serverInfoPath)
		command = append(command, "-serverinfo", serverInfoPath)
	}

	if test.numRenegotiations > 0 || test.sendKeyUpdate {
		found := false
		for _, flag := range command[1:] {
			if flag == "-state" {
				found = true
				break
			}
		}

		if !found {
			panic("-state flag missing to OpenSSL, you need this if testing renegotiation or KeyUpdate")
		}
	}

	cmd := exec.Command(command[0], command[1:]...)
	stdin = opensslInput(make(chan opensslInputEvent))
	cmd.Stdin = stdin
	out := newOpensslOutputSink()
	cmd.Stdout = out
	cmd.Stderr = out
	if err := cmd.Start(); err != nil {
		return nil, nil, nil, nil, err
	}

	// OpenSSL does print an "ACCEPT" banner, but it does so *before*
	// opening the listening socket, so we can't use that to wait until it
	// has started listening. Thus we are forced to poll until we get a
	// connection.
	var tcpConn net.Conn
	for i := uint(0); i < 5; i++ {
		tcpConn, err = net.DialTCP("tcp", nil, &net.TCPAddr{
			IP:   net.IPv4(127, 0, 0, 1),
			Port: serverPort,
		})
		if err == nil {
			break
		}
		time.Sleep((1 << i) * 5 * time.Millisecond)
	}
	if err != nil {
		close(stdin)
		cmd.Process.Kill()
		err = fmt.Errorf("error connecting to the OpenSSL server: %v (%v)\n\n%s", err, cmd.Wait(), out)
		return nil, nil, nil, nil, err
	}

	record := &recordingConn{
		Conn: tcpConn,
	}

	return record, cmd, stdin, out, nil
}

func (test *clientTest) dataPath() string {
	return filepath.Join("testdata", "Client-"+test.name)
}

func (test *clientTest) loadData() (flows [][]byte, err error) {
	in, err := os.Open(test.dataPath())
	if err != nil {
		return nil, err
	}
	defer in.Close()
	return parseTestData(in)
}

func (test *clientTest) run(t *testing.T, write bool) {
	var clientConn net.Conn
	var recordingConn *recordingConn
	var childProcess *exec.Cmd
	var stdin opensslInput
	var stdout *opensslOutputSink

	if write {
		var err error
		recordingConn, childProcess, stdin, stdout, err = test.connFromCommand()
		if err != nil {
			t.Fatalf("Failed to start subcommand: %s", err)
		}
		clientConn = recordingConn
		defer func() {
			if t.Failed() {
				t.Logf("OpenSSL output:\n\n%s", stdout.all)
			}
		}()
	} else {
		flows, err := test.loadData()
		if err != nil {
			t.Fatalf("failed to load data from %s: %v", test.dataPath(), err)
		}
		clientConn = &replayingConn{t: t, flows: flows, reading: false}
	}

	config := test.config
	if config == nil {
		config = testConfig
	}
	client := Client(clientConn, config)
	defer client.Close()

	if _, err := client.Write([]byte("hello\n")); err != nil {
		t.Errorf("Client.Write failed: %s", err)
		return
	}

	for i := 1; i <= test.numRenegotiations; i++ {
		// The initial handshake will generate a
		// handshakeComplete signal which needs to be quashed.
		if i == 1 && write {
			<-stdout.handshakeComplete
		}

		// OpenSSL will try to interleave application data and
		// a renegotiation if we send both concurrently.
		// Therefore: ask OpensSSL to start a renegotiation, run
		// a goroutine to call client.Read and thus process the
		// renegotiation request, watch for OpenSSL's stdout to
		// indicate that the handshake is complete and,
		// finally, have OpenSSL write something to cause
		// client.Read to complete.
		if write {
			stdin <- opensslRenegotiate
		}

		signalChan := make(chan struct{})

		go func() {
			defer close(signalChan)

			buf := make([]byte, 256)
			n, err := client.Read(buf)

			if test.checkRenegotiationError != nil {
				newErr := test.checkRenegotiationError(i, err)
				if err != nil && newErr == nil {
					return
				}
				err = newErr
			}

			if err != nil {
				t.Errorf("Client.Read failed after renegotiation #%d: %s", i, err)
				return
			}

			buf = buf[:n]
			if !bytes.Equal([]byte(opensslSentinel), buf) {
				t.Errorf("Client.Read returned %q, but wanted %q", string(buf), opensslSentinel)
			}

			if expected := i + 1; client.handshakes != expected {
				t.Errorf("client should have recorded %d handshakes, but believes that %d have occurred", expected, client.handshakes)
			}
		}()

		if write && test.renegotiationExpectedToFail != i {
			<-stdout.handshakeComplete
			stdin <- opensslSendSentinel
		}
		<-signalChan
	}

	if test.sendKeyUpdate {
		if write {
			<-stdout.handshakeComplete
			stdin <- opensslKeyUpdate
		}

		doneRead := make(chan struct{})

		go func() {
			defer close(doneRead)

			buf := make([]byte, 256)
			n, err := client.Read(buf)

			if err != nil {
				t.Errorf("Client.Read failed after KeyUpdate: %s", err)
				return
			}

			buf = buf[:n]
			if !bytes.Equal([]byte(opensslSentinel), buf) {
				t.Errorf("Client.Read returned %q, but wanted %q", string(buf), opensslSentinel)
			}
		}()

		if write {
			// There's no real reason to wait for the client KeyUpdate to
			// send data with the new server keys, except that s_server
			// drops writes if they are sent at the wrong time.
			<-stdout.readKeyUpdate
			stdin <- opensslSendSentinel
		}
		<-doneRead

		if _, err := client.Write([]byte("hello again\n")); err != nil {
			t.Errorf("Client.Write failed: %s", err)
			return
		}
	}

	if test.validate != nil {
		if err := test.validate(client.ConnectionState()); err != nil {
			t.Errorf("validate callback returned error: %s", err)
		}
	}

	// If the server sent us an alert after our last flight, give it a
	// chance to arrive.
	if write && test.renegotiationExpectedToFail == 0 {
		if err := peekError(client); err != nil {
			t.Errorf("final Read returned an error: %s", err)
		}
	}

	if write {
		client.Close()
		path := test.dataPath()
		out, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
		if err != nil {
			t.Fatalf("Failed to create output file: %s", err)
		}
		defer out.Close()
		recordingConn.Close()
		close(stdin)
		childProcess.Process.Kill()
		childProcess.Wait()
		if len(recordingConn.flows) < 3 {
			t.Fatalf("Client connection didn't work")
		}
		recordingConn.WriteTo(out)
		t.Logf("Wrote %s\n", path)
	}
}

// peekError does a read with a short timeout to check if the next read would
// cause an error, for example if there is an alert waiting on the wire.
func peekError(conn net.Conn) error {
	conn.SetReadDeadline(time.Now().Add(100 * time.Millisecond))
	if n, err := conn.Read(make([]byte, 1)); n != 0 {
		return errors.New("unexpectedly read data")
	} else if err != nil {
		if netErr, ok := err.(net.Error); !ok || !netErr.Timeout() {
			return err
		}
	}
	return nil
}

func runClientTestForVersion(t *testing.T, template *clientTest, version, option string) {
	// Make a deep copy of the template before going parallel.
	test := *template
	if template.config != nil {
		test.config = template.config.Clone()
	}
	test.name = version + "-" + test.name
	test.args = append([]string{option}, test.args...)

	runTestAndUpdateIfNeeded(t, version, test.run, false)
}

func runClientTestTLS10(t *testing.T, template *clientTest) {
	runClientTestForVersion(t, template, "TLSv10", "-tls1")
}

func runClientTestTLS11(t *testing.T, template *clientTest) {
	runClientTestForVersion(t, template, "TLSv11", "-tls1_1")
}

func runClientTestTLS12(t *testing.T, template *clientTest) {
	runClientTestForVersion(t, template, "TLSv12", "-tls1_2")
}

func runClientTestTLS13(t *testing.T, template *clientTest) {
	runClientTestForVersion(t, template, "TLSv13", "-tls1_3")
}

func TestHandshakeClientRSARC4(t *testing.T) {
	test := &clientTest{
		name: "RSA-RC4",
		args: []string{"-cipher", "RC4-SHA"},
	}
	runClientTestTLS10(t, test)
	runClientTestTLS11(t, test)
	runClientTestTLS12(t, test)
}

func TestHandshakeClientRSAAES128GCM(t *testing.T) {
	test := &clientTest{
		name: "AES128-GCM-SHA256",
		args: []string{"-cipher", "AES128-GCM-SHA256"},
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientRSAAES256GCM(t *testing.T) {
	test := &clientTest{
		name: "AES256-GCM-SHA384",
		args: []string{"-cipher", "AES256-GCM-SHA384"},
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHERSAAES(t *testing.T) {
	test := &clientTest{
		name: "ECDHE-RSA-AES",
		args: []string{"-cipher", "ECDHE-RSA-AES128-SHA"},
	}
	runClientTestTLS10(t, test)
	runClientTestTLS11(t, test)
	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHEECDSAAES(t *testing.T) {
	test := &clientTest{
		name: "ECDHE-ECDSA-AES",
		args: []string{"-cipher", "ECDHE-ECDSA-AES128-SHA"},
		cert: testECDSACertificate,
		key:  testECDSAPrivateKey,
	}
	runClientTestTLS10(t, test)
	runClientTestTLS11(t, test)
	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHEECDSAAESGCM(t *testing.T) {
	test := &clientTest{
		name: "ECDHE-ECDSA-AES-GCM",
		args: []string{"-cipher", "ECDHE-ECDSA-AES128-GCM-SHA256"},
		cert: testECDSACertificate,
		key:  testECDSAPrivateKey,
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientAES256GCMSHA384(t *testing.T) {
	test := &clientTest{
		name: "ECDHE-ECDSA-AES256-GCM-SHA384",
		args: []string{"-cipher", "ECDHE-ECDSA-AES256-GCM-SHA384"},
		cert: testECDSACertificate,
		key:  testECDSAPrivateKey,
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientAES128CBCSHA256(t *testing.T) {
	test := &clientTest{
		name: "AES128-SHA256",
		args: []string{"-cipher", "AES128-SHA256"},
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHERSAAES128CBCSHA256(t *testing.T) {
	test := &clientTest{
		name: "ECDHE-RSA-AES128-SHA256",
		args: []string{"-cipher", "ECDHE-RSA-AES128-SHA256"},
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHEECDSAAES128CBCSHA256(t *testing.T) {
	test := &clientTest{
		name: "ECDHE-ECDSA-AES128-SHA256",
		args: []string{"-cipher", "ECDHE-ECDSA-AES128-SHA256"},
		cert: testECDSACertificate,
		key:  testECDSAPrivateKey,
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientX25519(t *testing.T) {
	config := testConfig.Clone()
	config.CurvePreferences = []CurveID{X25519}

	test := &clientTest{
		name:   "X25519-ECDHE",
		args:   []string{"-cipher", "ECDHE-RSA-AES128-GCM-SHA256", "-curves", "X25519"},
		config: config,
	}

	runClientTestTLS12(t, test)
	runClientTestTLS13(t, test)
}

func TestHandshakeClientP256(t *testing.T) {
	config := testConfig.Clone()
	config.CurvePreferences = []CurveID{CurveP256}

	test := &clientTest{
		name:   "P256-ECDHE",
		args:   []string{"-cipher", "ECDHE-RSA-AES128-GCM-SHA256", "-curves", "P-256"},
		config: config,
	}

	runClientTestTLS12(t, test)
	runClientTestTLS13(t, test)
}

func TestHandshakeClientHelloRetryRequest(t *testing.T) {
	config := testConfig.Clone()
	config.CurvePreferences = []CurveID{X25519, CurveP256}

	test := &clientTest{
		name:   "HelloRetryRequest",
		args:   []string{"-cipher", "ECDHE-RSA-AES128-GCM-SHA256", "-curves", "P-256"},
		config: config,
		validate: func(cs ConnectionState) error {
			if !cs.testingOnlyDidHRR {
				return errors.New("expected HelloRetryRequest")
			}
			return nil
		},
	}

	runClientTestTLS13(t, test)
}

func TestHandshakeClientECDHERSAChaCha20(t *testing.T) {
	config := testConfig.Clone()
	config.CipherSuites = []uint16{TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256}

	test := &clientTest{
		name:   "ECDHE-RSA-CHACHA20-POLY1305",
		args:   []string{"-cipher", "ECDHE-RSA-CHACHA20-POLY1305"},
		config: config,
	}

	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHEECDSAChaCha20(t *testing.T) {
	config := testConfig.Clone()
	config.CipherSuites = []uint16{TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256}

	test := &clientTest{
		name:   "ECDHE-ECDSA-CHACHA20-POLY1305",
		args:   []string{"-cipher", "ECDHE-ECDSA-CHACHA20-POLY1305"},
		config: config,
		cert:   testECDSACertificate,
		key:    testECDSAPrivateKey,
	}

	runClientTestTLS12(t, test)
}

func TestHandshakeClientAES128SHA256(t *testing.T) {
	test := &clientTest{
		name: "AES128-SHA256",
		args: []string{"-ciphersuites", "TLS_AES_128_GCM_SHA256"},
	}
	runClientTestTLS13(t, test)
}
func TestHandshakeClientAES256SHA384(t *testing.T) {
	test := &clientTest{
		name: "AES256-SHA384",
		args: []string{"-ciphersuites", "TLS_AES_256_GCM_SHA384"},
	}
	runClientTestTLS13(t, test)
}
func TestHandshakeClientCHACHA20SHA256(t *testing.T) {
	test := &clientTest{
		name: "CHACHA20-SHA256",
		args: []string{"-ciphersuites", "TLS_CHACHA20_POLY1305_SHA256"},
	}
	runClientTestTLS13(t, test)
}

func TestHandshakeClientECDSATLS13(t *testing.T) {
	test := &clientTest{
		name: "ECDSA",
		cert: testECDSACertificate,
		key:  testECDSAPrivateKey,
	}
	runClientTestTLS13(t, test)
}

func TestHandshakeClientEd25519(t *testing.T) {
	test := &clientTest{
		name: "Ed25519",
		cert: testEd25519Certificate,
		key:  testEd25519PrivateKey,
	}
	runClientTestTLS12(t, test)
	runClientTestTLS13(t, test)

	config := testConfig.Clone()
	cert, _ := X509KeyPair([]byte(clientEd25519CertificatePEM), []byte(clientEd25519KeyPEM))
	config.Certificates = []Certificate{cert}

	test = &clientTest{
		name:   "ClientCert-Ed25519",
		args:   []string{"-Verify", "1"},
		config: config,
	}

	runClientTestTLS12(t, test)
	runClientTestTLS13(t, test)
}

func TestHandshakeClientCertRSA(t *testing.T) {
	config := testConfig.Clone()
	cert, _ := X509KeyPair([]byte(clientCertificatePEM), []byte(clientKeyPEM))
	config.Certificates = []Certificate{cert}

	test := &clientTest{
		name:   "ClientCert-RSA-RSA",
		args:   []string{"-cipher", "AES128", "-Verify", "1"},
		config: config,
	}

	runClientTestTLS10(t, test)
	runClientTestTLS12(t, test)

	test = &clientTest{
		name:   "ClientCert-RSA-ECDSA",
		args:   []string{"-cipher", "ECDHE-ECDSA-AES128-SHA", "-Verify", "1"},
		config: config,
		cert:   testECDSACertificate,
		key:    testECDSAPrivateKey,
	}

	runClientTestTLS10(t, test)
	runClientTestTLS12(t, test)
	runClientTestTLS13(t, test)

	test = &clientTest{
		name:   "ClientCert-RSA-AES256-GCM-SHA384",
		args:   []string{"-cipher", "ECDHE-RSA-AES256-GCM-SHA384", "-Verify", "1"},
		config: config,
		cert:   testRSACertificate,
		key:    testRSAPrivateKey,
	}

	runClientTestTLS12(t, test)
}

func TestHandshakeClientCertECDSA(t *testing.T) {
	config := testConfig.Clone()
	cert, _ := X509KeyPair([]byte(clientECDSACertificatePEM), []byte(clientECDSAKeyPEM))
	config.Certificates = []Certificate{cert}

	test := &clientTest{
		name:   "ClientCert-ECDSA-RSA",
		args:   []string{"-cipher", "AES128", "-Verify", "1"},
		config: config,
	}

	runClientTestTLS10(t, test)
	runClientTestTLS12(t, test)
	runClientTestTLS13(t, test)

	test = &clientTest{
		name:   "ClientCert-ECDSA-ECDSA",
		args:   []string{"-cipher", "ECDHE-ECDSA-AES128-SHA", "-Verify", "1"},
		config: config,
		cert:   testECDSACertificate,
		key:    testECDSAPrivateKey,
	}

	runClientTestTLS10(t, test)
	runClientTestTLS12(t, test)
}

// TestHandshakeClientCertRSAPSS tests rsa_pss_rsae_sha256 signatures from both
// client and server certificates. It also serves from both sides a certificate
// signed itself with RSA-PSS, mostly to check that crypto/x509 chain validation
// works.
func TestHandshakeClientCertRSAPSS(t *testing.T) {
	cert, err := x509.ParseCertificate(testRSAPSSCertificate)
	if err != nil {
		panic(err)
	}
	rootCAs := x509.NewCertPool()
	rootCAs.AddCert(cert)

	config := testConfig.Clone()
	// Use GetClientCertificate to bypass the client certificate selection logic.
	config.GetClientCertificate = func(*CertificateRequestInfo) (*Certificate, error) {
		return &Certificate{
			Certificate: [][]byte{testRSAPSSCertificate},
			PrivateKey:  testRSAPrivateKey,
		}, nil
	}
	config.RootCAs = rootCAs

	test := &clientTest{
		name: "ClientCert-RSA-RSAPSS",
		args: []string{"-cipher", "AES128", "-Verify", "1", "-client_sigalgs",
			"rsa_pss_rsae_sha256", "-sigalgs", "rsa_pss_rsae_sha256"},
		config: config,
		cert:   testRSAPSSCertificate,
		key:    testRSAPrivateKey,
	}
	runClientTestTLS12(t, test)
	runClientTestTLS13(t, test)
}

func TestHandshakeClientCertRSAPKCS1v15(t *testing.T) {
	config := testConfig.Clone()
	cert, _ := X509KeyPair([]byte(clientCertificatePEM), []byte(clientKeyPEM))
	config.Certificates = []Certificate{cert}

	test := &clientTest{
		name: "ClientCert-RSA-RSAPKCS1v15",
		args: []string{"-cipher", "AES128", "-Verify", "1", "-client_sigalgs",
			"rsa_pkcs1_sha256", "-sigalgs", "rsa_pkcs1_sha256"},
		config: config,
	}

	runClientTestTLS12(t, test)
}

func TestClientKeyUpdate(t *testing.T) {
	test := &clientTest{
		name:          "KeyUpdate",
		args:          []string{"-state"},
		sendKeyUpdate: true,
	}
	runClientTestTLS13(t, test)
}

func TestResumption(t *testing.T) {
	t.Run("TLSv12", func(t *testing.T) { testResumption(t, VersionTLS12) })
	t.Run("TLSv13", func(t *testing.T) { testResumption(t, VersionTLS13) })
}

func testResumption(t *testing.T, version uint16) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}

	// Note: using RSA 2048 test certificates because they are compatible with FIPS mode.
	testCertificates := []Certificate{{Certificate: [][]byte{testRSA2048Certificate}, PrivateKey: testRSA2048PrivateKey}}
	serverConfig := &Config{
		MaxVersion:   version,
		CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384},
		Certificates: testCertificates,
		Time:         testTime,
	}

	issuer, err := x509.ParseCertificate(testRSA2048CertificateIssuer)
	if err != nil {
		panic(err)
	}

	rootCAs := x509.NewCertPool()
	rootCAs.AddCert(issuer)

	clientConfig := &Config{
		MaxVersion:         version,
		CipherSuites:       []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
		ClientSessionCache: NewLRUClientSessionCache(32),
		RootCAs:            rootCAs,
		ServerName:         "example.golang",
		Time:               testTime,
	}

	testResumeState := func(test string, didResume bool) {
		t.Helper()
		_, hs, err := testHandshake(t, clientConfig, serverConfig)
		if err != nil {
			t.Fatalf("%s: handshake failed: %s", test, err)
		}
		if hs.DidResume != didResume {
			t.Fatalf("%s resumed: %v, expected: %v", test, hs.DidResume, didResume)
		}
		if didResume && (hs.PeerCertificates == nil || hs.VerifiedChains == nil) {
			t.Fatalf("expected non-nil certificates after resumption. Got peerCertificates: %#v, verifiedCertificates: %#v", hs.PeerCertificates, hs.VerifiedChains)
		}
		if got, want := hs.ServerName, clientConfig.ServerName; got != want {
			t.Errorf("%s: server name %s, want %s", test, got, want)
		}
	}

	getTicket := func() []byte {
		return clientConfig.ClientSessionCache.(*lruSessionCache).q.Front().Value.(*lruSessionCacheEntry).state.session.ticket
	}
	deleteTicket := func() {
		ticketKey := clientConfig.ClientSessionCache.(*lruSessionCache).q.Front().Value.(*lruSessionCacheEntry).sessionKey
		clientConfig.ClientSessionCache.Put(ticketKey, nil)
	}
	corruptTicket := func() {
		clientConfig.ClientSessionCache.(*lruSessionCache).q.Front().Value.(*lruSessionCacheEntry).state.session.secret[0] ^= 0xff
	}
	randomKey := func() [32]byte {
		var k [32]byte
		if _, err := io.ReadFull(serverConfig.rand(), k[:]); err != nil {
			t.Fatalf("Failed to read new SessionTicketKey: %s", err)
		}
		return k
	}

	testResumeState("Handshake", false)
	ticket := getTicket()
	testResumeState("Resume", true)
	if bytes.Equal(ticket, getTicket()) {
		t.Fatal("ticket didn't change after resumption")
	}

	// An old session ticket is replaced with a ticket encrypted with a fresh key.
	ticket = getTicket()
	serverConfig.Time = func() time.Time { return testTime().Add(24*time.Hour + time.Minute) }
	testResumeState("ResumeWithOldTicket", true)
	if bytes.Equal(ticket, getTicket()) {
		t.Fatal("old first ticket matches the fresh one")
	}

	// Once the session master secret is expired, a full handshake should occur.
	ticket = getTicket()
	serverConfig.Time = func() time.Time { return testTime().Add(24*8*time.Hour + time.Minute) }
	testResumeState("ResumeWithExpiredTicket", false)
	if bytes.Equal(ticket, getTicket()) {
		t.Fatal("expired first ticket matches the fresh one")
	}

	serverConfig.Time = testTime // reset the time back
	key1 := randomKey()
	serverConfig.SetSessionTicketKeys([][32]byte{key1})

	testResumeState("InvalidSessionTicketKey", false)
	testResumeState("ResumeAfterInvalidSessionTicketKey", true)

	key2 := randomKey()
	serverConfig.SetSessionTicketKeys([][32]byte{key2, key1})
	ticket = getTicket()
	testResumeState("KeyChange", true)
	if bytes.Equal(ticket, getTicket()) {
		t.Fatal("new ticket wasn't included while resuming")
	}
	testResumeState("KeyChangeFinish", true)

	// Age the session ticket a bit, but not yet expired.
	serverConfig.Time = func() time.Time { return testTime().Add(24*time.Hour + time.Minute) }
	testResumeState("OldSessionTicket", true)
	ticket = getTicket()
	// Expire the session ticket, which would force a full handshake.
	serverConfig.Time = func() time.Time { return testTime().Add(24*8*time.Hour + 2*time.Minute) }
	testResumeState("ExpiredSessionTicket", false)
	if bytes.Equal(ticket, getTicket()) {
		t.Fatal("new ticket wasn't provided after old ticket expired")
	}

	// Age the session ticket a bit at a time, but don't expire it.
	d := 0 * time.Hour
	serverConfig.Time = func() time.Time { return testTime().Add(d) }
	deleteTicket()
	testResumeState("GetFreshSessionTicket", false)
	for i := 0; i < 13; i++ {
		d += 12 * time.Hour
		testResumeState("OldSessionTicket", true)
	}
	// Expire it (now a little more than 7 days) and make sure a full
	// handshake occurs for TLS 1.2. Resumption should still occur for
	// TLS 1.3 since the client should be using a fresh ticket sent over
	// by the server.
	d += 12*time.Hour + time.Minute
	if version == VersionTLS13 {
		testResumeState("ExpiredSessionTicket", true)
	} else {
		testResumeState("ExpiredSessionTicket", false)
	}
	if bytes.Equal(ticket, getTicket()) {
		t.Fatal("new ticket wasn't provided after old ticket expired")
	}

	// Reset serverConfig to ensure that calling SetSessionTicketKeys
	// before the serverConfig is used works.
	serverConfig = &Config{
		MaxVersion:   version,
		CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384},
		Certificates: testCertificates,
		Time:         testTime,
	}
	serverConfig.SetSessionTicketKeys([][32]byte{key2})

	testResumeState("FreshConfig", true)

	// In TLS 1.3, cross-cipher suite resumption is allowed as long as the KDF
	// hash matches. Also, Config.CipherSuites does not apply to TLS 1.3.
	if version != VersionTLS13 {
		clientConfig.CipherSuites = []uint16{TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384}
		testResumeState("DifferentCipherSuite", false)
		testResumeState("DifferentCipherSuiteRecovers", true)
	}

	deleteTicket()
	testResumeState("WithoutSessionTicket", false)

	// In TLS 1.3, HelloRetryRequest is sent after incorrect key share.
	// See https://www.rfc-editor.org/rfc/rfc8446#page-14.
	if version == VersionTLS13 {
		deleteTicket()
		serverConfig = &Config{
			// Use a different curve than the client to force a HelloRetryRequest.
			CurvePreferences: []CurveID{CurveP521, CurveP384, CurveP256},
			MaxVersion:       version,
			Certificates:     testCertificates,
			Time:             testTime,
		}
		testResumeState("InitialHandshake", false)
		testResumeState("WithHelloRetryRequest", true)

		// Reset serverConfig back.
		serverConfig = &Config{
			MaxVersion:   version,
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384},
			Certificates: testCertificates,
			Time:         testTime,
		}
	}

	// Session resumption should work when using client certificates
	deleteTicket()
	serverConfig.ClientCAs = rootCAs
	serverConfig.ClientAuth = RequireAndVerifyClientCert
	clientConfig.Certificates = serverConfig.Certificates
	testResumeState("InitialHandshake", false)
	testResumeState("WithClientCertificates", true)
	serverConfig.ClientAuth = NoClientCert

	// Tickets should be removed from the session cache on TLS handshake
	// failure, and the client should recover from a corrupted PSK
	testResumeState("FetchTicketToCorrupt", false)
	corruptTicket()
	_, _, err = testHandshake(t, clientConfig, serverConfig)
	if err == nil {
		t.Fatalf("handshake did not fail with a corrupted client secret")
	}
	testResumeState("AfterHandshakeFailure", false)

	clientConfig.ClientSessionCache = nil
	testResumeState("WithoutSessionCache", false)

	clientConfig.ClientSessionCache = &serializingClientCache{t: t}
	testResumeState("BeforeSerializingCache", false)
	testResumeState("WithSerializingCache", true)
}

type serializingClientCache struct {
	t *testing.T

	ticket, state []byte
}

func (c *serializingClientCache) Get(sessionKey string) (session *ClientSessionState, ok bool) {
	if c.ticket == nil {
		return nil, false
	}
	state, err := ParseSessionState(c.state)
	if err != nil {
		c.t.Error(err)
		return nil, false
	}
	cs, err := NewResumptionState(c.ticket, state)
	if err != nil {
		c.t.Error(err)
		return nil, false
	}
	return cs, true
}

func (c *serializingClientCache) Put(sessionKey string, cs *ClientSessionState) {
	if cs == nil {
		c.ticket, c.state = nil, nil
		return
	}
	ticket, state, err := cs.ResumptionState()
	if err != nil {
		c.t.Error(err)
		return
	}
	stateBytes, err := state.Bytes()
	if err != nil {
		c.t.Error(err)
		return
	}
	c.ticket, c.state = ticket, stateBytes
}

func TestLRUClientSessionCache(t *testing.T) {
	// Initialize cache of capacity 4.
	cache := NewLRUClientSessionCache(4)
	cs := make([]ClientSessionState, 6)
	keys := []string{"0", "1", "2", "3", "4", "5", "6"}

	// Add 4 entries to the cache and look them up.
	for i := 0; i < 4; i++ {
		cache.Put(keys[i], &cs[i])
	}
	for i := 0; i < 4; i++ {
		if s, ok := cache.Get(keys[i]); !ok || s != &cs[i] {
			t.Fatalf("session cache failed lookup for added key: %s", keys[i])
		}
	}

	// Add 2 more entries to the cache. First 2 should be evicted.
	for i := 4; i < 6; i++ {
		cache.Put(keys[i], &cs[i])
	}
	for i := 0; i < 2; i++ {
		if s, ok := cache.Get(keys[i]); ok || s != nil {
			t.Fatalf("session cache should have evicted key: %s", keys[i])
		}
	}

	// Touch entry 2. LRU should evict 3 next.
	cache.Get(keys[2])
	cache.Put(keys[0], &cs[0])
	if s, ok := cache.Get(keys[3]); ok || s != nil {
		t.Fatalf("session cache should have evicted key 3")
	}

	// Update entry 0 in place.
	cache.Put(keys[0], &cs[3])
	if s, ok := cache.Get(keys[0]); !ok || s != &cs[3] {
		t.Fatalf("session cache failed update for key 0")
	}

	// Calling Put with a nil entry deletes the key.
	cache.Put(keys[0], nil)
	if _, ok := cache.Get(keys[0]); ok {
		t.Fatalf("session cache failed to delete key 0")
	}

	// Delete entry 2. LRU should keep 4 and 5
	cache.Put(keys[2], nil)
	if _, ok := cache.Get(keys[2]); ok {
		t.Fatalf("session cache failed to delete key 4")
	}
	for i := 4; i < 6; i++ {
		if s, ok := cache.Get(keys[i]); !ok || s != &cs[i] {
			t.Fatalf("session cache should not have deleted key: %s", keys[i])
		}
	}
}

func TestKeyLogTLS12(t *testing.T) {
	var serverBuf, clientBuf bytes.Buffer

	clientConfig := testConfig.Clone()
	clientConfig.KeyLogWriter = &clientBuf
	clientConfig.MaxVersion = VersionTLS12

	serverConfig := testConfig.Clone()
	serverConfig.KeyLogWriter = &serverBuf
	serverConfig.MaxVersion = VersionTLS12

	c, s := localPipe(t)
	done := make(chan bool)

	go func() {
		defer close(done)

		if err := Server(s, serverConfig).Handshake(); err != nil {
			t.Errorf("server: %s", err)
			return
		}
		s.Close()
	}()

	if err := Client(c, clientConfig).Handshake(); err != nil {
		t.Fatalf("client: %s", err)
	}

	c.Close()
	<-done

	checkKeylogLine := func(side, loggedLine string) {
		if len(loggedLine) == 0 {
			t.Fatalf("%s: no keylog line was produced", side)
		}
		const expectedLen = 13 /* "CLIENT_RANDOM" */ +
			1 /* space */ +
			32*2 /* hex client nonce */ +
			1 /* space */ +
			48*2 /* hex master secret */ +
			1 /* new line */
		if len(loggedLine) != expectedLen {
			t.Fatalf("%s: keylog line has incorrect length (want %d, got %d): %q", side, expectedLen, len(loggedLine), loggedLine)
		}
		if !strings.HasPrefix(loggedLine, "CLIENT_RANDOM "+strings.Repeat("0", 64)+" ") {
			t.Fatalf("%s: keylog line has incorrect structure or nonce: %q", side, loggedLine)
		}
	}

	checkKeylogLine("client", clientBuf.String())
	checkKeylogLine("server", serverBuf.String())
}

func TestKeyLogTLS13(t *testing.T) {
	var serverBuf, clientBuf bytes.Buffer

	clientConfig := testConfig.Clone()
	clientConfig.KeyLogWriter = &clientBuf

	serverConfig := testConfig.Clone()
	serverConfig.KeyLogWriter = &serverBuf

	c, s := localPipe(t)
	done := make(chan bool)

	go func() {
		defer close(done)

		if err := Server(s, serverConfig).Handshake(); err != nil {
			t.Errorf("server: %s", err)
			return
		}
		s.Close()
	}()

	if err := Client(c, clientConfig).Handshake(); err != nil {
		t.Fatalf("client: %s", err)
	}

	c.Close()
	<-done

	checkKeylogLines := func(side, loggedLines string) {
		loggedLines = strings.TrimSpace(loggedLines)
		lines := strings.Split(loggedLines, "\n")
		if len(lines) != 4 {
			t.Errorf("Expected the %s to log 4 lines, got %d", side, len(lines))
		}
	}

	checkKeylogLines("client", clientBuf.String())
	checkKeylogLines("server", serverBuf.String())
}

func TestHandshakeClientALPNMatch(t *testing.T) {
	config := testConfig.Clone()
	config.NextProtos = []string{"proto2", "proto1"}

	test := &clientTest{
		name: "ALPN",
		// Note that this needs OpenSSL 1.0.2 because that is the first
		// version that supports the -alpn flag.
		args:   []string{"-alpn", "proto1,proto2"},
		config: config,
		validate: func(state ConnectionState) error {
			// The server's preferences should override the client.
			if state.NegotiatedProtocol != "proto1" {
				return fmt.Errorf("Got protocol %q, wanted proto1", state.NegotiatedProtocol)
			}
			return nil
		},
	}
	runClientTestTLS12(t, test)
	runClientTestTLS13(t, test)
}

func TestServerSelectingUnconfiguredApplicationProtocol(t *testing.T) {
	// This checks that the server can't select an application protocol that the
	// client didn't offer.

	c, s := localPipe(t)
	errChan := make(chan error, 1)

	go func() {
		client := Client(c, &Config{
			ServerName:   "foo",
			CipherSuites: []uint16{TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256},
			NextProtos:   []string{"http", "something-else"},
		})
		errChan <- client.Handshake()
	}()

	var header [5]byte
	if _, err := io.ReadFull(s, header[:]); err != nil {
		t.Fatal(err)
	}
	recordLen := int(header[3])<<8 | int(header[4])

	record := make([]byte, recordLen)
	if _, err := io.ReadFull(s, record); err != nil {
		t.Fatal(err)
	}

	serverHello := &serverHelloMsg{
		vers:         VersionTLS12,
		random:       make([]byte, 32),
		cipherSuite:  TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		alpnProtocol: "how-about-this",
	}
	serverHelloBytes := mustMarshal(t, serverHello)

	s.Write([]byte{
		byte(recordTypeHandshake),
		byte(VersionTLS12 >> 8),
		byte(VersionTLS12 & 0xff),
		byte(len(serverHelloBytes) >> 8),
		byte(len(serverHelloBytes)),
	})
	s.Write(serverHelloBytes)
	s.Close()

	if err := <-errChan; !strings.Contains(err.Error(), "server selected unadvertised ALPN protocol") {
		t.Fatalf("Expected error about unconfigured ALPN protocol but got %q", err)
	}
}

// sctsBase64 contains data from `openssl s_client -serverinfo 18 -connect ritter.vg:443`
const sctsBase64 = "ABIBaQFnAHUApLkJkLQYWBSHuxOizGdwCjw1mAT5G9+443fNDsgN3BAAAAFHl5nuFgAABAMARjBEAiAcS4JdlW5nW9sElUv2zvQyPoZ6ejKrGGB03gjaBZFMLwIgc1Qbbn+hsH0RvObzhS+XZhr3iuQQJY8S9G85D9KeGPAAdgBo9pj4H2SCvjqM7rkoHUz8cVFdZ5PURNEKZ6y7T0/7xAAAAUeX4bVwAAAEAwBHMEUCIDIhFDgG2HIuADBkGuLobU5a4dlCHoJLliWJ1SYT05z6AiEAjxIoZFFPRNWMGGIjskOTMwXzQ1Wh2e7NxXE1kd1J0QsAdgDuS723dc5guuFCaR+r4Z5mow9+X7By2IMAxHuJeqj9ywAAAUhcZIqHAAAEAwBHMEUCICmJ1rBT09LpkbzxtUC+Hi7nXLR0J+2PmwLp+sJMuqK+AiEAr0NkUnEVKVhAkccIFpYDqHOlZaBsuEhWWrYpg2RtKp0="

func TestHandshakClientSCTs(t *testing.T) {
	config := testConfig.Clone()

	scts, err := base64.StdEncoding.DecodeString(sctsBase64)
	if err != nil {
		t.Fatal(err)
	}

	// Note that this needs OpenSSL 1.0.2 because that is the first
	// version that supports the -serverinfo flag.
	test := &clientTest{
		name:       "SCT",
		config:     config,
		extensions: [][]byte{scts},
		validate: func(state ConnectionState) error {
			expectedSCTs := [][]byte{
				scts[8:125],
				scts[127:245],
				scts[247:],
			}
			if n := len(state.SignedCertificateTimestamps); n != len(expectedSCTs) {
				return fmt.Errorf("Got %d scts, wanted %d", n, len(expectedSCTs))
			}
			for i, expected := range expectedSCTs {
				if sct := state.SignedCertificateTimestamps[i]; !bytes.Equal(sct, expected) {
					return fmt.Errorf("SCT #%d contained %x, expected %x", i, sct, expected)
				}
			}
			return nil
		},
	}
	runClientTestTLS12(t, test)

	// TLS 1.3 moved SCTs to the Certificate extensions and -serverinfo only
	// supports ServerHello extensions.
}

func TestRenegotiationRejected(t *testing.T) {
	config := testConfig.Clone()
	test := &clientTest{
		name:                        "RenegotiationRejected",
		args:                        []string{"-state"},
		config:                      config,
		numRenegotiations:           1,
		renegotiationExpectedToFail: 1,
		checkRenegotiationError: func(renegotiationNum int, err error) error {
			if err == nil {
				return errors.New("expected error from renegotiation but got nil")
			}
			if !strings.Contains(err.Error(), "no renegotiation") {
				return fmt.Errorf("expected renegotiation to be rejected but got %q", err)
			}
			return nil
		},
	}
	runClientTestTLS12(t, test)
}

func TestRenegotiateOnce(t *testing.T) {
	config := testConfig.Clone()
	config.Renegotiation = RenegotiateOnceAsClient

	test := &clientTest{
		name:              "RenegotiateOnce",
		args:              []string{"-state"},
		config:            config,
		numRenegotiations: 1,
	}

	runClientTestTLS12(t, test)
}

func TestRenegotiateTwice(t *testing.T) {
	config := testConfig.Clone()
	config.Renegotiation = RenegotiateFreelyAsClient

	test := &clientTest{
		name:              "RenegotiateTwice",
		args:              []string{"-state"},
		config:            config,
		numRenegotiations: 2,
	}

	runClientTestTLS12(t, test)
}

func TestRenegotiateTwiceRejected(t *testing.T) {
	config := testConfig.Clone()
	config.Renegotiation = RenegotiateOnceAsClient

	test := &clientTest{
		name:                        "RenegotiateTwiceRejected",
		args:                        []string{"-state"},
		config:                      config,
		numRenegotiations:           2,
		renegotiationExpectedToFail: 2,
		checkRenegotiationError: func(renegotiationNum int, err error) error {
			if renegotiationNum == 1 {
				return err
			}

			if err == nil {
				return errors.New("expected error from renegotiation but got nil")
			}
			if !strings.Contains(err.Error(), "no renegotiation") {
				return fmt.Errorf("expected renegotiation to be rejected but got %q", err)
			}
			return nil
		},
	}

	runClientTestTLS12(t, test)
}

func TestHandshakeClientExportKeyingMaterial(t *testing.T) {
	test := &clientTest{
		name:   "ExportKeyingMaterial",
		config: testConfig.Clone(),
		validate: func(state ConnectionState) error {
			if km, err := state.ExportKeyingMaterial("test", nil, 42); err != nil {
				return fmt.Errorf("ExportKeyingMaterial failed: %v", err)
			} else if len(km) != 42 {
				return fmt.Errorf("Got %d bytes from ExportKeyingMaterial, wanted %d", len(km), 42)
			}
			return nil
		},
	}
	runClientTestTLS10(t, test)
	runClientTestTLS12(t, test)
	runClientTestTLS13(t, test)
}

var hostnameInSNITests = []struct {
	in, out string
}{
	// Opaque string
	{"", ""},
	{"localhost", "localhost"},
	{"foo, bar, baz and qux", "foo, bar, baz and qux"},

	// DNS hostname
	{"golang.org", "golang.org"},
	{"golang.org.", "golang.org"},

	// Literal IPv4 address
	{"1.2.3.4", ""},

	// Literal IPv6 address
	{"::1", ""},
	{"::1%lo0", ""}, // with zone identifier
	{"[::1]", ""},   // as per RFC 5952 we allow the [] style as IPv6 literal
	{"[::1%lo0]", ""},
}

func TestHostnameInSNI(t *testing.T) {
	for _, tt := range hostnameInSNITests {
		c, s := localPipe(t)

		go func(host string) {
			Client(c, &Config{ServerName: host, InsecureSkipVerify: true}).Handshake()
		}(tt.in)

		var header [5]byte
		if _, err := io.ReadFull(s, header[:]); err != nil {
			t.Fatal(err)
		}
		recordLen := int(header[3])<<8 | int(header[4])

		record := make([]byte, recordLen)
		if _, err := io.ReadFull(s, record[:]); err != nil {
			t.Fatal(err)
		}

		c.Close()
		s.Close()

		var m clientHelloMsg
		if !m.unmarshal(record) {
			t.Errorf("unmarshaling ClientHello for %q failed", tt.in)
			continue
		}
		if tt.in != tt.out && m.serverName == tt.in {
			t.Errorf("prohibited %q found in ClientHello: %x", tt.in, record)
		}
		if m.serverName != tt.out {
			t.Errorf("expected %q not found in ClientHello: %x", tt.out, record)
		}
	}
}

func TestServerSelectingUnconfiguredCipherSuite(t *testing.T) {
	// This checks that the server can't select a cipher suite that the
	// client didn't offer. See #13174.

	c, s := localPipe(t)
	errChan := make(chan error, 1)

	go func() {
		client := Client(c, &Config{
			ServerName:   "foo",
			CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
		})
		errChan <- client.Handshake()
	}()

	var header [5]byte
	if _, err := io.ReadFull(s, header[:]); err != nil {
		t.Fatal(err)
	}
	recordLen := int(header[3])<<8 | int(header[4])

	record := make([]byte, recordLen)
	if _, err := io.ReadFull(s, record); err != nil {
		t.Fatal(err)
	}

	// Create a ServerHello that selects a different cipher suite than the
	// sole one that the client offered.
	serverHello := &serverHelloMsg{
		vers:        VersionTLS12,
		random:      make([]byte, 32),
		cipherSuite: TLS_RSA_WITH_AES_256_GCM_SHA384,
	}
	serverHelloBytes := mustMarshal(t, serverHello)

	s.Write([]byte{
		byte(recordTypeHandshake),
		byte(VersionTLS12 >> 8),
		byte(VersionTLS12 & 0xff),
		byte(len(serverHelloBytes) >> 8),
		byte(len(serverHelloBytes)),
	})
	s.Write(serverHelloBytes)
	s.Close()

	if err := <-errChan; !strings.Contains(err.Error(), "unconfigured cipher") {
		t.Fatalf("Expected error about unconfigured cipher suite but got %q", err)
	}
}

func TestVerifyConnection(t *testing.T) {
	t.Run("TLSv12", func(t *testing.T) { testVerifyConnection(t, VersionTLS12) })
	t.Run("TLSv13", func(t *testing.T) { testVerifyConnection(t, VersionTLS13) })
}

func testVerifyConnection(t *testing.T, version uint16) {
	checkFields := func(c ConnectionState, called *int, errorType string) error {
		if c.Version != version {
			return fmt.Errorf("%s: got Version %v, want %v", errorType, c.Version, version)
		}
		if c.HandshakeComplete {
			return fmt.Errorf("%s: got HandshakeComplete, want false", errorType)
		}
		if c.ServerName != "example.golang" {
			return fmt.Errorf("%s: got ServerName %s, want %s", errorType, c.ServerName, "example.golang")
		}
		if c.NegotiatedProtocol != "protocol1" {
			return fmt.Errorf("%s: got NegotiatedProtocol %s, want %s", errorType, c.NegotiatedProtocol, "protocol1")
		}
		if c.CipherSuite == 0 {
			return fmt.Errorf("%s: got CipherSuite 0, want non-zero", errorType)
		}
		wantDidResume := false
		if *called == 2 { // if this is the second time, then it should be a resumption
			wantDidResume = true
		}
		if c.DidResume != wantDidResume {
			return fmt.Errorf("%s: got DidResume %t, want %t", errorType, c.DidResume, wantDidResume)
		}
		return nil
	}

	tests := []struct {
		name            string
		configureServer func(*Config, *int)
		configureClient func(*Config, *int)
	}{
		{
			name: "RequireAndVerifyClientCert",
			configureServer: func(config *Config, called *int) {
				config.ClientAuth = RequireAndVerifyClientCert
				config.VerifyConnection = func(c ConnectionState) error {
					*called++
					if l := len(c.PeerCertificates); l != 1 {
						return fmt.Errorf("server: got len(PeerCertificates) = %d, wanted 1", l)
					}
					if len(c.VerifiedChains) == 0 {
						return fmt.Errorf("server: got len(VerifiedChains) = 0, wanted non-zero")
					}
					return checkFields(c, called, "server")
				}
			},
			configureClient: func(config *Config, called *int) {
				config.VerifyConnection = func(c ConnectionState) error {
					*called++
					if l := len(c.PeerCertificates); l != 1 {
						return fmt.Errorf("client: got len(PeerCertificates) = %d, wanted 1", l)
					}
					if len(c.VerifiedChains) == 0 {
						return fmt.Errorf("client: got len(VerifiedChains) = 0, wanted non-zero")
					}
					if c.DidResume {
						return nil
						// The SCTs and OCSP Response are dropped on resumption.
						// See http://golang.org/issue/39075.
					}
					if len(c.OCSPResponse) == 0 {
						return fmt.Errorf("client: got len(OCSPResponse) = 0, wanted non-zero")
					}
					if len(c.SignedCertificateTimestamps) == 0 {
						return fmt.Errorf("client: got len(SignedCertificateTimestamps) = 0, wanted non-zero")
					}
					return checkFields(c, called, "client")
				}
			},
		},
		{
			name: "InsecureSkipVerify",
			configureServer: func(config *Config, called *int) {
				config.ClientAuth = RequireAnyClientCert
				config.InsecureSkipVerify = true
				config.VerifyConnection = func(c ConnectionState) error {
					*called++
					if l := len(c.PeerCertificates); l != 1 {
						return fmt.Errorf("server: got len(PeerCertificates) = %d, wanted 1", l)
					}
					if c.VerifiedChains != nil {
						return fmt.Errorf("server: got Verified Chains %v, want nil", c.VerifiedChains)
					}
					return checkFields(c, called, "server")
				}
			},
			configureClient: func(config *Config, called *int) {
				config.InsecureSkipVerify = true
				config.VerifyConnection = func(c ConnectionState) error {
					*called++
					if l := len(c.PeerCertificates); l != 1 {
						return fmt.Errorf("client: got len(PeerCertificates) = %d, wanted 1", l)
					}
					if c.VerifiedChains != nil {
						return fmt.Errorf("server: got Verified Chains %v, want nil", c.VerifiedChains)
					}
					if c.DidResume {
						return nil
						// The SCTs and OCSP Response are dropped on resumption.
						// See http://golang.org/issue/39075.
					}
					if len(c.OCSPResponse) == 0 {
						return fmt.Errorf("client: got len(OCSPResponse) = 0, wanted non-zero")
					}
					if len(c.SignedCertificateTimestamps) == 0 {
						return fmt.Errorf("client: got len(SignedCertificateTimestamps) = 0, wanted non-zero")
					}
					return checkFields(c, called, "client")
				}
			},
		},
		{
			name: "NoClientCert",
			configureServer: func(config *Config, called *int) {
				config.ClientAuth = NoClientCert
				config.VerifyConnection = func(c ConnectionState) error {
					*called++
					return checkFields(c, called, "server")
				}
			},
			configureClient: func(config *Config, called *int) {
				config.VerifyConnection = func(c ConnectionState) error {
					*called++
					return checkFields(c, called, "client")
				}
			},
		},
		{
			name: "RequestClientCert",
			configureServer: func(config *Config, called *int) {
				config.ClientAuth = RequestClientCert
				config.VerifyConnection = func(c ConnectionState) error {
					*called++
					return checkFields(c, called, "server")
				}
			},
			configureClient: func(config *Config, called *int) {
				config.Certificates = nil // clear the client cert
				config.VerifyConnection = func(c ConnectionState) error {
					*called++
					if l := len(c.PeerCertificates); l != 1 {
						return fmt.Errorf("client: got len(PeerCertificates) = %d, wanted 1", l)
					}
					if len(c.VerifiedChains) == 0 {
						return fmt.Errorf("client: got len(VerifiedChains) = 0, wanted non-zero")
					}
					if c.DidResume {
						return nil
						// The SCTs and OCSP Response are dropped on resumption.
						// See http://golang.org/issue/39075.
					}
					if len(c.OCSPResponse) == 0 {
						return fmt.Errorf("client: got len(OCSPResponse) = 0, wanted non-zero")
					}
					if len(c.SignedCertificateTimestamps) == 0 {
						return fmt.Errorf("client: got len(SignedCertificateTimestamps) = 0, wanted non-zero")
					}
					return checkFields(c, called, "client")
				}
			},
		},
	}
	for _, test := range tests {
		// Note: using RSA 2048 test certificates because they are compatible with FIPS mode.
		testCertificates := []Certificate{{Certificate: [][]byte{testRSA2048Certificate}, PrivateKey: testRSA2048PrivateKey}}

		issuer, err := x509.ParseCertificate(testRSA2048CertificateIssuer)
		if err != nil {
			panic(err)
		}
		rootCAs := x509.NewCertPool()
		rootCAs.AddCert(issuer)

		var serverCalled, clientCalled int

		serverConfig := &Config{
			MaxVersion:   version,
			Certificates: testCertificates,
			Time:         testTime,
			ClientCAs:    rootCAs,
			NextProtos:   []string{"protocol1"},
		}
		serverConfig.Certificates[0].SignedCertificateTimestamps = [][]byte{[]byte("dummy sct 1"), []byte("dummy sct 2")}
		serverConfig.Certificates[0].OCSPStaple = []byte("dummy ocsp")
		test.configureServer(serverConfig, &serverCalled)

		clientConfig := &Config{
			MaxVersion:         version,
			ClientSessionCache: NewLRUClientSessionCache(32),
			RootCAs:            rootCAs,
			ServerName:         "example.golang",
			Certificates:       testCertificates,
			Time:               testTime,
			NextProtos:         []string{"protocol1"},
		}
		test.configureClient(clientConfig, &clientCalled)

		testHandshakeState := func(name string, didResume bool) {
			_, hs, err := testHandshake(t, clientConfig, serverConfig)
			if err != nil {
				t.Fatalf("%s: handshake failed: %s", name, err)
			}
			if hs.DidResume != didResume {
				t.Errorf("%s: resumed: %v, expected: %v", name, hs.DidResume, didResume)
			}
			wantCalled := 1
			if didResume {
				wantCalled = 2 // resumption would mean this is the second time it was called in this test
			}
			if clientCalled != wantCalled {
				t.Errorf("%s: expected client VerifyConnection called %d times, did %d times", name, wantCalled, clientCalled)
			}
			if serverCalled != wantCalled {
				t.Errorf("%s: expected server VerifyConnection called %d times, did %d times", name, wantCalled, serverCalled)
			}
		}
		testHandshakeState(fmt.Sprintf("%s-FullHandshake", test.name), false)
		testHandshakeState(fmt.Sprintf("%s-Resumption", test.name), true)
	}
}

func TestVerifyPeerCertificate(t *testing.T) {
	t.Run("TLSv12", func(t *testing.T) { testVerifyPeerCertificate(t, VersionTLS12) })
	t.Run("TLSv13", func(t *testing.T) { testVerifyPeerCertificate(t, VersionTLS13) })
}

func testVerifyPeerCertificate(t *testing.T, version uint16) {
	// Note: using RSA 2048 test certificates because they are compatible with FIPS mode.
	issuer, err := x509.ParseCertificate(testRSA2048CertificateIssuer)
	if err != nil {
		panic(err)
	}

	rootCAs := x509.NewCertPool()
	rootCAs.AddCert(issuer)

	sentinelErr := errors.New("TestVerifyPeerCertificate")

	verifyPeerCertificateCallback := func(called *bool, rawCerts [][]byte, validatedChains [][]*x509.Certificate) error {
		if l := len(rawCerts); l != 1 {
			return fmt.Errorf("got len(rawCerts) = %d, wanted 1", l)
		}
		if len(validatedChains) == 0 {
			return errors.New("got len(validatedChains) = 0, wanted non-zero")
		}
		*called = true
		return nil
	}
	verifyConnectionCallback := func(called *bool, isClient bool, c ConnectionState) error {
		if l := len(c.PeerCertificates); l != 1 {
			return fmt.Errorf("got len(PeerCertificates) = %d, wanted 1", l)
		}
		if len(c.VerifiedChains) == 0 {
			return fmt.Errorf("got len(VerifiedChains) = 0, wanted non-zero")
		}
		if isClient && len(c.OCSPResponse) == 0 {
			return fmt.Errorf("got len(OCSPResponse) = 0, wanted non-zero")
		}
		*called = true
		return nil
	}

	tests := []struct {
		configureServer func(*Config, *bool)
		configureClient func(*Config, *bool)
		validate        func(t *testing.T, testNo int, clientCalled, serverCalled bool, clientErr, serverErr error)
	}{
		{
			configureServer: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyPeerCertificate = func(rawCerts [][]byte, validatedChains [][]*x509.Certificate) error {
					return verifyPeerCertificateCallback(called, rawCerts, validatedChains)
				}
			},
			configureClient: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyPeerCertificate = func(rawCerts [][]byte, validatedChains [][]*x509.Certificate) error {
					return verifyPeerCertificateCallback(called, rawCerts, validatedChains)
				}
			},
			validate: func(t *testing.T, testNo int, clientCalled, serverCalled bool, clientErr, serverErr error) {
				if clientErr != nil {
					t.Errorf("test[%d]: client handshake failed: %v", testNo, clientErr)
				}
				if serverErr != nil {
					t.Errorf("test[%d]: server handshake failed: %v", testNo, serverErr)
				}
				if !clientCalled {
					t.Errorf("test[%d]: client did not call callback", testNo)
				}
				if !serverCalled {
					t.Errorf("test[%d]: server did not call callback", testNo)
				}
			},
		},
		{
			configureServer: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyPeerCertificate = func(rawCerts [][]byte, validatedChains [][]*x509.Certificate) error {
					return sentinelErr
				}
			},
			configureClient: func(config *Config, called *bool) {
				config.VerifyPeerCertificate = nil
			},
			validate: func(t *testing.T, testNo int, clientCalled, serverCalled bool, clientErr, serverErr error) {
				if serverErr != sentinelErr {
					t.Errorf("#%d: got server error %v, wanted sentinelErr", testNo, serverErr)
				}
			},
		},
		{
			configureServer: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
			},
			configureClient: func(config *Config, called *bool) {
				config.VerifyPeerCertificate = func(rawCerts [][]byte, validatedChains [][]*x509.Certificate) error {
					return sentinelErr
				}
			},
			validate: func(t *testing.T, testNo int, clientCalled, serverCalled bool, clientErr, serverErr error) {
				if clientErr != sentinelErr {
					t.Errorf("#%d: got client error %v, wanted sentinelErr", testNo, clientErr)
				}
			},
		},
		{
			configureServer: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
			},
			configureClient: func(config *Config, called *bool) {
				config.InsecureSkipVerify = true
				config.VerifyPeerCertificate = func(rawCerts [][]byte, validatedChains [][]*x509.Certificate) error {
					if l := len(rawCerts); l != 1 {
						return fmt.Errorf("got len(rawCerts) = %d, wanted 1", l)
					}
					// With InsecureSkipVerify set, this
					// callback should still be called but
					// validatedChains must be empty.
					if l := len(validatedChains); l != 0 {
						return fmt.Errorf("got len(validatedChains) = %d, wanted zero", l)
					}
					*called = true
					return nil
				}
			},
			validate: func(t *testing.T, testNo int, clientCalled, serverCalled bool, clientErr, serverErr error) {
				if clientErr != nil {
					t.Errorf("test[%d]: client handshake failed: %v", testNo, clientErr)
				}
				if serverErr != nil {
					t.Errorf("test[%d]: server handshake failed: %v", testNo, serverErr)
				}
				if !clientCalled {
					t.Errorf("test[%d]: client did not call callback", testNo)
				}
			},
		},
		{
			configureServer: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyConnection = func(c ConnectionState) error {
					return verifyConnectionCallback(called, false, c)
				}
			},
			configureClient: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyConnection = func(c ConnectionState) error {
					return verifyConnectionCallback(called, true, c)
				}
			},
			validate: func(t *testing.T, testNo int, clientCalled, serverCalled bool, clientErr, serverErr error) {
				if clientErr != nil {
					t.Errorf("test[%d]: client handshake failed: %v", testNo, clientErr)
				}
				if serverErr != nil {
					t.Errorf("test[%d]: server handshake failed: %v", testNo, serverErr)
				}
				if !clientCalled {
					t.Errorf("test[%d]: client did not call callback", testNo)
				}
				if !serverCalled {
					t.Errorf("test[%d]: server did not call callback", testNo)
				}
			},
		},
		{
			configureServer: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyConnection = func(c ConnectionState) error {
					return sentinelErr
				}
			},
			configureClient: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyConnection = nil
			},
			validate: func(t *testing.T, testNo int, clientCalled, serverCalled bool, clientErr, serverErr error) {
				if serverErr != sentinelErr {
					t.Errorf("#%d: got server error %v, wanted sentinelErr", testNo, serverErr)
				}
			},
		},
		{
			configureServer: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyConnection = nil
			},
			configureClient: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyConnection = func(c ConnectionState) error {
					return sentinelErr
				}
			},
			validate: func(t *testing.T, testNo int, clientCalled, serverCalled bool, clientErr, serverErr error) {
				if clientErr != sentinelErr {
					t.Errorf("#%d: got client error %v, wanted sentinelErr", testNo, clientErr)
				}
			},
		},
		{
			configureServer: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyPeerCertificate = func(rawCerts [][]byte, validatedChains [][]*x509.Certificate) error {
					return verifyPeerCertificateCallback(called, rawCerts, validatedChains)
				}
				config.VerifyConnection = func(c ConnectionState) error {
					return sentinelErr
				}
			},
			configureClient: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyPeerCertificate = nil
				config.VerifyConnection = nil
			},
			validate: func(t *testing.T, testNo int, clientCalled, serverCalled bool, clientErr, serverErr error) {
				if serverErr != sentinelErr {
					t.Errorf("#%d: got server error %v, wanted sentinelErr", testNo, serverErr)
				}
				if !serverCalled {
					t.Errorf("test[%d]: server did not call callback", testNo)
				}
			},
		},
		{
			configureServer: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyPeerCertificate = nil
				config.VerifyConnection = nil
			},
			configureClient: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyPeerCertificate = func(rawCerts [][]byte, validatedChains [][]*x509.Certificate) error {
					return verifyPeerCertificateCallback(called, rawCerts, validatedChains)
				}
				config.VerifyConnection = func(c ConnectionState) error {
					return sentinelErr
				}
			},
			validate: func(t *testing.T, testNo int, clientCalled, serverCalled bool, clientErr, serverErr error) {
				if clientErr != sentinelErr {
					t.Errorf("#%d: got client error %v, wanted sentinelErr", testNo, clientErr)
				}
				if !clientCalled {
					t.Errorf("test[%d]: client did not call callback", testNo)
				}
			},
		},
	}

	for i, test := range tests {
		c, s := localPipe(t)
		done := make(chan error)

		var clientCalled, serverCalled bool

		go func() {
			config := testConfig.Clone()
			config.ServerName = "example.golang"
			config.ClientAuth = RequireAndVerifyClientCert
			config.ClientCAs = rootCAs
			config.Time = testTime
			config.MaxVersion = version
			config.Certificates = make([]Certificate, 1)
			config.Certificates[0].Certificate = [][]byte{testRSA2048Certificate}
			config.Certificates[0].PrivateKey = testRSA2048PrivateKey
			config.Certificates[0].SignedCertificateTimestamps = [][]byte{[]byte("dummy sct 1"), []byte("dummy sct 2")}
			config.Certificates[0].OCSPStaple = []byte("dummy ocsp")
			test.configureServer(config, &serverCalled)

			err = Server(s, config).Handshake()
			s.Close()
			done <- err
		}()

		config := testConfig.Clone()
		config.Certificates = []Certificate{{Certificate: [][]byte{testRSA2048Certificate}, PrivateKey: testRSA2048PrivateKey}}
		config.ServerName = "example.golang"
		config.RootCAs = rootCAs
		config.Time = testTime
		config.MaxVersion = version
		test.configureClient(config, &clientCalled)
		clientErr := Client(c, config).Handshake()
		c.Close()
		serverErr := <-done

		test.validate(t, i, clientCalled, serverCalled, clientErr, serverErr)
	}
}

// brokenConn wraps a net.Conn and causes all Writes after a certain number to
// fail with brokenConnErr.
type brokenConn struct {
	net.Conn

	// breakAfter is the number of successful writes that will be allowed
	// before all subsequent writes fail.
	breakAfter int

	// numWrites is the number of writes that have been done.
	numWrites int
}

// brokenConnErr is the error that brokenConn returns once exhausted.
var brokenConnErr = errors.New("too many writes to brokenConn")

func (b *brokenConn) Write(data []byte) (int, error) {
	if b.numWrites >= b.breakAfter {
		return 0, brokenConnErr
	}

	b.numWrites++
	return b.Conn.Write(data)
}

func TestFailedWrite(t *testing.T) {
	// Test that a write error during the handshake is returned.
	for _, breakAfter := range []int{0, 1} {
		c, s := localPipe(t)
		done := make(chan bool)

		go func() {
			Server(s, testConfig).Handshake()
			s.Close()
			done <- true
		}()

		brokenC := &brokenConn{Conn: c, breakAfter: breakAfter}
		err := Client(brokenC, testConfig).Handshake()
		if err != brokenConnErr {
			t.Errorf("#%d: expected error from brokenConn but got %q", breakAfter, err)
		}
		brokenC.Close()

		<-done
	}
}

// writeCountingConn wraps a net.Conn and counts the number of Write calls.
type writeCountingConn struct {
	net.Conn

	// numWrites is the number of writes that have been done.
	numWrites int
}

func (wcc *writeCountingConn) Write(data []byte) (int, error) {
	wcc.numWrites++
	return wcc.Conn.Write(data)
}

func TestBuffering(t *testing.T) {
	t.Run("TLSv12", func(t *testing.T) { testBuffering(t, VersionTLS12) })
	t.Run("TLSv13", func(t *testing.T) { testBuffering(t, VersionTLS13) })
}

func testBuffering(t *testing.T, version uint16) {
	c, s := localPipe(t)
	done := make(chan bool)

	clientWCC := &writeCountingConn{Conn: c}
	serverWCC := &writeCountingConn{Conn: s}

	go func() {
		config := testConfig.Clone()
		config.MaxVersion = version
		Server(serverWCC, config).Handshake()
		serverWCC.Close()
		done <- true
	}()

	err := Client(clientWCC, testConfig).Handshake()
	if err != nil {
		t.Fatal(err)
	}
	clientWCC.Close()
	<-done

	var expectedClient, expectedServer int
	if version == VersionTLS13 {
		expectedClient = 2
		expectedServer = 1
	} else {
		expectedClient = 2
		expectedServer = 2
	}

	if n := clientWCC.numWrites; n != expectedClient {
		t.Errorf("expected client handshake to complete with %d writes, but saw %d", expectedClient, n)
	}

	if n := serverWCC.numWrites; n != expectedServer {
		t.Errorf("expected server handshake to complete with %d writes, but saw %d", expectedServer, n)
	}
}

func TestAlertFlushing(t *testing.T) {
	c, s := localPipe(t)
	done := make(chan bool)

	clientWCC := &writeCountingConn{Conn: c}
	serverWCC := &writeCountingConn{Conn: s}

	serverConfig := testConfig.Clone()

	// Cause a signature-time error
	brokenKey := rsa.PrivateKey{PublicKey: testRSAPrivateKey.PublicKey}
	brokenKey.D = big.NewInt(42)
	serverConfig.Certificates = []Certificate{{
		Certificate: [][]byte{testRSACertificate},
		PrivateKey:  &brokenKey,
	}}

	go func() {
		Server(serverWCC, serverConfig).Handshake()
		serverWCC.Close()
		done <- true
	}()

	err := Client(clientWCC, testConfig).Handshake()
	if err == nil {
		t.Fatal("client unexpectedly returned no error")
	}

	const expectedError = "remote error: tls: internal error"
	if e := err.Error(); !strings.Contains(e, expectedError) {
		t.Fatalf("expected to find %q in error but error was %q", expectedError, e)
	}
	clientWCC.Close()
	<-done

	if n := serverWCC.numWrites; n != 1 {
		t.Errorf("expected server handshake to complete with one write, but saw %d", n)
	}
}

func TestHandshakeRace(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	t.Parallel()
	// This test races a Read and Write to try and complete a handshake in
	// order to provide some evidence that there are no races or deadlocks
	// in the handshake locking.
	for i := 0; i < 32; i++ {
		c, s := localPipe(t)

		go func() {
			server := Server(s, testConfig)
			if err := server.Handshake(); err != nil {
				panic(err)
			}

			var request [1]byte
			if n, err := server.Read(request[:]); err != nil || n != 1 {
				panic(err)
			}

			server.Write(request[:])
			server.Close()
		}()

		startWrite := make(chan struct{})
		startRead := make(chan struct{})
		readDone := make(chan struct{}, 1)

		client := Client(c, testConfig)
		go func() {
			<-startWrite
			var request [1]byte
			client.Write(request[:])
		}()

		go func() {
			<-startRead
			var reply [1]byte
			if _, err := io.ReadFull(client, reply[:]); err != nil {
				panic(err)
			}
			c.Close()
			readDone <- struct{}{}
		}()

		if i&1 == 1 {
			startWrite <- struct{}{}
			startRead <- struct{}{}
		} else {
			startRead <- struct{}{}
			startWrite <- struct{}{}
		}
		<-readDone
	}
}

var getClientCertificateTests = []struct {
	setup               func(*Config, *Config)
	expectedClientError string
	verify              func(*testing.T, int, *ConnectionState)
}{
	{
		func(clientConfig, serverConfig *Config) {
			// Returning a Certificate with no certificate data
			// should result in an empty message being sent to the
			// server.
			serverConfig.ClientCAs = nil
			clientConfig.GetClientCertificate = func(cri *CertificateRequestInfo) (*Certificate, error) {
				if len(cri.SignatureSchemes) == 0 {
					panic("empty SignatureSchemes")
				}
				if len(cri.AcceptableCAs) != 0 {
					panic("AcceptableCAs should have been empty")
				}
				return new(Certificate), nil
			}
		},
		"",
		func(t *testing.T, testNum int, cs *ConnectionState) {
			if l := len(cs.PeerCertificates); l != 0 {
				t.Errorf("#%d: expected no certificates but got %d", testNum, l)
			}
		},
	},
	{
		func(clientConfig, serverConfig *Config) {
			// With TLS 1.1, the SignatureSchemes should be
			// synthesised from the supported certificate types.
			clientConfig.MaxVersion = VersionTLS11
			clientConfig.GetClientCertificate = func(cri *CertificateRequestInfo) (*Certificate, error) {
				if len(cri.SignatureSchemes) == 0 {
					panic("empty SignatureSchemes")
				}
				return new(Certificate), nil
			}
		},
		"",
		func(t *testing.T, testNum int, cs *ConnectionState) {
			if l := len(cs.PeerCertificates); l != 0 {
				t.Errorf("#%d: expected no certificates but got %d", testNum, l)
			}
		},
	},
	{
		func(clientConfig, serverConfig *Config) {
			// Returning an error should abort the handshake with
			// that error.
			clientConfig.GetClientCertificate = func(cri *CertificateRequestInfo) (*Certificate, error) {
				return nil, errors.New("GetClientCertificate")
			}
		},
		"GetClientCertificate",
		func(t *testing.T, testNum int, cs *ConnectionState) {
		},
	},
	{
		func(clientConfig, serverConfig *Config) {
			clientConfig.GetClientCertificate = func(cri *CertificateRequestInfo) (*Certificate, error) {
				if len(cri.AcceptableCAs) == 0 {
					panic("empty AcceptableCAs")
				}
				cert := &Certificate{
					Certificate: [][]byte{testRSA2048Certificate},
					PrivateKey:  testRSA2048PrivateKey,
				}
				return cert, nil
			}
		},
		"",
		func(t *testing.T, testNum int, cs *ConnectionState) {
			if len(cs.VerifiedChains) == 0 {
				t.Errorf("#%d: expected some verified chains, but found none", testNum)
			}
		},
	},
}

func TestGetClientCertificate(t *testing.T) {
	t.Run("TLSv12", func(t *testing.T) { testGetClientCertificate(t, VersionTLS12) })
	t.Run("TLSv13", func(t *testing.T) { testGetClientCertificate(t, VersionTLS13) })
}

func testGetClientCertificate(t *testing.T, version uint16) {
	// Note: using RSA 2048 test certificates because they are compatible with FIPS mode.
	issuer, err := x509.ParseCertificate(testRSA2048CertificateIssuer)
	if err != nil {
		panic(err)
	}

	for i, test := range getClientCertificateTests {
		serverConfig := testConfig.Clone()
		serverConfig.Certificates = []Certificate{{Certificate: [][]byte{testRSA2048Certificate}, PrivateKey: testRSA2048PrivateKey}}
		serverConfig.ClientAuth = VerifyClientCertIfGiven
		serverConfig.RootCAs = x509.NewCertPool()
		serverConfig.RootCAs.AddCert(issuer)
		serverConfig.ClientCAs = serverConfig.RootCAs
		serverConfig.Time = testTime
		serverConfig.MaxVersion = version

		clientConfig := testConfig.Clone()
		clientConfig.Certificates = []Certificate{{Certificate: [][]byte{testRSA2048Certificate}, PrivateKey: testRSA2048PrivateKey}}
		clientConfig.MaxVersion = version

		test.setup(clientConfig, serverConfig)

		// TLS 1.1 isn't available for FIPS required
		if fips140tls.Required() && clientConfig.MaxVersion == VersionTLS11 {
			t.Logf("skipping test %d for FIPS mode", i)
			continue
		}

		type serverResult struct {
			cs  ConnectionState
			err error
		}

		c, s := localPipe(t)
		done := make(chan serverResult)

		go func() {
			defer s.Close()
			server := Server(s, serverConfig)
			err := server.Handshake()

			var cs ConnectionState
			if err == nil {
				cs = server.ConnectionState()
			}
			done <- serverResult{cs, err}
		}()

		clientErr := Client(c, clientConfig).Handshake()
		c.Close()

		result := <-done

		if clientErr != nil {
			if len(test.expectedClientError) == 0 {
				t.Errorf("#%d: client error: %v", i, clientErr)
			} else if got := clientErr.Error(); got != test.expectedClientError {
				t.Errorf("#%d: expected client error %q, but got %q", i, test.expectedClientError, got)
			} else {
				test.verify(t, i, &result.cs)
			}
		} else if len(test.expectedClientError) > 0 {
			t.Errorf("#%d: expected client error %q, but got no error", i, test.expectedClientError)
		} else if err := result.err; err != nil {
			t.Errorf("#%d: server error: %v", i, err)
		} else {
			test.verify(t, i, &result.cs)
		}
	}
}

func TestRSAPSSKeyError(t *testing.T) {
	// crypto/tls does not support the rsa_pss_pss_* SignatureSchemes. If support for
	// public keys with OID RSASSA-PSS is added to crypto/x509, they will be misused with
	// the rsa_pss_rsae_* SignatureSchemes. Assert that RSASSA-PSS certificates don't
	// parse, or that they don't carry *rsa.PublicKey keys.
	b, _ := pem.Decode([]byte(`
-----BEGIN CERTIFICATE-----
MIIDZTCCAhygAwIBAgIUCF2x0FyTgZG0CC9QTDjGWkB5vgEwPgYJKoZIhvcNAQEK
MDGgDTALBglghkgBZQMEAgGhGjAYBgkqhkiG9w0BAQgwCwYJYIZIAWUDBAIBogQC
AgDeMBIxEDAOBgNVBAMMB1JTQS1QU1MwHhcNMTgwNjI3MjI0NDM2WhcNMTgwNzI3
MjI0NDM2WjASMRAwDgYDVQQDDAdSU0EtUFNTMIIBIDALBgkqhkiG9w0BAQoDggEP
ADCCAQoCggEBANxDm0f76JdI06YzsjB3AmmjIYkwUEGxePlafmIASFjDZl/elD0Z
/a7xLX468b0qGxLS5al7XCcEprSdsDR6DF5L520+pCbpfLyPOjuOvGmk9KzVX4x5
b05YXYuXdsQ0Kjxcx2i3jjCday6scIhMJVgBZxTEyMj1thPQM14SHzKCd/m6HmCL
QmswpH2yMAAcBRWzRpp/vdH5DeOJEB3aelq7094no731mrLUCHRiZ1htq8BDB3ou
czwqgwspbqZ4dnMXl2MvfySQ5wJUxQwILbiuAKO2lVVPUbFXHE9pgtznNoPvKwQT
JNcX8ee8WIZc2SEGzofjk3NpjR+2ADB2u3sCAwEAAaNTMFEwHQYDVR0OBBYEFNEz
AdyJ2f+fU+vSCS6QzohnOnprMB8GA1UdIwQYMBaAFNEzAdyJ2f+fU+vSCS6Qzohn
OnprMA8GA1UdEwEB/wQFMAMBAf8wPgYJKoZIhvcNAQEKMDGgDTALBglghkgBZQME
AgGhGjAYBgkqhkiG9w0BAQgwCwYJYIZIAWUDBAIBogQCAgDeA4IBAQCjEdrR5aab
sZmCwrMeKidXgfkmWvfuLDE+TCbaqDZp7BMWcMQXT9O0UoUT5kqgKj2ARm2pEW0Z
H3Z1vj3bbds72qcDIJXp+l0fekyLGeCrX/CbgnMZXEP7+/+P416p34ChR1Wz4dU1
KD3gdsUuTKKeMUog3plxlxQDhRQmiL25ygH1LmjLd6dtIt0GVRGr8lj3euVeprqZ
bZ3Uq5eLfsn8oPgfC57gpO6yiN+UURRTlK3bgYvLh4VWB3XXk9UaQZ7Mq1tpXjoD
HYFybkWzibkZp4WRo+Fa28rirH+/wHt0vfeN7UCceURZEx4JaxIIfe4ku7uDRhJi
RwBA9Xk1KBNF
-----END CERTIFICATE-----`))
	if b == nil {
		t.Fatal("Failed to decode certificate")
	}
	cert, err := x509.ParseCertificate(b.Bytes)
	if err != nil {
		return
	}
	if _, ok := cert.PublicKey.(*rsa.PublicKey); ok {
		t.Error("A RSASSA-PSS certificate was parsed like a PKCS#1 v1.5 one, and it will be mistakenly used with rsa_pss_rsae_* signature algorithms")
	}
}

func TestCloseClientConnectionOnIdleServer(t *testing.T) {
	clientConn, serverConn := localPipe(t)
	client := Client(clientConn, testConfig.Clone())
	go func() {
		var b [1]byte
		serverConn.Read(b[:])
		client.Close()
	}()
	client.SetWriteDeadline(time.Now().Add(time.Minute))
	err := client.Handshake()
	if err != nil {
		if err, ok := err.(net.Error); ok && err.Timeout() {
			t.Errorf("Expected a closed network connection error but got '%s'", err.Error())
		}
	} else {
		t.Errorf("Error expected, but no error returned")
	}
}

func testDowngradeCanary(t *testing.T, clientVersion, serverVersion uint16) error {
	defer func() { testingOnlyForceDowngradeCanary = false }()
	testingOnlyForceDowngradeCanary = true

	clientConfig := testConfig.Clone()
	clientConfig.MaxVersion = clientVersion
	serverConfig := testConfig.Clone()
	serverConfig.MaxVersion = serverVersion
	_, _, err := testHandshake(t, clientConfig, serverConfig)
	return err
}

func TestDowngradeCanary(t *testing.T) {
	if err := testDowngradeCanary(t, VersionTLS13, VersionTLS12); err == nil {
		t.Errorf("downgrade from TLS 1.3 to TLS 1.2 was not detected")
	}
	if testing.Short() {
		t.Skip("skipping the rest of the checks in short mode")
	}
	if err := testDowngradeCanary(t, VersionTLS13, VersionTLS11); err == nil {
		t.Errorf("downgrade from TLS 1.3 to TLS 1.1 was not detected")
	}
	if err := testDowngradeCanary(t, VersionTLS13, VersionTLS10); err == nil {
		t.Errorf("downgrade from TLS 1.3 to TLS 1.0 was not detected")
	}
	if err := testDowngradeCanary(t, VersionTLS12, VersionTLS11); err == nil {
		t.Errorf("downgrade from TLS 1.2 to TLS 1.1 was not detected")
	}
	if err := testDowngradeCanary(t, VersionTLS12, VersionTLS10); err == nil {
		t.Errorf("downgrade from TLS 1.2 to TLS 1.0 was not detected")
	}
	if err := testDowngradeCanary(t, VersionTLS13, VersionTLS13); err != nil {
		t.Errorf("server unexpectedly sent downgrade canary for TLS 1.3")
	}
	if err := testDowngradeCanary(t, VersionTLS12, VersionTLS12); err != nil {
		t.Errorf("client didn't ignore expected TLS 1.2 canary")
	}
	if !fips140tls.Required() {
		if err := testDowngradeCanary(t, VersionTLS11, VersionTLS11); err != nil {
			t.Errorf("client unexpectedly reacted to a canary in TLS 1.1")
		}
		if err := testDowngradeCanary(t, VersionTLS10, VersionTLS10); err != nil {
			t.Errorf("client unexpectedly reacted to a canary in TLS 1.0")
		}
	} else {
		t.Logf("skiping TLS 1.1 and TLS 1.0 downgrade canary checks in FIPS mode")
	}
}

func TestResumptionKeepsOCSPAndSCT(t *testing.T) {
	t.Run("TLSv12", func(t *testing.T) { testResumptionKeepsOCSPAndSCT(t, VersionTLS12) })
	t.Run("TLSv13", func(t *testing.T) { testResumptionKeepsOCSPAndSCT(t, VersionTLS13) })
}

func testResumptionKeepsOCSPAndSCT(t *testing.T, ver uint16) {
	// Note: using RSA 2048 test certificates because they are compatible with FIPS mode.
	issuer, err := x509.ParseCertificate(testRSA2048CertificateIssuer)
	if err != nil {
		t.Fatalf("failed to parse test issuer")
	}
	roots := x509.NewCertPool()
	roots.AddCert(issuer)
	clientConfig := &Config{
		MaxVersion:         ver,
		ClientSessionCache: NewLRUClientSessionCache(32),
		ServerName:         "example.golang",
		RootCAs:            roots,
		Time:               testTime,
	}
	serverConfig := testConfig.Clone()
	serverConfig.Certificates = []Certificate{{Certificate: [][]byte{testRSA2048Certificate}, PrivateKey: testRSA2048PrivateKey}}
	serverConfig.MaxVersion = ver
	serverConfig.Certificates[0].OCSPStaple = []byte{1, 2, 3}
	serverConfig.Certificates[0].SignedCertificateTimestamps = [][]byte{{4, 5, 6}}

	_, ccs, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	// after a new session we expect to see OCSPResponse and
	// SignedCertificateTimestamps populated as usual
	if !bytes.Equal(ccs.OCSPResponse, serverConfig.Certificates[0].OCSPStaple) {
		t.Errorf("client ConnectionState contained unexpected OCSPResponse: wanted %v, got %v",
			serverConfig.Certificates[0].OCSPStaple, ccs.OCSPResponse)
	}
	if !reflect.DeepEqual(ccs.SignedCertificateTimestamps, serverConfig.Certificates[0].SignedCertificateTimestamps) {
		t.Errorf("client ConnectionState contained unexpected SignedCertificateTimestamps: wanted %v, got %v",
			serverConfig.Certificates[0].SignedCertificateTimestamps, ccs.SignedCertificateTimestamps)
	}

	// if the server doesn't send any SCTs, repopulate the old SCTs
	oldSCTs := serverConfig.Certificates[0].SignedCertificateTimestamps
	serverConfig.Certificates[0].SignedCertificateTimestamps = nil
	_, ccs, err = testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if !ccs.DidResume {
		t.Fatalf("expected session to be resumed")
	}
	// after a resumed session we also expect to see OCSPResponse
	// and SignedCertificateTimestamps populated
	if !bytes.Equal(ccs.OCSPResponse, serverConfig.Certificates[0].OCSPStaple) {
		t.Errorf("client ConnectionState contained unexpected OCSPResponse after resumption: wanted %v, got %v",
			serverConfig.Certificates[0].OCSPStaple, ccs.OCSPResponse)
	}
	if !reflect.DeepEqual(ccs.SignedCertificateTimestamps, oldSCTs) {
		t.Errorf("client ConnectionState contained unexpected SignedCertificateTimestamps after resumption: wanted %v, got %v",
			oldSCTs, ccs.SignedCertificateTimestamps)
	}

	//  Only test overriding the SCTs for TLS 1.2, since in 1.3
	// the server won't send the message containing them
	if ver == VersionTLS13 {
		return
	}

	// if the server changes the SCTs it sends, they should override the saved SCTs
	serverConfig.Certificates[0].SignedCertificateTimestamps = [][]byte{{7, 8, 9}}
	_, ccs, err = testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
	if !ccs.DidResume {
		t.Fatalf("expected session to be resumed")
	}
	if !reflect.DeepEqual(ccs.SignedCertificateTimestamps, serverConfig.Certificates[0].SignedCertificateTimestamps) {
		t.Errorf("client ConnectionState contained unexpected SignedCertificateTimestamps after resumption: wanted %v, got %v",
			serverConfig.Certificates[0].SignedCertificateTimestamps, ccs.SignedCertificateTimestamps)
	}
}

// TestClientHandshakeContextCancellation tests that canceling
// the context given to the client side conn.HandshakeContext
// interrupts the in-progress handshake.
func TestClientHandshakeContextCancellation(t *testing.T) {
	c, s := localPipe(t)
	ctx, cancel := context.WithCancel(context.Background())
	unblockServer := make(chan struct{})
	defer close(unblockServer)
	go func() {
		cancel()
		<-unblockServer
		_ = s.Close()
	}()
	cli := Client(c, testConfig)
	// Initiates client side handshake, which will block until the client hello is read
	// by the server, unless the cancellation works.
	err := cli.HandshakeContext(ctx)
	if err == nil {
		t.Fatal("Client handshake did not error when the context was canceled")
	}
	if err != context.Canceled {
		t.Errorf("Unexpected client handshake error: %v", err)
	}
	if runtime.GOOS == "js" || runtime.GOOS == "wasip1" {
		t.Skip("conn.Close does not error as expected when called multiple times on GOOS=js or GOOS=wasip1")
	}
	err = cli.Close()
	if err == nil {
		t.Error("Client connection was not closed when the context was canceled")
	}
}

// TestTLS13OnlyClientHelloCipherSuite tests that when a client states that
// it only supports TLS 1.3, it correctly advertises only TLS 1.3 ciphers.
func TestTLS13OnlyClientHelloCipherSuite(t *testing.T) {
	tls13Tests := []struct {
		name    string
		ciphers []uint16
	}{
		{
			name:    "nil",
			ciphers: nil,
		},
		{
			name:    "empty",
			ciphers: []uint16{},
		},
		{
			name:    "some TLS 1.2 cipher",
			ciphers: []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
		},
		{
			name:    "some TLS 1.3 cipher",
			ciphers: []uint16{TLS_AES_128_GCM_SHA256},
		},
		{
			name:    "some TLS 1.2 and 1.3 ciphers",
			ciphers: []uint16{TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384, TLS_AES_256_GCM_SHA384},
		},
	}
	for _, tt := range tls13Tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			testTLS13OnlyClientHelloCipherSuite(t, tt.ciphers)
		})
	}
}

func testTLS13OnlyClientHelloCipherSuite(t *testing.T, ciphers []uint16) {
	serverConfig := &Config{
		Certificates: testConfig.Certificates,
		GetConfigForClient: func(chi *ClientHelloInfo) (*Config, error) {
			expectedCiphersuites := defaultCipherSuitesTLS13NoAES
			if fips140tls.Required() {
				expectedCiphersuites = allowedCipherSuitesTLS13FIPS
			}
			if len(chi.CipherSuites) != len(expectedCiphersuites) {
				t.Errorf("only TLS 1.3 suites should be advertised, got=%x", chi.CipherSuites)
			} else {
				for i := range expectedCiphersuites {
					if want, got := expectedCiphersuites[i], chi.CipherSuites[i]; want != got {
						t.Errorf("cipher at index %d does not match, want=%x, got=%x", i, want, got)
					}
				}
			}
			return nil, nil
		},
	}
	clientConfig := &Config{
		MinVersion:         VersionTLS13, // client only supports TLS 1.3
		CipherSuites:       ciphers,
		InsecureSkipVerify: true,
	}
	if _, _, err := testHandshake(t, clientConfig, serverConfig); err != nil {
		t.Fatalf("handshake failed: %s", err)
	}
}

// discardConn wraps a net.Conn but discards all writes, but reports that they happened.
type discardConn struct {
	net.Conn
}

func (dc *discardConn) Write(data []byte) (int, error) {
	return len(data), nil
}

// largeRSAKeyCertPEM contains a 8193 bit RSA key
const largeRSAKeyCertPEM = `-----BEGIN CERTIFICATE-----
MIIInjCCBIWgAwIBAgIBAjANBgkqhkiG9w0BAQsFADASMRAwDgYDVQQDEwd0ZXN0
aW5nMB4XDTIzMDYwNzIxMjMzNloXDTIzMDYwNzIzMjMzNlowEjEQMA4GA1UEAxMH
dGVzdGluZzCCBCIwDQYJKoZIhvcNAQEBBQADggQPADCCBAoCggQBAWdHsf6Rh2Ca
n2SQwn4t4OQrOjbLLdGE1pM6TBKKrHUFy62uEL8atNjlcfXIsa4aEu3xNGiqxqur
ZectlkZbm0FkaaQ1Wr9oikDY3KfjuaXdPdO/XC/h8AKNxlDOylyXwUSK/CuYb+1j
gy8yF5QFvVfwW/xwTlHmhUeSkVSQPosfQ6yXNNsmMzkd+ZPWLrfq4R+wiNtwYGu0
WSBcI/M9o8/vrNLnIppoiBJJ13j9CR1ToEAzOFh9wwRWLY10oZhoh1ONN1KQURx4
qedzvvP2DSjZbUccdvl2rBGvZpzfOiFdm1FCnxB0c72Cqx+GTHXBFf8bsa7KHky9
sNO1GUanbq17WoDNgwbY6H51bfShqv0CErxatwWox3we4EcAmFHPVTCYL1oWVMGo
a3Eth91NZj+b/nGhF9lhHKGzXSv9brmLLkfvM1jA6XhNhA7BQ5Vz67lj2j3XfXdh
t/BU5pBXbL4Ut4mIhT1YnKXAjX2/LF5RHQTE8Vwkx5JAEKZyUEGOReD/B+7GOrLp
HduMT9vZAc5aR2k9I8qq1zBAzsL69lyQNAPaDYd1BIAjUety9gAYaSQffCgAgpRO
Gt+DYvxS+7AT/yEd5h74MU2AH7KrAkbXOtlwupiGwhMVTstncDJWXMJqbBhyHPF8
3UmZH0hbL4PYmzSj9LDWQQXI2tv6vrCpfts3Cqhqxz9vRpgY7t1Wu6l/r+KxYYz3
1pcGpPvRmPh0DJm7cPTiXqPnZcPt+ulSaSdlxmd19OnvG5awp0fXhxryZVwuiT8G
VDkhyARrxYrdjlINsZJZbQjO0t8ketXAELJOnbFXXzeCOosyOHkLwsqOO96AVJA8
45ZVL5m95ClGy0RSrjVIkXsxTAMVG6SPAqKwk6vmTdRGuSPS4rhgckPVDHmccmuq
dfnT2YkX+wB2/M3oCgU+s30fAHGkbGZ0pCdNbFYFZLiH0iiMbTDl/0L/z7IdK0nH
GLHVE7apPraKC6xl6rPWsD2iSfrmtIPQa0+rqbIVvKP5JdfJ8J4alI+OxFw/znQe
V0/Rez0j22Fe119LZFFSXhRv+ZSvcq20xDwh00mzcumPWpYuCVPozA18yIhC9tNn
ALHndz0tDseIdy9vC71jQWy9iwri3ueN0DekMMF8JGzI1Z6BAFzgyAx3DkHtwHg7
B7qD0jPG5hJ5+yt323fYgJsuEAYoZ8/jzZ01pkX8bt+UsVN0DGnSGsI2ktnIIk3J
l+8krjmUy6EaW79nITwoOqaeHOIp8m3UkjEcoKOYrzHRKqRy+A09rY+m/cAQaafW
4xp0Zv7qZPLwnu0jsqB4jD8Ll9yPB02ndsoV6U5PeHzTkVhPml19jKUAwFfs7TJg
kXy+/xFhYVUCAwEAATANBgkqhkiG9w0BAQsFAAOCBAIAAQnZY77pMNeypfpba2WK
aDasT7dk2JqP0eukJCVPTN24Zca+xJNPdzuBATm/8SdZK9lddIbjSnWRsKvTnO2r
/rYdlPf3jM5uuJtb8+Uwwe1s+gszelGS9G/lzzq+ehWicRIq2PFcs8o3iQMfENiv
qILJ+xjcrvms5ZPDNahWkfRx3KCg8Q+/at2n5p7XYjMPYiLKHnDC+RE2b1qT20IZ
FhuK/fTWLmKbfYFNNga6GC4qcaZJ7x0pbm4SDTYp0tkhzcHzwKhidfNB5J2vNz6l
Ur6wiYwamFTLqcOwWo7rdvI+sSn05WQBv0QZlzFX+OAu0l7WQ7yU+noOxBhjvHds
14+r9qcQZg2q9kG+evopYZqYXRUNNlZKo9MRBXhfrISulFAc5lRFQIXMXnglvAu+
Ipz2gomEAOcOPNNVldhKAU94GAMJd/KfN0ZP7gX3YvPzuYU6XDhag5RTohXLm18w
5AF+ES3DOQ6ixu3DTf0D+6qrDuK+prdX8ivcdTQVNOQ+MIZeGSc6NWWOTaMGJ3lg
aZIxJUGdo6E7GBGiC1YTjgFKFbHzek1LRTh/LX3vbSudxwaG0HQxwsU9T4DWiMqa
Fkf2KteLEUA6HrR+0XlAZrhwoqAmrJ+8lCFX3V0gE9lpENfVHlFXDGyx10DpTB28
DdjnY3F7EPWNzwf9P3oNT69CKW3Bk6VVr3ROOJtDxVu1ioWo3TaXltQ0VOnap2Pu
sa5wfrpfwBDuAS9JCDg4ttNp2nW3F7tgXC6xPqw5pvGwUppEw9XNrqV8TZrxduuv
rQ3NyZ7KSzIpmFlD3UwV/fGfz3UQmHS6Ng1evrUID9DjfYNfRqSGIGjDfxGtYD+j
Z1gLJZuhjJpNtwBkKRtlNtrCWCJK2hidK/foxwD7kwAPo2I9FjpltxCRywZUs07X
KwXTfBR9v6ij1LV6K58hFS+8ezZyZ05CeVBFkMQdclTOSfuPxlMkQOtjp8QWDj+F
j/MYziT5KBkHvcbrjdRtUJIAi4N7zCsPZtjik918AK1WBNRVqPbrgq/XSEXMfuvs
6JbfK0B76vdBDRtJFC1JsvnIrGbUztxXzyQwFLaR/AjVJqpVlysLWzPKWVX6/+SJ
u1NQOl2E8P6ycyBsuGnO89p0S4F8cMRcI2X1XQsZ7/q0NBrOMaEp5T3SrWo9GiQ3
o2SBdbs3Y6MBPBtTu977Z/0RO63J3M5i2tjUiDfrFy7+VRLKr7qQ7JibohyB8QaR
9tedgjn2f+of7PnP/PEl1cCphUZeHM7QKUMPT8dbqwmKtlYY43EHXcvNOT5IBk3X
9lwJoZk/B2i+ZMRNSP34ztAwtxmasPt6RAWGQpWCn9qmttAHAnMfDqe7F7jVR6rS
u58=
-----END CERTIFICATE-----`

func TestHandshakeRSATooBig(t *testing.T) {
	testCert, _ := pem.Decode([]byte(largeRSAKeyCertPEM))

	c := &Conn{conn: &discardConn{}, config: testConfig.Clone()}

	expectedErr := "tls: server sent certificate containing RSA key larger than 8192 bits"
	err := c.verifyServerCertificate([][]byte{testCert.Bytes})
	if err == nil || err.Error() != expectedErr {
		t.Errorf("Conn.verifyServerCertificate unexpected error: want %q, got %q", expectedErr, err)
	}

	expectedErr = "tls: client sent certificate containing RSA key larger than 8192 bits"
	err = c.processCertsFromClient(Certificate{Certificate: [][]byte{testCert.Bytes}})
	if err == nil || err.Error() != expectedErr {
		t.Errorf("Conn.processCertsFromClient unexpected error: want %q, got %q", expectedErr, err)
	}
}

func TestTLS13ECHRejectionCallbacks(t *testing.T) {
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "test"},
		DNSNames:     []string{"example.golang"},
		NotBefore:    testConfig.Time().Add(-time.Hour),
		NotAfter:     testConfig.Time().Add(time.Hour),
	}
	certDER, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, k.Public(), k)
	if err != nil {
		t.Fatal(err)
	}
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		t.Fatal(err)
	}

	clientConfig, serverConfig := testConfig.Clone(), testConfig.Clone()
	serverConfig.Certificates = []Certificate{
		{
			Certificate: [][]byte{certDER},
			PrivateKey:  k,
		},
	}
	serverConfig.MinVersion = VersionTLS13
	clientConfig.RootCAs = x509.NewCertPool()
	clientConfig.RootCAs.AddCert(cert)
	clientConfig.MinVersion = VersionTLS13
	clientConfig.EncryptedClientHelloConfigList, _ = hex.DecodeString("0041fe0d003d0100200020204bed0a11fc0dde595a9b78d966b0011128eb83f65d3c91c1cc5ac786cd246f000400010001ff0e6578616d706c652e676f6c616e670000")
	clientConfig.ServerName = "example.golang"

	for _, tc := range []struct {
		name        string
		expectedErr string

		verifyConnection                    func(ConnectionState) error
		verifyPeerCertificate               func([][]byte, [][]*x509.Certificate) error
		encryptedClientHelloRejectionVerify func(ConnectionState) error
	}{
		{
			name:        "no callbacks",
			expectedErr: "tls: server rejected ECH",
		},
		{
			name: "EncryptedClientHelloRejectionVerify, no err",
			encryptedClientHelloRejectionVerify: func(ConnectionState) error {
				return nil
			},
			expectedErr: "tls: server rejected ECH",
		},
		{
			name: "EncryptedClientHelloRejectionVerify, err",
			encryptedClientHelloRejectionVerify: func(ConnectionState) error {
				return errors.New("callback err")
			},
			// testHandshake returns the server side error, so we just need to
			// check alertBadCertificate was sent
			expectedErr: "callback err",
		},
		{
			name: "VerifyConnection, err",
			verifyConnection: func(ConnectionState) error {
				return errors.New("callback err")
			},
			expectedErr: "tls: server rejected ECH",
		},
		{
			name: "VerifyPeerCertificate, err",
			verifyPeerCertificate: func([][]byte, [][]*x509.Certificate) error {
				return errors.New("callback err")
			},
			expectedErr: "tls: server rejected ECH",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, s := localPipe(t)
			done := make(chan error)

			go func() {
				serverErr := Server(s, serverConfig).Handshake()
				s.Close()
				done <- serverErr
			}()

			cConfig := clientConfig.Clone()
			cConfig.VerifyConnection = tc.verifyConnection
			cConfig.VerifyPeerCertificate = tc.verifyPeerCertificate
			cConfig.EncryptedClientHelloRejectionVerify = tc.encryptedClientHelloRejectionVerify

			clientErr := Client(c, cConfig).Handshake()
			c.Close()

			if tc.expectedErr == "" && clientErr != nil {
				t.Fatalf("unexpected err: %s", clientErr)
			} else if clientErr != nil && tc.expectedErr != clientErr.Error() {
				t.Fatalf("unexpected err: got %q, want %q", clientErr, tc.expectedErr)
			}
		})
	}
}

func TestECHTLS12Server(t *testing.T) {
	clientConfig, serverConfig := testConfig.Clone(), testConfig.Clone()

	serverConfig.MaxVersion = VersionTLS12
	clientConfig.MinVersion = 0

	clientConfig.EncryptedClientHelloConfigList, _ = hex.DecodeString("0041fe0d003d0100200020204bed0a11fc0dde595a9b78d966b0011128eb83f65d3c91c1cc5ac786cd246f000400010001ff0e6578616d706c652e676f6c616e670000")

	expectedErr := "server: tls: client offered only unsupported versions: [304]\nclient: remote error: tls: protocol version not supported"
	_, _, err := testHandshake(t, clientConfig, serverConfig)
	if err == nil || err.Error() != expectedErr {
		t.Fatalf("unexpected handshake error: got %q, want %q", err, expectedErr)
	}
}
