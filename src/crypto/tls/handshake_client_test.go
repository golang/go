// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/binary"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
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
)

const opensslSentinel = "SENTINEL\n"

type opensslInput chan opensslInputEvent

func (i opensslInput) Read(buf []byte) (n int, err error) {
	for event := range i {
		switch event {
		case opensslRenegotiate:
			return copy(buf, []byte("R\n")), nil
		case opensslSendSentinel:
			return copy(buf, []byte(opensslSentinel)), nil
		default:
			panic("unknown event")
		}
	}

	return 0, io.EOF
}

// opensslOutputSink is an io.Writer that receives the stdout and stderr from
// an `openssl` process and sends a value to handshakeComplete when it sees a
// log message from a completed server handshake.
type opensslOutputSink struct {
	handshakeComplete chan struct{}
	all               []byte
	line              []byte
}

func newOpensslOutputSink() *opensslOutputSink {
	return &opensslOutputSink{make(chan struct{}), nil, nil}
}

// opensslEndOfHandshake is a message that the “openssl s_server” tool will
// print when a handshake completes if run with “-state”.
const opensslEndOfHandshake = "SSL_accept:SSLv3/TLS write finished"

func (o *opensslOutputSink) Write(data []byte) (n int, err error) {
	o.line = append(o.line, data...)
	o.all = append(o.all, data...)

	for {
		i := bytes.Index(o.line, []byte{'\n'})
		if i < 0 {
			break
		}

		if bytes.Equal([]byte(opensslEndOfHandshake), o.line[:i]) {
			o.handshakeComplete <- struct{}{}
		}
		o.line = o.line[i+1:]
	}

	return len(data), nil
}

func (o *opensslOutputSink) WriteTo(w io.Writer) (int64, error) {
	n, err := w.Write(o.all)
	return int64(n), err
}

// clientTest represents a test of the TLS client handshake against a reference
// implementation.
type clientTest struct {
	// name is a freeform string identifying the test and the file in which
	// the expected results will be stored.
	name string
	// command, if not empty, contains a series of arguments for the
	// command to run for the reference server.
	command []string
	// config, if not nil, contains a custom Config to use for this test.
	config *Config
	// cert, if not empty, contains a DER-encoded certificate for the
	// reference server.
	cert []byte
	// key, if not nil, contains either a *rsa.PrivateKey or
	// *ecdsa.PrivateKey which is the private key for the reference server.
	key interface{}
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
}

var defaultServerCommand = []string{"openssl", "s_server"}

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

	var key interface{} = testRSAPrivateKey
	if test.key != nil {
		key = test.key
	}
	var pemType string
	var derBytes []byte
	switch key := key.(type) {
	case *rsa.PrivateKey:
		pemType = "RSA"
		derBytes = x509.MarshalPKCS1PrivateKey(key)
	case *ecdsa.PrivateKey:
		pemType = "EC"
		var err error
		derBytes, err = x509.MarshalECPrivateKey(key)
		if err != nil {
			panic(err)
		}
	default:
		panic("unknown key type")
	}

	var pemOut bytes.Buffer
	pem.Encode(&pemOut, &pem.Block{Type: pemType + " PRIVATE KEY", Bytes: derBytes})

	keyPath := tempFile(string(pemOut.Bytes()))
	defer os.Remove(keyPath)

	var command []string
	if len(test.command) > 0 {
		command = append(command, test.command...)
	} else {
		command = append(command, defaultServerCommand...)
	}
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
				Type:  fmt.Sprintf("SERVERINFO FOR EXTENSION %d", binary.BigEndian.Uint16(ext)),
				Bytes: ext,
			})
		}
		serverInfoPath := tempFile(serverInfo.String())
		defer os.Remove(serverInfoPath)
		command = append(command, "-serverinfo", serverInfoPath)
	}

	if test.numRenegotiations > 0 {
		found := false
		for _, flag := range command[1:] {
			if flag == "-state" {
				found = true
				break
			}
		}

		if !found {
			panic("-state flag missing to OpenSSL. You need this if testing renegotiation")
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
		out.WriteTo(os.Stdout)
		cmd.Process.Kill()
		return nil, nil, nil, nil, cmd.Wait()
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
	checkOpenSSLVersion(t)

	var clientConn, serverConn net.Conn
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
	} else {
		clientConn, serverConn = net.Pipe()
	}

	config := test.config
	if config == nil {
		config = testConfig
	}
	client := Client(clientConn, config)

	doneChan := make(chan bool)
	go func() {
		defer func() { doneChan <- true }()
		defer clientConn.Close()
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
				defer func() { signalChan <- struct{}{} }()

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

		if test.validate != nil {
			if err := test.validate(client.ConnectionState()); err != nil {
				t.Errorf("validate callback returned error: %s", err)
			}
		}
	}()

	if !write {
		flows, err := test.loadData()
		if err != nil {
			t.Fatalf("%s: failed to load data from %s: %v", test.name, test.dataPath(), err)
		}
		for i, b := range flows {
			if i%2 == 1 {
				serverConn.Write(b)
				continue
			}
			bb := make([]byte, len(b))
			_, err := io.ReadFull(serverConn, bb)
			if err != nil {
				t.Fatalf("%s #%d: %s", test.name, i, err)
			}
			if !bytes.Equal(b, bb) {
				t.Fatalf("%s #%d: mismatch on read: got:%x want:%x", test.name, i, bb, b)
			}
		}
		serverConn.Close()
	}

	<-doneChan

	if write {
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
			os.Stdout.Write(childProcess.Stdout.(*opensslOutputSink).all)
			t.Fatalf("Client connection didn't work")
		}
		recordingConn.WriteTo(out)
		fmt.Printf("Wrote %s\n", path)
	}
}

var (
	didParMu sync.Mutex
	didPar   = map[*testing.T]bool{}
)

// setParallel calls t.Parallel once. If you call it twice, it would
// panic.
func setParallel(t *testing.T) {
	didParMu.Lock()
	v := didPar[t]
	didPar[t] = true
	didParMu.Unlock()
	if !v {
		t.Parallel()
	}
}

func runClientTestForVersion(t *testing.T, template *clientTest, prefix, option string) {
	setParallel(t)

	test := *template
	test.name = prefix + test.name
	if len(test.command) == 0 {
		test.command = defaultClientCommand
	}
	test.command = append([]string(nil), test.command...)
	test.command = append(test.command, option)
	test.run(t, *update)
}

func runClientTestTLS10(t *testing.T, template *clientTest) {
	runClientTestForVersion(t, template, "TLSv10-", "-tls1")
}

func runClientTestTLS11(t *testing.T, template *clientTest) {
	runClientTestForVersion(t, template, "TLSv11-", "-tls1_1")
}

func runClientTestTLS12(t *testing.T, template *clientTest) {
	runClientTestForVersion(t, template, "TLSv12-", "-tls1_2")
}

func TestHandshakeClientRSARC4(t *testing.T) {
	test := &clientTest{
		name:    "RSA-RC4",
		command: []string{"openssl", "s_server", "-cipher", "RC4-SHA"},
	}
	runClientTestTLS10(t, test)
	runClientTestTLS11(t, test)
	runClientTestTLS12(t, test)
}

func TestHandshakeClientRSAAES128GCM(t *testing.T) {
	test := &clientTest{
		name:    "AES128-GCM-SHA256",
		command: []string{"openssl", "s_server", "-cipher", "AES128-GCM-SHA256"},
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientRSAAES256GCM(t *testing.T) {
	test := &clientTest{
		name:    "AES256-GCM-SHA384",
		command: []string{"openssl", "s_server", "-cipher", "AES256-GCM-SHA384"},
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHERSAAES(t *testing.T) {
	test := &clientTest{
		name:    "ECDHE-RSA-AES",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-RSA-AES128-SHA"},
	}
	runClientTestTLS10(t, test)
	runClientTestTLS11(t, test)
	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHEECDSAAES(t *testing.T) {
	test := &clientTest{
		name:    "ECDHE-ECDSA-AES",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-ECDSA-AES128-SHA"},
		cert:    testECDSACertificate,
		key:     testECDSAPrivateKey,
	}
	runClientTestTLS10(t, test)
	runClientTestTLS11(t, test)
	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHEECDSAAESGCM(t *testing.T) {
	test := &clientTest{
		name:    "ECDHE-ECDSA-AES-GCM",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-ECDSA-AES128-GCM-SHA256"},
		cert:    testECDSACertificate,
		key:     testECDSAPrivateKey,
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientAES256GCMSHA384(t *testing.T) {
	test := &clientTest{
		name:    "ECDHE-ECDSA-AES256-GCM-SHA384",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-ECDSA-AES256-GCM-SHA384"},
		cert:    testECDSACertificate,
		key:     testECDSAPrivateKey,
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientAES128CBCSHA256(t *testing.T) {
	test := &clientTest{
		name:    "AES128-SHA256",
		command: []string{"openssl", "s_server", "-cipher", "AES128-SHA256"},
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHERSAAES128CBCSHA256(t *testing.T) {
	test := &clientTest{
		name:    "ECDHE-RSA-AES128-SHA256",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-RSA-AES128-SHA256"},
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHEECDSAAES128CBCSHA256(t *testing.T) {
	test := &clientTest{
		name:    "ECDHE-ECDSA-AES128-SHA256",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-ECDSA-AES128-SHA256"},
		cert:    testECDSACertificate,
		key:     testECDSAPrivateKey,
	}
	runClientTestTLS12(t, test)
}

func TestHandshakeClientX25519(t *testing.T) {
	config := testConfig.Clone()
	config.CurvePreferences = []CurveID{X25519}

	test := &clientTest{
		name:    "X25519-ECDHE-RSA-AES-GCM",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-RSA-AES128-GCM-SHA256"},
		config:  config,
	}

	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHERSAChaCha20(t *testing.T) {
	config := testConfig.Clone()
	config.CipherSuites = []uint16{TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305}

	test := &clientTest{
		name:    "ECDHE-RSA-CHACHA20-POLY1305",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-RSA-CHACHA20-POLY1305"},
		config:  config,
	}

	runClientTestTLS12(t, test)
}

func TestHandshakeClientECDHEECDSAChaCha20(t *testing.T) {
	config := testConfig.Clone()
	config.CipherSuites = []uint16{TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305}

	test := &clientTest{
		name:    "ECDHE-ECDSA-CHACHA20-POLY1305",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-ECDSA-CHACHA20-POLY1305"},
		config:  config,
		cert:    testECDSACertificate,
		key:     testECDSAPrivateKey,
	}

	runClientTestTLS12(t, test)
}

func TestHandshakeClientCertRSA(t *testing.T) {
	config := testConfig.Clone()
	cert, _ := X509KeyPair([]byte(clientCertificatePEM), []byte(clientKeyPEM))
	config.Certificates = []Certificate{cert}

	test := &clientTest{
		name:    "ClientCert-RSA-RSA",
		command: []string{"openssl", "s_server", "-cipher", "AES128", "-verify", "1"},
		config:  config,
	}

	runClientTestTLS10(t, test)
	runClientTestTLS12(t, test)

	test = &clientTest{
		name:    "ClientCert-RSA-ECDSA",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-ECDSA-AES128-SHA", "-verify", "1"},
		config:  config,
		cert:    testECDSACertificate,
		key:     testECDSAPrivateKey,
	}

	runClientTestTLS10(t, test)
	runClientTestTLS12(t, test)

	test = &clientTest{
		name:    "ClientCert-RSA-AES256-GCM-SHA384",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-RSA-AES256-GCM-SHA384", "-verify", "1"},
		config:  config,
		cert:    testRSACertificate,
		key:     testRSAPrivateKey,
	}

	runClientTestTLS12(t, test)
}

func TestHandshakeClientCertECDSA(t *testing.T) {
	config := testConfig.Clone()
	cert, _ := X509KeyPair([]byte(clientECDSACertificatePEM), []byte(clientECDSAKeyPEM))
	config.Certificates = []Certificate{cert}

	test := &clientTest{
		name:    "ClientCert-ECDSA-RSA",
		command: []string{"openssl", "s_server", "-cipher", "AES128", "-verify", "1"},
		config:  config,
	}

	runClientTestTLS10(t, test)
	runClientTestTLS12(t, test)

	test = &clientTest{
		name:    "ClientCert-ECDSA-ECDSA",
		command: []string{"openssl", "s_server", "-cipher", "ECDHE-ECDSA-AES128-SHA", "-verify", "1"},
		config:  config,
		cert:    testECDSACertificate,
		key:     testECDSAPrivateKey,
	}

	runClientTestTLS10(t, test)
	runClientTestTLS12(t, test)
}

func TestClientResumption(t *testing.T) {
	serverConfig := &Config{
		CipherSuites: []uint16{TLS_RSA_WITH_RC4_128_SHA, TLS_ECDHE_RSA_WITH_RC4_128_SHA},
		Certificates: testConfig.Certificates,
	}

	issuer, err := x509.ParseCertificate(testRSACertificateIssuer)
	if err != nil {
		panic(err)
	}

	rootCAs := x509.NewCertPool()
	rootCAs.AddCert(issuer)

	clientConfig := &Config{
		CipherSuites:       []uint16{TLS_RSA_WITH_RC4_128_SHA},
		ClientSessionCache: NewLRUClientSessionCache(32),
		RootCAs:            rootCAs,
		ServerName:         "example.golang",
	}

	testResumeState := func(test string, didResume bool) {
		_, hs, err := testHandshake(clientConfig, serverConfig)
		if err != nil {
			t.Fatalf("%s: handshake failed: %s", test, err)
		}
		if hs.DidResume != didResume {
			t.Fatalf("%s resumed: %v, expected: %v", test, hs.DidResume, didResume)
		}
		if didResume && (hs.PeerCertificates == nil || hs.VerifiedChains == nil) {
			t.Fatalf("expected non-nil certificates after resumption. Got peerCertificates: %#v, verifiedCertificates: %#v", hs.PeerCertificates, hs.VerifiedChains)
		}
	}

	getTicket := func() []byte {
		return clientConfig.ClientSessionCache.(*lruSessionCache).q.Front().Value.(*lruSessionCacheEntry).state.sessionTicket
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
	if !bytes.Equal(ticket, getTicket()) {
		t.Fatal("first ticket doesn't match ticket after resumption")
	}

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

	// Reset serverConfig to ensure that calling SetSessionTicketKeys
	// before the serverConfig is used works.
	serverConfig = &Config{
		CipherSuites: []uint16{TLS_RSA_WITH_RC4_128_SHA, TLS_ECDHE_RSA_WITH_RC4_128_SHA},
		Certificates: testConfig.Certificates,
	}
	serverConfig.SetSessionTicketKeys([][32]byte{key2})

	testResumeState("FreshConfig", true)

	clientConfig.CipherSuites = []uint16{TLS_ECDHE_RSA_WITH_RC4_128_SHA}
	testResumeState("DifferentCipherSuite", false)
	testResumeState("DifferentCipherSuiteRecovers", true)

	clientConfig.ClientSessionCache = nil
	testResumeState("WithoutSessionCache", false)
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

	// Adding a nil entry is valid.
	cache.Put(keys[0], nil)
	if s, ok := cache.Get(keys[0]); !ok || s != nil {
		t.Fatalf("failed to add nil entry to cache")
	}
}

func TestKeyLog(t *testing.T) {
	var serverBuf, clientBuf bytes.Buffer

	clientConfig := testConfig.Clone()
	clientConfig.KeyLogWriter = &clientBuf

	serverConfig := testConfig.Clone()
	serverConfig.KeyLogWriter = &serverBuf

	c, s := net.Pipe()
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

	checkKeylogLine("client", string(clientBuf.Bytes()))
	checkKeylogLine("server", string(serverBuf.Bytes()))
}

func TestHandshakeClientALPNMatch(t *testing.T) {
	config := testConfig.Clone()
	config.NextProtos = []string{"proto2", "proto1"}

	test := &clientTest{
		name: "ALPN",
		// Note that this needs OpenSSL 1.0.2 because that is the first
		// version that supports the -alpn flag.
		command: []string{"openssl", "s_server", "-alpn", "proto1,proto2"},
		config:  config,
		validate: func(state ConnectionState) error {
			// The server's preferences should override the client.
			if state.NegotiatedProtocol != "proto1" {
				return fmt.Errorf("Got protocol %q, wanted proto1", state.NegotiatedProtocol)
			}
			return nil
		},
	}
	runClientTestTLS12(t, test)
}

// sctsBase64 contains data from `openssl s_client -serverinfo 18 -connect ritter.vg:443`
const sctsBase64 = "ABIBaQFnAHUApLkJkLQYWBSHuxOizGdwCjw1mAT5G9+443fNDsgN3BAAAAFHl5nuFgAABAMARjBEAiAcS4JdlW5nW9sElUv2zvQyPoZ6ejKrGGB03gjaBZFMLwIgc1Qbbn+hsH0RvObzhS+XZhr3iuQQJY8S9G85D9KeGPAAdgBo9pj4H2SCvjqM7rkoHUz8cVFdZ5PURNEKZ6y7T0/7xAAAAUeX4bVwAAAEAwBHMEUCIDIhFDgG2HIuADBkGuLobU5a4dlCHoJLliWJ1SYT05z6AiEAjxIoZFFPRNWMGGIjskOTMwXzQ1Wh2e7NxXE1kd1J0QsAdgDuS723dc5guuFCaR+r4Z5mow9+X7By2IMAxHuJeqj9ywAAAUhcZIqHAAAEAwBHMEUCICmJ1rBT09LpkbzxtUC+Hi7nXLR0J+2PmwLp+sJMuqK+AiEAr0NkUnEVKVhAkccIFpYDqHOlZaBsuEhWWrYpg2RtKp0="

func TestHandshakClientSCTs(t *testing.T) {
	config := testConfig.Clone()

	scts, err := base64.StdEncoding.DecodeString(sctsBase64)
	if err != nil {
		t.Fatal(err)
	}

	test := &clientTest{
		name: "SCT",
		// Note that this needs OpenSSL 1.0.2 because that is the first
		// version that supports the -serverinfo flag.
		command:    []string{"openssl", "s_server"},
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
}

func TestRenegotiationRejected(t *testing.T) {
	config := testConfig.Clone()
	test := &clientTest{
		name:                        "RenegotiationRejected",
		command:                     []string{"openssl", "s_server", "-state"},
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
		command:           []string{"openssl", "s_server", "-state"},
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
		command:           []string{"openssl", "s_server", "-state"},
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
		command:                     []string{"openssl", "s_server", "-state"},
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
		c, s := net.Pipe()

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

	c, s := net.Pipe()
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
	serverHelloBytes := serverHello.marshal()

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

func TestVerifyPeerCertificate(t *testing.T) {
	issuer, err := x509.ParseCertificate(testRSACertificateIssuer)
	if err != nil {
		panic(err)
	}

	rootCAs := x509.NewCertPool()
	rootCAs.AddCert(issuer)

	now := func() time.Time { return time.Unix(1476984729, 0) }

	sentinelErr := errors.New("TestVerifyPeerCertificate")

	verifyCallback := func(called *bool, rawCerts [][]byte, validatedChains [][]*x509.Certificate) error {
		if l := len(rawCerts); l != 1 {
			return fmt.Errorf("got len(rawCerts) = %d, wanted 1", l)
		}
		if len(validatedChains) == 0 {
			return errors.New("got len(validatedChains) = 0, wanted non-zero")
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
					return verifyCallback(called, rawCerts, validatedChains)
				}
			},
			configureClient: func(config *Config, called *bool) {
				config.InsecureSkipVerify = false
				config.VerifyPeerCertificate = func(rawCerts [][]byte, validatedChains [][]*x509.Certificate) error {
					return verifyCallback(called, rawCerts, validatedChains)
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
						return errors.New("got len(validatedChains) = 0, wanted zero")
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
	}

	for i, test := range tests {
		c, s := net.Pipe()
		done := make(chan error)

		var clientCalled, serverCalled bool

		go func() {
			config := testConfig.Clone()
			config.ServerName = "example.golang"
			config.ClientAuth = RequireAndVerifyClientCert
			config.ClientCAs = rootCAs
			config.Time = now
			test.configureServer(config, &serverCalled)

			err = Server(s, config).Handshake()
			s.Close()
			done <- err
		}()

		config := testConfig.Clone()
		config.ServerName = "example.golang"
		config.RootCAs = rootCAs
		config.Time = now
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
		c, s := net.Pipe()
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
	c, s := net.Pipe()
	done := make(chan bool)

	clientWCC := &writeCountingConn{Conn: c}
	serverWCC := &writeCountingConn{Conn: s}

	go func() {
		Server(serverWCC, testConfig).Handshake()
		serverWCC.Close()
		done <- true
	}()

	err := Client(clientWCC, testConfig).Handshake()
	if err != nil {
		t.Fatal(err)
	}
	clientWCC.Close()
	<-done

	if n := clientWCC.numWrites; n != 2 {
		t.Errorf("expected client handshake to complete with only two writes, but saw %d", n)
	}

	if n := serverWCC.numWrites; n != 2 {
		t.Errorf("expected server handshake to complete with only two writes, but saw %d", n)
	}
}

func TestAlertFlushing(t *testing.T) {
	c, s := net.Pipe()
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

	const expectedError = "remote error: tls: handshake failure"
	if e := err.Error(); !strings.Contains(e, expectedError) {
		t.Fatalf("expected to find %q in error but error was %q", expectedError, e)
	}
	clientWCC.Close()
	<-done

	if n := clientWCC.numWrites; n != 1 {
		t.Errorf("expected client handshake to complete with one write, but saw %d", n)
	}

	if n := serverWCC.numWrites; n != 1 {
		t.Errorf("expected server handshake to complete with one write, but saw %d", n)
	}
}

func TestHandshakeRace(t *testing.T) {
	t.Parallel()
	// This test races a Read and Write to try and complete a handshake in
	// order to provide some evidence that there are no races or deadlocks
	// in the handshake locking.
	for i := 0; i < 32; i++ {
		c, s := net.Pipe()

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
		readDone := make(chan struct{})

		client := Client(c, testConfig)
		go func() {
			<-startWrite
			var request [1]byte
			client.Write(request[:])
		}()

		go func() {
			<-startRead
			var reply [1]byte
			if n, err := client.Read(reply[:]); err != nil || n != 1 {
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

func TestTLS11SignatureSchemes(t *testing.T) {
	expected := tls11SignatureSchemesNumECDSA + tls11SignatureSchemesNumRSA
	if expected != len(tls11SignatureSchemes) {
		t.Errorf("expected to find %d TLS 1.1 signature schemes, but found %d", expected, len(tls11SignatureSchemes))
	}
}

var getClientCertificateTests = []struct {
	setup               func(*Config)
	expectedClientError string
	verify              func(*testing.T, int, *ConnectionState)
}{
	{
		func(clientConfig *Config) {
			// Returning a Certificate with no certificate data
			// should result in an empty message being sent to the
			// server.
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
		func(clientConfig *Config) {
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
		func(clientConfig *Config) {
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
		func(clientConfig *Config) {
			clientConfig.GetClientCertificate = func(cri *CertificateRequestInfo) (*Certificate, error) {
				return &testConfig.Certificates[0], nil
			}
		},
		"",
		func(t *testing.T, testNum int, cs *ConnectionState) {
			if l := len(cs.VerifiedChains); l != 0 {
				t.Errorf("#%d: expected some verified chains, but found none", testNum)
			}
		},
	},
}

func TestGetClientCertificate(t *testing.T) {
	issuer, err := x509.ParseCertificate(testRSACertificateIssuer)
	if err != nil {
		panic(err)
	}

	for i, test := range getClientCertificateTests {
		serverConfig := testConfig.Clone()
		serverConfig.ClientAuth = RequestClientCert
		serverConfig.RootCAs = x509.NewCertPool()
		serverConfig.RootCAs.AddCert(issuer)

		clientConfig := testConfig.Clone()

		test.setup(clientConfig)

		type serverResult struct {
			cs  ConnectionState
			err error
		}

		c, s := net.Pipe()
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
