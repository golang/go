// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bufio"
	"bytes"
	"context"
	"crypto/internal/boring"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"internal/testenv"
	"io"
	"net"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/cryptotest"
	"time"
)

// TLS reference tests run a connection against a reference implementation
// (OpenSSL) of TLS and record the bytes of the resulting connection. The Go
// code, during a test, is configured with deterministic randomness and so the
// reference test can be reproduced exactly in the future.
//
// In order to save everyone who wishes to run the tests from needing the
// reference implementation installed, the reference connections are saved in
// files in the testdata directory. Thus running the tests involves nothing
// external, but creating and updating them requires the reference
// implementation.
//
// Tests can be updated by running them with the -update flag. This will cause
// the test files for failing tests to be regenerated. Since the reference
// implementation will always generate fresh random numbers, large parts of the
// reference connection will always change.

var (
	update       = flag.Bool("update", false, "update golden files on failure")
	keyFile      = flag.String("keylog", "", "destination file for KeyLogWriter")
	bogoMode     = flag.Bool("bogo-mode", false, "Enabled bogo shim mode, ignore everything else")
	bogoFilter   = flag.String("bogo-filter", "", "BoGo test filter")
	bogoLocalDir = flag.String("bogo-local-dir", "",
		"If not-present, checkout BoGo into this dir, or otherwise use it as a pre-existing checkout")
	bogoReport = flag.String("bogo-html-report", "", "File path to render an HTML report with BoGo results")
)

func runTestAndUpdateIfNeeded(t *testing.T, name string, run func(t *testing.T, update bool)) {
	skipFIPS(t) // FIPS 140-3 mode changes the advertised parameters.

	// Go+BoringCrypto's boring.RandReader ignores the testing override set by
	// cryptotest.SetGlobalRandom, so e.g. ECDH key generation would be
	// non-deterministic. Setting cryptocustomrand=1 makes rand.CustomReader
	// pass the caller's reader (the testing source) through instead.
	if boring.Enabled {
		testenv.SetGODEBUG(t, "cryptocustomrand=1")
	}

	success := t.Run(name, func(t *testing.T) {
		cryptotest.SetGlobalRandom(t, 0)
		run(t, false)
	})

	if !success && *update {
		t.Run(name+"#update", func(t *testing.T) {
			cryptotest.SetGlobalRandom(t, 0)
			run(t, true)
		})
	}
}

// checkOpenSSLVersion ensures that the version of OpenSSL looks reasonable
// before updating the test data.
func checkOpenSSLVersion() error {
	if !*update {
		return nil
	}

	openssl := exec.Command("openssl", "version")
	output, err := openssl.CombinedOutput()
	if err != nil {
		return err
	}

	version := string(output)
	if strings.HasPrefix(version, "OpenSSL 1.1.1") {
		return nil
	}

	println("***********************************************")
	println("")
	println("You need to build OpenSSL 1.1.1 from source in order")
	println("to update the test data.")
	println("")
	println("Configure it with:")
	println("./Configure enable-weak-ssl-ciphers no-shared")
	println("and then add the apps/ directory at the front of your PATH.")
	println("***********************************************")

	return errors.New("version of OpenSSL does not appear to be suitable for updating test data")
}

// recordingConn is a net.Conn that records the traffic that passes through it.
// WriteTo can be used to produce output that can be later be loaded with
// ParseTestData.
type recordingConn struct {
	net.Conn
	sync.Mutex
	flows   [][]byte
	reading bool
}

func (r *recordingConn) Read(b []byte) (n int, err error) {
	if n, err = r.Conn.Read(b); n == 0 {
		return
	}
	b = b[:n]

	r.Lock()
	defer r.Unlock()

	if l := len(r.flows); l == 0 || !r.reading {
		buf := make([]byte, len(b))
		copy(buf, b)
		r.flows = append(r.flows, buf)
	} else {
		r.flows[l-1] = append(r.flows[l-1], b[:n]...)
	}
	r.reading = true
	return
}

func (r *recordingConn) Write(b []byte) (n int, err error) {
	if n, err = r.Conn.Write(b); n == 0 {
		return
	}
	b = b[:n]

	r.Lock()
	defer r.Unlock()

	if l := len(r.flows); l == 0 || r.reading {
		buf := make([]byte, len(b))
		copy(buf, b)
		r.flows = append(r.flows, buf)
	} else {
		r.flows[l-1] = append(r.flows[l-1], b[:n]...)
	}
	r.reading = false
	return
}

// WriteTo writes Go source code to w that contains the recorded traffic.
func (r *recordingConn) WriteTo(w io.Writer) (int64, error) {
	// TLS always starts with a client to server flow.
	clientToServer := true
	var written int64
	for i, flow := range r.flows {
		source, dest := "client", "server"
		if !clientToServer {
			source, dest = dest, source
		}
		n, err := fmt.Fprintf(w, ">>> Flow %d (%s to %s)\n", i+1, source, dest)
		written += int64(n)
		if err != nil {
			return written, err
		}
		dumper := hex.Dumper(w)
		n, err = dumper.Write(flow)
		written += int64(n)
		if err != nil {
			return written, err
		}
		err = dumper.Close()
		if err != nil {
			return written, err
		}
		clientToServer = !clientToServer
	}
	return written, nil
}

func parseTestData(r io.Reader) (flows [][]byte, err error) {
	var currentFlow []byte

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		// If the line starts with ">>> " then it marks the beginning
		// of a new flow.
		if strings.HasPrefix(line, ">>> ") {
			if len(currentFlow) > 0 || len(flows) > 0 {
				flows = append(flows, currentFlow)
				currentFlow = nil
			}
			continue
		}

		// Otherwise the line is a line of hex dump that looks like:
		// 00000170  fc f5 06 bf (...)  |.....X{&?......!|
		// (Some bytes have been omitted from the middle section.)
		_, after, ok := strings.Cut(line, " ")
		if !ok {
			return nil, errors.New("invalid test data")
		}
		line = after

		before, _, ok := strings.Cut(line, "|")
		if !ok {
			return nil, errors.New("invalid test data")
		}
		line = before

		hexBytes := strings.Fields(line)
		for _, hexByte := range hexBytes {
			val, err := strconv.ParseUint(hexByte, 16, 8)
			if err != nil {
				return nil, errors.New("invalid hex byte in test data: " + err.Error())
			}
			currentFlow = append(currentFlow, byte(val))
		}
	}

	if len(currentFlow) > 0 {
		flows = append(flows, currentFlow)
	}

	return flows, nil
}

// replayingConn is a net.Conn that replays flows recorded by recordingConn.
type replayingConn struct {
	t testing.TB
	sync.Mutex
	flows   [][]byte
	reading bool
}

var _ net.Conn = (*replayingConn)(nil)

func (r *replayingConn) Read(b []byte) (n int, err error) {
	r.Lock()
	defer r.Unlock()

	if !r.reading {
		r.t.Errorf("expected write, got read")
		return 0, fmt.Errorf("recording expected write, got read")
	}

	n = copy(b, r.flows[0])
	r.flows[0] = r.flows[0][n:]
	if len(r.flows[0]) == 0 {
		r.flows = r.flows[1:]
		if len(r.flows) == 0 {
			return n, io.EOF
		} else {
			r.reading = false
		}
	}
	return n, nil
}

func (r *replayingConn) Write(b []byte) (n int, err error) {
	r.Lock()
	defer r.Unlock()

	if r.reading {
		r.t.Errorf("expected read, got write")
		return 0, fmt.Errorf("recording expected read, got write")
	}

	if !bytes.HasPrefix(r.flows[0], b) {
		r.t.Errorf("write mismatch: expected %x, got %x", r.flows[0], b)
		return 0, fmt.Errorf("write mismatch")
	}
	r.flows[0] = r.flows[0][len(b):]
	if len(r.flows[0]) == 0 {
		r.flows = r.flows[1:]
		r.reading = true
	}
	return len(b), nil
}

func (r *replayingConn) Close() error {
	r.Lock()
	defer r.Unlock()

	if len(r.flows) > 0 {
		r.t.Errorf("closed with unfinished flows")
		return fmt.Errorf("unexpected close")
	}
	return nil
}

func (r *replayingConn) LocalAddr() net.Addr                { return nil }
func (r *replayingConn) RemoteAddr() net.Addr               { return nil }
func (r *replayingConn) SetDeadline(t time.Time) error      { return nil }
func (r *replayingConn) SetReadDeadline(t time.Time) error  { return nil }
func (r *replayingConn) SetWriteDeadline(t time.Time) error { return nil }

// tempFile creates a temp file containing contents and returns its path.
func tempFile(contents string) string {
	file, err := os.CreateTemp("", "go-tls-test")
	if err != nil {
		panic("failed to create temp file: " + err.Error())
	}
	path := file.Name()
	file.WriteString(contents)
	file.Close()
	return path
}

// localListener is set up by TestMain and used by localPipe to create Conn
// pairs like net.Pipe, but connected by an actual buffered TCP connection.
var localListener struct {
	mu   sync.Mutex
	addr net.Addr
	ch   chan net.Conn
}

const localFlakes = 0 // change to 1 or 2 to exercise localServer/localPipe handling of mismatches

func localServer(l net.Listener) {
	for n := 0; ; n++ {
		c, err := l.Accept()
		if err != nil {
			return
		}
		if localFlakes == 1 && n%2 == 0 {
			c.Close()
			continue
		}
		localListener.ch <- c
	}
}

var isConnRefused = func(err error) bool { return false }

func localPipe(t testing.TB) (net.Conn, net.Conn) {
	localListener.mu.Lock()
	defer localListener.mu.Unlock()

	addr := localListener.addr

	var err error
Dialing:
	// We expect a rare mismatch, but probably not 5 in a row.
	for i := 0; i < 5; i++ {
		tooSlow := time.NewTimer(1 * time.Second)
		defer tooSlow.Stop()
		var c1 net.Conn
		c1, err = net.Dial(addr.Network(), addr.String())
		if err != nil {
			if runtime.GOOS == "dragonfly" && (isConnRefused(err) || os.IsTimeout(err)) {
				// golang.org/issue/29583: Dragonfly sometimes returns a spurious
				// ECONNREFUSED or ETIMEDOUT.
				<-tooSlow.C
				continue
			}
			t.Fatalf("localPipe: %v", err)
		}
		if localFlakes == 2 && i == 0 {
			c1.Close()
			continue
		}
		for {
			select {
			case <-tooSlow.C:
				t.Logf("localPipe: timeout waiting for %v", c1.LocalAddr())
				c1.Close()
				continue Dialing

			case c2 := <-localListener.ch:
				if c2.RemoteAddr().String() == c1.LocalAddr().String() {
					t.Cleanup(func() { c1.Close() })
					t.Cleanup(func() { c2.Close() })
					return c1, c2
				}
				t.Logf("localPipe: unexpected connection: %v != %v", c2.RemoteAddr(), c1.LocalAddr())
				c2.Close()
			}
		}
	}

	t.Fatalf("localPipe: failed to connect: %v", err)
	panic("unreachable")
}

func TestMain(m *testing.M) {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage of %s:\n", os.Args)
		flag.PrintDefaults()
		if *bogoMode {
			os.Exit(89)
		}
	}

	flag.Parse()

	if *bogoMode {
		bogoShim()
		os.Exit(0)
	}

	os.Exit(runMain(m))
}

func runMain(m *testing.M) int {
	// Cipher suites preferences change based on the architecture. Force them to
	// the version without AES acceleration for test consistency.
	hasAESGCMHardwareSupport = false

	// Set up localPipe.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		l, err = net.Listen("tcp6", "[::1]:0")
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open local listener: %v", err)
		os.Exit(1)
	}
	localListener.ch = make(chan net.Conn)
	localListener.addr = l.Addr()
	defer l.Close()
	go localServer(l)

	if err := checkOpenSSLVersion(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v", err)
		os.Exit(1)
	}

	rootCAPath := tempFile(testRootCertPEM)
	defer os.Remove(rootCAPath)
	defaultClientCommand = []string{"openssl", "s_client", "-no_ticket",
		"-verify", "1", "-verify_return_error", "-CAfile", rootCAPath,
		"-servername", "test.golang.example", "-attime", fmt.Sprint(testTime().Unix())}

	clientRootCAPath := tempFile(testClientRootCertPEM)
	defer os.Remove(clientRootCAPath)
	serverCommand = []string{"openssl", "s_server", "-no_ticket", "-num_tickets", "0",
		"-naccept", "1", "-verify_return_error", "-verifyCAfile", clientRootCAPath,
		"-attime", fmt.Sprint(testTime().Unix())}

	if *keyFile != "" {
		f, err := os.OpenFile(*keyFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			panic("failed to open -keylog file: " + err.Error())
		}
		testConfigClient.KeyLogWriter = f
		testConfigServer.KeyLogWriter = f
		testConfigFIPS140.KeyLogWriter = f
		defer f.Close()
	}

	return m.Run()
}

func testHandshake(t *testing.T, clientConfig, serverConfig *Config) (serverState, clientState ConnectionState, err error) {
	const sentinel = "SENTINEL\n"
	c, s := localPipe(t)
	errChan := make(chan error, 1)
	go func() {
		cli := Client(c, clientConfig)
		err := cli.Handshake()
		if err != nil {
			errChan <- fmt.Errorf("client: %v", err)
			c.Close()
			return
		}
		clientState = cli.ConnectionState()
		buf, err := io.ReadAll(cli)
		if err != nil {
			if serverConfig.ClientAuth != NoClientCert && clientState.Version == VersionTLS13 {
				// In TLS 1.3, client certificates are sent after the server's
				// handshake has completed, and the client only learns about it
				// reading the alert after the handshake.
				errChan <- fmt.Errorf("client (from Read): %v", err)
				c.Close()
				return
			} else {
				t.Errorf("failed to call cli.Read: %v", err)
			}
		}
		defer func() { errChan <- nil }()
		if got := string(buf); got != sentinel {
			t.Errorf("read %q from TLS connection, but expected %q", got, sentinel)
		}
		// We discard the error because after ReadAll returns the server must
		// have already closed the connection. Sending data (the closeNotify
		// alert) can cause a reset, that will make Close return an error.
		cli.Close()
	}()
	server := Server(s, serverConfig)
	err = server.Handshake()
	if err == nil {
		serverState = server.ConnectionState()
		if _, err := io.WriteString(server, sentinel); err != nil {
			t.Errorf("failed to call server.Write: %v", err)
		}
		if err := server.Close(); err != nil {
			t.Errorf("failed to call server.Close: %v", err)
		}
	} else {
		err = fmt.Errorf("server: %v", err)
		s.Close()
	}
	err = errors.Join(err, <-errChan)
	return
}

func fromHex(s string) []byte {
	b, _ := hex.DecodeString(s)
	return b
}

func TestServerHelloTrailingMessage(t *testing.T) {
	// In TLS 1.3 the change cipher spec message is optional. If a CCS message
	// is not sent, after reading the ServerHello, the read traffic secret is
	// set, and all following messages must be encrypted. If the server sends
	// additional unencrypted messages in a record with the ServerHello, the
	// client must either fail or ignore the additional messages.

	c, s := localPipe(t)
	go func() {
		ctx := context.Background()
		srv := Server(s, testConfigServer.Clone())
		clientHello, _, err := srv.readClientHello(ctx)
		if err != nil {
			testFatal(t, err)
		}

		hs := serverHandshakeStateTLS13{
			c:           srv,
			ctx:         ctx,
			clientHello: clientHello,
		}
		if err := hs.processClientHello(); err != nil {
			testFatal(t, err)
		}
		if err := transcriptMsg(hs.clientHello, hs.transcript); err != nil {
			testFatal(t, err)
		}

		record, err := concatHandshakeMessages(hs.hello, &encryptedExtensionsMsg{alpnProtocol: "h2"})
		if err != nil {
			testFatal(t, err)
		}

		if _, err := s.Write(record); err != nil {
			testFatal(t, err)
		}
		srv.Close()
	}()

	cli := Client(c, testConfigClient.Clone())
	expectedErr := "tls: handshake buffer not empty before setting read traffic secret"
	if err := cli.Handshake(); err == nil {
		t.Fatal("expected error from incomplete handshake, got nil")
	} else if err.Error() != expectedErr {
		t.Fatalf("expected error %q, got %q", expectedErr, err.Error())
	}
}

func TestClientHelloTrailingMessage(t *testing.T) {
	// Same as TestServerHelloTrailingMessage but for the client side.

	c, s := localPipe(t)
	go func() {
		cli := Client(c, testConfigClient.Clone())

		hello, _, _, err := cli.makeClientHello()
		if err != nil {
			testFatal(t, err)
		}

		record, err := concatHandshakeMessages(hello, &certificateMsgTLS13{})
		if err != nil {
			testFatal(t, err)
		}

		if _, err := c.Write(record); err != nil {
			testFatal(t, err)
		}
		cli.Close()
	}()

	srv := Server(s, testConfigServer.Clone())
	expectedErr := "tls: handshake buffer not empty before setting read traffic secret"
	if err := srv.Handshake(); err == nil {
		t.Fatal("expected error from incomplete handshake, got nil")
	} else if err.Error() != expectedErr {
		t.Fatalf("expected error %q, got %q", expectedErr, err.Error())
	}
}

func TestDoubleClientHelloHRR(t *testing.T) {
	// If a client sends two ClientHello messages in a single record, and the
	// server sends a HRR after reading the first ClientHello, the server must
	// either fail or ignore the trailing ClientHello.

	c, s := localPipe(t)

	go func() {
		cli := Client(c, testConfigClient.Clone())

		hello, _, _, err := cli.makeClientHello()
		if err != nil {
			testFatal(t, err)
		}
		hello.keyShares = nil

		record, err := concatHandshakeMessages(hello, hello)
		if err != nil {
			testFatal(t, err)
		}

		if _, err := c.Write(record); err != nil {
			testFatal(t, err)
		}
		cli.Close()
	}()

	srv := Server(s, testConfigServer.Clone())
	expectedErr := "tls: handshake buffer not empty before HelloRetryRequest"
	if err := srv.Handshake(); err == nil {
		t.Fatal("expected error from incomplete handshake, got nil")
	} else if err.Error() != expectedErr {
		t.Fatalf("expected error %q, got %q", expectedErr, err.Error())
	}
}

// concatHandshakeMessages marshals and concatenates the given handshake
// messages into a single record.
func concatHandshakeMessages(msgs ...handshakeMessage) ([]byte, error) {
	var marshalled []byte
	for _, msg := range msgs {
		data, err := msg.marshal()
		if err != nil {
			return nil, err
		}
		marshalled = append(marshalled, data...)
	}
	m := len(marshalled)
	outBuf := make([]byte, recordHeaderLen)
	outBuf[0] = byte(recordTypeHandshake)
	vers := VersionTLS12
	outBuf[1] = byte(vers >> 8)
	outBuf[2] = byte(vers)
	outBuf[3] = byte(m >> 8)
	outBuf[4] = byte(m)
	outBuf = append(outBuf, marshalled...)
	return outBuf, nil
}

func TestMultipleKeyUpdate(t *testing.T) {
	for _, requestUpdate := range []bool{true, false} {
		t.Run(fmt.Sprintf("requestUpdate=%t", requestUpdate), func(t *testing.T) {

			c, s := localPipe(t)
			clientConfig := testConfigClient.Clone()
			clientConfig.MinVersion = VersionTLS13
			clientConfig.MaxVersion = VersionTLS13
			serverConfig := testConfigServer.Clone()
			serverConfig.MinVersion = VersionTLS13
			serverConfig.MaxVersion = VersionTLS13
			client := Client(c, clientConfig)
			server := Server(s, serverConfig)

			clientHandshakeDone := make(chan struct{})
			go func() {
				if err := client.Handshake(); err != nil {
				}
				close(clientHandshakeDone)
				io.Copy(io.Discard, server)
			}()

			if err := server.Handshake(); err != nil {
				t.Fatalf("server handshake failed: %v\n", err)
			}
			<-clientHandshakeDone

			c.SetReadDeadline(time.Now().Add(1 * time.Second))
			s.SetReadDeadline(time.Now().Add(1 * time.Second))

			kuMsg, err := (&keyUpdateMsg{updateRequested: requestUpdate}).marshal()
			if err != nil {
				t.Fatalf("failed to marshal key update message: %v", err)
			}

			client.out.Lock()
			if _, err := client.writeRecordLocked(recordTypeHandshake, append(kuMsg, kuMsg...)); err != nil {
				t.Fatalf("failed to write key update messages: %v", err)
			}
			client.out.Unlock()

			_, err = io.Copy(io.Discard, client)
			if err == nil {
				t.Fatal("expected multiple key update messages to cause an error, got nil")
			} else if !strings.HasSuffix(err.Error(), "tls: unexpected message") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
