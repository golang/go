// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ecdh"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/internal/hpke"
	"crypto/rand"
	"crypto/tls/internal/fips140tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"math"
	"math/big"
	"net"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"golang.org/x/crypto/cryptobyte"
)

var rsaCertPEM = `-----BEGIN CERTIFICATE-----
MIIB0zCCAX2gAwIBAgIJAI/M7BYjwB+uMA0GCSqGSIb3DQEBBQUAMEUxCzAJBgNV
BAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX
aWRnaXRzIFB0eSBMdGQwHhcNMTIwOTEyMjE1MjAyWhcNMTUwOTEyMjE1MjAyWjBF
MQswCQYDVQQGEwJBVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50
ZXJuZXQgV2lkZ2l0cyBQdHkgTHRkMFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBANLJ
hPHhITqQbPklG3ibCVxwGMRfp/v4XqhfdQHdcVfHap6NQ5Wok/4xIA+ui35/MmNa
rtNuC+BdZ1tMuVCPFZcCAwEAAaNQME4wHQYDVR0OBBYEFJvKs8RfJaXTH08W+SGv
zQyKn0H8MB8GA1UdIwQYMBaAFJvKs8RfJaXTH08W+SGvzQyKn0H8MAwGA1UdEwQF
MAMBAf8wDQYJKoZIhvcNAQEFBQADQQBJlffJHybjDGxRMqaRmDhX0+6v02TUKZsW
r5QuVbpQhH6u+0UgcW0jp9QwpxoPTLTWGXEWBBBurxFwiCBhkQ+V
-----END CERTIFICATE-----
`

var rsaKeyPEM = testingKey(`-----BEGIN RSA TESTING KEY-----
MIIBOwIBAAJBANLJhPHhITqQbPklG3ibCVxwGMRfp/v4XqhfdQHdcVfHap6NQ5Wo
k/4xIA+ui35/MmNartNuC+BdZ1tMuVCPFZcCAwEAAQJAEJ2N+zsR0Xn8/Q6twa4G
6OB1M1WO+k+ztnX/1SvNeWu8D6GImtupLTYgjZcHufykj09jiHmjHx8u8ZZB/o1N
MQIhAPW+eyZo7ay3lMz1V01WVjNKK9QSn1MJlb06h/LuYv9FAiEA25WPedKgVyCW
SmUwbPw8fnTcpqDWE3yTO3vKcebqMSsCIBF3UmVue8YU3jybC3NxuXq3wNm34R8T
xVLHwDXh/6NJAiEAl2oHGGLz64BuAfjKrqwz7qMYr9HCLIe/YsoWq/olzScCIQDi
D2lWusoe2/nEqfDVVWGWlyJ7yOmqaVm/iNUN9B2N2g==
-----END RSA TESTING KEY-----
`)

// keyPEM is the same as rsaKeyPEM, but declares itself as just
// "PRIVATE KEY", not "RSA PRIVATE KEY".  https://golang.org/issue/4477
var keyPEM = testingKey(`-----BEGIN TESTING KEY-----
MIIBOwIBAAJBANLJhPHhITqQbPklG3ibCVxwGMRfp/v4XqhfdQHdcVfHap6NQ5Wo
k/4xIA+ui35/MmNartNuC+BdZ1tMuVCPFZcCAwEAAQJAEJ2N+zsR0Xn8/Q6twa4G
6OB1M1WO+k+ztnX/1SvNeWu8D6GImtupLTYgjZcHufykj09jiHmjHx8u8ZZB/o1N
MQIhAPW+eyZo7ay3lMz1V01WVjNKK9QSn1MJlb06h/LuYv9FAiEA25WPedKgVyCW
SmUwbPw8fnTcpqDWE3yTO3vKcebqMSsCIBF3UmVue8YU3jybC3NxuXq3wNm34R8T
xVLHwDXh/6NJAiEAl2oHGGLz64BuAfjKrqwz7qMYr9HCLIe/YsoWq/olzScCIQDi
D2lWusoe2/nEqfDVVWGWlyJ7yOmqaVm/iNUN9B2N2g==
-----END TESTING KEY-----
`)

var ecdsaCertPEM = `-----BEGIN CERTIFICATE-----
MIIB/jCCAWICCQDscdUxw16XFDAJBgcqhkjOPQQBMEUxCzAJBgNVBAYTAkFVMRMw
EQYDVQQIEwpTb21lLVN0YXRlMSEwHwYDVQQKExhJbnRlcm5ldCBXaWRnaXRzIFB0
eSBMdGQwHhcNMTIxMTE0MTI0MDQ4WhcNMTUxMTE0MTI0MDQ4WjBFMQswCQYDVQQG
EwJBVTETMBEGA1UECBMKU29tZS1TdGF0ZTEhMB8GA1UEChMYSW50ZXJuZXQgV2lk
Z2l0cyBQdHkgTHRkMIGbMBAGByqGSM49AgEGBSuBBAAjA4GGAAQBY9+my9OoeSUR
lDQdV/x8LsOuLilthhiS1Tz4aGDHIPwC1mlvnf7fg5lecYpMCrLLhauAc1UJXcgl
01xoLuzgtAEAgv2P/jgytzRSpUYvgLBt1UA0leLYBy6mQQbrNEuqT3INapKIcUv8
XxYP0xMEUksLPq6Ca+CRSqTtrd/23uTnapkwCQYHKoZIzj0EAQOBigAwgYYCQXJo
A7Sl2nLVf+4Iu/tAX/IF4MavARKC4PPHK3zfuGfPR3oCCcsAoz3kAzOeijvd0iXb
H5jBImIxPL4WxQNiBTexAkF8D1EtpYuWdlVQ80/h/f4pBcGiXPqX5h2PQSQY7hP1
+jwM1FGS4fREIOvlBYr/SzzQRtwrvrzGYxDEDbsC0ZGRnA==
-----END CERTIFICATE-----
`

var ecdsaKeyPEM = testingKey(`-----BEGIN EC PARAMETERS-----
BgUrgQQAIw==
-----END EC PARAMETERS-----
-----BEGIN EC TESTING KEY-----
MIHcAgEBBEIBrsoKp0oqcv6/JovJJDoDVSGWdirrkgCWxrprGlzB9o0X8fV675X0
NwuBenXFfeZvVcwluO7/Q9wkYoPd/t3jGImgBwYFK4EEACOhgYkDgYYABAFj36bL
06h5JRGUNB1X/Hwuw64uKW2GGJLVPPhoYMcg/ALWaW+d/t+DmV5xikwKssuFq4Bz
VQldyCXTXGgu7OC0AQCC/Y/+ODK3NFKlRi+AsG3VQDSV4tgHLqZBBus0S6pPcg1q
kohxS/xfFg/TEwRSSws+roJr4JFKpO2t3/be5OdqmQ==
-----END EC TESTING KEY-----
`)

var keyPairTests = []struct {
	algo string
	cert string
	key  string
}{
	{"ECDSA", ecdsaCertPEM, ecdsaKeyPEM},
	{"RSA", rsaCertPEM, rsaKeyPEM},
	{"RSA-untyped", rsaCertPEM, keyPEM}, // golang.org/issue/4477
}

func TestX509KeyPair(t *testing.T) {
	t.Parallel()
	var pem []byte
	for _, test := range keyPairTests {
		pem = []byte(test.cert + test.key)
		if _, err := X509KeyPair(pem, pem); err != nil {
			t.Errorf("Failed to load %s cert followed by %s key: %s", test.algo, test.algo, err)
		}
		pem = []byte(test.key + test.cert)
		if _, err := X509KeyPair(pem, pem); err != nil {
			t.Errorf("Failed to load %s key followed by %s cert: %s", test.algo, test.algo, err)
		}
	}
}

func TestX509KeyPairErrors(t *testing.T) {
	_, err := X509KeyPair([]byte(rsaKeyPEM), []byte(rsaCertPEM))
	if err == nil {
		t.Fatalf("X509KeyPair didn't return an error when arguments were switched")
	}
	if subStr := "been switched"; !strings.Contains(err.Error(), subStr) {
		t.Fatalf("Expected %q in the error when switching arguments to X509KeyPair, but the error was %q", subStr, err)
	}

	_, err = X509KeyPair([]byte(rsaCertPEM), []byte(rsaCertPEM))
	if err == nil {
		t.Fatalf("X509KeyPair didn't return an error when both arguments were certificates")
	}
	if subStr := "certificate"; !strings.Contains(err.Error(), subStr) {
		t.Fatalf("Expected %q in the error when both arguments to X509KeyPair were certificates, but the error was %q", subStr, err)
	}

	const nonsensePEM = `
-----BEGIN NONSENSE-----
Zm9vZm9vZm9v
-----END NONSENSE-----
`

	_, err = X509KeyPair([]byte(nonsensePEM), []byte(nonsensePEM))
	if err == nil {
		t.Fatalf("X509KeyPair didn't return an error when both arguments were nonsense")
	}
	if subStr := "NONSENSE"; !strings.Contains(err.Error(), subStr) {
		t.Fatalf("Expected %q in the error when both arguments to X509KeyPair were nonsense, but the error was %q", subStr, err)
	}
}

func TestX509MixedKeyPair(t *testing.T) {
	if _, err := X509KeyPair([]byte(rsaCertPEM), []byte(ecdsaKeyPEM)); err == nil {
		t.Error("Load of RSA certificate succeeded with ECDSA private key")
	}
	if _, err := X509KeyPair([]byte(ecdsaCertPEM), []byte(rsaKeyPEM)); err == nil {
		t.Error("Load of ECDSA certificate succeeded with RSA private key")
	}
}

func newLocalListener(t testing.TB) net.Listener {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		ln, err = net.Listen("tcp6", "[::1]:0")
	}
	if err != nil {
		t.Fatal(err)
	}
	return ln
}

func runWithFIPSEnabled(t *testing.T, testFunc func(t *testing.T)) {
	originalFIPS := fips140tls.Required()
	defer func() {
		if originalFIPS {
			fips140tls.Force()
		} else {
			fips140tls.TestingOnlyAbandon()
		}
	}()

	fips140tls.Force()
	t.Run("fips140tls", testFunc)
}

func runWithFIPSDisabled(t *testing.T, testFunc func(t *testing.T)) {
	originalFIPS := fips140tls.Required()
	defer func() {
		if originalFIPS {
			fips140tls.Force()
		} else {
			fips140tls.TestingOnlyAbandon()
		}
	}()

	fips140tls.TestingOnlyAbandon()
	t.Run("no-fips140tls", testFunc)
}

func skipFIPS(t *testing.T) {
	if fips140tls.Required() {
		t.Skip("skipping test in FIPS mode")
	}
}

func TestDialTimeout(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	timeout := 100 * time.Microsecond
	for !t.Failed() {
		acceptc := make(chan net.Conn)
		listener := newLocalListener(t)
		go func() {
			for {
				conn, err := listener.Accept()
				if err != nil {
					close(acceptc)
					return
				}
				acceptc <- conn
			}
		}()

		addr := listener.Addr().String()
		dialer := &net.Dialer{
			Timeout: timeout,
		}
		if conn, err := DialWithDialer(dialer, "tcp", addr, nil); err == nil {
			conn.Close()
			t.Errorf("DialWithTimeout unexpectedly completed successfully")
		} else if !isTimeoutError(err) {
			t.Errorf("resulting error not a timeout: %v\nType %T: %#v", err, err, err)
		}

		listener.Close()

		// We're looking for a timeout during the handshake, so check that the
		// Listener actually accepted the connection to initiate it. (If the server
		// takes too long to accept the connection, we might cancel before the
		// underlying net.Conn is ever dialed — without ever attempting a
		// handshake.)
		lconn, ok := <-acceptc
		if ok {
			// The Listener accepted a connection, so assume that it was from our
			// Dial: we triggered the timeout at the point where we wanted it!
			t.Logf("Listener accepted a connection from %s", lconn.RemoteAddr())
			lconn.Close()
		}
		// Close any spurious extra connections from the listener. (This is
		// possible if there are, for example, stray Dial calls from other tests.)
		for extraConn := range acceptc {
			t.Logf("spurious extra connection from %s", extraConn.RemoteAddr())
			extraConn.Close()
		}
		if ok {
			break
		}

		t.Logf("with timeout %v, DialWithDialer returned before listener accepted any connections; retrying", timeout)
		timeout *= 2
	}
}

func TestDeadlineOnWrite(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	ln := newLocalListener(t)
	defer ln.Close()

	srvCh := make(chan *Conn, 1)

	go func() {
		sconn, err := ln.Accept()
		if err != nil {
			srvCh <- nil
			return
		}
		srv := Server(sconn, testConfig.Clone())
		if err := srv.Handshake(); err != nil {
			srvCh <- nil
			return
		}
		srvCh <- srv
	}()

	clientConfig := testConfig.Clone()
	clientConfig.MaxVersion = VersionTLS12
	conn, err := Dial("tcp", ln.Addr().String(), clientConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	srv := <-srvCh
	if srv == nil {
		t.Error(err)
	}

	// Make sure the client/server is setup correctly and is able to do a typical Write/Read
	buf := make([]byte, 6)
	if _, err := srv.Write([]byte("foobar")); err != nil {
		t.Errorf("Write err: %v", err)
	}
	if n, err := conn.Read(buf); n != 6 || err != nil || string(buf) != "foobar" {
		t.Errorf("Read = %d, %v, data %q; want 6, nil, foobar", n, err, buf)
	}

	// Set a deadline which should cause Write to timeout
	if err = srv.SetDeadline(time.Now()); err != nil {
		t.Fatalf("SetDeadline(time.Now()) err: %v", err)
	}
	if _, err = srv.Write([]byte("should fail")); err == nil {
		t.Fatal("Write should have timed out")
	}

	// Clear deadline and make sure it still times out
	if err = srv.SetDeadline(time.Time{}); err != nil {
		t.Fatalf("SetDeadline(time.Time{}) err: %v", err)
	}
	if _, err = srv.Write([]byte("This connection is permanently broken")); err == nil {
		t.Fatal("Write which previously failed should still time out")
	}

	// Verify the error
	if ne := err.(net.Error); ne.Temporary() != false {
		t.Error("Write timed out but incorrectly classified the error as Temporary")
	}
	if !isTimeoutError(err) {
		t.Error("Write timed out but did not classify the error as a Timeout")
	}
}

type readerFunc func([]byte) (int, error)

func (f readerFunc) Read(b []byte) (int, error) { return f(b) }

// TestDialer tests that tls.Dialer.DialContext can abort in the middle of a handshake.
// (The other cases are all handled by the existing dial tests in this package, which
// all also flow through the same code shared code paths)
func TestDialer(t *testing.T) {
	ln := newLocalListener(t)
	defer ln.Close()

	unblockServer := make(chan struct{}) // close-only
	defer close(unblockServer)
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		<-unblockServer
	}()

	ctx, cancel := context.WithCancel(context.Background())
	d := Dialer{Config: &Config{
		Rand: readerFunc(func(b []byte) (n int, err error) {
			// By the time crypto/tls wants randomness, that means it has a TCP
			// connection, so we're past the Dialer's dial and now blocked
			// in a handshake. Cancel our context and see if we get unstuck.
			// (Our TCP listener above never reads or writes, so the Handshake
			// would otherwise be stuck forever)
			cancel()
			return len(b), nil
		}),
		ServerName: "foo",
	}}
	_, err := d.DialContext(ctx, "tcp", ln.Addr().String())
	if err != context.Canceled {
		t.Errorf("err = %v; want context.Canceled", err)
	}
}

func isTimeoutError(err error) bool {
	if ne, ok := err.(net.Error); ok {
		return ne.Timeout()
	}
	return false
}

// tests that Conn.Read returns (non-zero, io.EOF) instead of
// (non-zero, nil) when a Close (alertCloseNotify) is sitting right
// behind the application data in the buffer.
func TestConnReadNonzeroAndEOF(t *testing.T) {
	// This test is racy: it assumes that after a write to a
	// localhost TCP connection, the peer TCP connection can
	// immediately read it. Because it's racy, we skip this test
	// in short mode, and then retry it several times with an
	// increasing sleep in between our final write (via srv.Close
	// below) and the following read.
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	var err error
	for delay := time.Millisecond; delay <= 64*time.Millisecond; delay *= 2 {
		if err = testConnReadNonzeroAndEOF(t, delay); err == nil {
			return
		}
	}
	t.Error(err)
}

func testConnReadNonzeroAndEOF(t *testing.T, delay time.Duration) error {
	ln := newLocalListener(t)
	defer ln.Close()

	srvCh := make(chan *Conn, 1)
	var serr error
	go func() {
		sconn, err := ln.Accept()
		if err != nil {
			serr = err
			srvCh <- nil
			return
		}
		serverConfig := testConfig.Clone()
		srv := Server(sconn, serverConfig)
		if err := srv.Handshake(); err != nil {
			serr = fmt.Errorf("handshake: %v", err)
			srvCh <- nil
			return
		}
		srvCh <- srv
	}()

	clientConfig := testConfig.Clone()
	// In TLS 1.3, alerts are encrypted and disguised as application data, so
	// the opportunistic peek won't work.
	clientConfig.MaxVersion = VersionTLS12
	conn, err := Dial("tcp", ln.Addr().String(), clientConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	srv := <-srvCh
	if srv == nil {
		return serr
	}

	buf := make([]byte, 6)

	srv.Write([]byte("foobar"))
	n, err := conn.Read(buf)
	if n != 6 || err != nil || string(buf) != "foobar" {
		return fmt.Errorf("Read = %d, %v, data %q; want 6, nil, foobar", n, err, buf)
	}

	srv.Write([]byte("abcdef"))
	srv.Close()
	time.Sleep(delay)
	n, err = conn.Read(buf)
	if n != 6 || string(buf) != "abcdef" {
		return fmt.Errorf("Read = %d, buf= %q; want 6, abcdef", n, buf)
	}
	if err != io.EOF {
		return fmt.Errorf("Second Read error = %v; want io.EOF", err)
	}
	return nil
}

func TestTLSUniqueMatches(t *testing.T) {
	ln := newLocalListener(t)
	defer ln.Close()

	serverTLSUniques := make(chan []byte)
	parentDone := make(chan struct{})
	childDone := make(chan struct{})
	defer close(parentDone)
	go func() {
		defer close(childDone)
		for i := 0; i < 2; i++ {
			sconn, err := ln.Accept()
			if err != nil {
				t.Error(err)
				return
			}
			serverConfig := testConfig.Clone()
			serverConfig.MaxVersion = VersionTLS12 // TLSUnique is not defined in TLS 1.3
			srv := Server(sconn, serverConfig)
			if err := srv.Handshake(); err != nil {
				t.Error(err)
				return
			}
			select {
			case <-parentDone:
				return
			case serverTLSUniques <- srv.ConnectionState().TLSUnique:
			}
		}
	}()

	clientConfig := testConfig.Clone()
	clientConfig.ClientSessionCache = NewLRUClientSessionCache(1)
	conn, err := Dial("tcp", ln.Addr().String(), clientConfig)
	if err != nil {
		t.Fatal(err)
	}

	var serverTLSUniquesValue []byte
	select {
	case <-childDone:
		return
	case serverTLSUniquesValue = <-serverTLSUniques:
	}

	if !bytes.Equal(conn.ConnectionState().TLSUnique, serverTLSUniquesValue) {
		t.Error("client and server channel bindings differ")
	}
	if serverTLSUniquesValue == nil || bytes.Equal(serverTLSUniquesValue, make([]byte, 12)) {
		t.Error("tls-unique is empty or zero")
	}
	conn.Close()

	conn, err = Dial("tcp", ln.Addr().String(), clientConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if !conn.ConnectionState().DidResume {
		t.Error("second session did not use resumption")
	}

	select {
	case <-childDone:
		return
	case serverTLSUniquesValue = <-serverTLSUniques:
	}

	if !bytes.Equal(conn.ConnectionState().TLSUnique, serverTLSUniquesValue) {
		t.Error("client and server channel bindings differ when session resumption is used")
	}
	if serverTLSUniquesValue == nil || bytes.Equal(serverTLSUniquesValue, make([]byte, 12)) {
		t.Error("resumption tls-unique is empty or zero")
	}
}

func TestVerifyHostname(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	c, err := Dial("tcp", "www.google.com:https", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.VerifyHostname("www.google.com"); err != nil {
		t.Fatalf("verify www.google.com: %v", err)
	}
	if err := c.VerifyHostname("www.yahoo.com"); err == nil {
		t.Fatalf("verify www.yahoo.com succeeded")
	}

	c, err = Dial("tcp", "www.google.com:https", &Config{InsecureSkipVerify: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.VerifyHostname("www.google.com"); err == nil {
		t.Fatalf("verify www.google.com succeeded with InsecureSkipVerify=true")
	}
}

func TestConnCloseBreakingWrite(t *testing.T) {
	ln := newLocalListener(t)
	defer ln.Close()

	srvCh := make(chan *Conn, 1)
	var serr error
	var sconn net.Conn
	go func() {
		var err error
		sconn, err = ln.Accept()
		if err != nil {
			serr = err
			srvCh <- nil
			return
		}
		serverConfig := testConfig.Clone()
		srv := Server(sconn, serverConfig)
		if err := srv.Handshake(); err != nil {
			serr = fmt.Errorf("handshake: %v", err)
			srvCh <- nil
			return
		}
		srvCh <- srv
	}()

	cconn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer cconn.Close()

	conn := &changeImplConn{
		Conn: cconn,
	}

	clientConfig := testConfig.Clone()
	tconn := Client(conn, clientConfig)
	if err := tconn.Handshake(); err != nil {
		t.Fatal(err)
	}

	srv := <-srvCh
	if srv == nil {
		t.Fatal(serr)
	}
	defer sconn.Close()

	connClosed := make(chan struct{})
	conn.closeFunc = func() error {
		close(connClosed)
		return nil
	}

	inWrite := make(chan bool, 1)
	var errConnClosed = errors.New("conn closed for test")
	conn.writeFunc = func(p []byte) (n int, err error) {
		inWrite <- true
		<-connClosed
		return 0, errConnClosed
	}

	closeReturned := make(chan bool, 1)
	go func() {
		<-inWrite
		tconn.Close() // test that this doesn't block forever.
		closeReturned <- true
	}()

	_, err = tconn.Write([]byte("foo"))
	if err != errConnClosed {
		t.Errorf("Write error = %v; want errConnClosed", err)
	}

	<-closeReturned
	if err := tconn.Close(); err != net.ErrClosed {
		t.Errorf("Close error = %v; want net.ErrClosed", err)
	}
}

func TestConnCloseWrite(t *testing.T) {
	ln := newLocalListener(t)
	defer ln.Close()

	clientDoneChan := make(chan struct{})

	serverCloseWrite := func() error {
		sconn, err := ln.Accept()
		if err != nil {
			return fmt.Errorf("accept: %v", err)
		}
		defer sconn.Close()

		serverConfig := testConfig.Clone()
		srv := Server(sconn, serverConfig)
		if err := srv.Handshake(); err != nil {
			return fmt.Errorf("handshake: %v", err)
		}
		defer srv.Close()

		data, err := io.ReadAll(srv)
		if err != nil {
			return err
		}
		if len(data) > 0 {
			return fmt.Errorf("Read data = %q; want nothing", data)
		}

		if err := srv.CloseWrite(); err != nil {
			return fmt.Errorf("server CloseWrite: %v", err)
		}

		// Wait for clientCloseWrite to finish, so we know we
		// tested the CloseWrite before we defer the
		// sconn.Close above, which would also cause the
		// client to unblock like CloseWrite.
		<-clientDoneChan
		return nil
	}

	clientCloseWrite := func() error {
		defer close(clientDoneChan)

		clientConfig := testConfig.Clone()
		conn, err := Dial("tcp", ln.Addr().String(), clientConfig)
		if err != nil {
			return err
		}
		if err := conn.Handshake(); err != nil {
			return err
		}
		defer conn.Close()

		if err := conn.CloseWrite(); err != nil {
			return fmt.Errorf("client CloseWrite: %v", err)
		}

		if _, err := conn.Write([]byte{0}); err != errShutdown {
			return fmt.Errorf("CloseWrite error = %v; want errShutdown", err)
		}

		data, err := io.ReadAll(conn)
		if err != nil {
			return err
		}
		if len(data) > 0 {
			return fmt.Errorf("Read data = %q; want nothing", data)
		}
		return nil
	}

	errChan := make(chan error, 2)

	go func() { errChan <- serverCloseWrite() }()
	go func() { errChan <- clientCloseWrite() }()

	for i := 0; i < 2; i++ {
		select {
		case err := <-errChan:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(10 * time.Second):
			t.Fatal("deadlock")
		}
	}

	// Also test CloseWrite being called before the handshake is
	// finished:
	{
		ln2 := newLocalListener(t)
		defer ln2.Close()

		netConn, err := net.Dial("tcp", ln2.Addr().String())
		if err != nil {
			t.Fatal(err)
		}
		defer netConn.Close()
		conn := Client(netConn, testConfig.Clone())

		if err := conn.CloseWrite(); err != errEarlyCloseWrite {
			t.Errorf("CloseWrite error = %v; want errEarlyCloseWrite", err)
		}
	}
}

func TestWarningAlertFlood(t *testing.T) {
	ln := newLocalListener(t)
	defer ln.Close()

	server := func() error {
		sconn, err := ln.Accept()
		if err != nil {
			return fmt.Errorf("accept: %v", err)
		}
		defer sconn.Close()

		serverConfig := testConfig.Clone()
		srv := Server(sconn, serverConfig)
		if err := srv.Handshake(); err != nil {
			return fmt.Errorf("handshake: %v", err)
		}
		defer srv.Close()

		_, err = io.ReadAll(srv)
		if err == nil {
			return errors.New("unexpected lack of error from server")
		}
		const expected = "too many ignored"
		if str := err.Error(); !strings.Contains(str, expected) {
			return fmt.Errorf("expected error containing %q, but saw: %s", expected, str)
		}

		return nil
	}

	errChan := make(chan error, 1)
	go func() { errChan <- server() }()

	clientConfig := testConfig.Clone()
	clientConfig.MaxVersion = VersionTLS12 // there are no warning alerts in TLS 1.3
	conn, err := Dial("tcp", ln.Addr().String(), clientConfig)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if err := conn.Handshake(); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < maxUselessRecords+1; i++ {
		conn.sendAlert(alertNoRenegotiation)
	}

	if err := <-errChan; err != nil {
		t.Fatal(err)
	}
}

func TestCloneFuncFields(t *testing.T) {
	const expectedCount = 9
	called := 0

	c1 := Config{
		Time: func() time.Time {
			called |= 1 << 0
			return time.Time{}
		},
		GetCertificate: func(*ClientHelloInfo) (*Certificate, error) {
			called |= 1 << 1
			return nil, nil
		},
		GetClientCertificate: func(*CertificateRequestInfo) (*Certificate, error) {
			called |= 1 << 2
			return nil, nil
		},
		GetConfigForClient: func(*ClientHelloInfo) (*Config, error) {
			called |= 1 << 3
			return nil, nil
		},
		VerifyPeerCertificate: func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
			called |= 1 << 4
			return nil
		},
		VerifyConnection: func(ConnectionState) error {
			called |= 1 << 5
			return nil
		},
		UnwrapSession: func(identity []byte, cs ConnectionState) (*SessionState, error) {
			called |= 1 << 6
			return nil, nil
		},
		WrapSession: func(cs ConnectionState, ss *SessionState) ([]byte, error) {
			called |= 1 << 7
			return nil, nil
		},
		EncryptedClientHelloRejectionVerify: func(ConnectionState) error {
			called |= 1 << 8
			return nil
		},
	}

	c2 := c1.Clone()

	c2.Time()
	c2.GetCertificate(nil)
	c2.GetClientCertificate(nil)
	c2.GetConfigForClient(nil)
	c2.VerifyPeerCertificate(nil, nil)
	c2.VerifyConnection(ConnectionState{})
	c2.UnwrapSession(nil, ConnectionState{})
	c2.WrapSession(ConnectionState{}, nil)
	c2.EncryptedClientHelloRejectionVerify(ConnectionState{})

	if called != (1<<expectedCount)-1 {
		t.Fatalf("expected %d calls but saw calls %b", expectedCount, called)
	}
}

func TestCloneNonFuncFields(t *testing.T) {
	var c1 Config
	v := reflect.ValueOf(&c1).Elem()

	typ := v.Type()
	for i := 0; i < typ.NumField(); i++ {
		f := v.Field(i)
		// testing/quick can't handle functions or interfaces and so
		// isn't used here.
		switch fn := typ.Field(i).Name; fn {
		case "Rand":
			f.Set(reflect.ValueOf(io.Reader(os.Stdin)))
		case "Time", "GetCertificate", "GetConfigForClient", "VerifyPeerCertificate", "VerifyConnection", "GetClientCertificate", "WrapSession", "UnwrapSession", "EncryptedClientHelloRejectionVerify":
			// DeepEqual can't compare functions. If you add a
			// function field to this list, you must also change
			// TestCloneFuncFields to ensure that the func field is
			// cloned.
		case "Certificates":
			f.Set(reflect.ValueOf([]Certificate{
				{Certificate: [][]byte{{'b'}}},
			}))
		case "NameToCertificate":
			f.Set(reflect.ValueOf(map[string]*Certificate{"a": nil}))
		case "RootCAs", "ClientCAs":
			f.Set(reflect.ValueOf(x509.NewCertPool()))
		case "ClientSessionCache":
			f.Set(reflect.ValueOf(NewLRUClientSessionCache(10)))
		case "KeyLogWriter":
			f.Set(reflect.ValueOf(io.Writer(os.Stdout)))
		case "NextProtos":
			f.Set(reflect.ValueOf([]string{"a", "b"}))
		case "ServerName":
			f.Set(reflect.ValueOf("b"))
		case "ClientAuth":
			f.Set(reflect.ValueOf(VerifyClientCertIfGiven))
		case "InsecureSkipVerify", "SessionTicketsDisabled", "DynamicRecordSizingDisabled", "PreferServerCipherSuites":
			f.Set(reflect.ValueOf(true))
		case "MinVersion", "MaxVersion":
			f.Set(reflect.ValueOf(uint16(VersionTLS12)))
		case "SessionTicketKey":
			f.Set(reflect.ValueOf([32]byte{}))
		case "CipherSuites":
			f.Set(reflect.ValueOf([]uint16{1, 2}))
		case "CurvePreferences":
			f.Set(reflect.ValueOf([]CurveID{CurveP256}))
		case "Renegotiation":
			f.Set(reflect.ValueOf(RenegotiateOnceAsClient))
		case "EncryptedClientHelloConfigList":
			f.Set(reflect.ValueOf([]byte{'x'}))
		case "EncryptedClientHelloKeys":
			f.Set(reflect.ValueOf([]EncryptedClientHelloKey{
				{Config: []byte{1}, PrivateKey: []byte{1}},
			}))
		case "mutex", "autoSessionTicketKeys", "sessionTicketKeys":
			continue // these are unexported fields that are handled separately
		default:
			t.Errorf("all fields must be accounted for, but saw unknown field %q", fn)
		}
	}
	// Set the unexported fields related to session ticket keys, which are copied with Clone().
	c1.autoSessionTicketKeys = []ticketKey{c1.ticketKeyFromBytes(c1.SessionTicketKey)}
	c1.sessionTicketKeys = []ticketKey{c1.ticketKeyFromBytes(c1.SessionTicketKey)}

	c2 := c1.Clone()
	if !reflect.DeepEqual(&c1, c2) {
		t.Errorf("clone failed to copy a field")
	}
}

func TestCloneNilConfig(t *testing.T) {
	var config *Config
	if cc := config.Clone(); cc != nil {
		t.Fatalf("Clone with nil should return nil, got: %+v", cc)
	}
}

// changeImplConn is a net.Conn which can change its Write and Close
// methods.
type changeImplConn struct {
	net.Conn
	writeFunc func([]byte) (int, error)
	closeFunc func() error
}

func (w *changeImplConn) Write(p []byte) (n int, err error) {
	if w.writeFunc != nil {
		return w.writeFunc(p)
	}
	return w.Conn.Write(p)
}

func (w *changeImplConn) Close() error {
	if w.closeFunc != nil {
		return w.closeFunc()
	}
	return w.Conn.Close()
}

func throughput(b *testing.B, version uint16, totalBytes int64, dynamicRecordSizingDisabled bool) {
	ln := newLocalListener(b)
	defer ln.Close()

	N := b.N

	// Less than 64KB because Windows appears to use a TCP rwin < 64KB.
	// See Issue #15899.
	const bufsize = 32 << 10

	go func() {
		buf := make([]byte, bufsize)
		for i := 0; i < N; i++ {
			sconn, err := ln.Accept()
			if err != nil {
				// panic rather than synchronize to avoid benchmark overhead
				// (cannot call b.Fatal in goroutine)
				panic(fmt.Errorf("accept: %v", err))
			}
			serverConfig := testConfig.Clone()
			serverConfig.CipherSuites = nil // the defaults may prefer faster ciphers
			serverConfig.DynamicRecordSizingDisabled = dynamicRecordSizingDisabled
			srv := Server(sconn, serverConfig)
			if err := srv.Handshake(); err != nil {
				panic(fmt.Errorf("handshake: %v", err))
			}
			if _, err := io.CopyBuffer(srv, srv, buf); err != nil {
				panic(fmt.Errorf("copy buffer: %v", err))
			}
		}
	}()

	b.SetBytes(totalBytes)
	clientConfig := testConfig.Clone()
	clientConfig.CipherSuites = nil // the defaults may prefer faster ciphers
	clientConfig.DynamicRecordSizingDisabled = dynamicRecordSizingDisabled
	clientConfig.MaxVersion = version

	buf := make([]byte, bufsize)
	chunks := int(math.Ceil(float64(totalBytes) / float64(len(buf))))
	for i := 0; i < N; i++ {
		conn, err := Dial("tcp", ln.Addr().String(), clientConfig)
		if err != nil {
			b.Fatal(err)
		}
		for j := 0; j < chunks; j++ {
			_, err := conn.Write(buf)
			if err != nil {
				b.Fatal(err)
			}
			_, err = io.ReadFull(conn, buf)
			if err != nil {
				b.Fatal(err)
			}
		}
		conn.Close()
	}
}

func BenchmarkThroughput(b *testing.B) {
	for _, mode := range []string{"Max", "Dynamic"} {
		for size := 1; size <= 64; size <<= 1 {
			name := fmt.Sprintf("%sPacket/%dMB", mode, size)
			b.Run(name, func(b *testing.B) {
				b.Run("TLSv12", func(b *testing.B) {
					throughput(b, VersionTLS12, int64(size<<20), mode == "Max")
				})
				b.Run("TLSv13", func(b *testing.B) {
					throughput(b, VersionTLS13, int64(size<<20), mode == "Max")
				})
			})
		}
	}
}

type slowConn struct {
	net.Conn
	bps int
}

func (c *slowConn) Write(p []byte) (int, error) {
	if c.bps == 0 {
		panic("too slow")
	}
	t0 := time.Now()
	wrote := 0
	for wrote < len(p) {
		time.Sleep(100 * time.Microsecond)
		allowed := int(time.Since(t0).Seconds()*float64(c.bps)) / 8
		if allowed > len(p) {
			allowed = len(p)
		}
		if wrote < allowed {
			n, err := c.Conn.Write(p[wrote:allowed])
			wrote += n
			if err != nil {
				return wrote, err
			}
		}
	}
	return len(p), nil
}

func latency(b *testing.B, version uint16, bps int, dynamicRecordSizingDisabled bool) {
	ln := newLocalListener(b)
	defer ln.Close()

	N := b.N

	go func() {
		for i := 0; i < N; i++ {
			sconn, err := ln.Accept()
			if err != nil {
				// panic rather than synchronize to avoid benchmark overhead
				// (cannot call b.Fatal in goroutine)
				panic(fmt.Errorf("accept: %v", err))
			}
			serverConfig := testConfig.Clone()
			serverConfig.DynamicRecordSizingDisabled = dynamicRecordSizingDisabled
			srv := Server(&slowConn{sconn, bps}, serverConfig)
			if err := srv.Handshake(); err != nil {
				panic(fmt.Errorf("handshake: %v", err))
			}
			io.Copy(srv, srv)
		}
	}()

	clientConfig := testConfig.Clone()
	clientConfig.DynamicRecordSizingDisabled = dynamicRecordSizingDisabled
	clientConfig.MaxVersion = version

	buf := make([]byte, 16384)
	peek := make([]byte, 1)

	for i := 0; i < N; i++ {
		conn, err := Dial("tcp", ln.Addr().String(), clientConfig)
		if err != nil {
			b.Fatal(err)
		}
		// make sure we're connected and previous connection has stopped
		if _, err := conn.Write(buf[:1]); err != nil {
			b.Fatal(err)
		}
		if _, err := io.ReadFull(conn, peek); err != nil {
			b.Fatal(err)
		}
		if _, err := conn.Write(buf); err != nil {
			b.Fatal(err)
		}
		if _, err = io.ReadFull(conn, peek); err != nil {
			b.Fatal(err)
		}
		conn.Close()
	}
}

func BenchmarkLatency(b *testing.B) {
	for _, mode := range []string{"Max", "Dynamic"} {
		for _, kbps := range []int{200, 500, 1000, 2000, 5000} {
			name := fmt.Sprintf("%sPacket/%dkbps", mode, kbps)
			b.Run(name, func(b *testing.B) {
				b.Run("TLSv12", func(b *testing.B) {
					latency(b, VersionTLS12, kbps*1000, mode == "Max")
				})
				b.Run("TLSv13", func(b *testing.B) {
					latency(b, VersionTLS13, kbps*1000, mode == "Max")
				})
			})
		}
	}
}

func TestConnectionStateMarshal(t *testing.T) {
	cs := &ConnectionState{}
	_, err := json.Marshal(cs)
	if err != nil {
		t.Errorf("json.Marshal failed on ConnectionState: %v", err)
	}
}

func TestConnectionState(t *testing.T) {
	issuer, err := x509.ParseCertificate(testRSA2048CertificateIssuer)
	if err != nil {
		panic(err)
	}
	rootCAs := x509.NewCertPool()
	rootCAs.AddCert(issuer)

	const alpnProtocol = "golang"
	const serverName = "example.golang"
	var scts = [][]byte{[]byte("dummy sct 1"), []byte("dummy sct 2")}
	var ocsp = []byte("dummy ocsp")

	checkConnectionState := func(t *testing.T, cs ConnectionState, version uint16, isClient bool) {
		if cs.Version != version {
			t.Errorf("got Version %x, expected %x", cs.Version, version)
		}

		if !cs.HandshakeComplete {
			t.Errorf("got HandshakeComplete %v, expected true", cs.HandshakeComplete)
		}

		if cs.DidResume {
			t.Errorf("got DidResume %v, expected false", cs.DidResume)
		}

		if cs.CipherSuite == 0 {
			t.Errorf("got zero CipherSuite")
		}

		if cs.CurveID == 0 {
			t.Errorf("got zero CurveID")
		}

		if cs.NegotiatedProtocol != alpnProtocol {
			t.Errorf("got ALPN protocol %q, expected %q", cs.NegotiatedProtocol, alpnProtocol)
		}

		if !cs.NegotiatedProtocolIsMutual {
			t.Errorf("got NegotiatedProtocolIsMutual %v, expected true", cs.NegotiatedProtocolIsMutual)
		}

		if cs.ServerName != serverName {
			t.Errorf("got ServerName %q, expected %q", cs.ServerName, serverName)
		}

		if len(cs.PeerCertificates) != 1 {
			t.Errorf("got %d PeerCertificates, expected %d", len(cs.PeerCertificates), 1)
		} else if !bytes.Equal(cs.PeerCertificates[0].Raw, testRSA2048Certificate) {
			t.Errorf("got PeerCertificates %x, expected %x", cs.PeerCertificates[0].Raw, testRSA2048Certificate)
		}

		if len(cs.VerifiedChains) != 1 {
			t.Errorf("got %d long verified chain, expected %d", len(cs.VerifiedChains), 1)
		} else if len(cs.VerifiedChains[0]) != 2 {
			t.Errorf("got %d verified chain, expected %d", len(cs.VerifiedChains[0]), 2)
		} else if !bytes.Equal(cs.VerifiedChains[0][0].Raw, testRSA2048Certificate) {
			t.Errorf("got verified chain[0][0] %x, expected %x", cs.VerifiedChains[0][0].Raw, testRSA2048Certificate)
		} else if !bytes.Equal(cs.VerifiedChains[0][1].Raw, testRSA2048CertificateIssuer) {
			t.Errorf("got verified chain[0][1] %x, expected %x", cs.VerifiedChains[0][1].Raw, testRSA2048CertificateIssuer)
		}

		// Only TLS 1.3 supports OCSP and SCTs on client certs.
		if isClient || version == VersionTLS13 {
			if len(cs.SignedCertificateTimestamps) != 2 {
				t.Errorf("got %d SCTs, expected %d", len(cs.SignedCertificateTimestamps), 2)
			} else if !bytes.Equal(cs.SignedCertificateTimestamps[0], scts[0]) {
				t.Errorf("got SCTs %x, expected %x", cs.SignedCertificateTimestamps[0], scts[0])
			} else if !bytes.Equal(cs.SignedCertificateTimestamps[1], scts[1]) {
				t.Errorf("got SCTs %x, expected %x", cs.SignedCertificateTimestamps[1], scts[1])
			}
			if !bytes.Equal(cs.OCSPResponse, ocsp) {
				t.Errorf("got OCSP %x, expected %x", cs.OCSPResponse, ocsp)
			}
		} else {
			if cs.SignedCertificateTimestamps != nil {
				t.Errorf("got %d SCTs, expected nil", len(cs.SignedCertificateTimestamps))
			}
			if cs.OCSPResponse != nil {
				t.Errorf("got OCSP %x, expected nil", cs.OCSPResponse)
			}
		}

		if version == VersionTLS13 {
			if cs.TLSUnique != nil {
				t.Errorf("got TLSUnique %x, expected nil", cs.TLSUnique)
			}
		} else {
			if cs.TLSUnique == nil {
				t.Errorf("got nil TLSUnique")
			}
		}
	}

	compareConnectionStates := func(t *testing.T, cs1, cs2 ConnectionState) {
		if cs1.Version != cs2.Version {
			t.Errorf("Version mismatch: %x != %x", cs1.Version, cs2.Version)
		}
		if cs1.HandshakeComplete != cs2.HandshakeComplete {
			t.Errorf("HandshakeComplete mismatch: %v != %v", cs1.HandshakeComplete, cs2.HandshakeComplete)
		}
		// DidResume is expected to be different.
		if cs1.CipherSuite != cs2.CipherSuite {
			t.Errorf("CipherSuite mismatch: %x != %x", cs1.CipherSuite, cs2.CipherSuite)
		}
		if cs1.CurveID != cs2.CurveID {
			t.Errorf("CurveID mismatch: %s != %s", cs1.CurveID, cs2.CurveID)
		}
		if cs1.NegotiatedProtocol != cs2.NegotiatedProtocol {
			t.Errorf("NegotiatedProtocol mismatch: %q != %q", cs1.NegotiatedProtocol, cs2.NegotiatedProtocol)
		}
		if cs1.NegotiatedProtocolIsMutual != cs2.NegotiatedProtocolIsMutual {
			t.Errorf("NegotiatedProtocolIsMutual mismatch: %v != %v", cs1.NegotiatedProtocolIsMutual, cs2.NegotiatedProtocolIsMutual)
		}
		if cs1.ServerName != cs2.ServerName {
			t.Errorf("ServerName mismatch: %q != %q", cs1.ServerName, cs2.ServerName)
		}
		if !reflect.DeepEqual(cs1.PeerCertificates, cs2.PeerCertificates) {
			t.Errorf("PeerCertificates mismatch")
		}
		if !reflect.DeepEqual(cs1.VerifiedChains, cs2.VerifiedChains) {
			t.Errorf("VerifiedChains mismatch")
		}
		if !reflect.DeepEqual(cs1.SignedCertificateTimestamps, cs2.SignedCertificateTimestamps) {
			t.Errorf("SignedCertificateTimestamps mismatch: %x != %x", cs1.SignedCertificateTimestamps, cs2.SignedCertificateTimestamps)
		}
		if !bytes.Equal(cs1.OCSPResponse, cs2.OCSPResponse) {
			t.Errorf("OCSPResponse mismatch: %x != %x", cs1.OCSPResponse, cs2.OCSPResponse)
		}
		// TLSUnique is expected to be different.
	}

	for _, v := range []uint16{VersionTLS10, VersionTLS12, VersionTLS13} {
		if !isFIPSVersion(v) && fips140tls.Required() {
			t.Skipf("skipping test in FIPS 140-3 mode for non-FIPS version %x", v)
		}
		var name string
		switch v {
		case VersionTLS10:
			name = "TLSv10"
		case VersionTLS12:
			name = "TLSv12"
		case VersionTLS13:
			name = "TLSv13"
		}
		t.Run(name, func(t *testing.T) {
			config := &Config{
				Time:               testTime,
				Certificates:       make([]Certificate, 1),
				MinVersion:         v,
				MaxVersion:         v,
				RootCAs:            rootCAs,
				ClientCAs:          rootCAs,
				ClientAuth:         RequireAndVerifyClientCert,
				NextProtos:         []string{alpnProtocol},
				ServerName:         serverName,
				ClientSessionCache: NewLRUClientSessionCache(1),
			}
			config.Certificates[0].Certificate = [][]byte{testRSA2048Certificate}
			config.Certificates[0].PrivateKey = testRSA2048PrivateKey
			config.Certificates[0].SignedCertificateTimestamps = scts
			config.Certificates[0].OCSPStaple = ocsp

			ss, cs, err := testHandshake(t, config, config)
			if err != nil {
				t.Fatalf("handshake failed: %v", err)
			}

			t.Run("Client", func(t *testing.T) { checkConnectionState(t, cs, v, true) })
			t.Run("Server", func(t *testing.T) { checkConnectionState(t, ss, v, false) })

			t.Run("Resume", func(t *testing.T) {
				// TODO: test changing parameters between original and resumed
				// connection when the protocol allows it.

				ss1, cs1, err := testHandshake(t, config, config)
				if err != nil {
					t.Fatalf("handshake failed: %v", err)
				}

				if !cs1.DidResume || !ss1.DidResume {
					t.Errorf("DidResume is false")
				}

				t.Run("Client", func(t *testing.T) { compareConnectionStates(t, cs, cs1) })
				t.Run("Server", func(t *testing.T) { compareConnectionStates(t, ss, ss1) })
			})
		})
	}
}

// Issue 28744: Ensure that we don't modify memory
// that Config doesn't own such as Certificates.
func TestBuildNameToCertificate_doesntModifyCertificates(t *testing.T) {
	c0 := Certificate{
		Certificate: [][]byte{testRSACertificate},
		PrivateKey:  testRSAPrivateKey,
	}
	c1 := Certificate{
		Certificate: [][]byte{testSNICertificate},
		PrivateKey:  testRSAPrivateKey,
	}
	config := testConfig.Clone()
	config.Certificates = []Certificate{c0, c1}

	config.BuildNameToCertificate()
	got := config.Certificates
	want := []Certificate{c0, c1}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Certificates were mutated by BuildNameToCertificate\nGot: %#v\nWant: %#v\n", got, want)
	}
}

func testingKey(s string) string { return strings.ReplaceAll(s, "TESTING KEY", "PRIVATE KEY") }

func TestClientHelloInfo_SupportsCertificate(t *testing.T) {
	skipFIPS(t) // Test certificates not FIPS compatible.

	rsaCert := &Certificate{
		Certificate: [][]byte{testRSACertificate},
		PrivateKey:  testRSAPrivateKey,
	}
	pkcs1Cert := &Certificate{
		Certificate:                  [][]byte{testRSACertificate},
		PrivateKey:                   testRSAPrivateKey,
		SupportedSignatureAlgorithms: []SignatureScheme{PKCS1WithSHA1, PKCS1WithSHA256},
	}
	ecdsaCert := &Certificate{
		// ECDSA P-256 certificate
		Certificate: [][]byte{testP256Certificate},
		PrivateKey:  testP256PrivateKey,
	}
	ed25519Cert := &Certificate{
		Certificate: [][]byte{testEd25519Certificate},
		PrivateKey:  testEd25519PrivateKey,
	}

	tests := []struct {
		c       *Certificate
		chi     *ClientHelloInfo
		wantErr string
	}{
		{rsaCert, &ClientHelloInfo{
			ServerName:        "example.golang",
			SignatureSchemes:  []SignatureScheme{PSSWithSHA256},
			SupportedVersions: []uint16{VersionTLS13},
		}, ""},
		{ecdsaCert, &ClientHelloInfo{
			SignatureSchemes:  []SignatureScheme{PSSWithSHA256, ECDSAWithP256AndSHA256},
			SupportedVersions: []uint16{VersionTLS13, VersionTLS12},
		}, ""},
		{rsaCert, &ClientHelloInfo{
			ServerName:        "example.com",
			SignatureSchemes:  []SignatureScheme{PSSWithSHA256},
			SupportedVersions: []uint16{VersionTLS13},
		}, "not valid for requested server name"},
		{ecdsaCert, &ClientHelloInfo{
			SignatureSchemes:  []SignatureScheme{ECDSAWithP384AndSHA384},
			SupportedVersions: []uint16{VersionTLS13},
		}, "signature algorithms"},
		{pkcs1Cert, &ClientHelloInfo{
			SignatureSchemes:  []SignatureScheme{PSSWithSHA256, ECDSAWithP256AndSHA256},
			SupportedVersions: []uint16{VersionTLS13},
		}, "signature algorithms"},

		{rsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
			SignatureSchemes:  []SignatureScheme{PKCS1WithSHA1},
			SupportedVersions: []uint16{VersionTLS13, VersionTLS12},
		}, "signature algorithms"},
		{rsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
			SignatureSchemes:  []SignatureScheme{PKCS1WithSHA1},
			SupportedVersions: []uint16{VersionTLS13, VersionTLS12},
			config: &Config{
				CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
				MaxVersion:   VersionTLS12,
			},
		}, ""}, // Check that mutual version selection works.

		{ecdsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{CurveP256},
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SignatureSchemes:  []SignatureScheme{ECDSAWithP256AndSHA256},
			SupportedVersions: []uint16{VersionTLS12},
		}, ""},
		{ecdsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{CurveP256},
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SignatureSchemes:  []SignatureScheme{ECDSAWithP384AndSHA384},
			SupportedVersions: []uint16{VersionTLS12},
		}, ""}, // TLS 1.2 does not restrict curves based on the SignatureScheme.
		{ecdsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{CurveP256},
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SignatureSchemes:  nil,
			SupportedVersions: []uint16{VersionTLS12},
		}, ""}, // TLS 1.2 comes with default signature schemes.
		{ecdsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{CurveP256},
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SignatureSchemes:  []SignatureScheme{ECDSAWithP256AndSHA256},
			SupportedVersions: []uint16{VersionTLS12},
		}, "cipher suite"},
		{ecdsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{CurveP256},
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SignatureSchemes:  []SignatureScheme{ECDSAWithP256AndSHA256},
			SupportedVersions: []uint16{VersionTLS12},
			config: &Config{
				CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
			},
		}, "cipher suite"},
		{ecdsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{CurveP384},
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SignatureSchemes:  []SignatureScheme{ECDSAWithP256AndSHA256},
			SupportedVersions: []uint16{VersionTLS12},
		}, "certificate curve"},
		{ecdsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{CurveP256},
			SupportedPoints:   []uint8{1},
			SignatureSchemes:  []SignatureScheme{ECDSAWithP256AndSHA256},
			SupportedVersions: []uint16{VersionTLS12},
		}, "doesn't support ECDHE"},
		{ecdsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{CurveP256},
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SignatureSchemes:  []SignatureScheme{PSSWithSHA256},
			SupportedVersions: []uint16{VersionTLS12},
		}, "signature algorithms"},

		{ed25519Cert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{CurveP256}, // only relevant for ECDHE support
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SignatureSchemes:  []SignatureScheme{Ed25519},
			SupportedVersions: []uint16{VersionTLS12},
		}, ""},
		{ed25519Cert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{CurveP256}, // only relevant for ECDHE support
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SignatureSchemes:  []SignatureScheme{Ed25519},
			SupportedVersions: []uint16{VersionTLS10},
			config:            &Config{MinVersion: VersionTLS10},
		}, "doesn't support Ed25519"},
		{ed25519Cert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256},
			SupportedCurves:   []CurveID{},
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SignatureSchemes:  []SignatureScheme{Ed25519},
			SupportedVersions: []uint16{VersionTLS12},
		}, "doesn't support ECDHE"},

		{rsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA},
			SupportedCurves:   []CurveID{CurveP256}, // only relevant for ECDHE support
			SupportedPoints:   []uint8{pointFormatUncompressed},
			SupportedVersions: []uint16{VersionTLS10},
			config:            &Config{MinVersion: VersionTLS10},
		}, ""},
		{rsaCert, &ClientHelloInfo{
			CipherSuites:      []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
			SupportedVersions: []uint16{VersionTLS12},
			config: &Config{
				CipherSuites: []uint16{TLS_RSA_WITH_AES_128_GCM_SHA256},
			},
		}, ""}, // static RSA fallback
	}
	for i, tt := range tests {
		err := tt.chi.SupportsCertificate(tt.c)
		switch {
		case tt.wantErr == "" && err != nil:
			t.Errorf("%d: unexpected error: %v", i, err)
		case tt.wantErr != "" && err == nil:
			t.Errorf("%d: unexpected success", i)
		case tt.wantErr != "" && !strings.Contains(err.Error(), tt.wantErr):
			t.Errorf("%d: got error %q, expected %q", i, err, tt.wantErr)
		}
	}
}

func TestCipherSuites(t *testing.T) {
	var lastID uint16
	for _, c := range CipherSuites() {
		if lastID > c.ID {
			t.Errorf("CipherSuites are not ordered by ID: got %#04x after %#04x", c.ID, lastID)
		} else {
			lastID = c.ID
		}

		if c.Insecure {
			t.Errorf("%#04x: Insecure CipherSuite returned by CipherSuites()", c.ID)
		}
	}
	lastID = 0
	for _, c := range InsecureCipherSuites() {
		if lastID > c.ID {
			t.Errorf("InsecureCipherSuites are not ordered by ID: got %#04x after %#04x", c.ID, lastID)
		} else {
			lastID = c.ID
		}

		if !c.Insecure {
			t.Errorf("%#04x: not Insecure CipherSuite returned by InsecureCipherSuites()", c.ID)
		}
	}

	CipherSuiteByID := func(id uint16) *CipherSuite {
		for _, c := range CipherSuites() {
			if c.ID == id {
				return c
			}
		}
		for _, c := range InsecureCipherSuites() {
			if c.ID == id {
				return c
			}
		}
		return nil
	}

	for _, c := range cipherSuites {
		cc := CipherSuiteByID(c.id)
		if cc == nil {
			t.Errorf("%#04x: no CipherSuite entry", c.id)
			continue
		}

		if tls12Only := c.flags&suiteTLS12 != 0; tls12Only && len(cc.SupportedVersions) != 1 {
			t.Errorf("%#04x: suite is TLS 1.2 only, but SupportedVersions is %v", c.id, cc.SupportedVersions)
		} else if !tls12Only && len(cc.SupportedVersions) != 3 {
			t.Errorf("%#04x: suite TLS 1.0-1.2, but SupportedVersions is %v", c.id, cc.SupportedVersions)
		}

		if cc.Insecure {
			if slices.Contains(defaultCipherSuites(false), c.id) {
				t.Errorf("%#04x: insecure suite in default list", c.id)
			}
		} else {
			if !slices.Contains(defaultCipherSuites(false), c.id) {
				t.Errorf("%#04x: secure suite not in default list", c.id)
			}
		}

		if got := CipherSuiteName(c.id); got != cc.Name {
			t.Errorf("%#04x: unexpected CipherSuiteName: got %q, expected %q", c.id, got, cc.Name)
		}
	}
	for _, c := range cipherSuitesTLS13 {
		cc := CipherSuiteByID(c.id)
		if cc == nil {
			t.Errorf("%#04x: no CipherSuite entry", c.id)
			continue
		}

		if cc.Insecure {
			t.Errorf("%#04x: Insecure %v, expected false", c.id, cc.Insecure)
		}
		if len(cc.SupportedVersions) != 1 || cc.SupportedVersions[0] != VersionTLS13 {
			t.Errorf("%#04x: suite is TLS 1.3 only, but SupportedVersions is %v", c.id, cc.SupportedVersions)
		}

		if got := CipherSuiteName(c.id); got != cc.Name {
			t.Errorf("%#04x: unexpected CipherSuiteName: got %q, expected %q", c.id, got, cc.Name)
		}
	}

	if got := CipherSuiteName(0xabc); got != "0x0ABC" {
		t.Errorf("unexpected fallback CipherSuiteName: got %q, expected 0x0ABC", got)
	}

	if len(cipherSuitesPreferenceOrder) != len(cipherSuites) {
		t.Errorf("cipherSuitesPreferenceOrder is not the same size as cipherSuites")
	}
	if len(cipherSuitesPreferenceOrderNoAES) != len(cipherSuitesPreferenceOrder) {
		t.Errorf("cipherSuitesPreferenceOrderNoAES is not the same size as cipherSuitesPreferenceOrder")
	}

	// Check that disabled suites are marked insecure.
	for _, badSuites := range []map[uint16]bool{disabledCipherSuites, rsaKexCiphers} {
		for id := range badSuites {
			c := CipherSuiteByID(id)
			if c == nil {
				t.Errorf("%#04x: no CipherSuite entry", id)
				continue
			}
			if !c.Insecure {
				t.Errorf("%#04x: disabled by default but not marked insecure", id)
			}
		}
	}

	for i, prefOrder := range [][]uint16{cipherSuitesPreferenceOrder, cipherSuitesPreferenceOrderNoAES} {
		// Check that insecure and HTTP/2 bad cipher suites are at the end of
		// the preference lists.
		var sawInsecure, sawBad bool
		for _, id := range prefOrder {
			c := CipherSuiteByID(id)
			if c == nil {
				t.Errorf("%#04x: no CipherSuite entry", id)
				continue
			}

			if c.Insecure {
				sawInsecure = true
			} else if sawInsecure {
				t.Errorf("%#04x: secure suite after insecure one(s)", id)
			}

			if http2isBadCipher(id) {
				sawBad = true
			} else if sawBad {
				t.Errorf("%#04x: non-bad suite after bad HTTP/2 one(s)", id)
			}
		}

		// Check that the list is sorted according to the documented criteria.
		isBetter := func(a, b uint16) int {
			aSuite, bSuite := cipherSuiteByID(a), cipherSuiteByID(b)
			aName, bName := CipherSuiteName(a), CipherSuiteName(b)
			// * < RC4
			if !strings.Contains(aName, "RC4") && strings.Contains(bName, "RC4") {
				return -1
			} else if strings.Contains(aName, "RC4") && !strings.Contains(bName, "RC4") {
				return +1
			}
			// * < CBC_SHA256
			if !strings.Contains(aName, "CBC_SHA256") && strings.Contains(bName, "CBC_SHA256") {
				return -1
			} else if strings.Contains(aName, "CBC_SHA256") && !strings.Contains(bName, "CBC_SHA256") {
				return +1
			}
			// * < 3DES
			if !strings.Contains(aName, "3DES") && strings.Contains(bName, "3DES") {
				return -1
			} else if strings.Contains(aName, "3DES") && !strings.Contains(bName, "3DES") {
				return +1
			}
			// ECDHE < *
			if aSuite.flags&suiteECDHE != 0 && bSuite.flags&suiteECDHE == 0 {
				return -1
			} else if aSuite.flags&suiteECDHE == 0 && bSuite.flags&suiteECDHE != 0 {
				return +1
			}
			// AEAD < CBC
			if aSuite.aead != nil && bSuite.aead == nil {
				return -1
			} else if aSuite.aead == nil && bSuite.aead != nil {
				return +1
			}
			// AES < ChaCha20
			if strings.Contains(aName, "AES") && strings.Contains(bName, "CHACHA20") {
				// negative for cipherSuitesPreferenceOrder
				if i == 0 {
					return -1
				} else {
					return +1
				}
			} else if strings.Contains(aName, "CHACHA20") && strings.Contains(bName, "AES") {
				// negative for cipherSuitesPreferenceOrderNoAES
				if i != 0 {
					return -1
				} else {
					return +1
				}
			}
			// AES-128 < AES-256
			if strings.Contains(aName, "AES_128") && strings.Contains(bName, "AES_256") {
				return -1
			} else if strings.Contains(aName, "AES_256") && strings.Contains(bName, "AES_128") {
				return +1
			}
			// ECDSA < RSA
			if aSuite.flags&suiteECSign != 0 && bSuite.flags&suiteECSign == 0 {
				return -1
			} else if aSuite.flags&suiteECSign == 0 && bSuite.flags&suiteECSign != 0 {
				return +1
			}
			t.Fatalf("two ciphersuites are equal by all criteria: %v and %v", aName, bName)
			panic("unreachable")
		}
		if !slices.IsSortedFunc(prefOrder, isBetter) {
			t.Error("preference order is not sorted according to the rules")
		}
	}
}

func TestVersionName(t *testing.T) {
	if got, exp := VersionName(VersionTLS13), "TLS 1.3"; got != exp {
		t.Errorf("unexpected VersionName: got %q, expected %q", got, exp)
	}
	if got, exp := VersionName(0x12a), "0x012A"; got != exp {
		t.Errorf("unexpected fallback VersionName: got %q, expected %q", got, exp)
	}
}

// http2isBadCipher is copied from net/http.
// TODO: if it ends up exposed somewhere, use that instead.
func http2isBadCipher(cipher uint16) bool {
	switch cipher {
	case TLS_RSA_WITH_RC4_128_SHA,
		TLS_RSA_WITH_3DES_EDE_CBC_SHA,
		TLS_RSA_WITH_AES_128_CBC_SHA,
		TLS_RSA_WITH_AES_256_CBC_SHA,
		TLS_RSA_WITH_AES_128_CBC_SHA256,
		TLS_RSA_WITH_AES_128_GCM_SHA256,
		TLS_RSA_WITH_AES_256_GCM_SHA384,
		TLS_ECDHE_ECDSA_WITH_RC4_128_SHA,
		TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA,
		TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
		TLS_ECDHE_RSA_WITH_RC4_128_SHA,
		TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA,
		TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
		TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,
		TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256,
		TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256:
		return true
	default:
		return false
	}
}

type brokenSigner struct{ crypto.Signer }

func (s brokenSigner) Sign(rand io.Reader, digest []byte, opts crypto.SignerOpts) (signature []byte, err error) {
	// Replace opts with opts.HashFunc(), so rsa.PSSOptions are discarded.
	return s.Signer.Sign(rand, digest, opts.HashFunc())
}

// TestPKCS1OnlyCert uses a client certificate with a broken crypto.Signer that
// always makes PKCS #1 v1.5 signatures, so can't be used with RSA-PSS.
func TestPKCS1OnlyCert(t *testing.T) {
	clientConfig := testConfig.Clone()
	clientConfig.Certificates = []Certificate{{
		Certificate: [][]byte{testRSACertificate},
		PrivateKey:  brokenSigner{testRSAPrivateKey},
	}}
	serverConfig := testConfig.Clone()
	serverConfig.MaxVersion = VersionTLS12 // TLS 1.3 doesn't support PKCS #1 v1.5
	serverConfig.ClientAuth = RequireAnyClientCert

	// If RSA-PSS is selected, the handshake should fail.
	if _, _, err := testHandshake(t, clientConfig, serverConfig); err == nil {
		t.Fatal("expected broken certificate to cause connection to fail")
	}

	clientConfig.Certificates[0].SupportedSignatureAlgorithms =
		[]SignatureScheme{PKCS1WithSHA1, PKCS1WithSHA256}

	// But if the certificate restricts supported algorithms, RSA-PSS should not
	// be selected, and the handshake should succeed.
	if _, _, err := testHandshake(t, clientConfig, serverConfig); err != nil {
		t.Error(err)
	}
}

func TestVerifyCertificates(t *testing.T) {
	skipFIPS(t) // Test certificates not FIPS compatible.

	// See https://go.dev/issue/31641.
	t.Run("TLSv12", func(t *testing.T) { testVerifyCertificates(t, VersionTLS12) })
	t.Run("TLSv13", func(t *testing.T) { testVerifyCertificates(t, VersionTLS13) })
}

func testVerifyCertificates(t *testing.T, version uint16) {
	tests := []struct {
		name string

		InsecureSkipVerify bool
		ClientAuth         ClientAuthType
		ClientCertificates bool
	}{
		{
			name: "defaults",
		},
		{
			name:               "InsecureSkipVerify",
			InsecureSkipVerify: true,
		},
		{
			name:       "RequestClientCert with no certs",
			ClientAuth: RequestClientCert,
		},
		{
			name:               "RequestClientCert with certs",
			ClientAuth:         RequestClientCert,
			ClientCertificates: true,
		},
		{
			name:               "RequireAnyClientCert",
			ClientAuth:         RequireAnyClientCert,
			ClientCertificates: true,
		},
		{
			name:       "VerifyClientCertIfGiven with no certs",
			ClientAuth: VerifyClientCertIfGiven,
		},
		{
			name:               "VerifyClientCertIfGiven with certs",
			ClientAuth:         VerifyClientCertIfGiven,
			ClientCertificates: true,
		},
		{
			name:               "RequireAndVerifyClientCert",
			ClientAuth:         RequireAndVerifyClientCert,
			ClientCertificates: true,
		},
	}

	issuer, err := x509.ParseCertificate(testRSACertificateIssuer)
	if err != nil {
		t.Fatal(err)
	}
	rootCAs := x509.NewCertPool()
	rootCAs.AddCert(issuer)

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			var serverVerifyConnection, clientVerifyConnection bool
			var serverVerifyPeerCertificates, clientVerifyPeerCertificates bool

			clientConfig := testConfig.Clone()
			clientConfig.Time = testTime
			clientConfig.MaxVersion = version
			clientConfig.MinVersion = version
			clientConfig.RootCAs = rootCAs
			clientConfig.ServerName = "example.golang"
			clientConfig.ClientSessionCache = NewLRUClientSessionCache(1)
			serverConfig := clientConfig.Clone()
			serverConfig.ClientCAs = rootCAs

			clientConfig.VerifyConnection = func(cs ConnectionState) error {
				clientVerifyConnection = true
				return nil
			}
			clientConfig.VerifyPeerCertificate = func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
				clientVerifyPeerCertificates = true
				return nil
			}
			serverConfig.VerifyConnection = func(cs ConnectionState) error {
				serverVerifyConnection = true
				return nil
			}
			serverConfig.VerifyPeerCertificate = func(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
				serverVerifyPeerCertificates = true
				return nil
			}

			clientConfig.InsecureSkipVerify = test.InsecureSkipVerify
			serverConfig.ClientAuth = test.ClientAuth
			if !test.ClientCertificates {
				clientConfig.Certificates = nil
			}

			if _, _, err := testHandshake(t, clientConfig, serverConfig); err != nil {
				t.Fatal(err)
			}

			want := serverConfig.ClientAuth != NoClientCert
			if serverVerifyPeerCertificates != want {
				t.Errorf("VerifyPeerCertificates on the server: got %v, want %v",
					serverVerifyPeerCertificates, want)
			}
			if !clientVerifyPeerCertificates {
				t.Errorf("VerifyPeerCertificates not called on the client")
			}
			if !serverVerifyConnection {
				t.Error("VerifyConnection did not get called on the server")
			}
			if !clientVerifyConnection {
				t.Error("VerifyConnection did not get called on the client")
			}

			serverVerifyPeerCertificates, clientVerifyPeerCertificates = false, false
			serverVerifyConnection, clientVerifyConnection = false, false
			cs, _, err := testHandshake(t, clientConfig, serverConfig)
			if err != nil {
				t.Fatal(err)
			}
			if !cs.DidResume {
				t.Error("expected resumption")
			}

			if serverVerifyPeerCertificates {
				t.Error("VerifyPeerCertificates got called on the server on resumption")
			}
			if clientVerifyPeerCertificates {
				t.Error("VerifyPeerCertificates got called on the client on resumption")
			}
			if !serverVerifyConnection {
				t.Error("VerifyConnection did not get called on the server on resumption")
			}
			if !clientVerifyConnection {
				t.Error("VerifyConnection did not get called on the client on resumption")
			}
		})
	}
}

func TestHandshakeMLKEM(t *testing.T) {
	skipFIPS(t) // No X25519MLKEM768 in FIPS
	var tests = []struct {
		name                string
		clientConfig        func(*Config)
		serverConfig        func(*Config)
		preparation         func(*testing.T)
		expectClientSupport bool
		expectMLKEM         bool
		expectHRR           bool
	}{
		{
			name:                "Default",
			expectClientSupport: true,
			expectMLKEM:         true,
			expectHRR:           false,
		},
		{
			name: "ClientCurvePreferences",
			clientConfig: func(config *Config) {
				config.CurvePreferences = []CurveID{X25519}
			},
			expectClientSupport: false,
		},
		{
			name: "ServerCurvePreferencesX25519",
			serverConfig: func(config *Config) {
				config.CurvePreferences = []CurveID{X25519}
			},
			expectClientSupport: true,
			expectMLKEM:         false,
			expectHRR:           false,
		},
		{
			name: "ServerCurvePreferencesHRR",
			serverConfig: func(config *Config) {
				config.CurvePreferences = []CurveID{CurveP256}
			},
			expectClientSupport: true,
			expectMLKEM:         false,
			expectHRR:           true,
		},
		{
			name: "ClientMLKEMOnly",
			clientConfig: func(config *Config) {
				config.CurvePreferences = []CurveID{X25519MLKEM768}
			},
			expectClientSupport: true,
			expectMLKEM:         true,
		},
		{
			name: "ClientSortedCurvePreferences",
			clientConfig: func(config *Config) {
				config.CurvePreferences = []CurveID{CurveP256, X25519MLKEM768}
			},
			expectClientSupport: true,
			expectMLKEM:         true,
		},
		{
			name: "ClientTLSv12",
			clientConfig: func(config *Config) {
				config.MaxVersion = VersionTLS12
			},
			expectClientSupport: false,
		},
		{
			name: "ServerTLSv12",
			serverConfig: func(config *Config) {
				config.MaxVersion = VersionTLS12
			},
			expectClientSupport: true,
			expectMLKEM:         false,
		},
		{
			name: "GODEBUG",
			preparation: func(t *testing.T) {
				t.Setenv("GODEBUG", "tlsmlkem=0")
			},
			expectClientSupport: false,
		},
	}

	baseConfig := testConfig.Clone()
	baseConfig.CurvePreferences = nil
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.preparation != nil {
				test.preparation(t)
			} else {
				t.Parallel()
			}
			serverConfig := baseConfig.Clone()
			if test.serverConfig != nil {
				test.serverConfig(serverConfig)
			}
			serverConfig.GetConfigForClient = func(hello *ClientHelloInfo) (*Config, error) {
				if !test.expectClientSupport && slices.Contains(hello.SupportedCurves, X25519MLKEM768) {
					return nil, errors.New("client supports X25519MLKEM768")
				} else if test.expectClientSupport && !slices.Contains(hello.SupportedCurves, X25519MLKEM768) {
					return nil, errors.New("client does not support X25519MLKEM768")
				}
				return nil, nil
			}
			clientConfig := baseConfig.Clone()
			if test.clientConfig != nil {
				test.clientConfig(clientConfig)
			}
			ss, cs, err := testHandshake(t, clientConfig, serverConfig)
			if err != nil {
				t.Fatal(err)
			}
			if test.expectMLKEM {
				if ss.CurveID != X25519MLKEM768 {
					t.Errorf("got CurveID %v (server), expected %v", ss.CurveID, X25519MLKEM768)
				}
				if cs.CurveID != X25519MLKEM768 {
					t.Errorf("got CurveID %v (client), expected %v", cs.CurveID, X25519MLKEM768)
				}
			} else {
				if ss.CurveID == X25519MLKEM768 {
					t.Errorf("got CurveID %v (server), expected not X25519MLKEM768", ss.CurveID)
				}
				if cs.CurveID == X25519MLKEM768 {
					t.Errorf("got CurveID %v (client), expected not X25519MLKEM768", cs.CurveID)
				}
			}
			if test.expectHRR {
				if !ss.testingOnlyDidHRR {
					t.Error("server did not use HRR")
				}
				if !cs.testingOnlyDidHRR {
					t.Error("client did not use HRR")
				}
			} else {
				if ss.testingOnlyDidHRR {
					t.Error("server used HRR")
				}
				if cs.testingOnlyDidHRR {
					t.Error("client used HRR")
				}
			}
		})
	}
}

func TestX509KeyPairPopulateCertificate(t *testing.T) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER})
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "test"},
	}
	certDER, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, key.Public(), key)
	if err != nil {
		t.Fatal(err)
	}
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})

	t.Run("x509keypairleaf=0", func(t *testing.T) {
		t.Setenv("GODEBUG", "x509keypairleaf=0")
		cert, err := X509KeyPair(certPEM, keyPEM)
		if err != nil {
			t.Fatal(err)
		}
		if cert.Leaf != nil {
			t.Fatal("Leaf should not be populated")
		}
	})
	t.Run("x509keypairleaf=1", func(t *testing.T) {
		t.Setenv("GODEBUG", "x509keypairleaf=1")
		cert, err := X509KeyPair(certPEM, keyPEM)
		if err != nil {
			t.Fatal(err)
		}
		if cert.Leaf == nil {
			t.Fatal("Leaf should be populated")
		}
	})
	t.Run("GODEBUG unset", func(t *testing.T) {
		cert, err := X509KeyPair(certPEM, keyPEM)
		if err != nil {
			t.Fatal(err)
		}
		if cert.Leaf == nil {
			t.Fatal("Leaf should be populated")
		}
	})
}

func TestEarlyLargeCertMsg(t *testing.T) {
	client, server := localPipe(t)

	go func() {
		if _, err := client.Write([]byte{byte(recordTypeHandshake), 3, 4, 0, 4, typeCertificate, 1, 255, 255}); err != nil {
			t.Log(err)
		}
	}()

	expectedErr := "tls: handshake message of length 131071 bytes exceeds maximum of 65536 bytes"
	servConn := Server(server, testConfig)
	err := servConn.Handshake()
	if err == nil {
		t.Fatal("unexpected success")
	}
	if err.Error() != expectedErr {
		t.Fatalf("unexpected error: got %q, want %q", err, expectedErr)
	}
}

func TestLargeCertMsg(t *testing.T) {
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "test"},
		ExtraExtensions: []pkix.Extension{
			{
				Id: asn1.ObjectIdentifier{1, 2, 3},
				// Ballast to inflate the certificate beyond the
				// regular handshake record size.
				Value: make([]byte, 65536),
			},
		},
	}
	cert, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, k.Public(), k)
	if err != nil {
		t.Fatal(err)
	}

	clientConfig, serverConfig := testConfig.Clone(), testConfig.Clone()
	clientConfig.InsecureSkipVerify = true
	serverConfig.Certificates = []Certificate{
		{
			Certificate: [][]byte{cert},
			PrivateKey:  k,
		},
	}
	if _, _, err := testHandshake(t, clientConfig, serverConfig); err != nil {
		t.Fatalf("unexpected failure: %s", err)
	}
}

func TestECH(t *testing.T) {
	k, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		DNSNames:     []string{"public.example"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
	}
	publicCertDER, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, k.Public(), k)
	if err != nil {
		t.Fatal(err)
	}
	publicCert, err := x509.ParseCertificate(publicCertDER)
	if err != nil {
		t.Fatal(err)
	}
	tmpl.DNSNames[0] = "secret.example"
	secretCertDER, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, k.Public(), k)
	if err != nil {
		t.Fatal(err)
	}
	secretCert, err := x509.ParseCertificate(secretCertDER)
	if err != nil {
		t.Fatal(err)
	}

	marshalECHConfig := func(id uint8, pubKey []byte, publicName string, maxNameLen uint8) []byte {
		builder := cryptobyte.NewBuilder(nil)
		builder.AddUint16(extensionEncryptedClientHello)
		builder.AddUint16LengthPrefixed(func(builder *cryptobyte.Builder) {
			builder.AddUint8(id)
			builder.AddUint16(hpke.DHKEM_X25519_HKDF_SHA256) // The only DHKEM we support
			builder.AddUint16LengthPrefixed(func(builder *cryptobyte.Builder) {
				builder.AddBytes(pubKey)
			})
			builder.AddUint16LengthPrefixed(func(builder *cryptobyte.Builder) {
				for _, aeadID := range sortedSupportedAEADs {
					builder.AddUint16(hpke.KDF_HKDF_SHA256) // The only KDF we support
					builder.AddUint16(aeadID)
				}
			})
			builder.AddUint8(maxNameLen)
			builder.AddUint8LengthPrefixed(func(builder *cryptobyte.Builder) {
				builder.AddBytes([]byte(publicName))
			})
			builder.AddUint16(0) // extensions
		})

		return builder.BytesOrPanic()
	}

	echKey, err := ecdh.X25519().GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}

	echConfig := marshalECHConfig(123, echKey.PublicKey().Bytes(), "public.example", 32)

	builder := cryptobyte.NewBuilder(nil)
	builder.AddUint16LengthPrefixed(func(builder *cryptobyte.Builder) {
		builder.AddBytes(echConfig)
	})
	echConfigList := builder.BytesOrPanic()

	clientConfig, serverConfig := testConfig.Clone(), testConfig.Clone()
	clientConfig.InsecureSkipVerify = false
	clientConfig.Rand = rand.Reader
	clientConfig.Time = nil
	clientConfig.MinVersion = VersionTLS13
	clientConfig.ServerName = "secret.example"
	clientConfig.RootCAs = x509.NewCertPool()
	clientConfig.RootCAs.AddCert(secretCert)
	clientConfig.RootCAs.AddCert(publicCert)
	clientConfig.EncryptedClientHelloConfigList = echConfigList
	serverConfig.InsecureSkipVerify = false
	serverConfig.Rand = rand.Reader
	serverConfig.Time = nil
	serverConfig.MinVersion = VersionTLS13
	serverConfig.ServerName = "public.example"
	serverConfig.Certificates = []Certificate{
		{Certificate: [][]byte{publicCertDER}, PrivateKey: k},
		{Certificate: [][]byte{secretCertDER}, PrivateKey: k},
	}
	serverConfig.EncryptedClientHelloKeys = []EncryptedClientHelloKey{
		{Config: echConfig, PrivateKey: echKey.Bytes(), SendAsRetry: true},
	}

	ss, cs, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("unexpected failure: %s", err)
	}
	if !ss.ECHAccepted {
		t.Fatal("server ConnectionState shows ECH not accepted")
	}
	if !cs.ECHAccepted {
		t.Fatal("client ConnectionState shows ECH not accepted")
	}
	if cs.ServerName != "secret.example" || ss.ServerName != "secret.example" {
		t.Fatalf("unexpected ConnectionState.ServerName, want %q, got server:%q, client: %q", "secret.example", ss.ServerName, cs.ServerName)
	}
	if len(cs.VerifiedChains) != 1 {
		t.Fatal("unexpect number of certificate chains")
	}
	if len(cs.VerifiedChains[0]) != 1 {
		t.Fatal("unexpect number of certificates")
	}
	if !cs.VerifiedChains[0][0].Equal(secretCert) {
		t.Fatal("unexpected certificate")
	}
}
