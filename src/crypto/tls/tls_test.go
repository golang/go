// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"io/ioutil"
	"math"
	"net"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
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
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		ln, err = net.Listen("tcp6", "[::1]:0")
	}
	if err != nil {
		t.Fatal(err)
	}
	return ln
}

func TestDialTimeout(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	listener := newLocalListener(t)

	addr := listener.Addr().String()
	defer listener.Close()

	complete := make(chan bool)
	defer close(complete)

	go func() {
		conn, err := listener.Accept()
		if err != nil {
			t.Error(err)
			return
		}
		<-complete
		conn.Close()
	}()

	dialer := &net.Dialer{
		Timeout: 10 * time.Millisecond,
	}

	var err error
	if _, err = DialWithDialer(dialer, "tcp", addr, nil); err == nil {
		t.Fatal("DialWithTimeout completed successfully")
	}

	if !isTimeoutError(err) {
		t.Errorf("resulting error not a timeout: %v\nType %T: %#v", err, err, err)
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
	go func() {
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
			serverTLSUniques <- srv.ConnectionState().TLSUnique
		}
	}()

	clientConfig := testConfig.Clone()
	clientConfig.ClientSessionCache = NewLRUClientSessionCache(1)
	conn, err := Dial("tcp", ln.Addr().String(), clientConfig)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(conn.ConnectionState().TLSUnique, <-serverTLSUniques) {
		t.Error("client and server channel bindings differ")
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
	if !bytes.Equal(conn.ConnectionState().TLSUnique, <-serverTLSUniques) {
		t.Error("client and server channel bindings differ when session resumption is used")
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
	if err := tconn.Close(); err != errClosed {
		t.Errorf("Close error = %v; want errClosed", err)
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

		data, err := ioutil.ReadAll(srv)
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

		data, err := ioutil.ReadAll(conn)
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

		_, err = ioutil.ReadAll(srv)
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
	const expectedCount = 5
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
	}

	c2 := c1.Clone()

	c2.Time()
	c2.GetCertificate(nil)
	c2.GetClientCertificate(nil)
	c2.GetConfigForClient(nil)
	c2.VerifyPeerCertificate(nil, nil)

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
		if !f.CanSet() {
			// unexported field; not cloned.
			continue
		}

		// testing/quick can't handle functions or interfaces and so
		// isn't used here.
		switch fn := typ.Field(i).Name; fn {
		case "Rand":
			f.Set(reflect.ValueOf(io.Reader(os.Stdin)))
		case "Time", "GetCertificate", "GetConfigForClient", "VerifyPeerCertificate", "GetClientCertificate":
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
		default:
			t.Errorf("all fields must be accounted for, but saw unknown field %q", fn)
		}
	}

	c2 := c1.Clone()
	// DeepEqual also compares unexported fields, thus c2 needs to have run
	// serverInit in order to be DeepEqual to c1. Cloning it and discarding
	// the result is sufficient.
	c2.Clone()

	if !reflect.DeepEqual(&c1, c2) {
		t.Errorf("clone failed to copy a field")
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
	issuer, err := x509.ParseCertificate(testRSACertificateIssuer)
	if err != nil {
		panic(err)
	}
	rootCAs := x509.NewCertPool()
	rootCAs.AddCert(issuer)

	now := func() time.Time { return time.Unix(1476984729, 0) }

	const alpnProtocol = "golang"
	const serverName = "example.golang"
	var scts = [][]byte{[]byte("dummy sct 1"), []byte("dummy sct 2")}
	var ocsp = []byte("dummy ocsp")

	for _, v := range []uint16{VersionTLS12, VersionTLS13} {
		var name string
		switch v {
		case VersionTLS12:
			name = "TLSv12"
		case VersionTLS13:
			name = "TLSv13"
		}
		t.Run(name, func(t *testing.T) {
			config := &Config{
				Time:         now,
				Rand:         zeroSource{},
				Certificates: make([]Certificate, 1),
				MaxVersion:   v,
				RootCAs:      rootCAs,
				ClientCAs:    rootCAs,
				ClientAuth:   RequireAndVerifyClientCert,
				NextProtos:   []string{alpnProtocol},
				ServerName:   serverName,
			}
			config.Certificates[0].Certificate = [][]byte{testRSACertificate}
			config.Certificates[0].PrivateKey = testRSAPrivateKey
			config.Certificates[0].SignedCertificateTimestamps = scts
			config.Certificates[0].OCSPStaple = ocsp

			ss, cs, err := testHandshake(t, config, config)
			if err != nil {
				t.Fatalf("Handshake failed: %v", err)
			}

			if ss.Version != v || cs.Version != v {
				t.Errorf("Got versions %x (server) and %x (client), expected %x", ss.Version, cs.Version, v)
			}

			if !ss.HandshakeComplete || !cs.HandshakeComplete {
				t.Errorf("Got HandshakeComplete %v (server) and %v (client), expected true", ss.HandshakeComplete, cs.HandshakeComplete)
			}

			if ss.DidResume || cs.DidResume {
				t.Errorf("Got DidResume %v (server) and %v (client), expected false", ss.DidResume, cs.DidResume)
			}

			if ss.CipherSuite == 0 || cs.CipherSuite == 0 {
				t.Errorf("Got invalid cipher suite: %v (server) and %v (client)", ss.CipherSuite, cs.CipherSuite)
			}

			if ss.NegotiatedProtocol != alpnProtocol || cs.NegotiatedProtocol != alpnProtocol {
				t.Errorf("Got negotiated protocol %q (server) and %q (client), expected %q", ss.NegotiatedProtocol, cs.NegotiatedProtocol, alpnProtocol)
			}

			if !cs.NegotiatedProtocolIsMutual {
				t.Errorf("Got false NegotiatedProtocolIsMutual on the client side")
			}
			// NegotiatedProtocolIsMutual on the server side is unspecified.

			if ss.ServerName != serverName {
				t.Errorf("Got server name %q, expected %q", ss.ServerName, serverName)
			}
			if cs.ServerName != "" {
				t.Errorf("Got unexpected server name on the client side")
			}

			if len(ss.PeerCertificates) != 1 || len(cs.PeerCertificates) != 1 {
				t.Errorf("Got %d (server) and %d (client) peer certificates, expected %d", len(ss.PeerCertificates), len(cs.PeerCertificates), 1)
			}

			if len(ss.VerifiedChains) != 1 || len(cs.VerifiedChains) != 1 {
				t.Errorf("Got %d (server) and %d (client) verified chains, expected %d", len(ss.VerifiedChains), len(cs.VerifiedChains), 1)
			} else if len(ss.VerifiedChains[0]) != 2 || len(cs.VerifiedChains[0]) != 2 {
				t.Errorf("Got %d (server) and %d (client) long verified chain, expected %d", len(ss.VerifiedChains[0]), len(cs.VerifiedChains[0]), 2)
			}

			if len(cs.SignedCertificateTimestamps) != 2 {
				t.Errorf("Got %d SCTs, expected %d", len(cs.SignedCertificateTimestamps), 2)
			}
			if !bytes.Equal(cs.OCSPResponse, ocsp) {
				t.Errorf("Got OCSPs %x, expected %x", cs.OCSPResponse, ocsp)
			}
			// Only TLS 1.3 supports OCSP and SCTs on client certs.
			if v == VersionTLS13 {
				if len(ss.SignedCertificateTimestamps) != 2 {
					t.Errorf("Got %d client SCTs, expected %d", len(ss.SignedCertificateTimestamps), 2)
				}
				if !bytes.Equal(ss.OCSPResponse, ocsp) {
					t.Errorf("Got client OCSPs %x, expected %x", ss.OCSPResponse, ocsp)
				}
			}

			if v == VersionTLS13 {
				if ss.TLSUnique != nil || cs.TLSUnique != nil {
					t.Errorf("Got TLSUnique %x (server) and %x (client), expected nil in TLS 1.3", ss.TLSUnique, cs.TLSUnique)
				}
			} else {
				if ss.TLSUnique == nil || cs.TLSUnique == nil {
					t.Errorf("Got TLSUnique %x (server) and %x (client), expected non-nil", ss.TLSUnique, cs.TLSUnique)
				}
			}
		})
	}
}

// TestEscapeRoute tests that the library will still work if support for TLS 1.3
// is dropped later in the Go 1.12 cycle.
func TestEscapeRoute(t *testing.T) {
	defer func(savedSupportedVersions []uint16) {
		supportedVersions = savedSupportedVersions
	}(supportedVersions)
	supportedVersions = []uint16{
		VersionTLS12,
		VersionTLS11,
		VersionTLS10,
		VersionSSL30,
	}

	expectVersion(t, testConfig, testConfig, VersionTLS12)
}

func expectVersion(t *testing.T, clientConfig, serverConfig *Config, v uint16) {
	ss, cs, err := testHandshake(t, clientConfig, serverConfig)
	if err != nil {
		t.Fatalf("Handshake failed: %v", err)
	}
	if ss.Version != v {
		t.Errorf("Server negotiated version %x, expected %x", cs.Version, v)
	}
	if cs.Version != v {
		t.Errorf("Client negotiated version %x, expected %x", cs.Version, v)
	}
}

// TestTLS13Switch checks the behavior of GODEBUG=tls13=[0|1]. See Issue 30055.
func TestTLS13Switch(t *testing.T) {
	defer func(savedGODEBUG string) {
		os.Setenv("GODEBUG", savedGODEBUG)
	}(os.Getenv("GODEBUG"))

	os.Setenv("GODEBUG", "tls13=0")
	tls13Support.Once = sync.Once{} // reset the cache

	tls12Config := testConfig.Clone()
	tls12Config.MaxVersion = VersionTLS12
	expectVersion(t, testConfig, testConfig, VersionTLS12)
	expectVersion(t, tls12Config, testConfig, VersionTLS12)
	expectVersion(t, testConfig, tls12Config, VersionTLS12)
	expectVersion(t, tls12Config, tls12Config, VersionTLS12)

	os.Setenv("GODEBUG", "tls13=1")
	tls13Support.Once = sync.Once{} // reset the cache

	expectVersion(t, testConfig, testConfig, VersionTLS13)
	expectVersion(t, tls12Config, testConfig, VersionTLS12)
	expectVersion(t, testConfig, tls12Config, VersionTLS12)
	expectVersion(t, tls12Config, tls12Config, VersionTLS12)
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
