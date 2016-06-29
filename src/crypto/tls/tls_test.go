// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"bytes"
	"crypto/x509"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"math"
	"math/rand"
	"net"
	"os"
	"reflect"
	"strings"
	"testing"
	"testing/quick"
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

var rsaKeyPEM = `-----BEGIN RSA PRIVATE KEY-----
MIIBOwIBAAJBANLJhPHhITqQbPklG3ibCVxwGMRfp/v4XqhfdQHdcVfHap6NQ5Wo
k/4xIA+ui35/MmNartNuC+BdZ1tMuVCPFZcCAwEAAQJAEJ2N+zsR0Xn8/Q6twa4G
6OB1M1WO+k+ztnX/1SvNeWu8D6GImtupLTYgjZcHufykj09jiHmjHx8u8ZZB/o1N
MQIhAPW+eyZo7ay3lMz1V01WVjNKK9QSn1MJlb06h/LuYv9FAiEA25WPedKgVyCW
SmUwbPw8fnTcpqDWE3yTO3vKcebqMSsCIBF3UmVue8YU3jybC3NxuXq3wNm34R8T
xVLHwDXh/6NJAiEAl2oHGGLz64BuAfjKrqwz7qMYr9HCLIe/YsoWq/olzScCIQDi
D2lWusoe2/nEqfDVVWGWlyJ7yOmqaVm/iNUN9B2N2g==
-----END RSA PRIVATE KEY-----
`

// keyPEM is the same as rsaKeyPEM, but declares itself as just
// "PRIVATE KEY", not "RSA PRIVATE KEY".  https://golang.org/issue/4477
var keyPEM = `-----BEGIN PRIVATE KEY-----
MIIBOwIBAAJBANLJhPHhITqQbPklG3ibCVxwGMRfp/v4XqhfdQHdcVfHap6NQ5Wo
k/4xIA+ui35/MmNartNuC+BdZ1tMuVCPFZcCAwEAAQJAEJ2N+zsR0Xn8/Q6twa4G
6OB1M1WO+k+ztnX/1SvNeWu8D6GImtupLTYgjZcHufykj09jiHmjHx8u8ZZB/o1N
MQIhAPW+eyZo7ay3lMz1V01WVjNKK9QSn1MJlb06h/LuYv9FAiEA25WPedKgVyCW
SmUwbPw8fnTcpqDWE3yTO3vKcebqMSsCIBF3UmVue8YU3jybC3NxuXq3wNm34R8T
xVLHwDXh/6NJAiEAl2oHGGLz64BuAfjKrqwz7qMYr9HCLIe/YsoWq/olzScCIQDi
D2lWusoe2/nEqfDVVWGWlyJ7yOmqaVm/iNUN9B2N2g==
-----END PRIVATE KEY-----
`

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

var ecdsaKeyPEM = `-----BEGIN EC PARAMETERS-----
BgUrgQQAIw==
-----END EC PARAMETERS-----
-----BEGIN EC PRIVATE KEY-----
MIHcAgEBBEIBrsoKp0oqcv6/JovJJDoDVSGWdirrkgCWxrprGlzB9o0X8fV675X0
NwuBenXFfeZvVcwluO7/Q9wkYoPd/t3jGImgBwYFK4EEACOhgYkDgYYABAFj36bL
06h5JRGUNB1X/Hwuw64uKW2GGJLVPPhoYMcg/ALWaW+d/t+DmV5xikwKssuFq4Bz
VQldyCXTXGgu7OC0AQCC/Y/+ODK3NFKlRi+AsG3VQDSV4tgHLqZBBus0S6pPcg1q
kohxS/xfFg/TEwRSSws+roJr4JFKpO2t3/be5OdqmQ==
-----END EC PRIVATE KEY-----
`

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
		serverConfig := testConfig.clone()
		srv := Server(sconn, serverConfig)
		if err := srv.Handshake(); err != nil {
			serr = fmt.Errorf("handshake: %v", err)
			srvCh <- nil
			return
		}
		srvCh <- srv
	}()

	clientConfig := testConfig.clone()
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
				t.Fatal(err)
			}
			serverConfig := testConfig.clone()
			srv := Server(sconn, serverConfig)
			if err := srv.Handshake(); err != nil {
				t.Fatal(err)
			}
			serverTLSUniques <- srv.ConnectionState().TLSUnique
		}
	}()

	clientConfig := testConfig.clone()
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
	if err := c.VerifyHostname("www.yahoo.com"); err == nil {
		t.Fatalf("verify www.google.com succeeded with InsecureSkipVerify=true")
	}
}

func TestVerifyHostnameResumed(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	config := &Config{
		ClientSessionCache: NewLRUClientSessionCache(32),
	}
	for i := 0; i < 2; i++ {
		c, err := Dial("tcp", "www.google.com:https", config)
		if err != nil {
			t.Fatalf("Dial #%d: %v", i, err)
		}
		cs := c.ConnectionState()
		if i > 0 && !cs.DidResume {
			t.Fatalf("Subsequent connection unexpectedly didn't resume")
		}
		if cs.VerifiedChains == nil {
			t.Fatalf("Dial #%d: cs.VerifiedChains == nil", i)
		}
		if err := c.VerifyHostname("www.google.com"); err != nil {
			t.Fatalf("verify www.google.com #%d: %v", i, err)
		}
		c.Close()
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
		serverConfig := testConfig.clone()
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

	clientConfig := testConfig.clone()
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

func TestClone(t *testing.T) {
	var c1 Config
	v := reflect.ValueOf(&c1).Elem()

	rnd := rand.New(rand.NewSource(time.Now().Unix()))
	typ := v.Type()
	for i := 0; i < typ.NumField(); i++ {
		f := v.Field(i)
		if !f.CanSet() {
			// unexported field; not cloned.
			continue
		}

		// testing/quick can't handle functions or interfaces.
		fn := typ.Field(i).Name
		switch fn {
		case "Rand":
			f.Set(reflect.ValueOf(io.Reader(os.Stdin)))
			continue
		case "Time", "GetCertificate":
			// DeepEqual can't compare functions.
			continue
		case "Certificates":
			f.Set(reflect.ValueOf([]Certificate{
				{Certificate: [][]byte{[]byte{'b'}}},
			}))
			continue
		case "NameToCertificate":
			f.Set(reflect.ValueOf(map[string]*Certificate{"a": nil}))
			continue
		case "RootCAs", "ClientCAs":
			f.Set(reflect.ValueOf(x509.NewCertPool()))
			continue
		case "ClientSessionCache":
			f.Set(reflect.ValueOf(NewLRUClientSessionCache(10)))
			continue
		}

		q, ok := quick.Value(f.Type(), rnd)
		if !ok {
			t.Fatalf("quick.Value failed on field %s", fn)
		}
		f.Set(q)
	}

	c2 := c1.clone()

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

func throughput(b *testing.B, totalBytes int64, dynamicRecordSizingDisabled bool) {
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
			serverConfig := testConfig.clone()
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
	clientConfig := testConfig.clone()
	clientConfig.DynamicRecordSizingDisabled = dynamicRecordSizingDisabled

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
				throughput(b, int64(size<<20), mode == "Max")
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

func latency(b *testing.B, bps int, dynamicRecordSizingDisabled bool) {
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
			serverConfig := testConfig.clone()
			serverConfig.DynamicRecordSizingDisabled = dynamicRecordSizingDisabled
			srv := Server(&slowConn{sconn, bps}, serverConfig)
			if err := srv.Handshake(); err != nil {
				panic(fmt.Errorf("handshake: %v", err))
			}
			io.Copy(srv, srv)
		}
	}()

	clientConfig := testConfig.clone()
	clientConfig.DynamicRecordSizingDisabled = dynamicRecordSizingDisabled

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
				latency(b, kbps*1000, mode == "Max")
			})
		}
	}
}
