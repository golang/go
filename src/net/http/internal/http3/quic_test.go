// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"bytes"
	"context"
	"crypto/tls"
	"net"
	"net/netip"
	"slices"
	"sync"
	"testing"
	"time"

	"internal/gate"
	"net/http/internal/testcert"
	"golang.org/x/net/quic"
)

// TestNewQUICConfig verifies that newQUICConfig always produces a config with
// an "h3" ALPN, without modifying the caller's tls.Config.
func TestNewQUICConfig(t *testing.T) {
	for _, test := range []struct {
		name      string
		tlsConfig *tls.Config
	}{{
		// net/http.Transport.TLSClientConfig is nil by default.
		name:      "nil",
		tlsConfig: nil,
	}, {
		name:      "no NextProtos",
		tlsConfig: &tls.Config{ServerName: "example.tld"},
	}, {
		name:      "other NextProtos",
		tlsConfig: &tls.Config{NextProtos: []string{"http/1.1"}},
	}, {
		name:      "h3 already set",
		tlsConfig: &tls.Config{NextProtos: []string{"h3"}},
	}} {
		t.Run(test.name, func(t *testing.T) {
			var origNextProtos []string
			if test.tlsConfig != nil {
				origNextProtos = slices.Clone(test.tlsConfig.NextProtos)
			}

			config := newQUICConfig(nil, test.tlsConfig)
			if config == nil {
				t.Fatal("newQUICConfig returned nil config")
			}
			if config.TLSConfig == nil {
				t.Fatal("newQUICConfig returned config with nil TLSConfig")
			}
			if got := config.TLSConfig.NextProtos; !slices.Equal(got, []string{"h3"}) {
				t.Errorf("TLSConfig.NextProtos = %q, want [h3]", got)
			}
			if test.tlsConfig != nil {
				if got := test.tlsConfig.NextProtos; !slices.Equal(got, origNextProtos) {
					t.Errorf("newQUICConfig modified the caller's TLSConfig.NextProtos: got %q, want %q", got, origNextProtos)
				}
			}
		})
	}
}

// newLocalQUICEndpoint returns a QUIC Endpoint listening on localhost.
func newLocalQUICEndpoint(t *testing.T) *quic.Endpoint {
	t.Helper()
	conf := &quic.Config{
		TLSConfig: testTLSConfig,
	}
	e, err := quic.Listen("udp", "127.0.0.1:0", conf)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		e.Close(context.Background())
	})
	return e
}

// newQUICEndpointPair returns two QUIC endpoints on the same test network.
func newQUICEndpointPair(t testing.TB) (e1, e2 *quic.Endpoint) {
	config := &quic.Config{
		TLSConfig: testTLSConfig,
	}
	tn := &testNet{}
	e1 = tn.newQUICEndpoint(t, config)
	e2 = tn.newQUICEndpoint(t, config)
	return e1, e2
}

// newQUICStreamPair returns the two sides of a bidirectional QUIC stream.
func newQUICStreamPair(t testing.TB) (s1, s2 *quic.Stream) {
	t.Helper()
	config := &quic.Config{
		TLSConfig: testTLSConfig,
	}
	e1, e2 := newQUICEndpointPair(t)
	c1, err := e1.Dial(context.Background(), "udp", e2.LocalAddr().String(), config)
	if err != nil {
		t.Fatal(err)
	}
	c2, err := e2.Accept(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	s1, err = c1.NewStream(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	s1.Flush()
	s2, err = c2.AcceptStream(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	return s1, s2
}

// A testNet is a fake network of net.PacketConns.
type testNet struct {
	mu    sync.Mutex
	conns map[netip.AddrPort]*testPacketConn
}

// newPacketConn returns a new PacketConn with a unique source address.
func (tn *testNet) newPacketConn() *testPacketConn {
	tn.mu.Lock()
	defer tn.mu.Unlock()
	if tn.conns == nil {
		tn.conns = make(map[netip.AddrPort]*testPacketConn)
	}
	localAddr := netip.AddrPortFrom(
		netip.AddrFrom4([4]byte{
			127, 0, 0, byte(len(tn.conns)),
		}),
		443)
	tc := &testPacketConn{
		tn:        tn,
		localAddr: localAddr,
		gate:      gate.New(false),
	}
	tn.conns[localAddr] = tc
	return tc
}

func (tn *testNet) newQUICEndpoint(t testing.TB, config *quic.Config) *quic.Endpoint {
	t.Helper()
	pc := tn.newPacketConn()
	e, err := quic.NewEndpoint(pc, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		e.Close(t.Context())
	})
	return e
}

// connForAddr returns the conn with the given source address.
func (tn *testNet) connForAddr(srcAddr netip.AddrPort) *testPacketConn {
	tn.mu.Lock()
	defer tn.mu.Unlock()
	return tn.conns[srcAddr]
}

// A testPacketConn is a net.PacketConn on a testNet fake network.
type testPacketConn struct {
	tn        *testNet
	localAddr netip.AddrPort

	gate   gate.Gate
	queue  []testPacket
	closed bool
}

type testPacket struct {
	b   []byte
	src netip.AddrPort
}

func (tc *testPacketConn) unlock() {
	tc.gate.Unlock(tc.closed || len(tc.queue) > 0)
}

func (tc *testPacketConn) ReadFrom(p []byte) (n int, srcAddr net.Addr, err error) {
	if err := tc.gate.WaitAndLock(context.Background()); err != nil {
		return 0, nil, err
	}
	defer tc.unlock()
	if tc.closed {
		return 0, nil, net.ErrClosed
	}
	n = copy(p, tc.queue[0].b)
	srcAddr = net.UDPAddrFromAddrPort(tc.queue[0].src)
	tc.queue = tc.queue[1:]
	return n, srcAddr, nil
}

func (tc *testPacketConn) WriteTo(p []byte, dstAddr net.Addr) (n int, err error) {
	tc.gate.Lock()
	closed := tc.closed
	tc.unlock()
	if closed {
		return 0, net.ErrClosed
	}

	ap, err := addrPortFromAddr(dstAddr)
	if err != nil {
		return 0, err
	}
	dst := tc.tn.connForAddr(ap)
	if dst == nil {
		return len(p), nil // sent into the void
	}
	dst.gate.Lock()
	defer dst.unlock()
	dst.queue = append(dst.queue, testPacket{
		b:   bytes.Clone(p),
		src: tc.localAddr,
	})
	return len(p), nil
}

func (tc *testPacketConn) Close() error {
	tc.tn.mu.Lock()
	tc.tn.conns[tc.localAddr] = nil
	tc.tn.mu.Unlock()

	tc.gate.Lock()
	defer tc.unlock()
	tc.closed = true
	tc.queue = nil
	return nil
}

func (tc *testPacketConn) LocalAddr() net.Addr {
	return net.UDPAddrFromAddrPort(tc.localAddr)
}

func (tc *testPacketConn) SetDeadline(time.Time) error      { panic("unimplemented") }
func (tc *testPacketConn) SetReadDeadline(time.Time) error  { panic("unimplemented") }
func (tc *testPacketConn) SetWriteDeadline(time.Time) error { panic("unimplemented") }

func addrPortFromAddr(addr net.Addr) (netip.AddrPort, error) {
	switch a := addr.(type) {
	case *net.UDPAddr:
		return a.AddrPort(), nil
	}
	return netip.ParseAddrPort(addr.String())
}

var testTLSConfig = &tls.Config{
	InsecureSkipVerify: true,
	CipherSuites: []uint16{
		tls.TLS_AES_128_GCM_SHA256,
		tls.TLS_AES_256_GCM_SHA384,
		tls.TLS_CHACHA20_POLY1305_SHA256,
	},
	MinVersion:   tls.VersionTLS13,
	Certificates: []tls.Certificate{testCert},
	NextProtos:   []string{"h3"},
}

var testCert = func() tls.Certificate {
	cert, err := tls.X509KeyPair(testcert.LocalhostCert, testcert.LocalhostKey)
	if err != nil {
		panic(err)
	}
	return cert
}()
