// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"context"
	"crypto/tls"
	"internal/nettest"
	"net"
	"net/http/internal/testcert"
	"net/netip"
	"slices"
	"sync/atomic"
	"testing"

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
	pnet := newPacketNet(t)
	e1 = newQUICEndpoint(t, pnet, config)
	e2 = newQUICEndpoint(t, pnet, config)
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

func newPacketNet(t testing.TB) *nettest.PacketNet {
	pnet := nettest.NewPacketNet()
	t.Cleanup(pnet.Close)
	return pnet
}

var nextTestAddr atomic.Uint32

// newPacketConn returns a new PacketConn on pnet with a unique source address.
func newPacketConn(t testing.TB, pnet *nettest.PacketNet) *nettest.PacketConn {
	t.Helper()
	n := nextTestAddr.Add(1)
	addr := net.UDPAddrFromAddrPort(netip.AddrPortFrom(
		netip.AddrFrom4([4]byte{127, byte(n >> 16), byte(n >> 8), byte(n)}),
		443,
	))
	pc, err := pnet.NewConn(addr)
	if err != nil {
		t.Fatal(err)
	}
	return pc
}

func newQUICEndpoint(t testing.TB, pnet *nettest.PacketNet, config *quic.Config) *quic.Endpoint {
	t.Helper()
	pc := newPacketConn(t, pnet)
	e, err := quic.NewEndpoint(pc, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		e.Close(t.Context())
	})
	return e
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
