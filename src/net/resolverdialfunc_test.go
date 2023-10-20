// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that Resolver.Dial can be a func returning an in-memory net.Conn
// speaking DNS.

package net

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"

	"golang.org/x/net/dns/dnsmessage"
)

func TestResolverDialFunc(t *testing.T) {
	fake := fakeDNSServer{
		rh: func(_, _ string, q dnsmessage.Message, _ time.Time) (dnsmessage.Message, error) {
			r := dnsmessage.Message{
				Header: dnsmessage.Header{
					ID:       q.Header.ID,
					Response: true,
					RCode:    dnsmessage.RCodeSuccess,
				},
				Questions: q.Questions,
				Answers: []dnsmessage.Resource{
					{
						Header: dnsmessage.ResourceHeader{
							Name:  q.Questions[0].Name,
							Type:  q.Questions[0].Type,
							Class: dnsmessage.ClassINET,
						},
					},
					{
						Header: dnsmessage.ResourceHeader{
							Name:  q.Questions[0].Name,
							Type:  q.Questions[0].Type,
							Class: dnsmessage.ClassINET,
						},
					},
				},
			}

			switch q.Questions[0].Type {
			case dnsmessage.TypeA:
				r.Answers[0].Body = &dnsmessage.AResource{A: [4]byte{1, 2, 3, 4}}
				r.Answers[1].Body = &dnsmessage.AResource{A: [4]byte{5, 6, 7, 8}}
			case dnsmessage.TypeAAAA:
				r.Answers[0].Body = &dnsmessage.AAAAResource{AAAA: [16]byte{1: 1, 15: 15}}
				r.Answers[1].Body = &dnsmessage.AAAAResource{AAAA: [16]byte{2: 2, 14: 14}}
			case dnsmessage.TypeSRV:
				r.Answers[0].Body = &dnsmessage.SRVResource{
					Priority: 1,
					Weight:   2,
					Port:     80,
					Target:   dnsmessage.MustNewName("foo.bar."),
				}
				r.Answers[1].Body = &dnsmessage.SRVResource{
					Priority: 2,
					Weight:   3,
					Port:     81,
					Target:   dnsmessage.MustNewName("bar.baz."),
				}
			default:
				panic("unexpected DNS type")
			}
			return r, nil
		},
	}

	r := &Resolver{PreferGo: true, Dial: fake.DialContext}

	ctx := context.Background()
	const fakeDomain = "something-that-is-a-not-a-real-domain.fake-tld."

	t.Run("LookupIP", func(t *testing.T) {
		ips, err := r.LookupIP(ctx, "ip", fakeDomain)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := sortedIPStrings(ips), []string{"0:200::e00", "1.2.3.4", "1::f", "5.6.7.8"}; !reflect.DeepEqual(got, want) {
			t.Errorf("LookupIP wrong.\n got: %q\nwant: %q\n", got, want)
		}
	})

	t.Run("LookupSRV", func(t *testing.T) {
		_, got, err := r.LookupSRV(ctx, "some-service", "tcp", fakeDomain)
		if err != nil {
			t.Fatal(err)
		}
		want := []*SRV{
			{
				Target:   "foo.bar.",
				Port:     80,
				Priority: 1,
				Weight:   2,
			},
			{
				Target:   "bar.baz.",
				Port:     81,
				Priority: 2,
				Weight:   3,
			},
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("wrong result. got:")
			for _, r := range got {
				t.Logf("  - %+v", r)
			}
		}
	})
}

func sortedIPStrings(ips []IP) []string {
	ret := make([]string, len(ips))
	for i, ip := range ips {
		ret[i] = ip.String()
	}
	sort.Strings(ret)
	return ret
}

type fakeDNSServer struct {
	rh        func(n, s string, q dnsmessage.Message, t time.Time) (dnsmessage.Message, error)
	alwaysTCP bool
}

func (server *fakeDNSServer) DialContext(_ context.Context, n, s string) (Conn, error) {
	if server.alwaysTCP || n == "tcp" || n == "tcp4" || n == "tcp6" {
		return &fakeDNSConn{tcp: true, server: server, n: n, s: s}, nil
	}
	return &fakeDNSPacketConn{fakeDNSConn: fakeDNSConn{tcp: false, server: server, n: n, s: s}}, nil
}

type fakeDNSConn struct {
	Conn
	tcp    bool
	server *fakeDNSServer
	n      string
	s      string
	q      dnsmessage.Message
	t      time.Time
	buf    []byte
}

func (f *fakeDNSConn) Close() error {
	return nil
}

func (f *fakeDNSConn) Read(b []byte) (int, error) {
	if len(f.buf) > 0 {
		n := copy(b, f.buf)
		f.buf = f.buf[n:]
		return n, nil
	}

	resp, err := f.server.rh(f.n, f.s, f.q, f.t)
	if err != nil {
		return 0, err
	}

	bb := make([]byte, 2, 514)
	bb, err = resp.AppendPack(bb)
	if err != nil {
		return 0, fmt.Errorf("cannot marshal DNS message: %v", err)
	}

	if f.tcp {
		l := len(bb) - 2
		bb[0] = byte(l >> 8)
		bb[1] = byte(l)
		f.buf = bb
		return f.Read(b)
	}

	bb = bb[2:]
	if len(b) < len(bb) {
		return 0, errors.New("read would fragment DNS message")
	}

	copy(b, bb)
	return len(bb), nil
}

func (f *fakeDNSConn) Write(b []byte) (int, error) {
	if f.tcp && len(b) >= 2 {
		b = b[2:]
	}
	if f.q.Unpack(b) != nil {
		return 0, fmt.Errorf("cannot unmarshal DNS message fake %s (%d)", f.n, len(b))
	}
	return len(b), nil
}

func (f *fakeDNSConn) SetDeadline(t time.Time) error {
	f.t = t
	return nil
}

type fakeDNSPacketConn struct {
	PacketConn
	fakeDNSConn
}

func (f *fakeDNSPacketConn) SetDeadline(t time.Time) error {
	return f.fakeDNSConn.SetDeadline(t)
}

func (f *fakeDNSPacketConn) Close() error {
	return f.fakeDNSConn.Close()
}
