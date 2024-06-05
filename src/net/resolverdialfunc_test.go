// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that Resolver.Dial can be a func returning an in-memory net.Conn
// speaking DNS.

package net

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"testing"
	"time"

	"golang.org/x/net/dns/dnsmessage"
)

func TestResolverDialFunc(t *testing.T) {
	r := &Resolver{
		PreferGo: true,
		Dial: newResolverDialFunc(&resolverDialHandler{
			StartDial: func(network, address string) error {
				t.Logf("StartDial(%q, %q) ...", network, address)
				return nil
			},
			Question: func(h dnsmessage.Header, q dnsmessage.Question) {
				t.Logf("Header: %+v for %q (type=%v, class=%v)", h,
					q.Name.String(), q.Type, q.Class)
			},
			// TODO: add test without HandleA* hooks specified at all, that Go
			// doesn't issue retries; map to something terminal.
			HandleA: func(w AWriter, name string) error {
				w.AddIP([4]byte{1, 2, 3, 4})
				w.AddIP([4]byte{5, 6, 7, 8})
				return nil
			},
			HandleAAAA: func(w AAAAWriter, name string) error {
				w.AddIP([16]byte{1: 1, 15: 15})
				w.AddIP([16]byte{2: 2, 14: 14})
				return nil
			},
			HandleSRV: func(w SRVWriter, name string) error {
				w.AddSRV(1, 2, 80, "foo.bar.")
				w.AddSRV(2, 3, 81, "bar.baz.")
				return nil
			},
		}),
	}
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
	slices.Sort(ret)
	return ret
}

func newResolverDialFunc(h *resolverDialHandler) func(ctx context.Context, network, address string) (Conn, error) {
	return func(ctx context.Context, network, address string) (Conn, error) {
		a := &resolverFuncConn{
			h:       h,
			network: network,
			address: address,
			ttl:     10, // 10 second default if unset
		}
		if h.StartDial != nil {
			if err := h.StartDial(network, address); err != nil {
				return nil, err
			}
		}
		return a, nil
	}
}

type resolverDialHandler struct {
	// StartDial, if non-nil, is called when Go first calls Resolver.Dial.
	// Any error returned aborts the dial and is returned unwrapped.
	StartDial func(network, address string) error

	Question func(dnsmessage.Header, dnsmessage.Question)

	// err may be ErrNotExist or ErrRefused; others map to SERVFAIL (RCode2).
	// A nil error means success.
	HandleA    func(w AWriter, name string) error
	HandleAAAA func(w AAAAWriter, name string) error
	HandleSRV  func(w SRVWriter, name string) error
}

type ResponseWriter struct{ a *resolverFuncConn }

func (w ResponseWriter) header() dnsmessage.ResourceHeader {
	q := w.a.q
	return dnsmessage.ResourceHeader{
		Name:  q.Name,
		Type:  q.Type,
		Class: q.Class,
		TTL:   w.a.ttl,
	}
}

// SetTTL sets the TTL for subsequent written resources.
// Once a resource has been written, SetTTL calls are no-ops.
// That is, it can only be called at most once, before anything
// else is written.
func (w ResponseWriter) SetTTL(seconds uint32) {
	// ... intention is last one wins and mutates all previously
	// written records too, but that's a little annoying.
	// But it's also annoying if the requirement is it needs to be set
	// last.
	// And it's also annoying if it's possible for users to set
	// different TTLs per Answer.
	if w.a.wrote {
		return
	}
	w.a.ttl = seconds

}

type AWriter struct{ ResponseWriter }

func (w AWriter) AddIP(v4 [4]byte) {
	w.a.wrote = true
	err := w.a.builder.AResource(w.header(), dnsmessage.AResource{A: v4})
	if err != nil {
		panic(err)
	}
}

type AAAAWriter struct{ ResponseWriter }

func (w AAAAWriter) AddIP(v6 [16]byte) {
	w.a.wrote = true
	err := w.a.builder.AAAAResource(w.header(), dnsmessage.AAAAResource{AAAA: v6})
	if err != nil {
		panic(err)
	}
}

type SRVWriter struct{ ResponseWriter }

// AddSRV adds a SRV record. The target name must end in a period and
// be 63 bytes or fewer.
func (w SRVWriter) AddSRV(priority, weight, port uint16, target string) error {
	targetName, err := dnsmessage.NewName(target)
	if err != nil {
		return err
	}
	w.a.wrote = true
	err = w.a.builder.SRVResource(w.header(), dnsmessage.SRVResource{
		Priority: priority,
		Weight:   weight,
		Port:     port,
		Target:   targetName,
	})
	if err != nil {
		panic(err) // internal fault, not user
	}
	return nil
}

var (
	ErrNotExist = errors.New("name does not exist") // maps to RCode3, NXDOMAIN
	ErrRefused  = errors.New("refused")             // maps to RCode5, REFUSED
)

type resolverFuncConn struct {
	h       *resolverDialHandler
	network string
	address string
	builder *dnsmessage.Builder
	q       dnsmessage.Question
	ttl     uint32
	wrote   bool

	rbuf bytes.Buffer
}

func (*resolverFuncConn) Close() error                       { return nil }
func (*resolverFuncConn) LocalAddr() Addr                    { return someaddr{} }
func (*resolverFuncConn) RemoteAddr() Addr                   { return someaddr{} }
func (*resolverFuncConn) SetDeadline(t time.Time) error      { return nil }
func (*resolverFuncConn) SetReadDeadline(t time.Time) error  { return nil }
func (*resolverFuncConn) SetWriteDeadline(t time.Time) error { return nil }

func (a *resolverFuncConn) Read(p []byte) (n int, err error) {
	return a.rbuf.Read(p)
}

func (a *resolverFuncConn) Write(packet []byte) (n int, err error) {
	if len(packet) < 2 {
		return 0, fmt.Errorf("short write of %d bytes; want 2+", len(packet))
	}
	reqLen := int(packet[0])<<8 | int(packet[1])
	req := packet[2:]
	if len(req) != reqLen {
		return 0, fmt.Errorf("packet declared length %d doesn't match body length %d", reqLen, len(req))
	}

	var parser dnsmessage.Parser
	h, err := parser.Start(req)
	if err != nil {
		// TODO: hook
		return 0, err
	}
	q, err := parser.Question()
	hadQ := (err == nil)
	if err == nil && a.h.Question != nil {
		a.h.Question(h, q)
	}
	if err != nil && err != dnsmessage.ErrSectionDone {
		return 0, err
	}

	resh := h
	resh.Response = true
	resh.Authoritative = true
	if hadQ {
		resh.RCode = dnsmessage.RCodeSuccess
	} else {
		resh.RCode = dnsmessage.RCodeNotImplemented
	}
	a.rbuf.Grow(514)
	a.rbuf.WriteByte('X') // reserved header for beu16 length
	a.rbuf.WriteByte('Y') // reserved header for beu16 length
	builder := dnsmessage.NewBuilder(a.rbuf.Bytes(), resh)
	a.builder = &builder
	if hadQ {
		a.q = q
		a.builder.StartQuestions()
		err := a.builder.Question(q)
		if err != nil {
			return 0, fmt.Errorf("Question: %w", err)
		}
		a.builder.StartAnswers()
		switch q.Type {
		case dnsmessage.TypeA:
			if a.h.HandleA != nil {
				resh.RCode = mapRCode(a.h.HandleA(AWriter{ResponseWriter{a}}, q.Name.String()))
			}
		case dnsmessage.TypeAAAA:
			if a.h.HandleAAAA != nil {
				resh.RCode = mapRCode(a.h.HandleAAAA(AAAAWriter{ResponseWriter{a}}, q.Name.String()))
			}
		case dnsmessage.TypeSRV:
			if a.h.HandleSRV != nil {
				resh.RCode = mapRCode(a.h.HandleSRV(SRVWriter{ResponseWriter{a}}, q.Name.String()))
			}
		}
	}
	tcpRes, err := builder.Finish()
	if err != nil {
		return 0, fmt.Errorf("Finish: %w", err)
	}

	n = len(tcpRes) - 2
	tcpRes[0] = byte(n >> 8)
	tcpRes[1] = byte(n)
	a.rbuf.Write(tcpRes[2:])

	return len(packet), nil
}

type someaddr struct{}

func (someaddr) Network() string { return "unused" }
func (someaddr) String() string  { return "unused-someaddr" }

func mapRCode(err error) dnsmessage.RCode {
	switch err {
	case nil:
		return dnsmessage.RCodeSuccess
	case ErrNotExist:
		return dnsmessage.RCodeNameError
	case ErrRefused:
		return dnsmessage.RCodeRefused
	default:
		return dnsmessage.RCodeServerFailure
	}
}
