// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"net/internal/socktest"
	"os"
	"runtime"
	"testing"
)

func isTimeoutError(err error) bool {
	nerr, ok := err.(Error)
	return ok && nerr.Timeout()
}

func isTemporaryError(err error) bool {
	nerr, ok := err.(Error)
	return ok && nerr.Temporary()
}

func (e *OpError) isValid() error {
	if e.Op == "" {
		return fmt.Errorf("OpError.Op is empty: %v", e)
	}
	if e.Net == "" {
		return fmt.Errorf("OpError.Net is empty: %v", e)
	}
	switch addr := e.Addr.(type) {
	case *TCPAddr:
		if addr == nil {
			return fmt.Errorf("OpError.Addr is empty: %v", e)
		}
	case *UDPAddr:
		if addr == nil {
			return fmt.Errorf("OpError.Addr is empty: %v", e)
		}
	case *IPAddr:
		if addr == nil {
			return fmt.Errorf("OpError.Addr is empty: %v", e)
		}
	case *IPNet:
		if addr == nil {
			return fmt.Errorf("OpError.Addr is empty: %v", e)
		}
	case *UnixAddr:
		if addr == nil {
			return fmt.Errorf("OpError.Addr is empty: %v", e)
		}
	case *pipeAddr:
		if addr == nil {
			return fmt.Errorf("OpError.Addr is empty: %v", e)
		}
	}
	if e.Err == nil {
		return fmt.Errorf("OpError.Err is empty: %v", e)
	}
	return nil
}

// parseDialError parses nestedErr and reports whether it is a valid
// error value from Dial, Listen functions.
// It returns nil when nestedErr is valid.
func parseDialError(nestedErr error) error {
	if nestedErr == nil {
		return nil
	}

	switch err := nestedErr.(type) {
	case *OpError:
		if err := err.isValid(); err != nil {
			return err
		}
		nestedErr = err.Err
		goto second
	}
	return fmt.Errorf("unexpected type on 1st nested level: %T", nestedErr)

second:
	if isPlatformError(nestedErr) {
		return nil
	}
	switch err := nestedErr.(type) {
	case *AddrError, *DNSError, InvalidAddrError, *ParseError, UnknownNetworkError, *timeoutError:
		return nil
	case *DNSConfigError:
		nestedErr = err.Err
		goto third
	case *os.SyscallError:
		nestedErr = err.Err
		goto third
	}
	switch nestedErr {
	case errClosing, errMissingAddress:
		return nil
	}
	return fmt.Errorf("unexpected type on 2nd nested level: %T", nestedErr)

third:
	if isPlatformError(nestedErr) {
		return nil
	}
	return fmt.Errorf("unexpected type on 3rd nested level: %T", nestedErr)
}

var dialErrorTests = []struct {
	network, address string
}{
	{"foo", ""},
	{"bar", "baz"},
	{"datakit", "mh/astro/r70"},
	{"tcp", ""},
	{"tcp", "127.0.0.1:☺"},
	{"tcp", "no-such-name:80"},
	{"tcp", "mh/astro/r70:http"},

	{"tcp", "127.0.0.1:0"},
	{"udp", "127.0.0.1:0"},
	{"ip:icmp", "127.0.0.1"},

	{"unix", "/path/to/somewhere"},
	{"unixgram", "/path/to/somewhere"},
	{"unixpacket", "/path/to/somewhere"},
}

func TestDialError(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("%s does not have full support of socktest", runtime.GOOS)
	}

	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	testHookLookupIP = func(fn func(string) ([]IPAddr, error), host string) ([]IPAddr, error) {
		return nil, &DNSError{Err: "dial error test", Name: "name", Server: "server", IsTimeout: true}
	}
	sw.Set(socktest.FilterConnect, func(so *socktest.Status) (socktest.AfterFilter, error) {
		return nil, errOpNotSupported
	})
	defer sw.Set(socktest.FilterConnect, nil)

	d := Dialer{Timeout: someTimeout}
	for i, tt := range dialErrorTests {
		c, err := d.Dial(tt.network, tt.address)
		if err == nil {
			t.Errorf("#%d: should fail; %s:%s->%s", i, tt.network, c.LocalAddr(), c.RemoteAddr())
			c.Close()
			continue
		}
		if err = parseDialError(err); err != nil {
			t.Errorf("#%d: %v", i, err)
			continue
		}
	}
}

var listenErrorTests = []struct {
	network, address string
}{
	{"foo", ""},
	{"bar", "baz"},
	{"datakit", "mh/astro/r70"},
	{"tcp", "127.0.0.1:☺"},
	{"tcp", "no-such-name:80"},
	{"tcp", "mh/astro/r70:http"},
}

func TestListenError(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("%s does not have full support of socktest", runtime.GOOS)
	}

	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	testHookLookupIP = func(fn func(string) ([]IPAddr, error), host string) ([]IPAddr, error) {
		return nil, &DNSError{Err: "listen error test", Name: "name", Server: "server", IsTimeout: true}
	}
	sw.Set(socktest.FilterListen, func(so *socktest.Status) (socktest.AfterFilter, error) {
		return nil, errOpNotSupported
	})
	defer sw.Set(socktest.FilterListen, nil)

	for i, tt := range listenErrorTests {
		ln, err := Listen(tt.network, tt.address)
		if err == nil {
			t.Errorf("#%d: should fail; %s:%s->", i, tt.network, ln.Addr())
			ln.Close()
			continue
		}
		if err = parseDialError(err); err != nil {
			t.Errorf("#%d: %v", i, err)
			continue
		}
	}
}
