// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/internal/socktest"
	"os"
	"runtime"
	"testing"
	"time"
)

func (e *OpError) isValid() error {
	if e.Op == "" {
		return fmt.Errorf("OpError.Op is empty: %v", e)
	}
	if e.Net == "" {
		return fmt.Errorf("OpError.Net is empty: %v", e)
	}
	for _, addr := range []Addr{e.Source, e.Addr} {
		switch addr := addr.(type) {
		case nil:
		case *TCPAddr:
			if addr == nil {
				return fmt.Errorf("OpError.Source or Addr is non-nil interface: %#v, %v", addr, e)
			}
		case *UDPAddr:
			if addr == nil {
				return fmt.Errorf("OpError.Source or Addr is non-nil interface: %#v, %v", addr, e)
			}
		case *IPAddr:
			if addr == nil {
				return fmt.Errorf("OpError.Source or Addr is non-nil interface: %#v, %v", addr, e)
			}
		case *IPNet:
			if addr == nil {
				return fmt.Errorf("OpError.Source or Addr is non-nil interface: %#v, %v", addr, e)
			}
		case *UnixAddr:
			if addr == nil {
				return fmt.Errorf("OpError.Source or Addr is non-nil interface: %#v, %v", addr, e)
			}
		case *pipeAddr:
			if addr == nil {
				return fmt.Errorf("OpError.Source or Addr is non-nil interface: %#v, %v", addr, e)
			}
		case fileAddr:
			if addr == "" {
				return fmt.Errorf("OpError.Source or Addr is empty: %#v, %v", addr, e)
			}
		default:
			return fmt.Errorf("OpError.Source or Addr is unknown type: %T, %v", addr, e)
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
	case *AddrError, addrinfoErrno, *DNSError, InvalidAddrError, *ParseError, *timeoutError, UnknownNetworkError:
		return nil
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
		if c != nil {
			t.Errorf("Dial returned non-nil interface %T(%v) with err != nil", c, c)
		}
		if err = parseDialError(err); err != nil {
			t.Errorf("#%d: %v", i, err)
			continue
		}
	}
}

func TestProtocolDialError(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "solaris":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for _, network := range []string{"tcp", "udp", "ip:4294967296", "unix", "unixpacket", "unixgram"} {
		var err error
		switch network {
		case "tcp":
			_, err = DialTCP(network, nil, &TCPAddr{Port: 1 << 16})
		case "udp":
			_, err = DialUDP(network, nil, &UDPAddr{Port: 1 << 16})
		case "ip:4294967296":
			_, err = DialIP(network, nil, nil)
		case "unix", "unixpacket", "unixgram":
			_, err = DialUnix(network, nil, &UnixAddr{Name: "//"})
		}
		if err == nil {
			t.Errorf("%s: should fail", network)
			continue
		}
		if err = parseDialError(err); err != nil {
			t.Errorf("%s: %v", network, err)
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

	{"tcp", "127.0.0.1:0"},

	{"unix", "/path/to/somewhere"},
	{"unixpacket", "/path/to/somewhere"},
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
		if ln != nil {
			t.Errorf("Listen returned non-nil interface %T(%v) with err != nil", ln, ln)
		}
		if err = parseDialError(err); err != nil {
			t.Errorf("#%d: %v", i, err)
			continue
		}
	}
}

var listenPacketErrorTests = []struct {
	network, address string
}{
	{"foo", ""},
	{"bar", "baz"},
	{"datakit", "mh/astro/r70"},
	{"udp", "127.0.0.1:☺"},
	{"udp", "no-such-name:80"},
	{"udp", "mh/astro/r70:http"},
}

func TestListenPacketError(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("%s does not have full support of socktest", runtime.GOOS)
	}

	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	testHookLookupIP = func(fn func(string) ([]IPAddr, error), host string) ([]IPAddr, error) {
		return nil, &DNSError{Err: "listen error test", Name: "name", Server: "server", IsTimeout: true}
	}

	for i, tt := range listenPacketErrorTests {
		c, err := ListenPacket(tt.network, tt.address)
		if err == nil {
			t.Errorf("#%d: should fail; %s:%s->", i, tt.network, c.LocalAddr())
			c.Close()
			continue
		}
		if c != nil {
			t.Errorf("ListenPacket returned non-nil interface %T(%v) with err != nil", c, c)
		}
		if err = parseDialError(err); err != nil {
			t.Errorf("#%d: %v", i, err)
			continue
		}
	}
}

func TestProtocolListenError(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for _, network := range []string{"tcp", "udp", "ip:4294967296", "unix", "unixpacket", "unixgram"} {
		var err error
		switch network {
		case "tcp":
			_, err = ListenTCP(network, &TCPAddr{Port: 1 << 16})
		case "udp":
			_, err = ListenUDP(network, &UDPAddr{Port: 1 << 16})
		case "ip:4294967296":
			_, err = ListenIP(network, nil)
		case "unix", "unixpacket":
			_, err = ListenUnix(network, &UnixAddr{Name: "//"})
		case "unixgram":
			_, err = ListenUnixgram(network, &UnixAddr{Name: "//"})
		}
		if err == nil {
			t.Errorf("%s: should fail", network)
			continue
		}
		if err = parseDialError(err); err != nil {
			t.Errorf("%s: %v", network, err)
			continue
		}
	}
}

// parseReadError parses nestedErr and reports whether it is a valid
// error value from Read functions.
// It returns nil when nestedErr is valid.
func parseReadError(nestedErr error) error {
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
	if nestedErr == io.EOF {
		return nil
	}
	return fmt.Errorf("unexpected type on 1st nested level: %T", nestedErr)

second:
	if isPlatformError(nestedErr) {
		return nil
	}
	switch err := nestedErr.(type) {
	case *os.SyscallError:
		nestedErr = err.Err
		goto third
	}
	switch nestedErr {
	case errClosing, errTimeout:
		return nil
	}
	return fmt.Errorf("unexpected type on 2nd nested level: %T", nestedErr)

third:
	if isPlatformError(nestedErr) {
		return nil
	}
	return fmt.Errorf("unexpected type on 3rd nested level: %T", nestedErr)
}

// parseWriteError parses nestedErr and reports whether it is a valid
// error value from Write functions.
// It returns nil when nestedErr is valid.
func parseWriteError(nestedErr error) error {
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
	case *AddrError, addrinfoErrno, *DNSError, InvalidAddrError, *ParseError, *timeoutError, UnknownNetworkError:
		return nil
	case *os.SyscallError:
		nestedErr = err.Err
		goto third
	}
	switch nestedErr {
	case errClosing, errTimeout, ErrWriteToConnected, io.ErrUnexpectedEOF:
		return nil
	}
	return fmt.Errorf("unexpected type on 2nd nested level: %T", nestedErr)

third:
	if isPlatformError(nestedErr) {
		return nil
	}
	return fmt.Errorf("unexpected type on 3rd nested level: %T", nestedErr)
}

// parseCloseError parses nestedErr and reports whether it is a valid
// error value from Close functions.
// It returns nil when nestedErr is valid.
func parseCloseError(nestedErr error) error {
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
	case *os.SyscallError:
		nestedErr = err.Err
		goto third
	case *os.PathError: // for Plan 9
		nestedErr = err.Err
		goto third
	}
	switch nestedErr {
	case errClosing:
		return nil
	}
	return fmt.Errorf("unexpected type on 2nd nested level: %T", nestedErr)

third:
	if isPlatformError(nestedErr) {
		return nil
	}
	return fmt.Errorf("unexpected type on 3rd nested level: %T", nestedErr)
}

func TestCloseError(t *testing.T) {
	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	for i := 0; i < 3; i++ {
		err = c.(*TCPConn).CloseRead()
		if perr := parseCloseError(err); perr != nil {
			t.Errorf("#%d: %v", i, perr)
		}
	}
	for i := 0; i < 3; i++ {
		err = c.(*TCPConn).CloseWrite()
		if perr := parseCloseError(err); perr != nil {
			t.Errorf("#%d: %v", i, perr)
		}
	}
	for i := 0; i < 3; i++ {
		err = c.Close()
		if perr := parseCloseError(err); perr != nil {
			t.Errorf("#%d: %v", i, perr)
		}
		err = ln.Close()
		if perr := parseCloseError(err); perr != nil {
			t.Errorf("#%d: %v", i, perr)
		}
	}

	pc, err := ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer pc.Close()

	for i := 0; i < 3; i++ {
		err = pc.Close()
		if perr := parseCloseError(err); perr != nil {
			t.Errorf("#%d: %v", i, perr)
		}
	}
}

// parseAcceptError parses nestedErr and reports whether it is a valid
// error value from Accept functions.
// It returns nil when nestedErr is valid.
func parseAcceptError(nestedErr error) error {
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
	case *os.SyscallError:
		nestedErr = err.Err
		goto third
	}
	switch nestedErr {
	case errClosing, errTimeout:
		return nil
	}
	return fmt.Errorf("unexpected type on 2nd nested level: %T", nestedErr)

third:
	if isPlatformError(nestedErr) {
		return nil
	}
	return fmt.Errorf("unexpected type on 3rd nested level: %T", nestedErr)
}

func TestAcceptError(t *testing.T) {
	handler := func(ls *localServer, ln Listener) {
		for {
			ln.(*TCPListener).SetDeadline(time.Now().Add(5 * time.Millisecond))
			c, err := ln.Accept()
			if perr := parseAcceptError(err); perr != nil {
				t.Error(perr)
			}
			if err != nil {
				if c != nil {
					t.Errorf("Accept returned non-nil interface %T(%v) with err != nil", c, c)
				}
				if nerr, ok := err.(Error); !ok || (!nerr.Timeout() && !nerr.Temporary()) {
					return
				}
				continue
			}
			c.Close()
		}
	}
	ls, err := newLocalServer("tcp")
	if err != nil {
		t.Fatal(err)
	}
	if err := ls.buildup(handler); err != nil {
		ls.teardown()
		t.Fatal(err)
	}

	time.Sleep(100 * time.Millisecond)
	ls.teardown()
}

// parseCommonError parses nestedErr and reports whether it is a valid
// error value from miscellaneous functions.
// It returns nil when nestedErr is valid.
func parseCommonError(nestedErr error) error {
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
	case *os.SyscallError:
		nestedErr = err.Err
		goto third
	case *os.LinkError:
		nestedErr = err.Err
		goto third
	case *os.PathError:
		nestedErr = err.Err
		goto third
	}
	switch nestedErr {
	case errClosing:
		return nil
	}
	return fmt.Errorf("unexpected type on 2nd nested level: %T", nestedErr)

third:
	if isPlatformError(nestedErr) {
		return nil
	}
	return fmt.Errorf("unexpected type on 3rd nested level: %T", nestedErr)
}

func TestFileError(t *testing.T) {
	switch runtime.GOOS {
	case "windows":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	f, err := ioutil.TempFile("", "go-nettest")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	c, err := FileConn(f)
	if err != nil {
		if c != nil {
			t.Errorf("FileConn returned non-nil interface %T(%v) with err != nil", c, c)
		}
		if perr := parseCommonError(err); perr != nil {
			t.Error(perr)
		}
	} else {
		c.Close()
		t.Error("should fail")
	}
	ln, err := FileListener(f)
	if err != nil {
		if ln != nil {
			t.Errorf("FileListener returned non-nil interface %T(%v) with err != nil", ln, ln)
		}
		if perr := parseCommonError(err); perr != nil {
			t.Error(perr)
		}
	} else {
		ln.Close()
		t.Error("should fail")
	}
	pc, err := FilePacketConn(f)
	if err != nil {
		if pc != nil {
			t.Errorf("FilePacketConn returned non-nil interface %T(%v) with err != nil", pc, pc)
		}
		if perr := parseCommonError(err); perr != nil {
			t.Error(perr)
		}
	} else {
		pc.Close()
		t.Error("should fail")
	}

	ln, err = newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 3; i++ {
		f, err := ln.(*TCPListener).File()
		if err != nil {
			if perr := parseCommonError(err); perr != nil {
				t.Error(perr)
			}
		} else {
			f.Close()
		}
		ln.Close()
	}
}
