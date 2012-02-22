// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proxy

import (
	"errors"
	"io"
	"net"
	"strconv"
)

// SOCKS5 returns a Dialer that makes SOCKSv5 connections to the given address
// with an optional username and password. See RFC 1928.
func SOCKS5(network, addr string, auth *Auth, forward Dialer) (Dialer, error) {
	s := &socks5{
		network: network,
		addr:    addr,
		forward: forward,
	}
	if auth != nil {
		s.user = auth.User
		s.password = auth.Password
	}

	return s, nil
}

type socks5 struct {
	user, password string
	network, addr  string
	forward        Dialer
}

const socks5Version = 5

const (
	socks5AuthNone     = 0
	socks5AuthPassword = 2
)

const socks5Connect = 1

const (
	socks5IP4    = 1
	socks5Domain = 3
	socks5IP6    = 4
)

var socks5Errors = []string{
	"",
	"general failure",
	"connection forbidden",
	"network unreachable",
	"host unreachable",
	"connection refused",
	"TTL expired",
	"command not supported",
	"address type not supported",
}

// Dial connects to the address addr on the network net via the SOCKS5 proxy.
func (s *socks5) Dial(network, addr string) (net.Conn, error) {
	switch network {
	case "tcp", "tcp6", "tcp4":
		break
	default:
		return nil, errors.New("proxy: no support for SOCKS5 proxy connections of type " + network)
	}

	conn, err := s.forward.Dial(s.network, s.addr)
	if err != nil {
		return nil, err
	}
	closeConn := &conn
	defer func() {
		if closeConn != nil {
			(*closeConn).Close()
		}
	}()

	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}

	port, err := strconv.Atoi(portStr)
	if err != nil {
		return nil, errors.New("proxy: failed to parse port number: " + portStr)
	}
	if port < 1 || port > 0xffff {
		return nil, errors.New("proxy: port number out of range: " + portStr)
	}

	// the size here is just an estimate
	buf := make([]byte, 0, 6+len(host))

	buf = append(buf, socks5Version)
	if len(s.user) > 0 && len(s.user) < 256 && len(s.password) < 256 {
		buf = append(buf, 2 /* num auth methods */, socks5AuthNone, socks5AuthPassword)
	} else {
		buf = append(buf, 1 /* num auth methods */, socks5AuthNone)
	}

	if _, err = conn.Write(buf); err != nil {
		return nil, errors.New("proxy: failed to write greeting to SOCKS5 proxy at " + s.addr + ": " + err.Error())
	}

	if _, err = io.ReadFull(conn, buf[:2]); err != nil {
		return nil, errors.New("proxy: failed to read greeting from SOCKS5 proxy at " + s.addr + ": " + err.Error())
	}
	if buf[0] != 5 {
		return nil, errors.New("proxy: SOCKS5 proxy at " + s.addr + " has unexpected version " + strconv.Itoa(int(buf[0])))
	}
	if buf[1] == 0xff {
		return nil, errors.New("proxy: SOCKS5 proxy at " + s.addr + " requires authentication")
	}

	if buf[1] == socks5AuthPassword {
		buf = buf[:0]
		buf = append(buf, socks5Version)
		buf = append(buf, uint8(len(s.user)))
		buf = append(buf, s.user...)
		buf = append(buf, uint8(len(s.password)))
		buf = append(buf, s.password...)

		if _, err = conn.Write(buf); err != nil {
			return nil, errors.New("proxy: failed to write authentication request to SOCKS5 proxy at " + s.addr + ": " + err.Error())
		}

		if _, err = io.ReadFull(conn, buf[:2]); err != nil {
			return nil, errors.New("proxy: failed to read authentication reply from SOCKS5 proxy at " + s.addr + ": " + err.Error())
		}

		if buf[1] != 0 {
			return nil, errors.New("proxy: SOCKS5 proxy at " + s.addr + " rejected username/password")
		}
	}

	buf = buf[:0]
	buf = append(buf, socks5Version, socks5Connect, 0 /* reserved */)

	if ip := net.ParseIP(host); ip != nil {
		if len(ip) == 4 {
			buf = append(buf, socks5IP4)
		} else {
			buf = append(buf, socks5IP6)
		}
		buf = append(buf, []byte(ip)...)
	} else {
		buf = append(buf, socks5Domain)
		buf = append(buf, byte(len(host)))
		buf = append(buf, host...)
	}
	buf = append(buf, byte(port>>8), byte(port))

	if _, err = conn.Write(buf); err != nil {
		return nil, errors.New("proxy: failed to write connect request to SOCKS5 proxy at " + s.addr + ": " + err.Error())
	}

	if _, err = io.ReadFull(conn, buf[:4]); err != nil {
		return nil, errors.New("proxy: failed to read connect reply from SOCKS5 proxy at " + s.addr + ": " + err.Error())
	}

	failure := "unknown error"
	if int(buf[1]) < len(socks5Errors) {
		failure = socks5Errors[buf[1]]
	}

	if len(failure) > 0 {
		return nil, errors.New("proxy: SOCKS5 proxy at " + s.addr + " failed to connect: " + failure)
	}

	bytesToDiscard := 0
	switch buf[3] {
	case socks5IP4:
		bytesToDiscard = 4
	case socks5IP6:
		bytesToDiscard = 16
	case socks5Domain:
		_, err := io.ReadFull(conn, buf[:1])
		if err != nil {
			return nil, errors.New("proxy: failed to read domain length from SOCKS5 proxy at " + s.addr + ": " + err.Error())
		}
		bytesToDiscard = int(buf[0])
	default:
		return nil, errors.New("proxy: got unknown address type " + strconv.Itoa(int(buf[3])) + " from SOCKS5 proxy at " + s.addr)
	}

	if cap(buf) < bytesToDiscard {
		buf = make([]byte, bytesToDiscard)
	} else {
		buf = buf[:bytesToDiscard]
	}
	if _, err = io.ReadFull(conn, buf); err != nil {
		return nil, errors.New("proxy: failed to read address from SOCKS5 proxy at " + s.addr + ": " + err.Error())
	}

	// Also need to discard the port number
	if _, err = io.ReadFull(conn, buf[:2]); err != nil {
		return nil, errors.New("proxy: failed to read port from SOCKS5 proxy at " + s.addr + ": " + err.Error())
	}

	closeConn = nil
	return conn, nil
}
