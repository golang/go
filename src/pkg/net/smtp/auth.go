// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package smtp

import (
	"crypto/hmac"
	"crypto/md5"
	"errors"
	"fmt"
)

// Auth is implemented by an SMTP authentication mechanism.
type Auth interface {
	// Start begins an authentication with a server.
	// It returns the name of the authentication protocol
	// and optionally data to include in the initial AUTH message
	// sent to the server. It can return proto == "" to indicate
	// that the authentication should be skipped.
	// If it returns a non-nil error, the SMTP client aborts
	// the authentication attempt and closes the connection.
	Start(server *ServerInfo) (proto string, toServer []byte, err error)

	// Next continues the authentication. The server has just sent
	// the fromServer data. If more is true, the server expects a
	// response, which Next should return as toServer; otherwise
	// Next should return toServer == nil.
	// If Next returns a non-nil error, the SMTP client aborts
	// the authentication attempt and closes the connection.
	Next(fromServer []byte, more bool) (toServer []byte, err error)
}

// ServerInfo records information about an SMTP server.
type ServerInfo struct {
	Name string   // SMTP server name
	TLS  bool     // using TLS, with valid certificate for Name
	Auth []string // advertised authentication mechanisms
}

type plainAuth struct {
	identity, username, password string
	host                         string
}

// PlainAuth returns an Auth that implements the PLAIN authentication
// mechanism as defined in RFC 4616.
// The returned Auth uses the given username and password to authenticate
// on TLS connections to host and act as identity. Usually identity will be
// left blank to act as username.
func PlainAuth(identity, username, password, host string) Auth {
	return &plainAuth{identity, username, password, host}
}

func (a *plainAuth) Start(server *ServerInfo) (string, []byte, error) {
	if !server.TLS {
		return "", nil, errors.New("unencrypted connection")
	}
	if server.Name != a.host {
		return "", nil, errors.New("wrong host name")
	}
	resp := []byte(a.identity + "\x00" + a.username + "\x00" + a.password)
	return "PLAIN", resp, nil
}

func (a *plainAuth) Next(fromServer []byte, more bool) ([]byte, error) {
	if more {
		// We've already sent everything.
		return nil, errors.New("unexpected server challenge")
	}
	return nil, nil
}

type cramMD5Auth struct {
	username, secret string
}

// CRAMMD5Auth returns an Auth that implements the CRAM-MD5 authentication
// mechanism as defined in RFC 2195.
// The returned Auth uses the given username and secret to authenticate
// to the server using the challenge-response mechanism.
func CRAMMD5Auth(username, secret string) Auth {
	return &cramMD5Auth{username, secret}
}

func (a *cramMD5Auth) Start(server *ServerInfo) (string, []byte, error) {
	return "CRAM-MD5", nil, nil
}

func (a *cramMD5Auth) Next(fromServer []byte, more bool) ([]byte, error) {
	if more {
		d := hmac.New(md5.New, []byte(a.secret))
		d.Write(fromServer)
		s := make([]byte, 0, d.Size())
		return []byte(fmt.Sprintf("%s %x", a.username, d.Sum(s))), nil
	}
	return nil, nil
}
