// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"crypto/tls"
	"errors"
	"net"
)

const nextProtoUnencryptedHTTP2 = "unencrypted_http2"

// unencryptedNetConnFromTLSConn retrieves a net.Conn wrapped in a *tls.Conn.
//
// TLSNextProto functions accept a *tls.Conn.
//
// When passing an unencrypted HTTP/2 connection to a TLSNextProto function,
// we pass a *tls.Conn with an underlying net.Conn containing the unencrypted connection.
// To be extra careful about mistakes (accidentally dropping TLS encryption in a place
// where we want it), the tls.Conn contains a net.Conn with an UnencryptedNetConn method
// that returns the actual connection we want to use.
func unencryptedNetConnFromTLSConn(tc *tls.Conn) (net.Conn, error) {
	conner, ok := tc.NetConn().(interface {
		UnencryptedNetConn() net.Conn
	})
	if !ok {
		return nil, errors.New("http2: TLS conn unexpectedly found in unencrypted handoff")
	}
	return conner.UnencryptedNetConn(), nil
}
