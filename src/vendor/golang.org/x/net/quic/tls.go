// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"time"
)

// startTLS starts the TLS handshake.
func (c *Conn) startTLS(now time.Time, initialConnID []byte, peerHostname string, params transportParameters) error {
	tlsConfig := c.config.TLSConfig
	if a, _, err := net.SplitHostPort(peerHostname); err == nil {
		peerHostname = a
	}
	if tlsConfig.ServerName == "" && peerHostname != "" {
		tlsConfig = tlsConfig.Clone()
		tlsConfig.ServerName = peerHostname
	}

	c.keysInitial = initialKeys(initialConnID, c.side)

	qconfig := &tls.QUICConfig{TLSConfig: tlsConfig}
	if c.side == clientSide {
		c.tls = tls.QUICClient(qconfig)
	} else {
		c.tls = tls.QUICServer(qconfig)
	}
	c.tls.SetTransportParameters(marshalTransportParameters(params))
	// TODO: We don't need or want a context for cancellation here,
	// but users can use a context to plumb values through to hooks defined
	// in the tls.Config. Pass through a context.
	if err := c.tls.Start(context.TODO()); err != nil {
		return err
	}
	return c.handleTLSEvents(now)
}

func (c *Conn) handleTLSEvents(now time.Time) error {
	for {
		e := c.tls.NextEvent()
		if c.testHooks != nil {
			c.testHooks.handleTLSEvent(e)
		}
		switch e.Kind {
		case tls.QUICNoEvent:
			return nil
		case tls.QUICSetReadSecret:
			if err := checkCipherSuite(e.Suite); err != nil {
				return err
			}
			switch e.Level {
			case tls.QUICEncryptionLevelHandshake:
				c.keysHandshake.r.init(e.Suite, e.Data)
			case tls.QUICEncryptionLevelApplication:
				c.keysAppData.r.init(e.Suite, e.Data)
			}
		case tls.QUICSetWriteSecret:
			if err := checkCipherSuite(e.Suite); err != nil {
				return err
			}
			switch e.Level {
			case tls.QUICEncryptionLevelHandshake:
				c.keysHandshake.w.init(e.Suite, e.Data)
			case tls.QUICEncryptionLevelApplication:
				c.keysAppData.w.init(e.Suite, e.Data)
			}
		case tls.QUICWriteData:
			var space numberSpace
			switch e.Level {
			case tls.QUICEncryptionLevelInitial:
				space = initialSpace
			case tls.QUICEncryptionLevelHandshake:
				space = handshakeSpace
			case tls.QUICEncryptionLevelApplication:
				space = appDataSpace
			default:
				return fmt.Errorf("quic: internal error: write handshake data at level %v", e.Level)
			}
			c.crypto[space].write(e.Data)
		case tls.QUICHandshakeDone:
			if c.side == serverSide {
				// "[...] the TLS handshake is considered confirmed
				// at the server when the handshake completes."
				// https://www.rfc-editor.org/rfc/rfc9001#section-4.1.2-1
				c.confirmHandshake(now)
			}
			c.handshakeDone()
		case tls.QUICTransportParameters:
			params, err := unmarshalTransportParams(e.Data)
			if err != nil {
				return err
			}
			if err := c.receiveTransportParameters(params); err != nil {
				return err
			}
		}
	}
}

// handleCrypto processes data received in a CRYPTO frame.
func (c *Conn) handleCrypto(now time.Time, space numberSpace, off int64, data []byte) error {
	var level tls.QUICEncryptionLevel
	switch space {
	case initialSpace:
		level = tls.QUICEncryptionLevelInitial
	case handshakeSpace:
		level = tls.QUICEncryptionLevelHandshake
	case appDataSpace:
		level = tls.QUICEncryptionLevelApplication
	default:
		return errors.New("quic: internal error: received CRYPTO frame in unexpected number space")
	}
	return c.crypto[space].handleCrypto(off, data, func(b []byte) error {
		return c.tls.HandleData(level, b)
	})
}
