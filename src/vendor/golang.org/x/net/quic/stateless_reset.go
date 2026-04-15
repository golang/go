// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"hash"
	"sync"
)

const statelessResetTokenLen = 128 / 8

// A statelessResetToken is a stateless reset token.
// https://www.rfc-editor.org/rfc/rfc9000#section-10.3
type statelessResetToken [statelessResetTokenLen]byte

type statelessResetTokenGenerator struct {
	canReset bool

	// The hash.Hash interface is not concurrency safe,
	// so we need a mutex here.
	//
	// There shouldn't be much contention on stateless reset token generation.
	// If this proves to be a problem, we could avoid the mutex by using a separate
	// generator per Conn, or by using a concurrency-safe generator.
	mu  sync.Mutex
	mac hash.Hash
}

func (g *statelessResetTokenGenerator) init(secret [32]byte) {
	zero := true
	for _, b := range secret {
		if b != 0 {
			zero = false
			break
		}
	}
	if zero {
		// Generate tokens using a random secret, but don't send stateless resets.
		rand.Read(secret[:])
		g.canReset = false
	} else {
		g.canReset = true
	}
	g.mac = hmac.New(sha256.New, secret[:])
}

func (g *statelessResetTokenGenerator) tokenForConnID(cid []byte) (token statelessResetToken) {
	g.mu.Lock()
	defer g.mu.Unlock()
	defer g.mac.Reset()
	g.mac.Write(cid)
	copy(token[:], g.mac.Sum(nil))
	return token
}
