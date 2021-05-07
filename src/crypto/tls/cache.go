// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/x509"
	"sync"
)

// certificateCache is a x509.Certificate cache indexed by ASN.1 DER data.
type certificateCache struct {
	rwmutex sync.RWMutex
	cache   map[string]*x509.Certificate
}

var certCache = certificateCache{
	cache: make(map[string]*x509.Certificate),
}

// loadOrParseCertificate returns a single certificate from the given ASN.1 DER data.
func (c *certificateCache) loadOrParseCertificate(der []byte) (*x509.Certificate, error) {
	c.rwmutex.RLock()
	if cert, ok := c.cache[string(der)]; ok {
		c.rwmutex.RUnlock()
		return cert, nil
	}
	c.rwmutex.RUnlock()

	c.rwmutex.Lock()
	defer c.rwmutex.Unlock()

	cert, ok := c.cache[string(der)]
	if !ok {
		var err error
		cert, err = x509.ParseCertificate(der)
		if err != nil {
			return nil, err
		}
		c.cache[string(der)] = cert
	}
	return cert, nil
}
