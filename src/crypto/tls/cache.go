// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/x509"
	"runtime"
	"sync"
	"sync/atomic"
)

type cacheEntry struct {
	refs atomic.Int64
	cert *x509.Certificate
}

// certCache implements an intern table for reference counted x509.Certificates,
// implemented in a similar fashion to BoringSSL's CRYPTO_BUFFER_POOL. This
// allows for a single x509.Certificate to be kept in memory and referenced from
// multiple Conns. Returned references should not be mutated by callers. Certificates
// are still safe to use after they are removed from the cache.
//
// Certificates are returned wrapped in an activeCert struct that should be held by
// the caller. When references to the activeCert are freed, the number of references
// to the certificate in the cache is decremented. Once the number of references
// reaches zero, the entry is evicted from the cache.
//
// The main difference between this implementation and CRYPTO_BUFFER_POOL is that
// CRYPTO_BUFFER_POOL is a more  generic structure which supports blobs of data,
// rather than specific structures. Since we only care about x509.Certificates,
// certCache is implemented as a specific cache, rather than a generic one.
//
// See https://boringssl.googlesource.com/boringssl/+/master/include/openssl/pool.h
// and https://boringssl.googlesource.com/boringssl/+/master/crypto/pool/pool.c
// for the BoringSSL reference.
type certCache struct {
	sync.Map
}

var globalCertCache = new(certCache)

// activeCert is a handle to a certificate held in the cache. Once there are
// no alive activeCerts for a given certificate, the certificate is removed
// from the cache by a finalizer.
type activeCert struct {
	cert *x509.Certificate
}

// active increments the number of references to the entry, wraps the
// certificate in the entry in an activeCert, and sets the finalizer.
//
// Note that there is a race between active and the finalizer set on the
// returned activeCert, triggered if active is called after the ref count is
// decremented such that refs may be > 0 when evict is called. We consider this
// safe, since the caller holding an activeCert for an entry that is no longer
// in the cache is fine, with the only side effect being the memory overhead of
// there being more than one distinct reference to a certificate alive at once.
func (cc *certCache) active(e *cacheEntry) *activeCert {
	e.refs.Add(1)
	a := &activeCert{e.cert}
	runtime.SetFinalizer(a, func(_ *activeCert) {
		if e.refs.Add(-1) == 0 {
			cc.evict(e)
		}
	})
	return a
}

// evict removes a cacheEntry from the cache.
func (cc *certCache) evict(e *cacheEntry) {
	cc.Delete(string(e.cert.Raw))
}

// newCert returns a x509.Certificate parsed from der. If there is already a copy
// of the certificate in the cache, a reference to the existing certificate will
// be returned. Otherwise, a fresh certificate will be added to the cache, and
// the reference returned. The returned reference should not be mutated.
func (cc *certCache) newCert(der []byte) (*activeCert, error) {
	if entry, ok := cc.Load(string(der)); ok {
		return cc.active(entry.(*cacheEntry)), nil
	}

	cert, err := x509.ParseCertificate(der)
	if err != nil {
		return nil, err
	}

	entry := &cacheEntry{cert: cert}
	if entry, loaded := cc.LoadOrStore(string(der), entry); loaded {
		return cc.active(entry.(*cacheEntry)), nil
	}
	return cc.active(entry), nil
}
