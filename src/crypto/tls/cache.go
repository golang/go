// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/x509"
	"runtime"
	"sync"
	"weak"
)

// weakCertCache provides a cache of *x509.Certificates, allowing multiple
// connections to reuse parsed certificates, instead of re-parsing the
// certificate for every connection, which is an expensive operation.
type weakCertCache struct{ sync.Map }

func (wcc *weakCertCache) newCert(der []byte) (*x509.Certificate, error) {
	if entry, ok := wcc.Load(string(der)); ok {
		if v := entry.(weak.Pointer[x509.Certificate]).Value(); v != nil {
			return v, nil
		}
	}

	cert, err := x509.ParseCertificate(der)
	if err != nil {
		return nil, err
	}

	wp := weak.Make(cert)
	if entry, loaded := wcc.LoadOrStore(string(der), wp); !loaded {
		runtime.AddCleanup(cert, func(_ any) { wcc.CompareAndDelete(string(der), entry) }, any(string(der)))
	} else if v := entry.(weak.Pointer[x509.Certificate]).Value(); v != nil {
		return v, nil
	} else {
		if wcc.CompareAndSwap(string(der), entry, wp) {
			runtime.AddCleanup(cert, func(_ any) { wcc.CompareAndDelete(string(der), wp) }, any(string(der)))
		}
	}
	return cert, nil
}

var globalCertCache = new(weakCertCache)
