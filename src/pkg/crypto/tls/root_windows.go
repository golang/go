// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/x509"
	"syscall"
	"unsafe"
)

func loadStore(roots *x509.CertPool, name string) {
	store, err := syscall.CertOpenSystemStore(syscall.InvalidHandle, syscall.StringToUTF16Ptr(name))
	if err != nil {
		return
	}
	defer syscall.CertCloseStore(store, 0)

	var cert *syscall.CertContext
	for {
		cert, err = syscall.CertEnumCertificatesInStore(store, cert)
		if err != nil {
			return
		}

		buf := (*[1 << 20]byte)(unsafe.Pointer(cert.EncodedCert))[:]
		// ParseCertificate requires its own copy of certificate data to keep.
		buf2 := make([]byte, cert.Length)
		copy(buf2, buf)
		if c, err := x509.ParseCertificate(buf2); err == nil {
			roots.AddCert(c)
		}
	}
}

func initDefaultRoots() {
	roots := x509.NewCertPool()

	// Roots
	loadStore(roots, "ROOT")

	// Intermediates
	loadStore(roots, "CA")

	varDefaultRoots = roots
}
