// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/x509"
	"reflect"
	"syscall"
	"unsafe"
)

func loadStore(roots *x509.CertPool, name string) {
	store, errno := syscall.CertOpenSystemStore(syscall.InvalidHandle, syscall.StringToUTF16Ptr(name))
	if errno != 0 {
		return
	}

	var cert *syscall.CertContext
	for {
		cert = syscall.CertEnumCertificatesInStore(store, cert)
		if cert == nil {
			break
		}

		var asn1Slice []byte
		hdrp := (*reflect.SliceHeader)(unsafe.Pointer(&asn1Slice))
		hdrp.Data = cert.EncodedCert
		hdrp.Len = int(cert.Length)
		hdrp.Cap = int(cert.Length)

		buf := make([]byte, len(asn1Slice))
		copy(buf, asn1Slice)

		if cert, err := x509.ParseCertificate(buf); err == nil {
			roots.AddCert(cert)
		}
	}

	syscall.CertCloseStore(store, 0)
}

func initDefaultRoots() {
	roots := x509.NewCertPool()

	// Roots
	loadStore(roots, "ROOT")

	// Intermediates
	loadStore(roots, "CA")

	varDefaultRoots = roots
}
