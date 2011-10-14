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

	var prev *syscall.CertContext
	for {
		cur := syscall.CertEnumCertificatesInStore(store, prev)
		if cur == nil {
			break
		}

		var buf []byte
		hdrp := (*reflect.SliceHeader)(unsafe.Pointer(&buf))
		hdrp.Data = cur.EncodedCert
		hdrp.Len = int(cur.Length)
		hdrp.Cap = int(cur.Length)

		cert, err := x509.ParseCertificate(buf)
		if err != nil {
			continue
		}

		roots.AddCert(cert)
		prev = cur
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
