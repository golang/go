// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd linux openbsd netbsd

package tls

import (
	"crypto/x509"
	"io/ioutil"
)

// Possible certificate files; stop after finding one.
var certFiles = []string{
	"/etc/ssl/certs/ca-certificates.crt",     // Linux etc
	"/etc/pki/tls/certs/ca-bundle.crt",       // Fedora/RHEL
	"/etc/ssl/ca-bundle.pem",                 // OpenSUSE
	"/etc/ssl/cert.pem",                      // OpenBSD
	"/usr/local/share/certs/ca-root-nss.crt", // FreeBSD
}

func initDefaultRoots() {
	roots := x509.NewCertPool()
	for _, file := range certFiles {
		data, err := ioutil.ReadFile(file)
		if err == nil {
			roots.AppendCertsFromPEM(data)
			break
		}
	}
	varDefaultRoots = roots
}
