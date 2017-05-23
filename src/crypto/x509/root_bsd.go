// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd netbsd openbsd

package x509

// Possible certificate files; stop after finding one.
var certFiles = []string{
	"/usr/local/share/certs/ca-root-nss.crt", // FreeBSD/DragonFly
	"/etc/ssl/cert.pem",                      // OpenBSD
	"/etc/openssl/certs/ca-certificates.crt", // NetBSD
}
