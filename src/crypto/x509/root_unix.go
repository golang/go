// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix dragonfly freebsd js,wasm linux netbsd openbsd solaris

package x509

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

// Possible directories with certificate files; stop after successfully
// reading at least one file from a directory.
var certDirectories = []string{
	"/etc/ssl/certs",               // SLES10/SLES11, https://golang.org/issue/12139
	"/system/etc/security/cacerts", // Android
	"/usr/local/share/certs",       // FreeBSD
	"/etc/pki/tls/certs",           // Fedora/RHEL
	"/etc/openssl/certs",           // NetBSD
	"/var/ssl/certs",               // AIX
}

const (
	// certFileEnv is the environment variable which identifies where to locate
	// the SSL certificate file. If set this overrides the system default.
	certFileEnv = "SSL_CERT_FILE"

	// certDirEnv is the environment variable which identifies which directory
	// to check for SSL certificate files. If set this overrides the system default.
	// It is a colon separated list of directories.
	// See https://www.openssl.org/docs/man1.0.2/man1/c_rehash.html.
	certDirEnv = "SSL_CERT_DIR"
)

func (c *Certificate) systemVerify(opts *VerifyOptions) (chains [][]*Certificate, err error) {
	return nil, nil
}

func loadSystemRoots() (*CertPool, error) {
	roots := NewCertPool()

	files := certFiles
	if f := os.Getenv(certFileEnv); f != "" {
		files = []string{f}
	}

	var firstErr error
	for _, file := range files {
		data, err := ioutil.ReadFile(file)
		if err == nil {
			roots.AppendCertsFromPEM(data)
			break
		}
		if firstErr == nil && !os.IsNotExist(err) {
			firstErr = err
		}
	}

	dirs := certDirectories
	if d := os.Getenv(certDirEnv); d != "" {
		// OpenSSL and BoringSSL both use ":" as the SSL_CERT_DIR separator.
		// See:
		//  * https://golang.org/issue/35325
		//  * https://www.openssl.org/docs/man1.0.2/man1/c_rehash.html
		dirs = strings.Split(d, ":")
	}

	for _, directory := range dirs {
		fis, err := readUniqueDirectoryEntries(directory)
		if err != nil {
			if firstErr == nil && !os.IsNotExist(err) {
				firstErr = err
			}
			continue
		}
		for _, fi := range fis {
			data, err := ioutil.ReadFile(directory + "/" + fi.Name())
			if err == nil {
				roots.AppendCertsFromPEM(data)
			}
		}
	}

	if len(roots.certs) > 0 || firstErr == nil {
		return roots, nil
	}

	return nil, firstErr
}

// readUniqueDirectoryEntries is like ioutil.ReadDir but omits
// symlinks that point within the directory.
func readUniqueDirectoryEntries(dir string) ([]os.FileInfo, error) {
	fis, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	uniq := fis[:0]
	for _, fi := range fis {
		if !isSameDirSymlink(fi, dir) {
			uniq = append(uniq, fi)
		}
	}
	return uniq, nil
}

// isSameDirSymlink reports whether fi in dir is a symlink with a
// target not containing a slash.
func isSameDirSymlink(fi os.FileInfo, dir string) bool {
	if fi.Mode()&os.ModeSymlink == 0 {
		return false
	}
	target, err := os.Readlink(filepath.Join(dir, fi.Name()))
	return err == nil && !strings.Contains(target, "/")
}
