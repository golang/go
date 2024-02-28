// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package godebugs provides a table of known GODEBUG settings,
// for use by a variety of other packages, including internal/godebug,
// runtime, runtime/metrics, and cmd/go/internal/load.
package godebugs

// An Info describes a single known GODEBUG setting.
type Info struct {
	Name    string // name of the setting ("panicnil")
	Package string // package that uses the setting ("runtime")
	Changed int    // minor version when default changed, if any; 21 means Go 1.21
	Old     string // value that restores behavior prior to Changed
	Opaque  bool   // setting does not export information to runtime/metrics using [internal/godebug.Setting.IncNonDefault]
}

// All is the table of known settings, sorted by Name.
//
// Note: After adding entries to this table, run 'go generate runtime/metrics'
// to update the runtime/metrics doc comment.
// (Otherwise the runtime/metrics test will fail.)
//
// Note: After adding entries to this table, update the list in doc/godebug.md as well.
// (Otherwise the test in this package will fail.)
var All = []Info{
	{Name: "execerrdot", Package: "os/exec"},
	{Name: "gocachehash", Package: "cmd/go"},
	{Name: "gocachetest", Package: "cmd/go"},
	{Name: "gocacheverify", Package: "cmd/go"},
	{Name: "gotypesalias", Package: "go/types"},
	{Name: "http2client", Package: "net/http"},
	{Name: "http2debug", Package: "net/http", Opaque: true},
	{Name: "http2server", Package: "net/http"},
	{Name: "httplaxcontentlength", Package: "net/http", Changed: 22, Old: "1"},
	{Name: "httpmuxgo121", Package: "net/http", Changed: 22, Old: "1"},
	{Name: "installgoroot", Package: "go/build"},
	{Name: "jstmpllitinterp", Package: "html/template"},
	//{Name: "multipartfiles", Package: "mime/multipart"},
	{Name: "multipartmaxheaders", Package: "mime/multipart"},
	{Name: "multipartmaxparts", Package: "mime/multipart"},
	{Name: "multipathtcp", Package: "net"},
	{Name: "netdns", Package: "net", Opaque: true},
	{Name: "panicnil", Package: "runtime", Changed: 21, Old: "1"},
	{Name: "randautoseed", Package: "math/rand"},
	{Name: "tarinsecurepath", Package: "archive/tar"},
	{Name: "tls10server", Package: "crypto/tls", Changed: 22, Old: "1"},
	{Name: "tlsmaxrsasize", Package: "crypto/tls"},
	{Name: "tlsrsakex", Package: "crypto/tls", Changed: 22, Old: "1"},
	{Name: "tlsunsafeekm", Package: "crypto/tls", Changed: 22, Old: "1"},
	{Name: "winreadlinkvolume", Package: "os", Changed: 22, Old: "0"},
	{Name: "winsymlink", Package: "os", Changed: 22, Old: "0"},
	{Name: "x509sha1", Package: "crypto/x509"},
	{Name: "x509usefallbackroots", Package: "crypto/x509"},
	{Name: "x509usepolicies", Package: "crypto/x509"},
	{Name: "zipinsecurepath", Package: "archive/zip"},
}

// Lookup returns the Info with the given name.
func Lookup(name string) *Info {
	// binary search, avoiding import of sort.
	lo := 0
	hi := len(All)
	for lo < hi {
		m := int(uint(lo+hi) >> 1)
		mid := All[m].Name
		if name == mid {
			return &All[m]
		}
		if name < mid {
			hi = m
		} else {
			lo = m + 1
		}
	}
	return nil
}
