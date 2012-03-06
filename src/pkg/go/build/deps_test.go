// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file exercises the import parser but also checks that
// some low-level packages do not have new dependencies added.

package build_test

import (
	"go/build"
	"sort"
	"testing"
)

// pkgDeps defines the expected dependencies between packages in
// the Go source tree.  It is a statement of policy.
// Changes should not be made to this map without prior discussion.
//
// The map contains two kinds of entries:
// 1) Lower-case keys are standard import paths and list the
// allowed imports in that package.
// 2) Upper-case keys define aliases for package sets, which can then
// be used as dependencies by other rules.
//
// DO NOT CHANGE THIS DATA TO FIX BUILDS.
// 
var pkgDeps = map[string][]string{
	// L0 is the lowest level, core, nearly unavoidable packages.
	"errors":      {},
	"io":          {"errors", "sync"},
	"runtime":     {"unsafe"},
	"sync":        {"sync/atomic"},
	"sync/atomic": {"unsafe"},
	"unsafe":      {},

	"L0": {
		"errors",
		"io",
		"runtime",
		"sync",
		"sync/atomic",
		"unsafe",
	},

	// L1 adds simple data and functions, most notably
	// Unicode and strings processing.
	"bufio":         {"L0", "unicode/utf8", "bytes"},
	"bytes":         {"L0", "unicode", "unicode/utf8"},
	"math":          {"unsafe"},
	"math/cmplx":    {"math"},
	"math/rand":     {"L0", "math"},
	"path":          {"L0", "unicode/utf8", "strings"},
	"sort":          {"math"},
	"strconv":       {"L0", "unicode", "unicode/utf8", "math", "strings"},
	"strings":       {"L0", "unicode", "unicode/utf8"},
	"unicode":       {},
	"unicode/utf16": {},
	"unicode/utf8":  {},

	"L1": {
		"L0",
		"bufio",
		"bytes",
		"math",
		"math/cmplx",
		"math/rand",
		"path",
		"sort",
		"strconv",
		"strings",
		"unicode",
		"unicode/utf16",
		"unicode/utf8",
	},

	// L2 adds reflection and some basic utility packages
	// and interface definitions, but nothing that makes
	// system calls.
	"crypto":          {"L1", "hash"}, // interfaces
	"crypto/cipher":   {"L1"},         // interfaces
	"encoding/base32": {"L1"},
	"encoding/base64": {"L1"},
	"encoding/binary": {"L1", "reflect"},
	"hash":            {"L1"}, // interfaces
	"hash/adler32":    {"L1", "hash"},
	"hash/crc32":      {"L1", "hash"},
	"hash/crc64":      {"L1", "hash"},
	"hash/fnv":        {"L1", "hash"},
	"image":           {"L1", "image/color"}, // interfaces
	"image/color":     {"L1"},                // interfaces
	"reflect":         {"L1"},

	"L2": {
		"L1",
		"crypto",
		"crypto/cipher",
		"encoding/base32",
		"encoding/base64",
		"encoding/binary",
		"hash",
		"hash/adler32",
		"hash/crc32",
		"hash/crc64",
		"hash/fnv",
		"image",
		"image/color",
		"reflect",
	},

	// End of linear dependency definitions.

	// Operating system access.
	"syscall":       {"L0", "unicode/utf16"},
	"time":          {"L0", "syscall"},
	"os":            {"L0", "os", "syscall", "time", "unicode/utf16"},
	"path/filepath": {"L1", "os"},
	"io/ioutil":     {"L1", "os", "path/filepath", "time"},
	"os/exec":       {"L1", "os", "syscall"},
	"os/signal":     {"L1", "os", "syscall"},

	// OS enables basic operating system functionality,
	// but not direct use of package syscall, nor os/signal.
	"OS": {
		"io/ioutil",
		"os",
		"os/exec",
		"path/filepath",
		"time",
	},

	// Formatted I/O.
	"fmt": {"L1", "OS", "reflect"},
	"log": {"L1", "OS", "fmt"},

	// Packages used by testing must be low-level (L1+fmt).
	"regexp":         {"L1", "regexp/syntax"},
	"regexp/syntax":  {"L1"},
	"runtime/debug":  {"L1", "fmt", "io/ioutil", "os"},
	"runtime/pprof":  {"L1", "fmt", "text/tabwriter"},
	"text/tabwriter": {"L1"},

	"testing":        {"L1", "flag", "fmt", "os", "runtime/pprof", "time"},
	"testing/iotest": {"L1", "log"},
	"testing/quick":  {"L1", "flag", "fmt", "reflect"},

	// L3 is defined as L2+fmt+log+time, because in general once
	// you're using L2 packages, use of fmt, log, or time is not a big deal.
	"L3": {
		"L2",
		"fmt",
		"log",
		"time",
	},

	// Go parser.
	"go/ast":     {"L3", "OS", "go/scanner", "go/token"},
	"go/doc":     {"L3", "go/ast", "go/token", "regexp", "text/template"},
	"go/parser":  {"L3", "OS", "go/ast", "go/scanner", "go/token"},
	"go/printer": {"L3", "OS", "go/ast", "go/scanner", "go/token", "text/tabwriter"},
	"go/scanner": {"L3", "OS", "go/token"},
	"go/token":   {"L3"},

	"GOPARSER": {
		"go/ast",
		"go/doc",
		"go/parser",
		"go/printer",
		"go/scanner",
		"go/token",
	},

	// One of a kind.
	"archive/tar":         {"L3", "OS"},
	"archive/zip":         {"L3", "OS", "compress/flate"},
	"compress/bzip2":      {"L3"},
	"compress/flate":      {"L3"},
	"compress/gzip":       {"L3", "compress/flate"},
	"compress/lzw":        {"L3"},
	"compress/zlib":       {"L3", "compress/flate"},
	"database/sql":        {"L3", "database/sql/driver"},
	"database/sql/driver": {"L3", "time"},
	"debug/dwarf":         {"L3"},
	"debug/elf":           {"L3", "OS", "debug/dwarf"},
	"debug/gosym":         {"L3"},
	"debug/macho":         {"L3", "OS", "debug/dwarf"},
	"debug/pe":            {"L3", "OS", "debug/dwarf"},
	"encoding/ascii85":    {"L3"},
	"encoding/asn1":       {"L3", "math/big"},
	"encoding/csv":        {"L3"},
	"encoding/gob":        {"L3", "OS"},
	"encoding/hex":        {"L3"},
	"encoding/json":       {"L3"},
	"encoding/pem":        {"L3"},
	"encoding/xml":        {"L3"},
	"flag":                {"L3", "OS"},
	"go/build":            {"L3", "OS", "GOPARSER"},
	"html":                {"L3"},
	"image/draw":          {"L3"},
	"image/gif":           {"L3", "compress/lzw"},
	"image/jpeg":          {"L3"},
	"image/png":           {"L3", "compress/zlib"},
	"index/suffixarray":   {"L3", "regexp"},
	"math/big":            {"L3"},
	"mime":                {"L3", "OS", "syscall"},
	"net/url":             {"L3"},
	"text/scanner":        {"L3", "OS"},
	"text/template/parse": {"L3"},

	"html/template": {
		"L3", "OS", "encoding/json", "html", "text/template",
		"text/template/parse",
	},
	"text/template": {
		"L3", "OS", "net/url", "text/template/parse",
	},

	// Cgo.
	"runtime/cgo": {"L0", "C"},
	"CGO":         {"C", "runtime/cgo"},

	// Fake entry to satisfy the pseudo-import "C"
	// that shows up in programs that use cgo.
	"C": {},

	"os/user": {"L3", "CGO", "syscall"},

	// Basic networking.
	// TODO: Remove reflect, possibly math/rand.
	"net": {"L0", "CGO", "math/rand", "os", "reflect", "sort", "syscall", "time"},

	// NET enables use of basic network-related packages.
	"NET": {
		"net",
		"mime",
		"net/textproto",
		"net/url",
	},

	// Uses of networking.
	"log/syslog":    {"L3", "OS", "net"},
	"net/mail":      {"L3", "NET", "OS"},
	"net/textproto": {"L3", "OS", "net"},

	// Core crypto.
	"crypto/aes":    {"L2"},
	"crypto/des":    {"L2"},
	"crypto/hmac":   {"L2"},
	"crypto/md5":    {"L2"},
	"crypto/rc4":    {"L2"},
	"crypto/sha1":   {"L2"},
	"crypto/sha256": {"L2"},
	"crypto/sha512": {"L2"},
	"crypto/subtle": {"L2"},

	"CRYPTO": {
		"crypto/aes",
		"crypto/des",
		"crypto/hmac",
		"crypto/md5",
		"crypto/rc4",
		"crypto/sha1",
		"crypto/sha256",
		"crypto/sha512",
		"crypto/subtle",
	},

	// Random byte, number generation.
	// This would be part of core crypto except that it imports
	// math/big, which imports fmt.
	"crypto/rand": {"L3", "CRYPTO", "OS", "math/big", "syscall"},

	// Mathematical crypto: dependencies on fmt (L3) and math/big.
	// We could avoid some of the fmt, but math/big imports fmt anyway.
	"crypto/dsa":      {"L3", "CRYPTO", "math/big"},
	"crypto/ecdsa":    {"L3", "CRYPTO", "crypto/elliptic", "math/big"},
	"crypto/elliptic": {"L3", "CRYPTO", "math/big"},
	"crypto/rsa":      {"L3", "CRYPTO", "crypto/rand", "math/big"},

	"CRYPTO-MATH": {
		"CRYPTO",
		"crypto/dsa",
		"crypto/ecdsa",
		"crypto/elliptic",
		"crypto/rand",
		"crypto/rsa",
		"encoding/asn1",
		"math/big",
	},

	// SSL/TLS.
	"crypto/tls": {
		"L3", "CRYPTO-MATH", "CGO", "OS",
		"crypto/x509", "encoding/pem", "net", "syscall",
	},
	"crypto/x509":      {"L3", "CRYPTO-MATH", "crypto/x509/pkix", "encoding/pem"},
	"crypto/x509/pkix": {"L3", "CRYPTO-MATH"},

	// Simple net+crypto-aware packages.
	"mime/multipart": {"L3", "OS", "mime", "crypto/rand", "net/textproto"},
	"net/smtp":       {"L3", "CRYPTO", "NET", "crypto/tls"},

	// HTTP, kingpin of dependencies.
	"net/http": {
		"L3", "NET", "OS",
		"compress/gzip", "crypto/tls", "mime/multipart", "runtime/debug",
	},

	// HTTP-using packages.
	"expvar":            {"L3", "OS", "encoding/json", "net/http"},
	"net/http/cgi":      {"L3", "NET", "OS", "crypto/tls", "net/http", "regexp"},
	"net/http/fcgi":     {"L3", "NET", "OS", "net/http", "net/http/cgi"},
	"net/http/httptest": {"L3", "NET", "OS", "crypto/tls", "flag", "net/http"},
	"net/http/httputil": {"L3", "NET", "OS", "net/http"},
	"net/http/pprof":    {"L3", "OS", "html/template", "net/http", "runtime/pprof"},
	"net/rpc":           {"L3", "NET", "encoding/gob", "net/http", "text/template"},
	"net/rpc/jsonrpc":   {"L3", "NET", "encoding/json", "net/rpc"},
}

// isMacro reports whether p is a package dependency macro
// (uppercase name).
func isMacro(p string) bool {
	return 'A' <= p[0] && p[0] <= 'Z'
}

func allowed(pkg string) map[string]bool {
	m := map[string]bool{}
	var allow func(string)
	allow = func(p string) {
		if m[p] {
			return
		}
		m[p] = true // set even for macros, to avoid loop on cycle

		// Upper-case names are macro-expanded.
		if isMacro(p) {
			for _, pp := range pkgDeps[p] {
				allow(pp)
			}
		}
	}
	for _, pp := range pkgDeps[pkg] {
		allow(pp)
	}
	return m
}

var bools = []bool{false, true}
var geese = []string{"darwin", "freebsd", "linux", "netbsd", "openbsd", "plan9", "windows"}
var goarches = []string{"386", "amd64", "arm"}

type osPkg struct {
	goos, pkg string
}

// allowedErrors are the operating systems and packages known to contain errors
// (currently just "no Go source files")
var allowedErrors = map[osPkg]bool{
	osPkg{"windows", "log/syslog"}: true,
	osPkg{"plan9", "log/syslog"}:   true,
}

func TestDependencies(t *testing.T) {
	var all []string

	for k := range pkgDeps {
		all = append(all, k)
	}
	sort.Strings(all)

	ctxt := build.Default
	test := func(mustImport bool) {
		for _, pkg := range all {
			if isMacro(pkg) {
				continue
			}
			p, err := ctxt.Import(pkg, "", 0)
			if err != nil {
				if allowedErrors[osPkg{ctxt.GOOS, pkg}] {
					continue
				}
				// Some of the combinations we try might not
				// be reasonable (like arm,plan9,cgo), so ignore
				// errors for the auto-generated combinations.
				if !mustImport {
					continue
				}
				t.Errorf("%s/%s/cgo=%v %v", ctxt.GOOS, ctxt.GOARCH, ctxt.CgoEnabled, err)
				continue
			}
			ok := allowed(pkg)
			var bad []string
			for _, imp := range p.Imports {
				if !ok[imp] {
					bad = append(bad, imp)
				}
			}
			if bad != nil {
				t.Errorf("%s/%s/cgo=%v unexpected dependency: %s imports %v", ctxt.GOOS, ctxt.GOARCH, ctxt.CgoEnabled, pkg, bad)
			}
		}
	}
	test(true)

	if testing.Short() {
		t.Logf("skipping other systems")
		return
	}

	for _, ctxt.GOOS = range geese {
		for _, ctxt.GOARCH = range goarches {
			for _, ctxt.CgoEnabled = range bools {
				test(false)
			}
		}
	}
}
