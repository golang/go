// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file exercises the import parser but also checks that
// some low-level packages do not have new dependencies added.

package build

import (
	"runtime"
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
	"sync":        {"runtime", "sync/atomic", "unsafe"},
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

	// L1 adds simple functions and strings processing,
	// but not Unicode tables.
	"math":          {"unsafe"},
	"math/cmplx":    {"math"},
	"math/rand":     {"L0", "math"},
	"sort":          {},
	"strconv":       {"L0", "unicode/utf8", "math"},
	"unicode/utf16": {},
	"unicode/utf8":  {},

	"L1": {
		"L0",
		"math",
		"math/cmplx",
		"math/rand",
		"sort",
		"strconv",
		"unicode/utf16",
		"unicode/utf8",
	},

	// L2 adds Unicode and strings processing.
	"bufio":   {"L0", "unicode/utf8", "bytes"},
	"bytes":   {"L0", "unicode", "unicode/utf8"},
	"path":    {"L0", "unicode/utf8", "strings"},
	"strings": {"L0", "unicode", "unicode/utf8"},
	"unicode": {},

	"L2": {
		"L1",
		"bufio",
		"bytes",
		"path",
		"strings",
		"unicode",
	},

	// L3 adds reflection and some basic utility packages
	// and interface definitions, but nothing that makes
	// system calls.
	"crypto":              {"L2", "hash"},          // interfaces
	"crypto/cipher":       {"L2", "crypto/subtle"}, // interfaces
	"crypto/subtle":       {},
	"encoding/base32":     {"L2"},
	"encoding/base64":     {"L2"},
	"encoding/binary":     {"L2", "reflect"},
	"hash":                {"L2"}, // interfaces
	"hash/adler32":        {"L2", "hash"},
	"hash/crc32":          {"L2", "hash"},
	"hash/crc64":          {"L2", "hash"},
	"hash/fnv":            {"L2", "hash"},
	"image":               {"L2", "image/color"}, // interfaces
	"image/color":         {"L2"},                // interfaces
	"image/color/palette": {"L2", "image/color"},
	"reflect":             {"L2"},

	"L3": {
		"L2",
		"crypto",
		"crypto/cipher",
		"crypto/subtle",
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
		"image/color/palette",
		"reflect",
	},

	// End of linear dependency definitions.

	// Operating system access.
	"syscall":       {"L0", "unicode/utf16"},
	"time":          {"L0", "syscall"},
	"os":            {"L1", "os", "syscall", "time"},
	"path/filepath": {"L2", "os", "syscall"},
	"io/ioutil":     {"L2", "os", "path/filepath", "time"},
	"os/exec":       {"L2", "os", "path/filepath", "syscall"},
	"os/signal":     {"L2", "os", "syscall"},

	// OS enables basic operating system functionality,
	// but not direct use of package syscall, nor os/signal.
	"OS": {
		"io/ioutil",
		"os",
		"os/exec",
		"path/filepath",
		"time",
	},

	// Formatted I/O: few dependencies (L1) but we must add reflect.
	"fmt": {"L1", "os", "reflect"},
	"log": {"L1", "os", "fmt", "time"},

	// Packages used by testing must be low-level (L2+fmt).
	"regexp":         {"L2", "regexp/syntax"},
	"regexp/syntax":  {"L2"},
	"runtime/debug":  {"L2", "fmt", "io/ioutil", "os", "time"},
	"runtime/pprof":  {"L2", "fmt", "text/tabwriter"},
	"text/tabwriter": {"L2"},

	"testing":        {"L2", "flag", "fmt", "os", "runtime/pprof", "time"},
	"testing/iotest": {"L2", "log"},
	"testing/quick":  {"L2", "flag", "fmt", "reflect"},

	// L4 is defined as L3+fmt+log+time, because in general once
	// you're using L3 packages, use of fmt, log, or time is not a big deal.
	"L4": {
		"L3",
		"fmt",
		"log",
		"time",
	},

	// Go parser.
	"go/ast":     {"L4", "OS", "go/scanner", "go/token"},
	"go/doc":     {"L4", "go/ast", "go/token", "regexp", "text/template"},
	"go/parser":  {"L4", "OS", "go/ast", "go/scanner", "go/token"},
	"go/printer": {"L4", "OS", "go/ast", "go/scanner", "go/token", "text/tabwriter"},
	"go/scanner": {"L4", "OS", "go/token"},
	"go/token":   {"L4"},

	"GOPARSER": {
		"go/ast",
		"go/doc",
		"go/parser",
		"go/printer",
		"go/scanner",
		"go/token",
	},

	// One of a kind.
	"archive/tar":         {"L4", "OS", "syscall"},
	"archive/zip":         {"L4", "OS", "compress/flate"},
	"compress/bzip2":      {"L4"},
	"compress/flate":      {"L4"},
	"compress/gzip":       {"L4", "compress/flate"},
	"compress/lzw":        {"L4"},
	"compress/zlib":       {"L4", "compress/flate"},
	"database/sql":        {"L4", "container/list", "database/sql/driver"},
	"database/sql/driver": {"L4", "time"},
	"debug/dwarf":         {"L4"},
	"debug/elf":           {"L4", "OS", "debug/dwarf"},
	"debug/gosym":         {"L4"},
	"debug/macho":         {"L4", "OS", "debug/dwarf"},
	"debug/pe":            {"L4", "OS", "debug/dwarf"},
	"encoding":            {"L4"},
	"encoding/ascii85":    {"L4"},
	"encoding/asn1":       {"L4", "math/big"},
	"encoding/csv":        {"L4"},
	"encoding/gob":        {"L4", "OS", "encoding"},
	"encoding/hex":        {"L4"},
	"encoding/json":       {"L4", "encoding"},
	"encoding/pem":        {"L4"},
	"encoding/xml":        {"L4", "encoding"},
	"flag":                {"L4", "OS"},
	"go/build":            {"L4", "OS", "GOPARSER"},
	"html":                {"L4"},
	"image/draw":          {"L4"},
	"image/gif":           {"L4", "compress/lzw", "image/color/palette", "image/draw"},
	"image/jpeg":          {"L4"},
	"image/png":           {"L4", "compress/zlib"},
	"index/suffixarray":   {"L4", "regexp"},
	"math/big":            {"L4"},
	"mime":                {"L4", "OS", "syscall"},
	"net/url":             {"L4"},
	"text/scanner":        {"L4", "OS"},
	"text/template/parse": {"L4"},

	"html/template": {
		"L4", "OS", "encoding/json", "html", "text/template",
		"text/template/parse",
	},
	"text/template": {
		"L4", "OS", "net/url", "text/template/parse",
	},

	// Cgo.
	"runtime/cgo": {"L0", "C"},
	"CGO":         {"C", "runtime/cgo"},

	// Fake entry to satisfy the pseudo-import "C"
	// that shows up in programs that use cgo.
	"C": {},

	// Plan 9 alone needs io/ioutil and os.
	"os/user": {"L4", "CGO", "io/ioutil", "os", "syscall"},

	// Basic networking.
	// Because net must be used by any package that wants to
	// do networking portably, it must have a small dependency set: just L1+basic os.
	"net": {"L1", "CGO", "os", "syscall", "time"},

	// NET enables use of basic network-related packages.
	"NET": {
		"net",
		"mime",
		"net/textproto",
		"net/url",
	},

	// Uses of networking.
	"log/syslog":    {"L4", "OS", "net"},
	"net/mail":      {"L4", "NET", "OS"},
	"net/textproto": {"L4", "OS", "net"},

	// Core crypto.
	"crypto/aes":    {"L3"},
	"crypto/des":    {"L3"},
	"crypto/hmac":   {"L3"},
	"crypto/md5":    {"L3"},
	"crypto/rc4":    {"L3"},
	"crypto/sha1":   {"L3"},
	"crypto/sha256": {"L3"},
	"crypto/sha512": {"L3"},

	"CRYPTO": {
		"crypto/aes",
		"crypto/des",
		"crypto/hmac",
		"crypto/md5",
		"crypto/rc4",
		"crypto/sha1",
		"crypto/sha256",
		"crypto/sha512",
	},

	// Random byte, number generation.
	// This would be part of core crypto except that it imports
	// math/big, which imports fmt.
	"crypto/rand": {"L4", "CRYPTO", "OS", "math/big", "syscall", "internal/syscall"},

	// Mathematical crypto: dependencies on fmt (L4) and math/big.
	// We could avoid some of the fmt, but math/big imports fmt anyway.
	"crypto/dsa":      {"L4", "CRYPTO", "math/big"},
	"crypto/ecdsa":    {"L4", "CRYPTO", "crypto/elliptic", "math/big", "encoding/asn1"},
	"crypto/elliptic": {"L4", "CRYPTO", "math/big"},
	"crypto/rsa":      {"L4", "CRYPTO", "crypto/rand", "math/big"},

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
		"L4", "CRYPTO-MATH", "CGO", "OS",
		"container/list", "crypto/x509", "encoding/pem", "net", "syscall",
	},
	"crypto/x509": {
		"L4", "CRYPTO-MATH", "OS", "CGO",
		"crypto/x509/pkix", "encoding/pem", "encoding/hex", "net", "syscall",
	},
	"crypto/x509/pkix": {"L4", "CRYPTO-MATH"},

	// Simple net+crypto-aware packages.
	"mime/multipart": {"L4", "OS", "mime", "crypto/rand", "net/textproto"},
	"net/smtp":       {"L4", "CRYPTO", "NET", "crypto/tls"},

	// HTTP, kingpin of dependencies.
	"net/http": {
		"L4", "NET", "OS",
		"compress/gzip", "crypto/tls", "mime/multipart", "runtime/debug",
		"net/http/internal",
	},

	// HTTP-using packages.
	"expvar":            {"L4", "OS", "encoding/json", "net/http"},
	"net/http/cgi":      {"L4", "NET", "OS", "crypto/tls", "net/http", "regexp"},
	"net/http/fcgi":     {"L4", "NET", "OS", "net/http", "net/http/cgi"},
	"net/http/httptest": {"L4", "NET", "OS", "crypto/tls", "flag", "net/http"},
	"net/http/httputil": {"L4", "NET", "OS", "net/http", "net/http/internal"},
	"net/http/pprof":    {"L4", "OS", "html/template", "net/http", "runtime/pprof"},
	"net/rpc":           {"L4", "NET", "encoding/gob", "html/template", "net/http"},
	"net/rpc/jsonrpc":   {"L4", "NET", "encoding/json", "net/rpc"},
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
var geese = []string{"android", "darwin", "dragonfly", "freebsd", "linux", "nacl", "netbsd", "openbsd", "plan9", "solaris", "windows"}
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
	if runtime.GOOS == "nacl" {
		// NaCl tests run in a limited file system and we do not
		// provide access to every source file.
		t.Skip("skipping on NaCl")
	}
	var all []string

	for k := range pkgDeps {
		all = append(all, k)
	}
	sort.Strings(all)

	ctxt := Default
	test := func(mustImport bool) {
		for _, pkg := range all {
			if isMacro(pkg) {
				continue
			}
			if pkg == "runtime/cgo" && !ctxt.CgoEnabled {
				continue
			}
			p, err := ctxt.Import(pkg, "", 0)
			if err != nil {
				if allowedErrors[osPkg{ctxt.GOOS, pkg}] {
					continue
				}
				if !ctxt.CgoEnabled && pkg == "runtime/cgo" {
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
