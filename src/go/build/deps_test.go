// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file exercises the import parser but also checks that
// some low-level packages do not have new dependencies added.

package build

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// pkgDeps defines the expected dependencies between packages in
// the Go source tree. It is a statement of policy.
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
	"errors":                  {"runtime", "internal/reflectlite"},
	"io":                      {"errors", "sync", "sync/atomic"},
	"runtime":                 {"unsafe", "runtime/internal/atomic", "runtime/internal/sys", "runtime/internal/math", "internal/cpu", "internal/bytealg"},
	"runtime/internal/sys":    {},
	"runtime/internal/atomic": {"unsafe", "internal/cpu"},
	"runtime/internal/math":   {"runtime/internal/sys"},
	"internal/race":           {"runtime", "unsafe"},
	"sync":                    {"internal/race", "runtime", "sync/atomic", "unsafe"},
	"sync/atomic":             {"unsafe"},
	"unsafe":                  {},
	"internal/cpu":            {},
	"internal/bytealg":        {"unsafe", "internal/cpu"},
	"internal/reflectlite":    {"runtime", "unsafe"},

	"L0": {
		"errors",
		"io",
		"runtime",
		"runtime/internal/atomic",
		"sync",
		"sync/atomic",
		"unsafe",
		"internal/cpu",
		"internal/bytealg",
		"internal/reflectlite",
	},

	// L1 adds simple functions and strings processing,
	// but not Unicode tables.
	"math":          {"internal/cpu", "unsafe", "math/bits"},
	"math/bits":     {"unsafe"},
	"math/cmplx":    {"math"},
	"math/rand":     {"L0", "math"},
	"strconv":       {"L0", "unicode/utf8", "math", "math/bits"},
	"unicode/utf16": {},
	"unicode/utf8":  {},

	"L1": {
		"L0",
		"math",
		"math/bits",
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
	"crypto":                 {"L2", "hash"}, // interfaces
	"crypto/cipher":          {"L2", "crypto/subtle", "crypto/internal/subtle", "encoding/binary"},
	"crypto/internal/subtle": {"unsafe", "reflect"}, // reflect behind a appengine tag
	"crypto/subtle":          {},
	"encoding/base32":        {"L2"},
	"encoding/base64":        {"L2", "encoding/binary"},
	"encoding/binary":        {"L2", "reflect"},
	"hash":                   {"L2"}, // interfaces
	"hash/adler32":           {"L2", "hash"},
	"hash/crc32":             {"L2", "hash"},
	"hash/crc64":             {"L2", "hash"},
	"hash/fnv":               {"L2", "hash"},
	"hash/maphash":           {"L2", "hash"},
	"image":                  {"L2", "image/color"}, // interfaces
	"image/color":            {"L2"},                // interfaces
	"image/color/palette":    {"L2", "image/color"},
	"internal/fmtsort":       {"reflect", "sort"},
	"reflect":                {"L2"},
	"sort":                   {"internal/reflectlite"},

	"L3": {
		"L2",
		"crypto",
		"crypto/cipher",
		"crypto/internal/subtle",
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
		"internal/fmtsort",
		"internal/oserror",
		"reflect",
	},

	// End of linear dependency definitions.

	// Operating system access.
	"syscall":                           {"L0", "internal/oserror", "internal/race", "internal/syscall/windows/sysdll", "syscall/js", "unicode/utf16"},
	"syscall/js":                        {"L0"},
	"internal/oserror":                  {"L0"},
	"internal/syscall/unix":             {"L0", "syscall"},
	"internal/syscall/windows":          {"L0", "syscall", "internal/syscall/windows/sysdll", "unicode/utf16"},
	"internal/syscall/windows/registry": {"L0", "syscall", "internal/syscall/windows/sysdll", "unicode/utf16"},
	"internal/syscall/execenv":          {"L0", "syscall", "internal/syscall/windows", "unicode/utf16"},
	"time": {
		// "L0" without the "io" package:
		"errors",
		"runtime",
		"runtime/internal/atomic",
		"sync",
		"sync/atomic",
		"unsafe",
		// Other time dependencies:
		"internal/syscall/windows/registry",
		"syscall",
		"syscall/js",
	},

	"internal/cfg":     {"L0"},
	"internal/poll":    {"L0", "internal/oserror", "internal/race", "syscall", "time", "unicode/utf16", "unicode/utf8", "internal/syscall/windows", "internal/syscall/unix"},
	"internal/testlog": {"L0"},
	"os":               {"L1", "os", "syscall", "time", "internal/oserror", "internal/poll", "internal/syscall/windows", "internal/syscall/unix", "internal/syscall/execenv", "internal/testlog"},
	"path/filepath":    {"L2", "os", "syscall", "internal/syscall/windows"},
	"io/ioutil":        {"L2", "os", "path/filepath", "time"},
	"os/exec":          {"L2", "os", "context", "path/filepath", "syscall", "internal/syscall/execenv"},
	"os/signal":        {"L2", "os", "syscall"},

	// OS enables basic operating system functionality,
	// but not direct use of package syscall, nor os/signal.
	"OS": {
		"io/ioutil",
		"os",
		"os/exec",
		"path/filepath",
		"time",
	},

	// Formatted I/O: few dependencies (L1) but we must add reflect and internal/fmtsort.
	"fmt": {"L1", "os", "reflect", "internal/fmtsort"},
	"log": {"L1", "os", "fmt", "time"},

	// Packages used by testing must be low-level (L2+fmt).
	"regexp":         {"L2", "regexp/syntax"},
	"regexp/syntax":  {"L2"},
	"runtime/debug":  {"L2", "fmt", "io/ioutil", "os", "time"},
	"runtime/pprof":  {"L2", "compress/gzip", "context", "encoding/binary", "fmt", "io/ioutil", "os", "text/tabwriter", "time"},
	"runtime/trace":  {"L0", "context", "fmt"},
	"text/tabwriter": {"L2"},

	"testing":                  {"L2", "flag", "fmt", "internal/race", "os", "runtime/debug", "runtime/pprof", "runtime/trace", "time"},
	"testing/iotest":           {"L2", "log"},
	"testing/quick":            {"L2", "flag", "fmt", "reflect", "time"},
	"internal/obscuretestdata": {"L2", "OS", "encoding/base64"},
	"internal/testenv":         {"L2", "OS", "flag", "testing", "syscall", "internal/cfg"},
	"internal/lazyregexp":      {"L2", "OS", "regexp"},
	"internal/lazytemplate":    {"L2", "OS", "text/template"},

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
	"go/doc":     {"L4", "OS", "go/ast", "go/token", "regexp", "internal/lazyregexp", "text/template"},
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

	"go/format":       {"L4", "GOPARSER", "internal/format"},
	"internal/format": {"L4", "GOPARSER"},

	// Go type checking.
	"go/constant":               {"L4", "go/token", "math/big"},
	"go/importer":               {"L4", "go/build", "go/internal/gccgoimporter", "go/internal/gcimporter", "go/internal/srcimporter", "go/token", "go/types"},
	"go/internal/gcimporter":    {"L4", "OS", "go/build", "go/constant", "go/token", "go/types", "text/scanner"},
	"go/internal/gccgoimporter": {"L4", "OS", "debug/elf", "go/constant", "go/token", "go/types", "internal/xcoff", "text/scanner"},
	"go/internal/srcimporter":   {"L4", "OS", "fmt", "go/ast", "go/build", "go/parser", "go/token", "go/types", "path/filepath"},
	"go/types":                  {"L4", "GOPARSER", "container/heap", "go/constant"},

	// One of a kind.
	"archive/tar":                    {"L4", "OS", "syscall", "os/user"},
	"archive/zip":                    {"L4", "OS", "compress/flate"},
	"container/heap":                 {"sort"},
	"compress/bzip2":                 {"L4"},
	"compress/flate":                 {"L4"},
	"compress/gzip":                  {"L4", "compress/flate"},
	"compress/lzw":                   {"L4"},
	"compress/zlib":                  {"L4", "compress/flate"},
	"context":                        {"errors", "internal/reflectlite", "sync", "sync/atomic", "time"},
	"database/sql":                   {"L4", "container/list", "context", "database/sql/driver", "database/sql/internal"},
	"database/sql/driver":            {"L4", "context", "time", "database/sql/internal"},
	"debug/dwarf":                    {"L4"},
	"debug/elf":                      {"L4", "OS", "debug/dwarf", "compress/zlib"},
	"debug/gosym":                    {"L4"},
	"debug/macho":                    {"L4", "OS", "debug/dwarf", "compress/zlib"},
	"debug/pe":                       {"L4", "OS", "debug/dwarf", "compress/zlib"},
	"debug/plan9obj":                 {"L4", "OS"},
	"encoding":                       {"L4"},
	"encoding/ascii85":               {"L4"},
	"encoding/asn1":                  {"L4", "math/big"},
	"encoding/csv":                   {"L4"},
	"encoding/gob":                   {"L4", "OS", "encoding"},
	"encoding/hex":                   {"L4"},
	"encoding/json":                  {"L4", "encoding"},
	"encoding/pem":                   {"L4"},
	"encoding/xml":                   {"L4", "encoding"},
	"flag":                           {"L4", "OS"},
	"go/build":                       {"L4", "OS", "GOPARSER", "internal/goroot", "internal/goversion"},
	"html":                           {"L4"},
	"image/draw":                     {"L4", "image/internal/imageutil"},
	"image/gif":                      {"L4", "compress/lzw", "image/color/palette", "image/draw"},
	"image/internal/imageutil":       {"L4"},
	"image/jpeg":                     {"L4", "image/internal/imageutil"},
	"image/png":                      {"L4", "compress/zlib"},
	"index/suffixarray":              {"L4", "regexp"},
	"internal/goroot":                {"L4", "OS"},
	"internal/singleflight":          {"sync"},
	"internal/trace":                 {"L4", "OS", "container/heap"},
	"internal/xcoff":                 {"L4", "OS", "debug/dwarf"},
	"math/big":                       {"L4"},
	"mime":                           {"L4", "OS", "syscall", "internal/syscall/windows/registry"},
	"mime/quotedprintable":           {"L4"},
	"net/internal/socktest":          {"L4", "OS", "syscall", "internal/syscall/windows"},
	"net/url":                        {"L4"},
	"plugin":                         {"L0", "OS", "CGO"},
	"runtime/pprof/internal/profile": {"L4", "OS", "compress/gzip", "regexp"},
	"testing/internal/testdeps":      {"L4", "internal/testlog", "runtime/pprof", "regexp"},
	"text/scanner":                   {"L4", "OS"},
	"text/template/parse":            {"L4"},

	"html/template": {
		"L4", "OS", "encoding/json", "html", "text/template",
		"text/template/parse",
	},
	"text/template": {
		"L4", "OS", "net/url", "text/template/parse",
	},

	// Cgo.
	// If you add a dependency on CGO, you must add the package to
	// cgoPackages in cmd/dist/test.go.
	"runtime/cgo": {"L0", "C"},
	"CGO":         {"C", "runtime/cgo"},

	// Fake entry to satisfy the pseudo-import "C"
	// that shows up in programs that use cgo.
	"C": {},

	// Race detector/MSan uses cgo.
	"runtime/race": {"C"},
	"runtime/msan": {"C"},

	// Plan 9 alone needs io/ioutil and os.
	"os/user": {"L4", "CGO", "io/ioutil", "os", "syscall", "internal/syscall/windows", "internal/syscall/windows/registry"},

	// Internal package used only for testing.
	"os/signal/internal/pty": {"CGO", "fmt", "os", "syscall"},

	// Basic networking.
	// Because net must be used by any package that wants to
	// do networking portably, it must have a small dependency set: just L0+basic os.
	"net": {
		"L0", "CGO",
		"context", "math/rand", "os", "sort", "syscall", "time",
		"internal/nettrace", "internal/poll", "internal/syscall/unix",
		"internal/syscall/windows", "internal/singleflight", "internal/race",
		"golang.org/x/net/dns/dnsmessage", "golang.org/x/net/lif", "golang.org/x/net/route",
	},

	// NET enables use of basic network-related packages.
	"NET": {
		"net",
		"mime",
		"net/textproto",
		"net/url",
	},

	// Uses of networking.
	"log/syslog":    {"L4", "OS", "net"},
	"net/mail":      {"L4", "NET", "OS", "mime"},
	"net/textproto": {"L4", "OS", "net"},

	// Core crypto.
	"crypto/aes":               {"L3"},
	"crypto/des":               {"L3"},
	"crypto/hmac":              {"L3"},
	"crypto/internal/randutil": {"io", "sync"},
	"crypto/md5":               {"L3"},
	"crypto/rc4":               {"L3"},
	"crypto/sha1":              {"L3"},
	"crypto/sha256":            {"L3"},
	"crypto/sha512":            {"L3"},

	"CRYPTO": {
		"crypto/aes",
		"crypto/des",
		"crypto/hmac",
		"crypto/internal/randutil",
		"crypto/md5",
		"crypto/rc4",
		"crypto/sha1",
		"crypto/sha256",
		"crypto/sha512",
		"golang.org/x/crypto/chacha20poly1305",
		"golang.org/x/crypto/curve25519",
		"golang.org/x/crypto/poly1305",
	},

	// Random byte, number generation.
	// This would be part of core crypto except that it imports
	// math/big, which imports fmt.
	"crypto/rand": {"L4", "CRYPTO", "OS", "math/big", "syscall", "syscall/js", "internal/syscall/unix"},

	// Not part of CRYPTO because it imports crypto/rand and crypto/sha512.
	"crypto/ed25519":                       {"L3", "CRYPTO", "crypto/rand", "crypto/ed25519/internal/edwards25519"},
	"crypto/ed25519/internal/edwards25519": {"encoding/binary"},

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
		"L4", "CRYPTO-MATH", "OS", "golang.org/x/crypto/cryptobyte", "golang.org/x/crypto/hkdf",
		"container/list", "crypto/x509", "encoding/pem", "net", "syscall", "crypto/ed25519",
	},
	"crypto/x509": {
		"L4", "CRYPTO-MATH", "OS", "CGO", "crypto/ed25519",
		"crypto/x509/pkix", "encoding/pem", "encoding/hex", "net", "os/user", "syscall", "net/url",
		"golang.org/x/crypto/cryptobyte", "golang.org/x/crypto/cryptobyte/asn1",
	},
	"crypto/x509/pkix": {"L4", "CRYPTO-MATH", "encoding/hex"},

	// Simple net+crypto-aware packages.
	"mime/multipart": {"L4", "OS", "mime", "crypto/rand", "net/textproto", "mime/quotedprintable"},
	"net/smtp":       {"L4", "CRYPTO", "NET", "crypto/tls"},

	// HTTP, kingpin of dependencies.
	"net/http": {
		"L4", "NET", "OS",
		"compress/gzip",
		"container/list",
		"context",
		"crypto/rand",
		"crypto/tls",
		"golang.org/x/net/http/httpguts",
		"golang.org/x/net/http/httpproxy",
		"golang.org/x/net/http2/hpack",
		"golang.org/x/net/idna",
		"golang.org/x/text/unicode/norm",
		"golang.org/x/text/width",
		"internal/nettrace",
		"mime/multipart",
		"net/http/httptrace",
		"net/http/internal",
		"runtime/debug",
		"syscall/js",
	},
	"net/http/internal":  {"L4"},
	"net/http/httptrace": {"context", "crypto/tls", "internal/nettrace", "net", "net/textproto", "reflect", "time"},

	// HTTP-using packages.
	"expvar":             {"L4", "OS", "encoding/json", "net/http"},
	"net/http/cgi":       {"L4", "NET", "OS", "crypto/tls", "net/http", "regexp"},
	"net/http/cookiejar": {"L4", "NET", "net/http"},
	"net/http/fcgi":      {"L4", "NET", "OS", "context", "net/http", "net/http/cgi"},
	"net/http/httptest": {
		"L4", "NET", "OS", "crypto/tls", "flag", "net/http", "net/http/internal", "crypto/x509",
		"golang.org/x/net/http/httpguts",
	},
	"net/http/httputil": {"L4", "NET", "OS", "context", "net/http", "net/http/internal", "golang.org/x/net/http/httpguts"},
	"net/http/pprof":    {"L4", "OS", "html/template", "net/http", "runtime/pprof", "runtime/trace"},
	"net/rpc":           {"L4", "NET", "encoding/gob", "html/template", "net/http", "go/token"},
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

// listStdPkgs returns the same list of packages as "go list std".
func listStdPkgs(goroot string) ([]string, error) {
	// Based on cmd/go's matchPackages function.
	var pkgs []string

	src := filepath.Join(goroot, "src") + string(filepath.Separator)
	walkFn := func(path string, fi os.FileInfo, err error) error {
		if err != nil || !fi.IsDir() || path == src {
			return nil
		}

		base := filepath.Base(path)
		if strings.HasPrefix(base, ".") || strings.HasPrefix(base, "_") || base == "testdata" {
			return filepath.SkipDir
		}

		name := filepath.ToSlash(path[len(src):])
		if name == "builtin" || name == "cmd" || strings.Contains(name, "golang.org/x/") {
			return filepath.SkipDir
		}

		pkgs = append(pkgs, name)
		return nil
	}
	if err := filepath.Walk(src, walkFn); err != nil {
		return nil, err
	}
	return pkgs, nil
}

func TestDependencies(t *testing.T) {
	iOS := runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64")
	if iOS {
		// Tests run in a limited file system and we do not
		// provide access to every source file.
		t.Skipf("skipping on %s/%s, missing full GOROOT", runtime.GOOS, runtime.GOARCH)
	}

	ctxt := Default
	all, err := listStdPkgs(ctxt.GOROOT)
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(all)

	sawImport := map[string]map[string]bool{} // from package => to package => true

	for _, pkg := range all {
		imports, err := findImports(pkg)
		if err != nil {
			t.Error(err)
			continue
		}
		if sawImport[pkg] == nil {
			sawImport[pkg] = map[string]bool{}
		}
		ok := allowed(pkg)
		var bad []string
		for _, imp := range imports {
			sawImport[pkg][imp] = true
			if !ok[imp] {
				bad = append(bad, imp)
			}
		}
		if bad != nil {
			t.Errorf("unexpected dependency: %s imports %v", pkg, bad)
		}
	}

	// depPath returns the path between the given from and to packages.
	// It returns the empty string if there's no dependency path.
	var depPath func(string, string) string
	depPath = func(from, to string) string {
		if sawImport[from][to] {
			return from + " => " + to
		}
		for pkg := range sawImport[from] {
			if p := depPath(pkg, to); p != "" {
				return from + " => " + p
			}
		}
		return ""
	}

	// Also test some high-level policy goals are being met by not finding
	// these dependency paths:
	badPaths := []struct{ from, to string }{
		{"net", "unicode"},
		{"os", "unicode"},
	}

	for _, path := range badPaths {
		if how := depPath(path.from, path.to); how != "" {
			t.Errorf("policy violation: %s", how)
		}
	}

}

var buildIgnore = []byte("\n// +build ignore")

func findImports(pkg string) ([]string, error) {
	dir := filepath.Join(Default.GOROOT, "src", pkg)
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var imports []string
	var haveImport = map[string]bool{}
	for _, file := range files {
		name := file.Name()
		if name == "slice_go14.go" || name == "slice_go18.go" {
			// These files are for compiler bootstrap with older versions of Go and not built in the standard build.
			continue
		}
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			return nil, err
		}
		var imp []string
		data, err := readImports(f, false, &imp)
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("reading %v: %v", name, err)
		}
		if bytes.Contains(data, buildIgnore) {
			continue
		}
		for _, quoted := range imp {
			path, err := strconv.Unquote(quoted)
			if err != nil {
				continue
			}
			if !haveImport[path] {
				haveImport[path] = true
				imports = append(imports, path)
			}
		}
	}
	sort.Strings(imports)
	return imports, nil
}
