// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/build"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func runTest(t *testing.T, path string) {
	exitCode = 0

	*recursive = false
	if suffix := ".go"; strings.HasSuffix(path, suffix) {
		// single file
		path = filepath.Join(runtime.GOROOT(), "src/pkg", path)
		path, file := filepath.Split(path)
		*pkgName = file[:len(file)-len(suffix)]
		processFiles([]string{path}, true)
	} else {
		// package directory
		// TODO(gri) gotype should use the build package instead
		ctxt := build.Default
		ctxt.CgoEnabled = false
		pkg, err := ctxt.Import(path, "", 0)
		if err != nil {
			t.Errorf("build.Import error for path = %s: %s", path, err)
			return
		}
		// TODO(gri) there ought to be a more direct way using the build package...
		files := make([]string, len(pkg.GoFiles))
		for i, file := range pkg.GoFiles {
			files[i] = filepath.Join(pkg.Dir, file)
		}
		*pkgName = pkg.Name
		processFiles(files, true)
	}

	if exitCode != 0 {
		t.Errorf("processing %s failed: exitCode = %d", path, exitCode)
	}
}

var tests = []string{
	// individual files
	"exp/gotype/testdata/test1.go",

	// directories
	// Note: Packages that don't typecheck yet are commented out.
	// Unless there is a comment next to the commented out packages,
	// the package doesn't typecheck due to errors in the shift
	// expression checker.
	"archive/tar",
	"archive/zip",

	"bufio",
	"bytes",

	// "compress/bzip2",
	"compress/flate",
	"compress/gzip",
	// "compress/lzw",
	"compress/zlib",

	"container/heap",
	"container/list",
	"container/ring",

	"crypto",
	"crypto/aes",
	"crypto/cipher",
	"crypto/des",
	"crypto/dsa",
	"crypto/ecdsa",
	"crypto/elliptic",
	"crypto/hmac",
	"crypto/md5",
	// "crypto/rand",
	"crypto/rc4",
	// "crypto/rsa", // intermittent failure: /home/gri/go2/src/pkg/crypto/rsa/pkcs1v15.go:21:27: undeclared name: io
	"crypto/sha1",
	"crypto/sha256",
	"crypto/sha512",
	"crypto/subtle",
	"crypto/tls",
	"crypto/x509",
	"crypto/x509/pkix",

	"database/sql",
	"database/sql/driver",

	// "debug/dwarf",
	"debug/elf",
	"debug/gosym",
	"debug/macho",
	"debug/pe",

	"encoding/ascii85",
	// "encoding/asn1",
	"encoding/base32",
	"encoding/base64",
	"encoding/binary",
	"encoding/csv",
	"encoding/gob",
	"encoding/hex",
	"encoding/json",
	"encoding/pem",
	"encoding/xml",

	"errors",
	"expvar",
	"flag",
	"fmt",

	"exp/gotype",

	"go/ast",
	"go/build",
	"go/doc",
	"go/format",
	"go/parser",
	"go/printer",
	"go/scanner",
	// "go/token",
	"go/types",

	"hash/adler32",
	"hash/crc32",
	"hash/crc64",
	"hash/fnv",

	"image",
	"image/color",
	"image/draw",
	"image/gif",
	// "image/jpeg",
	"image/png",

	"index/suffixarray",

	"io",
	"io/ioutil",

	"log",
	"log/syslog",

	// "math",
	//"math/big",
	"math/cmplx",
	"math/rand",

	"mime",
	"mime/multipart",

	// "net",
	"net/http",
	"net/http/cgi",
	"net/http/fcgi",
	"net/http/httptest",
	"net/http/httputil",
	"net/http/pprof",
	"net/mail",
	"net/rpc",
	"net/rpc/jsonrpc",
	"net/smtp",
	// "net/textproto",
	"net/url",

	"path",
	"path/filepath",

	"reflect",

	"regexp",
	"regexp/syntax",

	// "runtime",
	"runtime/cgo",
	"runtime/debug",
	"runtime/pprof",

	"sort",
	// "strconv",
	"strings",

	"sync",
	"sync/atomic",

	// "syscall",

	"testing",
	"testing/iotest",
	"testing/quick",

	// "text/scanner",
	"text/tabwriter",
	"text/template",
	"text/template/parse",

	"time",
	"unicode",
	"unicode/utf16",
	"unicode/utf8",
}

func Test(t *testing.T) {
	for _, test := range tests {
		runTest(t, test)
	}
}
