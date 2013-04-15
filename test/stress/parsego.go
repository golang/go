// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path"
	"runtime"
	"strings"
)

func isGoFile(dir os.FileInfo) bool {
	return !dir.IsDir() &&
		!strings.HasPrefix(dir.Name(), ".") && // ignore .files
		path.Ext(dir.Name()) == ".go"
}

func isPkgFile(dir os.FileInfo) bool {
	return isGoFile(dir) &&
		!strings.HasSuffix(dir.Name(), "_test.go") // ignore test files
}

func pkgName(filename string) string {
	file, err := parser.ParseFile(token.NewFileSet(), filename, nil, parser.PackageClauseOnly)
	if err != nil || file == nil {
		return ""
	}
	return file.Name.Name
}

func parseDir(dirpath string) map[string]*ast.Package {
	// the package name is the directory name within its parent.
	// (use dirname instead of path because dirname is clean; it
	// has no trailing '/')
	_, pkgname := path.Split(dirpath)

	// filter function to select the desired .go files
	filter := func(d os.FileInfo) bool {
		if isPkgFile(d) {
			// Some directories contain main packages: Only accept
			// files that belong to the expected package so that
			// parser.ParsePackage doesn't return "multiple packages
			// found" errors.
			// Additionally, accept the special package name
			// fakePkgName if we are looking at cmd documentation.
			name := pkgName(dirpath + "/" + d.Name())
			return name == pkgname
		}
		return false
	}

	// get package AST
	pkgs, err := parser.ParseDir(token.NewFileSet(), dirpath, filter, parser.ParseComments)
	if err != nil {
		println("parse", dirpath, err.Error())
		panic("go ParseDir fail: " + err.Error())
	}
	return pkgs
}

func stressParseGo() {
	pkgroot := runtime.GOROOT() + "/src/pkg/"
	for {
		m := make(map[string]map[string]*ast.Package)
		for _, pkg := range packages {
			m[pkg] = parseDir(pkgroot + pkg)
			Println("parsed go package", pkg)
		}
	}
}

// find . -type d -not -path "./exp" -not -path "./exp/*" -printf "\t\"%p\",\n" | sort | sed "s/\.\///" | grep -v testdata
var packages = []string{
	"archive",
	"archive/tar",
	"archive/zip",
	"bufio",
	"builtin",
	"bytes",
	"compress",
	"compress/bzip2",
	"compress/flate",
	"compress/gzip",
	"compress/lzw",
	"compress/zlib",
	"container",
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
	"crypto/rand",
	"crypto/rc4",
	"crypto/rsa",
	"crypto/sha1",
	"crypto/sha256",
	"crypto/sha512",
	"crypto/subtle",
	"crypto/tls",
	"crypto/x509",
	"crypto/x509/pkix",
	"database",
	"database/sql",
	"database/sql/driver",
	"debug",
	"debug/dwarf",
	"debug/elf",
	"debug/gosym",
	"debug/macho",
	"debug/pe",
	"encoding",
	"encoding/ascii85",
	"encoding/asn1",
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
	"go",
	"go/ast",
	"go/build",
	"go/doc",
	"go/format",
	"go/parser",
	"go/printer",
	"go/scanner",
	"go/token",
	"hash",
	"hash/adler32",
	"hash/crc32",
	"hash/crc64",
	"hash/fnv",
	"html",
	"html/template",
	"image",
	"image/color",
	"image/draw",
	"image/gif",
	"image/jpeg",
	"image/png",
	"index",
	"index/suffixarray",
	"io",
	"io/ioutil",
	"log",
	"log/syslog",
	"math",
	"math/big",
	"math/cmplx",
	"math/rand",
	"mime",
	"mime/multipart",
	"net",
	"net/http",
	"net/http/cgi",
	"net/http/cookiejar",
	"net/http/fcgi",
	"net/http/httptest",
	"net/http/httputil",
	"net/http/pprof",
	"net/mail",
	"net/rpc",
	"net/rpc/jsonrpc",
	"net/smtp",
	"net/textproto",
	"net/url",
	"os",
	"os/exec",
	"os/signal",
	"os/user",
	"path",
	"path/filepath",
	"reflect",
	"regexp",
	"regexp/syntax",
	"runtime",
	"runtime/cgo",
	"runtime/debug",
	"runtime/pprof",
	"runtime/race",
	"sort",
	"strconv",
	"strings",
	"sync",
	"sync/atomic",
	"syscall",
	"testing",
	"testing/iotest",
	"testing/quick",
	"text",
	"text/scanner",
	"text/tabwriter",
	"text/template",
	"text/template/parse",
	"time",
	"unicode",
	"unicode/utf16",
	"unicode/utf8",
	"unsafe",
}
