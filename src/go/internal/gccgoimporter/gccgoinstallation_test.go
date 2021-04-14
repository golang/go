// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gccgoimporter

import (
	"go/types"
	"testing"
)

// importablePackages is a list of packages that we verify that we can
// import. This should be all standard library packages in all relevant
// versions of gccgo. Note that since gccgo follows a different release
// cycle, and since different systems have different versions installed,
// we can't use the last-two-versions rule of the gc toolchain.
var importablePackages = [...]string{
	"archive/tar",
	"archive/zip",
	"bufio",
	"bytes",
	"compress/bzip2",
	"compress/flate",
	"compress/gzip",
	"compress/lzw",
	"compress/zlib",
	"container/heap",
	"container/list",
	"container/ring",
	"crypto/aes",
	"crypto/cipher",
	"crypto/des",
	"crypto/dsa",
	"crypto/ecdsa",
	"crypto/elliptic",
	"crypto",
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
	"database/sql/driver",
	"database/sql",
	"debug/dwarf",
	"debug/elf",
	"debug/gosym",
	"debug/macho",
	"debug/pe",
	"encoding/ascii85",
	"encoding/asn1",
	"encoding/base32",
	"encoding/base64",
	"encoding/binary",
	"encoding/csv",
	"encoding/gob",
	// "encoding", // Added in GCC 4.9.
	"encoding/hex",
	"encoding/json",
	"encoding/pem",
	"encoding/xml",
	"errors",
	"expvar",
	"flag",
	"fmt",
	"go/ast",
	"go/build",
	"go/doc",
	// "go/format", // Added in GCC 4.8.
	"go/parser",
	"go/printer",
	"go/scanner",
	"go/token",
	"hash/adler32",
	"hash/crc32",
	"hash/crc64",
	"hash/fnv",
	"hash",
	"html",
	"html/template",
	"image/color",
	// "image/color/palette", // Added in GCC 4.9.
	"image/draw",
	"image/gif",
	"image",
	"image/jpeg",
	"image/png",
	"index/suffixarray",
	"io",
	"io/ioutil",
	"log",
	"log/syslog",
	"math/big",
	"math/cmplx",
	"math",
	"math/rand",
	"mime",
	"mime/multipart",
	"net",
	"net/http/cgi",
	// "net/http/cookiejar", // Added in GCC 4.8.
	"net/http/fcgi",
	"net/http",
	"net/http/httptest",
	"net/http/httputil",
	"net/http/pprof",
	"net/mail",
	"net/rpc",
	"net/rpc/jsonrpc",
	"net/smtp",
	"net/textproto",
	"net/url",
	"os/exec",
	"os",
	"os/signal",
	"os/user",
	"path/filepath",
	"path",
	"reflect",
	"regexp",
	"regexp/syntax",
	"runtime/debug",
	"runtime",
	"runtime/pprof",
	"sort",
	"strconv",
	"strings",
	"sync/atomic",
	"sync",
	"syscall",
	"testing",
	"testing/iotest",
	"testing/quick",
	"text/scanner",
	"text/tabwriter",
	"text/template",
	"text/template/parse",
	"time",
	"unicode",
	"unicode/utf16",
	"unicode/utf8",
}

func TestInstallationImporter(t *testing.T) {
	// This test relies on gccgo being around.
	gpath := gccgoPath()
	if gpath == "" {
		t.Skip("This test needs gccgo")
	}

	var inst GccgoInstallation
	err := inst.InitFromDriver(gpath)
	if err != nil {
		t.Fatal(err)
	}
	imp := inst.GetImporter(nil, nil)

	// Ensure we don't regress the number of packages we can parse. First import
	// all packages into the same map and then each individually.
	pkgMap := make(map[string]*types.Package)
	for _, pkg := range importablePackages {
		_, err = imp(pkgMap, pkg, ".", nil)
		if err != nil {
			t.Error(err)
		}
	}

	for _, pkg := range importablePackages {
		_, err = imp(make(map[string]*types.Package), pkg, ".", nil)
		if err != nil {
			t.Error(err)
		}
	}

	// Test for certain specific entities in the imported data.
	for _, test := range [...]importerTest{
		{pkgpath: "io", name: "Reader", want: "type Reader interface{Read(p []byte) (n int, err error)}"},
		{pkgpath: "io", name: "ReadWriter", want: "type ReadWriter interface{Reader; Writer}"},
		{pkgpath: "math", name: "Pi", want: "const Pi untyped float"},
		{pkgpath: "math", name: "Sin", want: "func Sin(x float64) float64"},
		{pkgpath: "sort", name: "Ints", want: "func Ints(a []int)"},
		{pkgpath: "unsafe", name: "Pointer", want: "type Pointer"},
	} {
		runImporterTest(t, imp, nil, &test)
	}
}
