// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(go1renameTests, go1pkgrename)
}

var go1renameTests = []testCase{
	{
		Name: "go1rename.0",
		In: `package main

import (
	"asn1"
	"big"
	"cmath"
	"csv"
	"exec"
	"exp/template/html"
	"gob"
	"http"
	"http/cgi"
	"http/fcgi"
	"http/httptest"
	"http/pprof"
	"json"
	"mail"
	"rand"
	"rpc"
	"rpc/jsonrpc"
	"scanner"
	"smtp"
	"syslog"
	"tabwriter"
	"template"
	"template/parse"
	"url"
	"utf16"
	"utf8"
	"xml"

	"crypto/bcrypt"
)
`,
		Out: `package main

import (
	"encoding/asn1"
	"encoding/csv"
	"encoding/gob"
	"encoding/json"
	"encoding/xml"
	"html/template"
	"log/syslog"
	"math/big"
	"math/cmplx"
	"math/rand"
	"net/http"
	"net/http/cgi"
	"net/http/fcgi"
	"net/http/httptest"
	"net/http/pprof"
	"net/mail"
	"net/rpc"
	"net/rpc/jsonrpc"
	"net/smtp"
	"net/url"
	"os/exec"
	"text/scanner"
	"text/tabwriter"
	"text/template"
	"text/template/parse"
	"unicode/utf16"
	"unicode/utf8"

	"code.google.com/p/go.crypto/bcrypt"
)
`,
	},
	{
		Name: "go1rename.1",
		In: `package main

import "cmath"
import poot "exp/template/html"

import (
	"ebnf"
	"old/regexp"
)

var _ = cmath.Sin
var _ = poot.Poot
`,
		Out: `package main

import "math/cmplx"
import poot "html/template"

import (
	"exp/ebnf"
	"old/regexp"
)

var _ = cmplx.Sin
var _ = poot.Poot
`,
	},
	{
		Name: "go1rename.2",
		In: `package foo

import (
	"fmt"
	"http"
	"url"

	"google/secret/project/go"
)

func main() {}
`,
		Out: `package foo

import (
	"fmt"
	"net/http"
	"net/url"

	"google/secret/project/go"
)

func main() {}
`,
	},
}
