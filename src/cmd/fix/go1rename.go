// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	register(go1renameFix)
}

var go1renameFix = fix{
	"go1rename",
	"2012-02-12",
	renameFix(go1renameReplace),
	`Rewrite package-level names that have been renamed in Go 1.

http://codereview.appspot.com/5625045/
http://codereview.appspot.com/5672072/
`,
}

var go1renameReplace = []rename{
	{
		OldImport: "crypto/aes",
		NewImport: "crypto/cipher",
		Old:       "*aes.Cipher",
		New:       "cipher.Block",
	},
	{
		OldImport: "crypto/des",
		NewImport: "crypto/cipher",
		Old:       "*des.Cipher",
		New:       "cipher.Block",
	},
	{
		OldImport: "crypto/des",
		NewImport: "crypto/cipher",
		Old:       "*des.TripleDESCipher",
		New:       "cipher.Block",
	},
	{
		OldImport: "encoding/json",
		NewImport: "",
		Old:       "json.MarshalForHTML",
		New:       "json.Marshal",
	},
	{
		OldImport: "net/url",
		NewImport: "",
		Old:       "url.ParseWithReference",
		New:       "url.Parse",
	},
	{
		OldImport: "net/url",
		NewImport: "",
		Old:       "url.ParseRequest",
		New:       "url.ParseRequestURI",
	},
	{
		OldImport: "os",
		NewImport: "syscall",
		Old:       "os.Exec",
		New:       "syscall.Exec",
	},
	{
		OldImport: "runtime",
		NewImport: "",
		Old:       "runtime.Cgocalls",
		New:       "runtime.NumCgoCall",
	},
	{
		OldImport: "runtime",
		NewImport: "",
		Old:       "runtime.Goroutines",
		New:       "runtime.NumGoroutine",
	},
}
