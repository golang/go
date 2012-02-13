// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var cryptotypeFix = fix{
	"cryptotype",
	"2012-02-12",
	renameFix(cryptotypeReplace),
	`Rewrite uses of concrete cipher types to refer to the generic cipher.Block.

http://codereview.appspot.com/5625045/
`,
}

var cryptotypeReplace = []rename{
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
}
