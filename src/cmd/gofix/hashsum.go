// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(hashSumFix)
}

var hashSumFix = fix{
	"hashsum",
	"2011-11-30",
	hashSumFn,
	`Pass a nil argument to calls to hash.Sum

This fix rewrites code so that it passes a nil argument to hash.Sum.
The additional argument will allow callers to avoid an
allocation in the future.

http://codereview.appspot.com/5448065
`,
}

// Type-checking configuration: tell the type-checker this basic
// information about types, functions, and variables in external packages.
var hashSumTypeConfig = &TypeConfig{
	Var: map[string]string{
		"crypto.MD4":       "crypto.Hash",
		"crypto.MD5":       "crypto.Hash",
		"crypto.SHA1":      "crypto.Hash",
		"crypto.SHA224":    "crypto.Hash",
		"crypto.SHA256":    "crypto.Hash",
		"crypto.SHA384":    "crypto.Hash",
		"crypto.SHA512":    "crypto.Hash",
		"crypto.MD5SHA1":   "crypto.Hash",
		"crypto.RIPEMD160": "crypto.Hash",
	},

	Func: map[string]string{
		"adler32.New":    "hash.Hash",
		"crc32.New":      "hash.Hash",
		"crc32.NewIEEE":  "hash.Hash",
		"crc64.New":      "hash.Hash",
		"fnv.New32a":     "hash.Hash",
		"fnv.New32":      "hash.Hash",
		"fnv.New64a":     "hash.Hash",
		"fnv.New64":      "hash.Hash",
		"hmac.New":       "hash.Hash",
		"hmac.NewMD5":    "hash.Hash",
		"hmac.NewSHA1":   "hash.Hash",
		"hmac.NewSHA256": "hash.Hash",
		"md4.New":        "hash.Hash",
		"md5.New":        "hash.Hash",
		"ripemd160.New":  "hash.Hash",
		"sha1.New224":    "hash.Hash",
		"sha1.New":       "hash.Hash",
		"sha256.New224":  "hash.Hash",
		"sha256.New":     "hash.Hash",
		"sha512.New384":  "hash.Hash",
		"sha512.New":     "hash.Hash",
	},

	Type: map[string]*Type{
		"crypto.Hash": {
			Method: map[string]string{
				"New": "func() hash.Hash",
			},
		},
	},
}

func hashSumFn(f *ast.File) bool {
	typeof, _ := typecheck(hashSumTypeConfig, f)

	fixed := false

	walk(f, func(n interface{}) {
		call, ok := n.(*ast.CallExpr)
		if ok && len(call.Args) == 0 {
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if ok && sel.Sel.Name == "Sum" && typeof[sel.X] == "hash.Hash" {
				call.Args = append(call.Args, ast.NewIdent("nil"))
				fixed = true
			}
		}
	})

	return fixed
}
