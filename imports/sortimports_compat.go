// +build !go1.2

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import "go/ast"

// Go 1.1 users don't get fancy package grouping.
// But this is still gofmt-compliant:

var sortImports = ast.SortImports
