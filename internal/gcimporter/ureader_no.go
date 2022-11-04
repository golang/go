// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.18
// +build !go1.18

package gcimporter

import (
	"fmt"
	"go/token"
	"go/types"
)

func UImportData(fset *token.FileSet, imports map[string]*types.Package, data []byte, path string) (_ int, pkg *types.Package, err error) {
	err = fmt.Errorf("go/tools compiled with a Go version earlier than 1.18 cannot read unified IR export data")
	return
}
