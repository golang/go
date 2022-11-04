// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.11
// +build !go1.11

package gcimporter

import "go/types"

func newInterface(methods []*types.Func, embeddeds []types.Type) *types.Interface {
	named := make([]*types.Named, len(embeddeds))
	for i, e := range embeddeds {
		var ok bool
		named[i], ok = e.(*types.Named)
		if !ok {
			panic("embedding of non-defined interfaces in interfaces is not supported before Go 1.11")
		}
	}
	return types.NewInterface(methods, named)
}
