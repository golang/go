// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/xlog"
)

func New() source.Cache {
	return &cache{}
}

type cache struct {
}

func (c *cache) NewSession(log xlog.Logger) source.Session {
	return &session{
		cache: c,
		log:   log,
	}
}
