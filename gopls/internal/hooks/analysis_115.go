// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.15
// +build !go1.15

package hooks

import "golang.org/x/tools/internal/lsp/source"

func updateAnalyzers(_ *source.Options) {}
