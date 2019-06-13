// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hooks adds all the standard gopls implementations.
// This can be used in tests without needing to use the gopls main, and is
// also the place to edit for custom builds of gopls.
package hooks // import "golang.org/x/tools/gopls/internal/hooks"

import (
	"context"

	"golang.org/x/tools/internal/lsp/cache"
)

func Install(ctx context.Context) context.Context {
	cache.UpdateAnalyzers = updateAnalyzers
	return ctx
}
