// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag adds support for telemetry tracing.
package trace

import (
	"context"

	"golang.org/x/tools/internal/lsp/telemetry/tag"
)

func StartSpan(ctx context.Context, name string, tags ...tag.Tag) (context.Context, func()) {
	return tag.With(ctx, tags...), func() {}
}
