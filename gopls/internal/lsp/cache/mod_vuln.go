// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"os"

	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/gopls/internal/vulncheck"
	"golang.org/x/tools/internal/memoize"
)

// ModVuln returns import vulnerability analysis for the given go.mod URI.
// Concurrent requests are combined into a single command.
func (s *snapshot) ModVuln(ctx context.Context, modURI span.URI) (*govulncheck.Result, error) {
	s.mu.Lock()
	entry, hit := s.modVulnHandles.Get(modURI)
	s.mu.Unlock()

	type modVuln struct {
		result *govulncheck.Result
		err    error
	}

	// Cache miss?
	if !hit {
		// If the file handle is an overlay, it may not be written to disk.
		// The go.mod file has to be on disk for vulncheck to work.
		//
		// TODO(hyangah): use overlays for vulncheck.
		fh, err := s.ReadFile(ctx, modURI)
		if err != nil {
			return nil, err
		}
		if _, ok := fh.(*Overlay); ok {
			if info, _ := os.Stat(modURI.Filename()); info == nil {
				return nil, source.ErrNoModOnDisk
			}
		}

		handle := memoize.NewPromise("modVuln", func(ctx context.Context, arg interface{}) interface{} {
			result, err := modVulnImpl(ctx, arg.(*snapshot), modURI)
			return modVuln{result, err}
		})

		entry = handle
		s.mu.Lock()
		s.modVulnHandles.Set(modURI, entry, nil)
		s.mu.Unlock()
	}

	// Await result.
	v, err := s.awaitPromise(ctx, entry.(*memoize.Promise))
	if err != nil {
		return nil, err
	}
	res := v.(modVuln)
	return res.result, res.err
}

func modVulnImpl(ctx context.Context, s *snapshot, uri span.URI) (*govulncheck.Result, error) {
	if vulncheck.VulnerablePackages == nil {
		return &govulncheck.Result{}, nil
	}
	fh, err := s.ReadFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	return vulncheck.VulnerablePackages(ctx, s, fh)
}
