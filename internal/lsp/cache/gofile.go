// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

func (v *view) CheckPackageHandles(ctx context.Context, f source.File) (source.Snapshot, []source.CheckPackageHandle, error) {
	// Get the snapshot that will be used for type-checking.
	s := v.getSnapshot()

	cphs, err := s.CheckPackageHandles(ctx, f)
	if err != nil {
		return nil, nil, err
	}
	if len(cphs) == 0 {
		return nil, nil, errors.Errorf("no CheckPackageHandles for %s", f.URI())
	}
	return s, cphs, nil
}

func (s *snapshot) CheckPackageHandles(ctx context.Context, f source.File) ([]source.CheckPackageHandle, error) {
	ctx = telemetry.File.With(ctx, f.URI())

	fh := s.Handle(ctx, f)

	// Determine if we need to type-check the package.
	m, cphs, load, check := s.shouldCheck(fh)

	// We may need to re-load package metadata.
	// We only need to this if it has been invalidated, and is therefore unvailable.
	if load {
		var err error
		m, err = s.load(ctx, source.FileURI(f.URI()))
		if err != nil {
			return nil, err
		}
		// If load has explicitly returned nil metadata and no error,
		// it means that we should not re-type-check the packages.
		if m == nil {
			return cphs, nil
		}
	}
	if check {
		var results []source.CheckPackageHandle
		for _, m := range m {
			cph, err := s.checkPackageHandle(ctx, m.id, source.ParseFull)
			if err != nil {
				return nil, err
			}
			results = append(results, cph)
		}
		cphs = results
	}
	if len(cphs) == 0 {
		return nil, errors.Errorf("no CheckPackageHandles for %s", f)
	}
	return cphs, nil
}

func (s *snapshot) shouldCheck(fh source.FileHandle) (m []*metadata, cphs []source.CheckPackageHandle, load, check bool) {
	// Get the metadata for the given file.
	m = s.getMetadataForURI(fh.Identity().URI)

	// If there is no metadata for the package, we definitely need to type-check again.
	if len(m) == 0 {
		return nil, nil, true, true
	}

	// If the metadata for the package had missing dependencies,
	// we _may_ need to re-check. If the missing dependencies haven't changed
	// since previous load, we will not check again.
	for _, m := range m {
		if len(m.missingDeps) != 0 {
			load = true
			check = true
		}
	}
	// We expect to see a checked package for each package ID,
	// and it should be parsed in full mode.
	cphs = s.getPackages(source.FileURI(fh.Identity().URI), source.ParseFull)
	if len(cphs) < len(m) {
		return m, nil, load, true
	}
	return m, cphs, load, check
}

func (v *view) GetActiveReverseDeps(ctx context.Context, f source.File) (results []source.CheckPackageHandle) {
	var (
		s     = v.getSnapshot()
		rdeps = transitiveReverseDependencies(ctx, f.URI(), s)
		files = v.openFiles(ctx, rdeps)
		seen  = make(map[span.URI]struct{})
	)
	for _, f := range files {
		if _, ok := seen[f.URI()]; ok {
			continue
		}
		cphs, err := s.CheckPackageHandles(ctx, f)
		if err != nil {
			continue
		}
		cph, err := source.WidestCheckPackageHandle(cphs)
		if err != nil {
			continue
		}
		for _, ph := range cph.Files() {
			seen[ph.File().Identity().URI] = struct{}{}
		}
		results = append(results, cph)
	}
	return results
}

func transitiveReverseDependencies(ctx context.Context, uri span.URI, s *snapshot) (result []span.URI) {
	var (
		seen         = make(map[packageID]struct{})
		uris         = make(map[span.URI]struct{})
		topLevelURIs = make(map[span.URI]struct{})
	)

	metadata := s.getMetadataForURI(uri)

	for _, m := range metadata {
		for _, uri := range m.files {
			topLevelURIs[uri] = struct{}{}
		}
		s.reverseDependencies(m.id, uris, seen)
	}
	// Filter out the URIs that belong to the original package.
	for uri := range uris {
		if _, ok := topLevelURIs[uri]; ok {
			continue
		}
		result = append(result, uri)
	}
	return result
}
