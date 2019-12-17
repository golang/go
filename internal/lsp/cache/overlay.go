// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

type overlay struct {
	session *session
	uri     span.URI
	data    []byte
	hash    string
	version float64
	kind    source.FileKind

	// sameContentOnDisk is true if a file has been saved on disk,
	// and therefore does not need to be part of the overlay sent to go/packages.
	sameContentOnDisk bool
}

func (o *overlay) FileSystem() source.FileSystem {
	return o.session
}

func (o *overlay) Identity() source.FileIdentity {
	return source.FileIdentity{
		URI:        o.uri,
		Identifier: o.hash,
		Version:    o.version,
		Kind:       o.kind,
	}
}
func (o *overlay) Read(ctx context.Context) ([]byte, string, error) {
	return o.data, o.hash, nil
}

func (s *session) setOverlay(f source.File, version float64, data []byte) {
	s.overlayMu.Lock()
	defer func() {
		s.overlayMu.Unlock()
		s.filesWatchMap.Notify(f.URI(), source.Change)
	}()

	if data == nil {
		delete(s.overlays, f.URI())
		return
	}

	s.overlays[f.URI()] = &overlay{
		session: s,
		uri:     f.URI(),
		kind:    f.Kind(),
		data:    data,
		hash:    hashContents(data),
		version: version,
	}
}

func (s *session) clearOverlay(uri span.URI) {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	delete(s.overlays, uri)
}

// openOverlay adds the file content to the overlay.
// It also checks if the provided content is equivalent to the file's content on disk.
func (s *session) openOverlay(ctx context.Context, uri span.URI, kind source.FileKind, version float64, data []byte) {
	s.overlayMu.Lock()
	defer func() {
		s.overlayMu.Unlock()
		s.filesWatchMap.Notify(uri, source.Open)
	}()
	s.overlays[uri] = &overlay{
		session: s,
		uri:     uri,
		kind:    kind,
		data:    data,
		hash:    hashContents(data),
		version: version,
	}
	// If the file is on disk, check if its content is the same as the overlay.
	if _, hash, err := s.cache.GetFile(uri, kind).Read(ctx); err == nil {
		if hash == s.overlays[uri].hash {
			s.overlays[uri].sameContentOnDisk = true
		}
	}
}

func (s *session) readOverlay(uri span.URI) *overlay {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	// We might have the content saved in an overlay.
	if overlay, ok := s.overlays[uri]; ok {
		return overlay
	}
	return nil
}

func (s *session) buildOverlay() map[string][]byte {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	overlays := make(map[string][]byte)
	for uri, overlay := range s.overlays {
		if overlay.sameContentOnDisk {
			continue
		}
		overlays[uri.Filename()] = overlay.data
	}
	return overlays
}
