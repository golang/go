// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
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

func (s *session) setOverlay(uri span.URI, version float64, data []byte) error {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	o, ok := s.overlays[uri]
	if !ok {
		return errors.Errorf("setting overlay for unopened file %s", uri)
	}
	s.overlays[uri] = &overlay{
		session: s,
		uri:     uri,
		kind:    o.kind,
		data:    data,
		hash:    hashContents(data),
		version: version,
	}
	return nil
}

func (s *session) closeOverlay(uri span.URI) error {
	s.openFiles.Delete(uri)

	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	_, ok := s.overlays[uri]
	if !ok {
		return errors.Errorf("closing unopened overlay %s", uri)
	}
	delete(s.overlays, uri)
	return nil
}

func (s *session) openOverlay(ctx context.Context, uri span.URI, languageID string, version float64, data []byte) error {
	kind := source.DetectLanguage(languageID, uri.Filename())
	if kind == source.UnknownKind {
		return errors.Errorf("openOverlay: unknown file kind for %s", uri)
	}

	s.openFiles.Store(uri, true)

	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

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
	return nil
}

func (s *session) saveOverlay(uri span.URI, version float64, data []byte) error {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	o, ok := s.overlays[uri]
	if !ok {
		return errors.Errorf("saveOverlay: unopened overlay %s", uri)
	}
	if o.version != version {
		return errors.Errorf("saveOverlay: saving %s at version %v, currently at %v", uri, version, o.version)
	}
	if data != nil {
		if !bytes.Equal(o.data, data) {
			return errors.Errorf("saveOverlay: overlay %s changed on save", uri)
		}
		o.data = data
	}
	o.sameContentOnDisk = true
	o.version = version

	return nil
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
		// TODO(rstambler): Make sure not to send overlays outside of the current view.
		if overlay.sameContentOnDisk {
			continue
		}
		overlays[uri.Filename()] = overlay.data
	}
	return overlays
}
