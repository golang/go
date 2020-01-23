// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

type overlay struct {
	session *session
	uri     span.URI
	text    []byte
	hash    string
	version float64
	kind    source.FileKind

	// saved is true if a file has been saved on disk,
	// and therefore does not need to be part of the overlay sent to go/packages.
	saved bool
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
	return o.text, o.hash, nil
}

func (s *session) updateOverlay(ctx context.Context, c source.FileModification) error {
	// Make sure that the file was not changed on disk.
	if c.OnDisk {
		return errors.Errorf("updateOverlay called for an on-disk change: %s", c.URI)
	}

	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	o, ok := s.overlays[c.URI]

	// Determine the file kind on open, otherwise, assume it has been cached.
	var kind source.FileKind
	switch c.Action {
	case source.Open:
		kind = source.DetectLanguage(c.LanguageID, c.URI.Filename())
	default:
		if !ok {
			return errors.Errorf("updateOverlay: modifying unopened overlay %v", c.URI)
		}
		kind = o.kind
	}
	if kind == source.UnknownKind {
		return errors.Errorf("updateOverlay: unknown file kind for %s", c.URI)
	}

	// Closing a file just deletes its overlay.
	if c.Action == source.Close {
		delete(s.overlays, c.URI)
		return nil
	}

	// If the file is on disk, check if its content is the same as the overlay.
	text := c.Text
	if text == nil {
		text = o.text
	}
	hash := hashContents(text)
	var sameContentOnDisk bool
	switch c.Action {
	case source.Open:
		_, h, err := s.cache.GetFile(c.URI).Read(ctx)
		sameContentOnDisk = (err == nil && h == hash)
	case source.Save:
		// Make sure the version and content (if present) is the same.
		if o.version != c.Version {
			return errors.Errorf("updateOverlay: saving %s at version %v, currently at %v", c.URI, c.Version, o.version)
		}
		if c.Text != nil && o.hash != hash {
			return errors.Errorf("updateOverlay: overlay %s changed on save", c.URI)
		}
		sameContentOnDisk = true
	}
	s.overlays[c.URI] = &overlay{
		session: s,
		uri:     c.URI,
		version: c.Version,
		text:    text,
		kind:    kind,
		hash:    hash,
		saved:   sameContentOnDisk,
	}
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
		if overlay.saved {
			continue
		}
		overlays[uri.Filename()] = overlay.text
	}
	return overlays
}
