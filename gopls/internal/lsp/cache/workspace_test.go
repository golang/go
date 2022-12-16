// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"os"

	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
)

// osFileSource is a fileSource that just reads from the operating system.
type osFileSource struct {
	overlays map[span.URI]fakeOverlay
}

type fakeOverlay struct {
	source.VersionedFileHandle
	uri     span.URI
	content string
	err     error
	saved   bool
}

func (o fakeOverlay) Saved() bool { return o.saved }

func (o fakeOverlay) Read() ([]byte, error) {
	if o.err != nil {
		return nil, o.err
	}
	return []byte(o.content), nil
}

func (o fakeOverlay) URI() span.URI {
	return o.uri
}

// change updates the file source with the given file content. For convenience,
// empty content signals a deletion. If saved is true, these changes are
// persisted to disk.
func (s *osFileSource) change(ctx context.Context, uri span.URI, content string, saved bool) (*fileChange, error) {
	if content == "" {
		delete(s.overlays, uri)
		if saved {
			if err := os.Remove(uri.Filename()); err != nil {
				return nil, err
			}
		}
		fh, err := s.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		data, err := fh.Read()
		return &fileChange{exists: err == nil, content: data, fileHandle: &closedFile{fh}}, nil
	}
	if s.overlays == nil {
		s.overlays = map[span.URI]fakeOverlay{}
	}
	s.overlays[uri] = fakeOverlay{uri: uri, content: content, saved: saved}
	return &fileChange{
		exists:     content != "",
		content:    []byte(content),
		fileHandle: s.overlays[uri],
	}, nil
}

func (s *osFileSource) GetFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	if overlay, ok := s.overlays[uri]; ok {
		return overlay, nil
	}
	fi, statErr := os.Stat(uri.Filename())
	if statErr != nil {
		return &fileHandle{
			err: statErr,
			uri: uri,
		}, nil
	}
	fh, err := readFile(ctx, uri, fi)
	if err != nil {
		return nil, err
	}
	return fh, nil
}
