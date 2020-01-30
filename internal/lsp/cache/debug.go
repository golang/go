// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"sort"

	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/span"
)

type debugView struct{ *view }

func (v debugView) ID() string             { return v.id }
func (v debugView) Session() debug.Session { return debugSession{v.session} }
func (v debugView) Env() []string          { return v.Options().Env }

type debugSession struct{ *session }

func (s debugSession) ID() string         { return s.id }
func (s debugSession) Cache() debug.Cache { return debugCache{s.cache} }
func (s debugSession) Files() []*debug.File {
	var files []*debug.File
	seen := make(map[span.URI]*debug.File)
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()
	for _, overlay := range s.overlays {
		f, ok := seen[overlay.uri]
		if !ok {
			f = &debug.File{Session: s, URI: overlay.uri}
			seen[overlay.uri] = f
			files = append(files, f)
		}
		f.Data = string(overlay.text)
		f.Error = nil
		f.Hash = overlay.hash
	}
	sort.Slice(files, func(i int, j int) bool {
		return files[i].URI < files[j].URI
	})
	return files
}

func (s debugSession) File(hash string) *debug.File {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()
	for _, overlay := range s.overlays {
		if overlay.hash == hash {
			return &debug.File{
				Session: s,
				URI:     overlay.uri,
				Data:    string(overlay.text),
				Error:   nil,
				Hash:    overlay.hash,
			}
		}
	}
	return &debug.File{
		Session: s,
		Hash:    hash,
	}
}
