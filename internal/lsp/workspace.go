// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

func (s *Server) didChangeWorkspaceFolders(ctx context.Context, params *protocol.DidChangeWorkspaceFoldersParams) error {
	event := params.Event
	for _, folder := range event.Removed {
		view := s.session.View(folder.Name)
		if view != nil {
			view.Shutdown(ctx)
		} else {
			return errors.Errorf("view %s for %v not found", folder.Name, folder.URI)
		}
	}
	return s.addFolders(ctx, event.Added)
}

func (s *Server) addView(ctx context.Context, name string, uri span.URI) (source.Snapshot, func(), error) {
	s.stateMu.Lock()
	state := s.state
	s.stateMu.Unlock()
	if state < serverInitialized {
		return nil, func() {}, errors.Errorf("addView called before server initialized")
	}
	options := s.session.Options().Clone()
	if err := s.fetchConfig(ctx, name, uri, options); err != nil {
		return nil, func() {}, err
	}
	_, snapshot, release, err := s.session.NewView(ctx, name, uri, options)
	return snapshot, release, err
}

func (s *Server) didChangeConfiguration(ctx context.Context, changed interface{}) error {
	// go through all the views getting the config
	for _, view := range s.session.Views() {
		options := s.session.Options().Clone()
		if err := s.fetchConfig(ctx, view.Name(), view.Folder(), options); err != nil {
			return err
		}
		view, err := view.SetOptions(ctx, options)
		if err != nil {
			return err
		}
		go func() {
			snapshot, release := view.Snapshot(ctx)
			defer release()
			s.diagnoseDetached(snapshot)
		}()
	}
	return nil
}
