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

func (s *Server) changeFolders(ctx context.Context, event protocol.WorkspaceFoldersChangeEvent) error {
	for _, folder := range event.Removed {
		view := s.session.View(folder.Name)
		if view != nil {
			view.Shutdown(ctx)
		} else {
			return errors.Errorf("view %s for %v not found", folder.Name, folder.URI)
		}
	}

	for _, folder := range event.Added {
		view, cphs, err := s.addView(ctx, folder.Name, span.NewURI(folder.URI))
		if err != nil {
			return err
		}
		go s.diagnoseSnapshot(view.Snapshot(), cphs)
	}
	return nil
}

func (s *Server) addView(ctx context.Context, name string, uri span.URI) (source.View, []source.CheckPackageHandle, error) {
	s.stateMu.Lock()
	state := s.state
	s.stateMu.Unlock()
	if state < serverInitialized {
		return nil, nil, errors.Errorf("addView called before server initialized")
	}

	options := s.session.Options()
	s.fetchConfig(ctx, name, uri, &options)

	return s.session.NewView(ctx, name, uri, options)
}

func (s *Server) updateConfiguration(ctx context.Context, changed interface{}) error {
	// go through all the views getting the config
	for _, view := range s.session.Views() {
		options := s.session.Options()
		s.fetchConfig(ctx, view.Name(), view.Folder(), &options)
		view.SetOptions(ctx, options)
	}
	return nil
}
