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

var wsIndex int64

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

func (s *Server) didChangeConfiguration(ctx context.Context, _ *protocol.DidChangeConfigurationParams) error {
	// Apply any changes to the session-level settings.
	options := s.session.Options().Clone()
	semanticTokensRegistered := options.SemanticTokens
	if err := s.fetchConfig(ctx, "", "", options); err != nil {
		return err
	}
	s.session.SetOptions(options)

	// Go through each view, getting and updating its configuration.
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

	registration := semanticTokenRegistration(options.SemanticTypes, options.SemanticMods)
	// Update any session-specific registrations or unregistrations.
	if !semanticTokensRegistered && options.SemanticTokens {
		if err := s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
			Registrations: []protocol.Registration{registration},
		}); err != nil {
			return err
		}
	} else if semanticTokensRegistered && !options.SemanticTokens {
		if err := s.client.UnregisterCapability(ctx, &protocol.UnregistrationParams{
			Unregisterations: []protocol.Unregistration{
				{
					ID:     registration.ID,
					Method: registration.Method,
				},
			},
		}); err != nil {
			return err
		}
	}
	return nil
}

func semanticTokenRegistration(tokenTypes, tokenModifiers []string) protocol.Registration {
	return protocol.Registration{
		ID:     "textDocument/semanticTokens",
		Method: "textDocument/semanticTokens",
		RegisterOptions: &protocol.SemanticTokensOptions{
			Legend: protocol.SemanticTokensLegend{
				// TODO(pjw): trim these to what we use (and an unused one
				// at position 0 of TokTypes, to catch typos)
				TokenTypes:     tokenTypes,
				TokenModifiers: tokenModifiers,
			},
			Full:  true,
			Range: true,
		},
	}
}
