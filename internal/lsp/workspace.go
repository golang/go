// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) changeFolders(ctx context.Context, event protocol.WorkspaceFoldersChangeEvent) error {
	s.log.Infof(ctx, "change folders")
	for _, folder := range event.Removed {
		if err := s.removeView(ctx, folder.Name, span.NewURI(folder.URI)); err != nil {
			return err
		}
	}

	for _, folder := range event.Added {
		if err := s.addView(ctx, folder.Name, span.NewURI(folder.URI)); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) addView(ctx context.Context, name string, uri span.URI) error {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	// We need a "detached" context so it does not get timeout cancelled.
	// TODO(iancottrell): Do we need to copy any values across?
	viewContext := context.Background()
	s.log.Infof(viewContext, "add view %v as %v", name, uri)
	folderPath, err := uri.Filename()
	if err != nil {
		return err
	}
	s.views = append(s.views, cache.NewView(viewContext, s.log, name, uri, &packages.Config{
		Context: viewContext,
		Dir:     folderPath,
		Env:     os.Environ(),
		Mode:    packages.LoadImports,
		Fset:    token.NewFileSet(),
		Overlay: make(map[string][]byte),
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
		},
		Tests: true,
	}))
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]source.View)
	return nil
}

func (s *Server) removeView(ctx context.Context, name string, uri span.URI) error {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]source.View)
	s.log.Infof(ctx, "drop view %v as %v", name, uri)
	for i, view := range s.views {
		if view.Name() == name {
			// delete this view... we don't care about order but we do want to make
			// sure we can garbage collect the view
			s.views[i] = s.views[len(s.views)-1]
			s.views[len(s.views)-1] = nil
			s.views = s.views[:len(s.views)-1]
			//TODO: shutdown the view in here
			return nil
		}
	}
	return fmt.Errorf("view %s for %v not found", name, uri)
}

// findView returns the view corresponding to the given URI.
// If the file is not already associated with a view, pick one using some heuristics.
func (s *Server) findView(ctx context.Context, uri span.URI) source.View {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()

	// check if we already know this file
	if v, found := s.viewMap[uri]; found {
		return v
	}

	// pick the best view for this file and memoize the result
	v := s.bestView(ctx, uri)
	s.viewMap[uri] = v
	return v
}

// bestView finds the best view to associate a given URI with.
// viewMu must be held when calling this method.
func (s *Server) bestView(ctx context.Context, uri span.URI) source.View {
	// we need to find the best view for this file
	var longest source.View
	for _, view := range s.views {
		if longest != nil && len(longest.Folder()) > len(view.Folder()) {
			continue
		}
		if strings.HasPrefix(string(uri), string(view.Folder())) {
			longest = view
		}
	}
	if longest != nil {
		return longest
	}
	//TODO: are there any more heuristics we can use?
	return s.views[0]
}
