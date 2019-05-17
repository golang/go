// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/span"
)

type session struct {
	cache *cache
	// the logger to use to communicate back with the client
	log xlog.Logger

	viewMu  sync.Mutex
	views   []*view
	viewMap map[span.URI]source.View

	overlayMu sync.Mutex
	overlays  map[span.URI][]byte
}

func (s *session) Shutdown(ctx context.Context) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	for _, view := range s.views {
		view.shutdown(ctx)
	}
	s.views = nil
	s.viewMap = nil
}

func (s *session) Cache() source.Cache {
	return s.cache
}

func (s *session) NewView(name string, folder span.URI) source.View {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	ctx := context.Background()
	backgroundCtx, cancel := context.WithCancel(ctx)
	v := &view{
		session:        s,
		baseCtx:        ctx,
		backgroundCtx:  backgroundCtx,
		cancel:         cancel,
		name:           name,
		folder:         folder,
		filesByURI:     make(map[span.URI]viewFile),
		filesByBase:    make(map[string][]viewFile),
		contentChanges: make(map[span.URI]func()),
		mcache: &metadataCache{
			packages: make(map[string]*metadata),
		},
		pcache: &packageCache{
			packages: make(map[string]*entry),
		},
		ignoredURIs: make(map[span.URI]struct{}),
	}
	s.views = append(s.views, v)
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]source.View)
	return v
}

// View returns the view by name.
func (s *session) View(name string) source.View {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	for _, view := range s.views {
		if view.Name() == name {
			return view
		}
	}
	return nil
}

// ViewOf returns a view corresponding to the given URI.
// If the file is not already associated with a view, pick one using some heuristics.
func (s *session) ViewOf(uri span.URI) source.View {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()

	// check if we already know this file
	if v, found := s.viewMap[uri]; found {
		return v
	}

	// pick the best view for this file and memoize the result
	v := s.bestView(uri)
	s.viewMap[uri] = v
	return v
}

func (s *session) Views() []source.View {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	result := make([]source.View, len(s.views))
	for i, v := range s.views {
		result[i] = v
	}
	return result
}

// bestView finds the best view to associate a given URI with.
// viewMu must be held when calling this method.
func (s *session) bestView(uri span.URI) source.View {
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

func (s *session) removeView(ctx context.Context, view *view) error {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]source.View)
	for i, v := range s.views {
		if view == v {
			// delete this view... we don't care about order but we do want to make
			// sure we can garbage collect the view
			s.views[i] = s.views[len(s.views)-1]
			s.views[len(s.views)-1] = nil
			s.views = s.views[:len(s.views)-1]
			v.shutdown(ctx)
			return nil
		}
	}
	return fmt.Errorf("view %s for %v not found", view.Name(), view.Folder())
}

func (s *session) Logger() xlog.Logger {
	return s.log
}

func (s *session) setOverlay(uri span.URI, content []byte) {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()
	//TODO: we also need to invalidate anything that depended on this "file"
	if content == nil {
		delete(s.overlays, uri)
		return
	}
	s.overlays[uri] = content
}

func (s *session) buildOverlay() map[string][]byte {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()
	overlay := make(map[string][]byte)
	for uri, content := range s.overlays {
		if filename, err := uri.Filename(); err == nil {
			overlay[filename] = content
		}
	}
	return overlay
}
