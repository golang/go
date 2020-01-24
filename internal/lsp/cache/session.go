// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/trace"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

type session struct {
	cache *cache
	id    string

	options source.Options

	viewMu  sync.Mutex
	views   []*view
	viewMap map[span.URI]*view

	overlayMu sync.Mutex
	overlays  map[span.URI]*overlay
}

func (s *session) Options() source.Options {
	return s.options
}

func (s *session) SetOptions(options source.Options) {
	s.options = options
}

func (s *session) Shutdown(ctx context.Context) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	for _, view := range s.views {
		view.shutdown(ctx)
	}
	s.views = nil
	s.viewMap = nil
	debug.DropSession(debugSession{s})
}

func (s *session) Cache() source.Cache {
	return s.cache
}

func (s *session) NewView(ctx context.Context, name string, folder span.URI, options source.Options) (source.View, source.Snapshot, error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	v, snapshot, err := s.createView(ctx, name, folder, options)
	if err != nil {
		return nil, nil, err
	}
	s.views = append(s.views, v)
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]*view)
	return v, snapshot, nil
}

func (s *session) createView(ctx context.Context, name string, folder span.URI, options source.Options) (*view, *snapshot, error) {
	index := atomic.AddInt64(&viewIndex, 1)
	// We want a true background context and not a detached context here
	// the spans need to be unrelated and no tag values should pollute it.
	baseCtx := trace.Detach(xcontext.Detach(ctx))
	backgroundCtx, cancel := context.WithCancel(baseCtx)

	v := &view{
		session:       s,
		initialized:   make(chan struct{}),
		id:            strconv.FormatInt(index, 10),
		options:       options,
		baseCtx:       baseCtx,
		backgroundCtx: backgroundCtx,
		cancel:        cancel,
		name:          name,
		folder:        folder,
		filesByURI:    make(map[span.URI]*fileBase),
		filesByBase:   make(map[string][]*fileBase),
		snapshot: &snapshot{
			packages:          make(map[packageKey]*packageHandle),
			ids:               make(map[span.URI][]packageID),
			metadata:          make(map[packageID]*metadata),
			files:             make(map[span.URI]source.FileHandle),
			importedBy:        make(map[packageID][]packageID),
			actions:           make(map[actionKey]*actionHandle),
			workspacePackages: make(map[packageID]packagePath),
			unloadableFiles:   make(map[span.URI]struct{}),
		},
		ignoredURIs: make(map[span.URI]struct{}),
	}
	v.snapshot.view = v

	if v.session.cache.options != nil {
		v.session.cache.options(&v.options)
	}
	// Set the module-specific information.
	if err := v.setBuildInformation(ctx, folder, options.Env, v.options.TempModfile); err != nil {
		return nil, nil, err
	}

	// Initialize the view without blocking.
	go v.initialize(xcontext.Detach(ctx), v.snapshot)

	debug.AddView(debugView{v})
	return v, v.snapshot, nil
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
func (s *session) ViewOf(uri span.URI) (source.View, error) {
	return s.viewOf(uri)
}

func (s *session) viewOf(uri span.URI) (*view, error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()

	// Check if we already know this file.
	if v, found := s.viewMap[uri]; found {
		return v, nil
	}
	// Pick the best view for this file and memoize the result.
	v, err := s.bestView(uri)
	if err != nil {
		return nil, err
	}
	s.viewMap[uri] = v
	return v, nil
}

func (s *session) viewsOf(uri span.URI) []*view {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()

	var views []*view
	for _, view := range s.views {
		if strings.HasPrefix(string(uri), string(view.Folder())) {
			views = append(views, view)
		}
	}
	return views
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
func (s *session) bestView(uri span.URI) (*view, error) {
	if len(s.views) == 0 {
		return nil, errors.Errorf("no views in the session")
	}
	// we need to find the best view for this file
	var longest *view
	for _, view := range s.views {
		if longest != nil && len(longest.Folder()) > len(view.Folder()) {
			continue
		}
		if strings.HasPrefix(string(uri), string(view.Folder())) {
			longest = view
		}
	}
	if longest != nil {
		return longest, nil
	}
	// TODO: are there any more heuristics we can use?
	return s.views[0], nil
}

func (s *session) removeView(ctx context.Context, view *view) error {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	i, err := s.dropView(ctx, view)
	if err != nil {
		return err
	}
	// delete this view... we don't care about order but we do want to make
	// sure we can garbage collect the view
	s.views[i] = s.views[len(s.views)-1]
	s.views[len(s.views)-1] = nil
	s.views = s.views[:len(s.views)-1]
	return nil
}

func (s *session) updateView(ctx context.Context, view *view, options source.Options) (*view, *snapshot, error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	i, err := s.dropView(ctx, view)
	if err != nil {
		return nil, nil, err
	}
	v, snapshot, err := s.createView(ctx, view.name, view.folder, options)
	if err != nil {
		// we have dropped the old view, but could not create the new one
		// this should not happen and is very bad, but we still need to clean
		// up the view array if it happens
		s.views[i] = s.views[len(s.views)-1]
		s.views[len(s.views)-1] = nil
		s.views = s.views[:len(s.views)-1]
	}
	// substitute the new view into the array where the old view was
	s.views[i] = v
	return v, snapshot, nil
}

func (s *session) dropView(ctx context.Context, v *view) (int, error) {
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]*view)
	for i := range s.views {
		if v == s.views[i] {
			// we found the view, drop it and return the index it was found at
			s.views[i] = nil
			v.shutdown(ctx)
			return i, nil
		}
	}
	return -1, errors.Errorf("view %s for %v not found", v.Name(), v.Folder())
}

func (s *session) DidModifyFiles(ctx context.Context, changes []source.FileModification) ([]source.Snapshot, error) {
	views := make(map[*view][]span.URI)

	for _, c := range changes {
		// Only update overlays for in-editor changes.
		if !c.OnDisk {
			if err := s.updateOverlay(ctx, c); err != nil {
				return nil, err
			}
		}
		for _, view := range s.viewsOf(c.URI) {
			if view.Ignore(c.URI) {
				return nil, errors.Errorf("ignored file %v", c.URI)
			}
			// If the file change is on-disk and not a create,
			// make sure the file is known to the view already.
			if c.OnDisk {
				switch c.Action {
				case source.Change, source.Delete:
					if !view.knownFile(c.URI) {
						continue
					}
				}
			}
			// Make sure that the file is added to the view.
			if _, err := view.getFile(c.URI); err != nil {
				return nil, err
			}
			views[view] = append(views[view], c.URI)
		}
	}
	var snapshots []source.Snapshot
	for view, uris := range views {
		snapshots = append(snapshots, view.invalidateContent(ctx, uris))
	}
	return snapshots, nil
}

func (s *session) IsOpen(uri span.URI) bool {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	_, open := s.overlays[uri]
	return open
}

func (s *session) GetFile(uri span.URI) source.FileHandle {
	if overlay := s.readOverlay(uri); overlay != nil {
		return overlay
	}
	// Fall back to the cache-level file system.
	return s.cache.GetFile(uri)
}
