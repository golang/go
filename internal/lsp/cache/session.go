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

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

type Session struct {
	cache *Cache
	id    string

	options source.Options

	viewMu  sync.Mutex
	views   []*view
	viewMap map[span.URI]*view

	overlayMu sync.Mutex
	overlays  map[span.URI]*overlay
}

type overlay struct {
	session *Session
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
		SessionID:  o.session.id,
		Version:    o.version,
		Kind:       o.kind,
	}
}
func (o *overlay) Read(ctx context.Context) ([]byte, string, error) {
	return o.text, o.hash, nil
}

func (s *Session) Options() source.Options {
	return s.options
}

func (s *Session) SetOptions(options source.Options) {
	s.options = options
}

func (s *Session) Shutdown(ctx context.Context) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	for _, view := range s.views {
		view.shutdown(ctx)
	}
	s.views = nil
	s.viewMap = nil
	if di := debug.GetInstance(ctx); di != nil {
		di.State.DropSession(DebugSession{s})
	}
}

func (s *Session) Cache() source.Cache {
	return s.cache
}

func (s *Session) NewView(ctx context.Context, name string, folder span.URI, options source.Options) (source.View, source.Snapshot, error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	v, snapshot, err := s.createView(ctx, name, folder, options, 0)
	if err != nil {
		return nil, nil, err
	}
	s.views = append(s.views, v)
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]*view)
	return v, snapshot, nil
}

func (s *Session) createView(ctx context.Context, name string, folder span.URI, options source.Options, snapshotID uint64) (*view, *snapshot, error) {
	index := atomic.AddInt64(&viewIndex, 1)
	// We want a true background context and not a detached context here
	// the spans need to be unrelated and no tag values should pollute it.
	baseCtx := event.Detach(xcontext.Detach(ctx))
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
			id:                snapshotID,
			packages:          make(map[packageKey]*packageHandle),
			ids:               make(map[span.URI][]packageID),
			metadata:          make(map[packageID]*metadata),
			files:             make(map[span.URI]source.FileHandle),
			importedBy:        make(map[packageID][]packageID),
			actions:           make(map[actionKey]*actionHandle),
			workspacePackages: make(map[packageID]packagePath),
			unloadableFiles:   make(map[span.URI]struct{}),
			modHandles:        make(map[span.URI]*modHandle),
		},
		ignoredURIs: make(map[span.URI]struct{}),
		gocmdRunner: &gocommand.Runner{},
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

	if di := debug.GetInstance(ctx); di != nil {
		di.State.AddView(debugView{v})
	}
	return v, v.snapshot, nil
}

// View returns the view by name.
func (s *Session) View(name string) source.View {
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
func (s *Session) ViewOf(uri span.URI) (source.View, error) {
	return s.viewOf(uri)
}

func (s *Session) viewOf(uri span.URI) (*view, error) {
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

func (s *Session) viewsOf(uri span.URI) []*view {
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

func (s *Session) Views() []source.View {
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
func (s *Session) bestView(uri span.URI) (*view, error) {
	if len(s.views) == 0 {
		return nil, errors.Errorf("no views in the session")
	}
	// we need to find the best view for this file
	var longest *view
	for _, view := range s.views {
		if longest != nil && len(longest.Folder()) > len(view.Folder()) {
			continue
		}
		if view.contains(uri) {
			longest = view
		}
	}
	if longest != nil {
		return longest, nil
	}
	// TODO: are there any more heuristics we can use?
	return s.views[0], nil
}

func (s *Session) removeView(ctx context.Context, view *view) error {
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

func (s *Session) updateView(ctx context.Context, view *view, options source.Options) (*view, *snapshot, error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	i, err := s.dropView(ctx, view)
	if err != nil {
		return nil, nil, err
	}
	// Preserve the snapshot ID if we are recreating the view.
	view.snapshotMu.Lock()
	snapshotID := view.snapshot.id
	view.snapshotMu.Unlock()
	v, snapshot, err := s.createView(ctx, view.name, view.folder, options, snapshotID)
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

func (s *Session) dropView(ctx context.Context, v *view) (int, error) {
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

func (s *Session) DidModifyFiles(ctx context.Context, changes []source.FileModification) ([]source.Snapshot, error) {
	views := make(map[*view]map[span.URI]source.FileHandle)

	overlays, err := s.updateOverlays(ctx, changes)
	if err != nil {
		return nil, err
	}
	forceReloadMetadata := false
	for _, c := range changes {
		if c.Action == source.InvalidateMetadata {
			forceReloadMetadata = true
		}
		// Do nothing if the file is open in the editor and we receive
		// an on-disk action. The editor is the source of truth.
		if s.isOpen(c.URI) && c.OnDisk {
			continue
		}
		// Look through all of the session's views, invalidating the file for
		// all of the views to which it is known.
		for _, view := range s.views {
			if view.Ignore(c.URI) {
				return nil, errors.Errorf("ignored file %v", c.URI)
			}
			// Don't propagate changes that are outside of the view's scope
			// or knowledge.
			if !view.relevantChange(c) {
				continue
			}
			// Make sure that the file is added to the view.
			if _, err := view.getFile(c.URI); err != nil {
				return nil, err
			}
			if _, ok := views[view]; !ok {
				views[view] = make(map[span.URI]source.FileHandle)
			}
			if o, ok := overlays[c.URI]; ok {
				views[view][c.URI] = o
			} else {
				views[view][c.URI] = s.cache.GetFile(c.URI)
			}
		}
	}
	var snapshots []source.Snapshot
	for view, uris := range views {
		snapshots = append(snapshots, view.invalidateContent(ctx, uris, forceReloadMetadata))
	}
	return snapshots, nil
}

func (s *Session) isOpen(uri span.URI) bool {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	_, open := s.overlays[uri]
	return open
}

func (s *Session) updateOverlays(ctx context.Context, changes []source.FileModification) (map[span.URI]*overlay, error) {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	for _, c := range changes {
		// Don't update overlays for on-disk changes or metadata invalidations.
		if c.OnDisk || c.Action == source.InvalidateMetadata {
			continue
		}

		o, ok := s.overlays[c.URI]

		// Determine the file kind on open, otherwise, assume it has been cached.
		var kind source.FileKind
		switch c.Action {
		case source.Open:
			kind = source.DetectLanguage(c.LanguageID, c.URI.Filename())
		default:
			if !ok {
				return nil, errors.Errorf("updateOverlays: modifying unopened overlay %v", c.URI)
			}
			kind = o.kind
		}
		if kind == source.UnknownKind {
			return nil, errors.Errorf("updateOverlays: unknown file kind for %s", c.URI)
		}

		// Closing a file just deletes its overlay.
		if c.Action == source.Close {
			delete(s.overlays, c.URI)
			continue
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
				return nil, errors.Errorf("updateOverlays: saving %s at version %v, currently at %v", c.URI, c.Version, o.version)
			}
			if c.Text != nil && o.hash != hash {
				return nil, errors.Errorf("updateOverlays: overlay %s changed on save", c.URI)
			}
			sameContentOnDisk = true
		}
		o = &overlay{
			session: s,
			uri:     c.URI,
			version: c.Version,
			text:    text,
			kind:    kind,
			hash:    hash,
			saved:   sameContentOnDisk,
		}
		s.overlays[c.URI] = o
	}

	// Get the overlays for each change while the session's overlay map is locked.
	overlays := make(map[span.URI]*overlay)
	for _, c := range changes {
		if o, ok := s.overlays[c.URI]; ok {
			overlays[c.URI] = o
		}
	}
	return overlays, nil
}

// GetFile implements the source.FileSystem interface.
func (s *Session) GetFile(uri span.URI) source.FileHandle {
	if overlay := s.readOverlay(uri); overlay != nil {
		return overlay
	}
	// Fall back to the cache-level file system.
	return s.cache.GetFile(uri)
}

func (s *Session) readOverlay(uri span.URI) *overlay {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	if overlay, ok := s.overlays[uri]; ok {
		return overlay
	}
	return nil
}

func (s *Session) UnsavedFiles() []span.URI {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	var unsaved []span.URI
	for uri, overlay := range s.overlays {
		if !overlay.saved {
			unsaved = append(unsaved, uri)
		}
	}
	return unsaved
}
