// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/source/typerefs"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/persistent"
	"golang.org/x/tools/internal/xcontext"
)

type Session struct {
	// Unique identifier for this session.
	id string

	// Immutable attributes shared across views.
	cache       *Cache            // shared cache
	gocmdRunner *gocommand.Runner // limits go command concurrency

	optionsMu sync.Mutex
	options   *source.Options

	viewMu  sync.Mutex
	views   []*View
	viewMap map[span.URI]*View // map of URI->best view

	parseCache *parseCache

	*overlayFS
}

// ID returns the unique identifier for this session on this server.
func (s *Session) ID() string     { return s.id }
func (s *Session) String() string { return s.id }

// GoCommandRunner returns the gocommand Runner for this session.
func (s *Session) GoCommandRunner() *gocommand.Runner {
	return s.gocmdRunner
}

// Options returns a copy of the SessionOptions for this session.
func (s *Session) Options() *source.Options {
	s.optionsMu.Lock()
	defer s.optionsMu.Unlock()
	return s.options
}

// SetOptions sets the options of this session to new values.
func (s *Session) SetOptions(options *source.Options) {
	s.optionsMu.Lock()
	defer s.optionsMu.Unlock()
	s.options = options
}

// Shutdown the session and all views it has created.
func (s *Session) Shutdown(ctx context.Context) {
	var views []*View
	s.viewMu.Lock()
	views = append(views, s.views...)
	s.views = nil
	s.viewMap = nil
	s.viewMu.Unlock()
	for _, view := range views {
		view.shutdown()
	}
	s.parseCache.stop()
	event.Log(ctx, "Shutdown session", KeyShutdownSession.Of(s))
}

// Cache returns the cache that created this session, for debugging only.
func (s *Session) Cache() *Cache {
	return s.cache
}

// NewView creates a new View, returning it and its first snapshot. If a
// non-empty tempWorkspace directory is provided, the View will record a copy
// of its gopls workspace module in that directory, so that client tooling
// can execute in the same main module.  On success it also returns a release
// function that must be called when the Snapshot is no longer needed.
func (s *Session) NewView(ctx context.Context, name string, folder span.URI, options *source.Options) (*View, source.Snapshot, func(), error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	for _, view := range s.views {
		if span.SameExistingFile(view.folder, folder) {
			return nil, nil, nil, source.ErrViewExists
		}
	}
	view, snapshot, release, err := s.createView(ctx, name, folder, options, 0)
	if err != nil {
		return nil, nil, nil, err
	}
	s.views = append(s.views, view)
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]*View)
	return view, snapshot, release, nil
}

// TODO(rfindley): clarify that createView can never be cancelled (with the
// possible exception of server shutdown).
// On success, the caller becomes responsible for calling the release function once.
func (s *Session) createView(ctx context.Context, name string, folder span.URI, options *source.Options, seqID uint64) (*View, *snapshot, func(), error) {
	index := atomic.AddInt64(&viewIndex, 1)

	// Get immutable workspace information.
	info, err := s.getWorkspaceInformation(ctx, folder, options)
	if err != nil {
		return nil, nil, nil, err
	}

	gowork, _ := info.GOWORK()
	wsModFiles, wsModFilesErr := computeWorkspaceModFiles(ctx, info.gomod, gowork, info.effectiveGO111MODULE(), s)

	// We want a true background context and not a detached context here
	// the spans need to be unrelated and no tag values should pollute it.
	baseCtx := event.Detach(xcontext.Detach(ctx))
	backgroundCtx, cancel := context.WithCancel(baseCtx)

	v := &View{
		id:                   strconv.FormatInt(index, 10),
		gocmdRunner:          s.gocmdRunner,
		initialWorkspaceLoad: make(chan struct{}),
		initializationSema:   make(chan struct{}, 1),
		options:              options,
		baseCtx:              baseCtx,
		name:                 name,
		folder:               folder,
		moduleUpgrades:       map[span.URI]map[string]string{},
		vulns:                map[span.URI]*govulncheck.Result{},
		parseCache:           s.parseCache,
		fs:                   s.overlayFS,
		workspaceInformation: info,
	}
	v.importsState = &importsState{
		ctx: backgroundCtx,
		processEnv: &imports.ProcessEnv{
			GocmdRunner: s.gocmdRunner,
			SkipPathInScan: func(dir string) bool {
				prefix := strings.TrimSuffix(string(v.folder), "/") + "/"
				uri := strings.TrimSuffix(string(span.URIFromPath(dir)), "/")
				if !strings.HasPrefix(uri+"/", prefix) {
					return false
				}
				filterer := source.NewFilterer(options.DirectoryFilters)
				rel := strings.TrimPrefix(uri, prefix)
				disallow := filterer.Disallow(rel)
				return disallow
			},
		},
	}
	v.snapshot = &snapshot{
		sequenceID:           seqID,
		globalID:             nextSnapshotID(),
		view:                 v,
		backgroundCtx:        backgroundCtx,
		cancel:               cancel,
		store:                s.cache.store,
		packages:             persistent.NewMap(packageIDLessInterface),
		meta:                 new(metadataGraph),
		files:                newFilesMap(),
		activePackages:       persistent.NewMap(packageIDLessInterface),
		symbolizeHandles:     persistent.NewMap(uriLessInterface),
		workspacePackages:    make(map[PackageID]PackagePath),
		unloadableFiles:      make(map[span.URI]struct{}),
		parseModHandles:      persistent.NewMap(uriLessInterface),
		parseWorkHandles:     persistent.NewMap(uriLessInterface),
		modTidyHandles:       persistent.NewMap(uriLessInterface),
		modVulnHandles:       persistent.NewMap(uriLessInterface),
		modWhyHandles:        persistent.NewMap(uriLessInterface),
		knownSubdirs:         newKnownDirsSet(),
		workspaceModFiles:    wsModFiles,
		workspaceModFilesErr: wsModFilesErr,
		pkgIndex:             typerefs.NewPackageIndex(),
	}
	// Save one reference in the view.
	v.releaseSnapshot = v.snapshot.Acquire()

	// Record the environment of the newly created view in the log.
	event.Log(ctx, viewEnv(v))

	// Initialize the view without blocking.
	initCtx, initCancel := context.WithCancel(xcontext.Detach(ctx))
	v.initCancelFirstAttempt = initCancel
	snapshot := v.snapshot

	// Pass a second reference to the background goroutine.
	bgRelease := snapshot.Acquire()
	go func() {
		defer bgRelease()
		snapshot.initialize(initCtx, true)
	}()

	// Return a third reference to the caller.
	return v, snapshot, snapshot.Acquire(), nil
}

// ViewByName returns a view with a matching name, if the session has one.
func (s *Session) ViewByName(name string) *View {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	for _, view := range s.views {
		if view.Name() == name {
			return view
		}
	}
	return nil
}

// View returns the view with a matching id, if present.
func (s *Session) View(id string) (*View, error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	for _, view := range s.views {
		if view.ID() == id {
			return view, nil
		}
	}
	return nil, fmt.Errorf("no view with ID %q", id)
}

// ViewOf returns a view corresponding to the given URI.
// If the file is not already associated with a view, pick one using some heuristics.
func (s *Session) ViewOf(uri span.URI) (*View, error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	return s.viewOfLocked(uri)
}

// Precondition: caller holds s.viewMu lock.
func (s *Session) viewOfLocked(uri span.URI) (*View, error) {
	// Check if we already know this file.
	if v, found := s.viewMap[uri]; found {
		return v, nil
	}
	// Pick the best view for this file and memoize the result.
	if len(s.views) == 0 {
		return nil, fmt.Errorf("no views in session")
	}
	s.viewMap[uri] = bestViewForURI(uri, s.views)
	return s.viewMap[uri], nil
}

func (s *Session) Views() []*View {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	result := make([]*View, len(s.views))
	copy(result, s.views)
	return result
}

// bestViewForURI returns the most closely matching view for the given URI
// out of the given set of views.
func bestViewForURI(uri span.URI, views []*View) *View {
	// we need to find the best view for this file
	var longest *View
	for _, view := range views {
		if longest != nil && len(longest.Folder()) > len(view.Folder()) {
			continue
		}
		// TODO(rfindley): this should consider the workspace layout (i.e.
		// go.work).
		if view.contains(uri) {
			longest = view
		}
	}
	if longest != nil {
		return longest
	}
	// Try our best to return a view that knows the file.
	for _, view := range views {
		if view.knownFile(uri) {
			return view
		}
	}
	// TODO: are there any more heuristics we can use?
	return views[0]
}

// RemoveView removes the view v from the session
func (s *Session) RemoveView(view *View) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	i := s.dropView(view)
	if i == -1 { // error reported elsewhere
		return
	}
	// delete this view... we don't care about order but we do want to make
	// sure we can garbage collect the view
	s.views = removeElement(s.views, i)
}

// updateView recreates the view with the given options.
//
// If the resulting error is non-nil, the view may or may not have already been
// dropped from the session.
func (s *Session) updateView(ctx context.Context, view *View, options *source.Options) (*View, error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()

	return s.updateViewLocked(ctx, view, options)
}

func (s *Session) updateViewLocked(ctx context.Context, view *View, options *source.Options) (*View, error) {
	// Preserve the snapshot ID if we are recreating the view.
	view.snapshotMu.Lock()
	if view.snapshot == nil {
		view.snapshotMu.Unlock()
		panic("updateView called after View was already shut down")
	}
	seqID := view.snapshot.sequenceID // Preserve sequence IDs when updating a view in place.
	view.snapshotMu.Unlock()

	i := s.dropView(view)
	if i == -1 {
		return nil, fmt.Errorf("view %q not found", view.id)
	}

	v, snapshot, release, err := s.createView(ctx, view.name, view.folder, options, seqID)
	if err != nil {
		// we have dropped the old view, but could not create the new one
		// this should not happen and is very bad, but we still need to clean
		// up the view array if it happens
		s.views = removeElement(s.views, i)
		return nil, err
	}
	defer release()

	// The new snapshot has lost the history of the previous view. As a result,
	// it may not see open files that aren't in its build configuration (as it
	// would have done via didOpen notifications). This can lead to inconsistent
	// behavior when configuration is changed mid-session.
	//
	// Ensure the new snapshot observes all open files.
	for _, o := range v.fs.Overlays() {
		_, _ = snapshot.ReadFile(ctx, o.URI())
	}

	// substitute the new view into the array where the old view was
	s.views[i] = v
	return v, nil
}

// removeElement removes the ith element from the slice replacing it with the last element.
// TODO(adonovan): generics, someday.
func removeElement(slice []*View, index int) []*View {
	last := len(slice) - 1
	slice[index] = slice[last]
	slice[last] = nil // aid GC
	return slice[:last]
}

// dropView removes v from the set of views for the receiver s and calls
// v.shutdown, returning the index of v in s.views (if found), or -1 if v was
// not found. s.viewMu must be held while calling this function.
func (s *Session) dropView(v *View) int {
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]*View)
	for i := range s.views {
		if v == s.views[i] {
			// we found the view, drop it and return the index it was found at
			s.views[i] = nil
			v.shutdown()
			return i
		}
	}
	// TODO(rfindley): it looks wrong that we don't shutdown v in this codepath.
	// We should never get here.
	bug.Reportf("tried to drop nonexistent view %q", v.id)
	return -1
}

func (s *Session) ModifyFiles(ctx context.Context, changes []source.FileModification) error {
	_, release, err := s.DidModifyFiles(ctx, changes)
	release()
	return err
}

// TODO(rfindley): fileChange seems redundant with source.FileModification.
// De-dupe into a common representation for changes.
type fileChange struct {
	content    []byte
	exists     bool
	fileHandle source.FileHandle

	// isUnchanged indicates whether the file action is one that does not
	// change the actual contents of the file. Opens and closes should not
	// be treated like other changes, since the file content doesn't change.
	isUnchanged bool
}

// DidModifyFiles reports a file modification to the session. It returns
// the new snapshots after the modifications have been applied, paired with
// the affected file URIs for those snapshots.
// On success, it returns a release function that
// must be called when the snapshots are no longer needed.
//
// TODO(rfindley): what happens if this function fails? It must leave us in a
// broken state, which we should surface to the user, probably as a request to
// restart gopls.
func (s *Session) DidModifyFiles(ctx context.Context, changes []source.FileModification) (map[source.Snapshot][]span.URI, func(), error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()

	// Update overlays.
	//
	// TODO(rfindley): I think we do this while holding viewMu to prevent views
	// from seeing the updated file content before they have processed
	// invalidations, which could lead to a partial view of the changes (i.e.
	// spurious diagnostics). However, any such view would immediately be
	// invalidated here, so it is possible that we could update overlays before
	// acquiring viewMu.
	if err := s.updateOverlays(ctx, changes); err != nil {
		return nil, nil, err
	}

	// Re-create views whose definition may have changed.
	//
	// checkViews controls whether to re-evaluate view definitions when
	// collecting views below. Any addition or deletion of a go.mod or go.work
	// file may have affected the definition of the view.
	checkViews := false

	for _, c := range changes {
		if isGoMod(c.URI) || isGoWork(c.URI) {
			// Change, InvalidateMetadata, and UnknownFileAction actions do not cause
			// us to re-evaluate views.
			redoViews := (c.Action != source.Change &&
				c.Action != source.InvalidateMetadata &&
				c.Action != source.UnknownFileAction)

			if redoViews {
				checkViews = true
				break
			}
		}
	}

	if checkViews {
		for _, view := range s.views {
			// TODO(rfindley): can we avoid running the go command (go env)
			// synchronously to change processing? Can we assume that the env did not
			// change, and derive go.work using a combination of the configured
			// GOWORK value and filesystem?
			info, err := s.getWorkspaceInformation(ctx, view.folder, view.Options())
			if err != nil {
				// Catastrophic failure, equivalent to a failure of session
				// initialization and therefore should almost never happen. One
				// scenario where this failure mode could occur is if some file
				// permissions have changed preventing us from reading go.mod
				// files.
				//
				// TODO(rfindley): consider surfacing this error more loudly. We
				// could report a bug, but it's not really a bug.
				event.Error(ctx, "fetching workspace information", err)
			}

			if info != view.workspaceInformation {
				_, err := s.updateViewLocked(ctx, view, view.Options())
				if err != nil {
					// More catastrophic failure. The view may or may not still exist.
					// The best we can do is log and move on.
					event.Error(ctx, "recreating view", err)
				}
			}
		}
	}

	// Collect information about views affected by these changes.
	views := make(map[*View]map[span.URI]*fileChange)
	affectedViews := map[span.URI][]*View{}
	// forceReloadMetadata records whether any change is the magic
	// source.InvalidateMetadata action.
	forceReloadMetadata := false
	for _, c := range changes {
		if c.Action == source.InvalidateMetadata {
			forceReloadMetadata = true
		}
		// Build the list of affected views.
		var changedViews []*View
		for _, view := range s.views {
			// Don't propagate changes that are outside of the view's scope
			// or knowledge.
			if !view.relevantChange(c) {
				continue
			}
			changedViews = append(changedViews, view)
		}
		// If the change is not relevant to any view, but the change is
		// happening in the editor, assign it the most closely matching view.
		if len(changedViews) == 0 {
			if c.OnDisk {
				continue
			}
			bestView, err := s.viewOfLocked(c.URI)
			if err != nil {
				return nil, nil, err
			}
			changedViews = append(changedViews, bestView)
		}
		affectedViews[c.URI] = changedViews

		isUnchanged := c.Action == source.Open || c.Action == source.Close

		// Apply the changes to all affected views.
		for _, view := range changedViews {
			// Make sure that the file is added to the view's seenFiles set.
			view.markKnown(c.URI)
			if _, ok := views[view]; !ok {
				views[view] = make(map[span.URI]*fileChange)
			}
			fh, err := s.ReadFile(ctx, c.URI)
			if err != nil {
				return nil, nil, err
			}
			content, err := fh.Content()
			if err != nil {
				// Ignore the error: the file may be deleted.
				content = nil
			}
			views[view][c.URI] = &fileChange{
				content:     content,
				exists:      err == nil,
				fileHandle:  fh,
				isUnchanged: isUnchanged,
			}
		}
	}

	var releases []func()
	viewToSnapshot := map[*View]*snapshot{}
	for view, changed := range views {
		snapshot, release := view.invalidateContent(ctx, changed, forceReloadMetadata)
		releases = append(releases, release)
		viewToSnapshot[view] = snapshot
	}

	// The release function is called when the
	// returned URIs no longer need to be valid.
	release := func() {
		for _, release := range releases {
			release()
		}
	}

	// We only want to diagnose each changed file once, in the view to which
	// it "most" belongs. We do this by picking the best view for each URI,
	// and then aggregating the set of snapshots and their URIs (to avoid
	// diagnosing the same snapshot multiple times).
	snapshotURIs := map[source.Snapshot][]span.URI{}
	for _, mod := range changes {
		viewSlice, ok := affectedViews[mod.URI]
		if !ok || len(viewSlice) == 0 {
			continue
		}
		view := bestViewForURI(mod.URI, viewSlice)
		snapshot, ok := viewToSnapshot[view]
		if !ok {
			panic(fmt.Sprintf("no snapshot for view %s", view.Folder()))
		}
		snapshotURIs[snapshot] = append(snapshotURIs[snapshot], mod.URI)
	}

	return snapshotURIs, release, nil
}

// ExpandModificationsToDirectories returns the set of changes with the
// directory changes removed and expanded to include all of the files in
// the directory.
func (s *Session) ExpandModificationsToDirectories(ctx context.Context, changes []source.FileModification) []source.FileModification {
	var snapshots []*snapshot
	s.viewMu.Lock()
	for _, v := range s.views {
		snapshot, release, err := v.getSnapshot()
		if err != nil {
			continue // view is shut down; continue with others
		}
		defer release()
		snapshots = append(snapshots, snapshot)
	}
	s.viewMu.Unlock()

	knownDirs := knownDirectories(ctx, snapshots)
	defer knownDirs.Destroy()

	var result []source.FileModification
	for _, c := range changes {
		if !knownDirs.Contains(c.URI) {
			result = append(result, c)
		} else {
			for uri := range knownFilesInDir(ctx, snapshots, c.URI) {
				result = append(result, source.FileModification{
					URI:        uri,
					Action:     c.Action,
					LanguageID: "",
					OnDisk:     c.OnDisk,
					// changes to directories cannot include text or versions
				})
			}
		}
	}
	return result
}

// knownDirectories returns all of the directories known to the given
// snapshots, including workspace directories and their subdirectories.
// It is responsibility of the caller to destroy the returned set.
func knownDirectories(ctx context.Context, snapshots []*snapshot) knownDirsSet {
	result := newKnownDirsSet()
	for _, snapshot := range snapshots {
		dirs := snapshot.dirs(ctx)
		for _, dir := range dirs {
			result.Insert(dir)
		}
		knownSubdirs := snapshot.getKnownSubdirs(dirs)
		result.SetAll(knownSubdirs)
		knownSubdirs.Destroy()
	}
	return result
}

// knownFilesInDir returns the files known to the snapshots in the session.
// It does not respect symlinks.
func knownFilesInDir(ctx context.Context, snapshots []*snapshot, dir span.URI) map[span.URI]struct{} {
	files := map[span.URI]struct{}{}

	for _, snapshot := range snapshots {
		for _, uri := range snapshot.knownFilesInDir(ctx, dir) {
			files[uri] = struct{}{}
		}
	}
	return files
}

// Precondition: caller holds s.viewMu lock.
// TODO(rfindley): move this to fs_overlay.go.
func (fs *overlayFS) updateOverlays(ctx context.Context, changes []source.FileModification) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	for _, c := range changes {
		// Don't update overlays for metadata invalidations.
		if c.Action == source.InvalidateMetadata {
			continue
		}

		o, ok := fs.overlays[c.URI]

		// If the file is not opened in an overlay and the change is on disk,
		// there's no need to update an overlay. If there is an overlay, we
		// may need to update the overlay's saved value.
		if !ok && c.OnDisk {
			continue
		}

		// Determine the file kind on open, otherwise, assume it has been cached.
		var kind source.FileKind
		switch c.Action {
		case source.Open:
			kind = source.FileKindForLang(c.LanguageID)
		default:
			if !ok {
				return fmt.Errorf("updateOverlays: modifying unopened overlay %v", c.URI)
			}
			kind = o.kind
		}

		// Closing a file just deletes its overlay.
		if c.Action == source.Close {
			delete(fs.overlays, c.URI)
			continue
		}

		// If the file is on disk, check if its content is the same as in the
		// overlay. Saves and on-disk file changes don't come with the file's
		// content.
		text := c.Text
		if text == nil && (c.Action == source.Save || c.OnDisk) {
			if !ok {
				return fmt.Errorf("no known content for overlay for %s", c.Action)
			}
			text = o.content
		}
		// On-disk changes don't come with versions.
		version := c.Version
		if c.OnDisk || c.Action == source.Save {
			version = o.version
		}
		hash := source.HashOf(text)
		var sameContentOnDisk bool
		switch c.Action {
		case source.Delete:
			// Do nothing. sameContentOnDisk should be false.
		case source.Save:
			// Make sure the version and content (if present) is the same.
			if false && o.version != version { // Client no longer sends the version
				return fmt.Errorf("updateOverlays: saving %s at version %v, currently at %v", c.URI, c.Version, o.version)
			}
			if c.Text != nil && o.hash != hash {
				return fmt.Errorf("updateOverlays: overlay %s changed on save", c.URI)
			}
			sameContentOnDisk = true
		default:
			fh, err := fs.delegate.ReadFile(ctx, c.URI)
			if err != nil {
				return err
			}
			_, readErr := fh.Content()
			sameContentOnDisk = (readErr == nil && fh.FileIdentity().Hash == hash)
		}
		o = &Overlay{
			uri:     c.URI,
			version: version,
			content: text,
			kind:    kind,
			hash:    hash,
			saved:   sameContentOnDisk,
		}

		// NOTE: previous versions of this code checked here that the overlay had a
		// view and file kind (but we don't know why).

		fs.overlays[c.URI] = o
	}

	return nil
}

// FileWatchingGlobPatterns returns a new set of glob patterns to
// watch every directory known by the view. For views within a module,
// this is the module root, any directory in the module root, and any
// replace targets.
func (s *Session) FileWatchingGlobPatterns(ctx context.Context) map[string]struct{} {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	patterns := map[string]struct{}{}
	for _, view := range s.views {
		snapshot, release, err := view.getSnapshot()
		if err != nil {
			continue // view is shut down; continue with others
		}
		for k, v := range snapshot.fileWatchingGlobPatterns(ctx) {
			patterns[k] = v
		}
		release()
	}
	return patterns
}
