// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

type Session struct {
	cache *Cache
	id    string

	options *source.Options

	viewMu  sync.Mutex
	views   []*View
	viewMap map[span.URI]*View

	overlayMu sync.Mutex
	overlays  map[span.URI]*overlay

	// gocmdRunner guards go command calls from concurrency errors.
	gocmdRunner *gocommand.Runner
}

type overlay struct {
	session *Session
	uri     span.URI
	text    []byte
	hash    string
	version float64
	kind    source.FileKind

	// saved is true if a file matches the state on disk,
	// and therefore does not need to be part of the overlay sent to go/packages.
	saved bool
}

func (o *overlay) Read() ([]byte, error) {
	return o.text, nil
}

func (o *overlay) FileIdentity() source.FileIdentity {
	return source.FileIdentity{
		URI:  o.uri,
		Hash: o.hash,
		Kind: o.kind,
	}
}

func (o *overlay) VersionedFileIdentity() source.VersionedFileIdentity {
	return source.VersionedFileIdentity{
		URI:       o.uri,
		SessionID: o.session.id,
		Version:   o.version,
	}
}

func (o *overlay) Kind() source.FileKind {
	return o.kind
}

func (o *overlay) URI() span.URI {
	return o.uri
}

func (o *overlay) Version() float64 {
	return o.version
}

func (o *overlay) Session() string {
	return o.session.id
}

func (o *overlay) Saved() bool {
	return o.saved
}

// closedFile implements LSPFile for a file that the editor hasn't told us about.
type closedFile struct {
	source.FileHandle
}

func (c *closedFile) VersionedFileIdentity() source.VersionedFileIdentity {
	return source.VersionedFileIdentity{
		URI:       c.FileHandle.URI(),
		SessionID: "",
		Version:   0,
	}
}

func (c *closedFile) Session() string {
	return ""
}

func (c *closedFile) Version() float64 {
	return 0
}

func (s *Session) ID() string     { return s.id }
func (s *Session) String() string { return s.id }

func (s *Session) Options() *source.Options {
	return s.options
}

func (s *Session) SetOptions(options *source.Options) {
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
	event.Log(ctx, "Shutdown session", KeyShutdownSession.Of(s))
}

func (s *Session) Cache() interface{} {
	return s.cache
}

func (s *Session) NewView(ctx context.Context, name string, folder span.URI, options *source.Options) (source.View, source.Snapshot, func(), error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	view, snapshot, release, err := s.createView(ctx, name, folder, options, 0)
	if err != nil {
		return nil, nil, func() {}, err
	}
	s.views = append(s.views, view)
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]*View)
	return view, snapshot, release, nil
}

func (s *Session) createView(ctx context.Context, name string, folder span.URI, options *source.Options, snapshotID uint64) (*View, *snapshot, func(), error) {
	index := atomic.AddInt64(&viewIndex, 1)

	if s.cache.options != nil {
		s.cache.options(options)
	}

	// Set the module-specific information.
	ws, err := s.getWorkspaceInformation(ctx, folder, options)
	if err != nil {
		return nil, nil, func() {}, err
	}

	// If workspace module mode is enabled, find all of the modules in the
	// workspace. By default, we just find the root module.
	var modules map[span.URI]*moduleRoot
	modules, err = findWorkspaceModules(ctx, ws.rootURI, options)
	if err != nil {
		return nil, nil, func() {}, err
	}

	// We want a true background context and not a detached context here
	// the spans need to be unrelated and no tag values should pollute it.
	baseCtx := event.Detach(xcontext.Detach(ctx))
	backgroundCtx, cancel := context.WithCancel(baseCtx)

	v := &View{
		session:              s,
		initialized:          make(chan struct{}),
		initializationSema:   make(chan struct{}, 1),
		initializeOnce:       &sync.Once{},
		id:                   strconv.FormatInt(index, 10),
		options:              options,
		baseCtx:              baseCtx,
		backgroundCtx:        backgroundCtx,
		cancel:               cancel,
		name:                 name,
		folder:               folder,
		filesByURI:           make(map[span.URI]*fileBase),
		filesByBase:          make(map[string][]*fileBase),
		workspaceInformation: *ws,
	}
	v.importsState = &importsState{
		ctx: backgroundCtx,
		processEnv: &imports.ProcessEnv{
			GocmdRunner: s.gocmdRunner,
			WorkingDir:  folder.Filename(),
			Env:         ws.goEnv,
		},
	}
	v.snapshot = &snapshot{
		id:                snapshotID,
		view:              v,
		generation:        s.cache.store.Generation(generationName(v, 0)),
		packages:          make(map[packageKey]*packageHandle),
		ids:               make(map[span.URI][]packageID),
		metadata:          make(map[packageID]*metadata),
		files:             make(map[span.URI]source.VersionedFileHandle),
		goFiles:           make(map[parseKey]*parseGoHandle),
		importedBy:        make(map[packageID][]packageID),
		actions:           make(map[actionKey]*actionHandle),
		workspacePackages: make(map[packageID]packagePath),
		unloadableFiles:   make(map[span.URI]struct{}),
		parseModHandles:   make(map[span.URI]*parseModHandle),
		modTidyHandles:    make(map[span.URI]*modTidyHandle),
		modUpgradeHandles: make(map[span.URI]*modUpgradeHandle),
		modWhyHandles:     make(map[span.URI]*modWhyHandle),
		modules:           modules,
	}

	v.snapshot.workspaceDirectories = v.snapshot.findWorkspaceDirectories(ctx)

	// Initialize the view without blocking.
	initCtx, initCancel := context.WithCancel(xcontext.Detach(ctx))
	v.initCancelFirstAttempt = initCancel
	snapshot := v.snapshot
	release := snapshot.generation.Acquire(initCtx)
	go func() {
		snapshot.initialize(initCtx, true)
		release()
	}()
	return v, snapshot, snapshot.generation.Acquire(ctx), nil
}

// findWorkspaceModules walks the view's root folder, looking for go.mod files.
// Any that are found are added to the view's set of modules, which are then
// used to construct the workspace module.
//
// It assumes that the caller has not yet created the view, and therefore does
// not lock any of the internal data structures before accessing them.
//
// TODO(rstambler): Check overlays for go.mod files.
func findWorkspaceModules(ctx context.Context, root span.URI, options *source.Options) (map[span.URI]*moduleRoot, error) {
	// Walk the view's folder to find all modules in the view.
	modules := make(map[span.URI]*moduleRoot)
	if !options.ExperimentalWorkspaceModule {
		path := filepath.Join(root.Filename(), "go.mod")
		if info, _ := os.Stat(path); info != nil {
			if m := getViewModule(ctx, root, span.URIFromPath(path), options); m != nil {
				modules[m.rootURI] = m
			}
		}
		return modules, nil
	}
	return modules, filepath.Walk(root.Filename(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// Probably a permission error. Keep looking.
			return filepath.SkipDir
		}
		// For any path that is not the workspace folder, check if the path
		// would be ignored by the go command. Vendor directories also do not
		// contain workspace modules.
		if info.IsDir() && path != root.Filename() {
			suffix := strings.TrimPrefix(path, root.Filename())
			switch {
			case checkIgnored(suffix),
				strings.Contains(filepath.ToSlash(suffix), "/vendor/"):
				return filepath.SkipDir
			}
		}
		// We're only interested in go.mod files.
		if filepath.Base(path) == "go.mod" {
			if m := getViewModule(ctx, root, span.URIFromPath(path), options); m != nil {
				modules[m.rootURI] = m
			}
		}
		return nil
	})
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

func (s *Session) viewOf(uri span.URI) (*View, error) {
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

func (s *Session) viewsOf(uri span.URI) []*View {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()

	var views []*View
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
func (s *Session) bestView(uri span.URI) (*View, error) {
	if len(s.views) == 0 {
		return nil, errors.Errorf("no views in the session")
	}
	// we need to find the best view for this file
	var longest *View
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
	// Try our best to return a view that knows the file.
	for _, view := range s.views {
		if view.knownFile(uri) {
			return view, nil
		}
	}
	// TODO: are there any more heuristics we can use?
	return s.views[0], nil
}

func (s *Session) removeView(ctx context.Context, view *View) error {
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

func (s *Session) updateView(ctx context.Context, view *View, options *source.Options) (*View, error) {
	s.viewMu.Lock()
	defer s.viewMu.Unlock()
	i, err := s.dropView(ctx, view)
	if err != nil {
		return nil, err
	}
	// Preserve the snapshot ID if we are recreating the view.
	view.snapshotMu.Lock()
	snapshotID := view.snapshot.id
	view.snapshotMu.Unlock()
	v, _, release, err := s.createView(ctx, view.name, view.folder, options, snapshotID)
	release()
	if err != nil {
		// we have dropped the old view, but could not create the new one
		// this should not happen and is very bad, but we still need to clean
		// up the view array if it happens
		s.views[i] = s.views[len(s.views)-1]
		s.views[len(s.views)-1] = nil
		s.views = s.views[:len(s.views)-1]
		return nil, err
	}
	// substitute the new view into the array where the old view was
	s.views[i] = v
	return v, nil
}

func (s *Session) dropView(ctx context.Context, v *View) (int, error) {
	// we always need to drop the view map
	s.viewMap = make(map[span.URI]*View)
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

func (s *Session) ModifyFiles(ctx context.Context, changes []source.FileModification) error {
	_, _, releases, _, err := s.DidModifyFiles(ctx, changes)
	for _, release := range releases {
		release()
	}
	return err
}

func (s *Session) DidModifyFiles(ctx context.Context, changes []source.FileModification) (map[span.URI]source.View, map[source.View]source.Snapshot, []func(), []span.URI, error) {
	views := make(map[*View]map[span.URI]source.VersionedFileHandle)
	bestViews := map[span.URI]source.View{}
	// Keep track of deleted files so that we can clear their diagnostics.
	// A file might be re-created after deletion, so only mark files that
	// have truly been deleted.
	deletions := map[span.URI]struct{}{}

	overlays, err := s.updateOverlays(ctx, changes)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	var forceReloadMetadata bool
	for _, c := range changes {
		if c.Action == source.InvalidateMetadata {
			forceReloadMetadata = true
		}

		// Build the list of affected views.
		bestView, err := s.viewOf(c.URI)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		bestViews[c.URI] = bestView

		var changedViews []*View
		for _, view := range s.views {
			// Don't propagate changes that are outside of the view's scope
			// or knowledge.
			if !view.relevantChange(c) {
				continue
			}
			changedViews = append(changedViews, view)
		}
		// If no view matched the change, assign it to the best view.
		if len(changedViews) == 0 {
			changedViews = append(changedViews, bestView)
		}

		// Apply the changes to all affected views.
		for _, view := range changedViews {
			// Make sure that the file is added to the view.
			if _, err := view.getFile(c.URI); err != nil {
				return nil, nil, nil, nil, err
			}
			if _, ok := views[view]; !ok {
				views[view] = make(map[span.URI]source.VersionedFileHandle)
			}
			var (
				fh source.VersionedFileHandle
				ok bool
			)
			if fh, ok = overlays[c.URI]; ok {
				views[view][c.URI] = fh
				delete(deletions, c.URI)
			} else {
				fsFile, err := s.cache.getFile(ctx, c.URI)
				if err != nil {
					return nil, nil, nil, nil, err
				}
				fh = &closedFile{fsFile}
				views[view][c.URI] = fh
				if _, err := fh.Read(); err != nil {
					deletions[c.URI] = struct{}{}
				}
			}
		}
	}

	snapshots := map[source.View]source.Snapshot{}
	var releases []func()
	for view, uris := range views {
		snapshot, release := view.invalidateContent(ctx, uris, forceReloadMetadata)
		snapshots[view] = snapshot
		releases = append(releases, release)
	}
	var deletionsSlice []span.URI
	for uri := range deletions {
		deletionsSlice = append(deletionsSlice, uri)
	}
	return bestViews, snapshots, releases, deletionsSlice, nil
}

func (s *Session) updateOverlays(ctx context.Context, changes []source.FileModification) (map[span.URI]*overlay, error) {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	for _, c := range changes {
		// Don't update overlays for metadata invalidations.
		if c.Action == source.InvalidateMetadata {
			continue
		}

		o, ok := s.overlays[c.URI]

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

		// If the file is on disk, check if its content is the same as in the
		// overlay. Saves and on-disk file changes don't come with the file's
		// content.
		text := c.Text
		if text == nil && (c.Action == source.Save || c.OnDisk) {
			if !ok {
				return nil, fmt.Errorf("no known content for overlay for %s", c.Action)
			}
			text = o.text
		}
		// On-disk changes don't come with versions.
		version := c.Version
		if c.OnDisk {
			version = o.version
		}
		hash := hashContents(text)
		var sameContentOnDisk bool
		switch c.Action {
		case source.Delete:
			// Do nothing. sameContentOnDisk should be false.
		case source.Save:
			// Make sure the version and content (if present) is the same.
			if o.version != version {
				return nil, errors.Errorf("updateOverlays: saving %s at version %v, currently at %v", c.URI, c.Version, o.version)
			}
			if c.Text != nil && o.hash != hash {
				return nil, errors.Errorf("updateOverlays: overlay %s changed on save", c.URI)
			}
			sameContentOnDisk = true
		default:
			fh, err := s.cache.getFile(ctx, c.URI)
			if err != nil {
				return nil, err
			}
			_, readErr := fh.Read()
			sameContentOnDisk = (readErr == nil && fh.FileIdentity().Hash == hash)
		}
		o = &overlay{
			session: s,
			uri:     c.URI,
			version: version,
			text:    text,
			kind:    kind,
			hash:    hash,
			saved:   sameContentOnDisk,
		}
		s.overlays[c.URI] = o
	}

	// Get the overlays for each change while the session's overlay map is
	// locked.
	overlays := make(map[span.URI]*overlay)
	for _, c := range changes {
		if o, ok := s.overlays[c.URI]; ok {
			overlays[c.URI] = o
		}
	}
	return overlays, nil
}

func (s *Session) GetFile(ctx context.Context, uri span.URI) (source.FileHandle, error) {
	if overlay := s.readOverlay(uri); overlay != nil {
		return overlay, nil
	}
	// Fall back to the cache-level file system.
	return s.cache.getFile(ctx, uri)
}

func (s *Session) readOverlay(uri span.URI) *overlay {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	if overlay, ok := s.overlays[uri]; ok {
		return overlay
	}
	return nil
}

func (s *Session) Overlays() []source.Overlay {
	s.overlayMu.Lock()
	defer s.overlayMu.Unlock()

	overlays := make([]source.Overlay, 0, len(s.overlays))
	for _, overlay := range s.overlays {
		overlays = append(overlays, overlay)
	}
	return overlays
}

// goVersion returns the Go version in use for the given session.
func (s *Session) goVersion(ctx context.Context, folder string, env []string) (int, error) {
	// Check the go version by running "go list" with modules off.
	// Borrowed from internal/imports/mod.go:620.
	const format = `{{context.ReleaseTags}}`
	inv := gocommand.Invocation{
		Verb:       "list",
		Args:       []string{"-e", "-f", format},
		Env:        append(env, "GO111MODULE=off"),
		WorkingDir: folder,
	}
	stdoutBytes, err := s.gocmdRunner.Run(ctx, inv)
	if err != nil {
		return 0, err
	}
	stdout := stdoutBytes.String()
	if len(stdout) < 3 {
		return 0, fmt.Errorf("bad ReleaseTags output: %q", stdout)
	}
	// Split up "[go1.1 go1.15]"
	tags := strings.Fields(stdout[1 : len(stdout)-2])
	for i := len(tags) - 1; i >= 0; i-- {
		var version int
		if _, err := fmt.Sscanf(tags[i], "go1.%d", &version); err != nil {
			continue
		}
		return version, nil
	}
	return 0, fmt.Errorf("no parseable ReleaseTags in %v", tags)
}
