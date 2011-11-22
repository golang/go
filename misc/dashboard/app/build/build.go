// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"appengine"
	"appengine/datastore"
	"http"
)

// A Package describes a package that is listed on the dashboard.
type Package struct {
	Name string
	Path string // (empty for the main Go tree)
}

func (p *Package) Key(c appengine.Context) *datastore.Key {
	key := p.Path
	if key == "" {
		key = "go"
	}
	return datastore.NewKey(c, "Package", key, 0, nil)
}

// A Commit describes an individual commit in a package.
//
// Each Commit entity is a descendant of its associated Package entity.
// In other words, all Commits with the same PackagePath belong to the same
// datastore entity group.
type Commit struct {
	PackagePath string // (empty for Go commits)
	Num         int    // Internal monotonic counter unique to this package.
	Hash        string
	ParentHash  string

	User string
	Desc string `datastore:",noindex"`
	Time datastore.Time

	// Result is the Data string of each build Result for this Commit.
	// For non-Go commits, only the Results for the current Go tip, weekly,
	// and release Tags are stored here. This is purely de-normalized data.
	// The complete data set is stored in Result entities.
	Result []string `datastore:",noindex"`
}

func (com *Commit) Key(c appengine.Context) *datastore.Key {
	key := com.PackagePath + ":" + com.Hash
	return datastore.NewKey(c, "Commit", key, 0, nil)
}

// A Result describes a build result for a Commit on an OS/architecture.
//
// Each Result entity is a descendant of its associated Commit entity.
type Result struct {
	Builder     string // "arch-os[-note]"
	Hash        string
	PackagePath string // (empty for Go commits)

	// The Go Commit this was built against (empty for Go commits).
	GoHash string

	OK      bool
	Log     string `datastore:"-"`        // for JSON unmarshaling
	LogHash string `datastore:",noindex"` // Key to the Log record.
}

func (r *Result) Data() string {
	return fmt.Sprintf("%v|%v|%v|%v", r.Builder, r.OK, r.LogHash, r.GoHash)
}

// A Log is a gzip-compressed log file stored under the SHA1 hash of the
// uncompressed log text.
type Log struct {
	CompressedLog []byte
}

// A Tag is used to keep track of the most recent weekly and release tags.
// Typically there will be one Tag entity for each kind of hg tag.
type Tag struct {
	Kind string // "weekly", "release", or "tip"
	Name string // the tag itself (for example: "release.r60")
	Hash string
}

func (t *Tag) Key(c appengine.Context) *datastore.Key {
	return datastore.NewKey(c, "Tag", t.Kind, 0, nil)
}

// commitHandler records a new commit. It reads a JSON-encoded Commit value
// from the request body and creates a new Commit entity.
// commitHandler also updates the "tip" Tag for each new commit at tip.
//
// This handler is used by a gobuilder process in -commit mode.
func commitHandler(w http.ResponseWriter, r *http.Request)

// tagHandler records a new tag. It reads a JSON-encoded Tag value from the
// request body and updates the Tag entity for the Kind of tag provided.
//
// This handler is used by a gobuilder process in -commit mode.
func tagHandler(w http.ResponseWriter, r *http.Request)

// todoHandler returns a JSON-encoded string of the hash of the next of Commit
// to be built. It expects a "builder" query parameter.
//
// By default it scans the first 20 Go Commits in Num-descending order and
// returns the first one it finds that doesn't have a Result for this builder.
//
// If provided with additional packagePath and goHash query parameters,
// and scans the first 20 Commits in Num-descending order for the specified
// packagePath and returns the first that doesn't have a Result for this builder
// and goHash combination.
func todoHandler(w http.ResponseWriter, r *http.Request)

// resultHandler records a build result.
// It reads a JSON-encoded Result value from the request body,
// creates a new Result entity, and updates the relevant Commit entity.
// If the Log field is not empty, resultHandler creates a new Log entity
// and updates the LogHash field before putting the Commit entity.
func resultHandler(w http.ResponseWriter, r *http.Request)

// AuthHandler wraps a http.HandlerFunc with a handler that validates the
// supplied key and builder query parameters.
func AuthHandler(http.HandlerFunc) http.HandlerFunc

func init() {
	http.HandleFunc("/commit", AuthHandler(commitHandler))
	http.HandleFunc("/result", AuthHandler(commitHandler))
	http.HandleFunc("/tag", AuthHandler(tagHandler))
	http.HandleFunc("/todo", AuthHandler(todoHandler))
}
