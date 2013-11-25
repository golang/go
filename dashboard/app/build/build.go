// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

import (
	"bytes"
	"compress/gzip"
	"crypto/sha1"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"time"

	"appengine"
	"appengine/datastore"
)

const maxDatastoreStringLen = 500

// A Package describes a package that is listed on the dashboard.
type Package struct {
	Kind    string // "subrepo", "external", or empty for the main Go tree
	Name    string
	Path    string // (empty for the main Go tree)
	NextNum int    // Num of the next head Commit
}

func (p *Package) String() string {
	return fmt.Sprintf("%s: %q", p.Path, p.Name)
}

func (p *Package) Key(c appengine.Context) *datastore.Key {
	key := p.Path
	if key == "" {
		key = "go"
	}
	return datastore.NewKey(c, "Package", key, 0, nil)
}

// LastCommit returns the most recent Commit for this Package.
func (p *Package) LastCommit(c appengine.Context) (*Commit, error) {
	var commits []*Commit
	_, err := datastore.NewQuery("Commit").
		Ancestor(p.Key(c)).
		Order("-Time").
		Limit(1).
		GetAll(c, &commits)
	if _, ok := err.(*datastore.ErrFieldMismatch); ok {
		// Some fields have been removed, so it's okay to ignore this error.
		err = nil
	}
	if err != nil {
		return nil, err
	}
	if len(commits) != 1 {
		return nil, datastore.ErrNoSuchEntity
	}
	return commits[0], nil
}

// GetPackage fetches a Package by path from the datastore.
func GetPackage(c appengine.Context, path string) (*Package, error) {
	p := &Package{Path: path}
	err := datastore.Get(c, p.Key(c), p)
	if err == datastore.ErrNoSuchEntity {
		return nil, fmt.Errorf("package %q not found", path)
	}
	if _, ok := err.(*datastore.ErrFieldMismatch); ok {
		// Some fields have been removed, so it's okay to ignore this error.
		err = nil
	}
	return p, err
}

// A Commit describes an individual commit in a package.
//
// Each Commit entity is a descendant of its associated Package entity.
// In other words, all Commits with the same PackagePath belong to the same
// datastore entity group.
type Commit struct {
	PackagePath string // (empty for main repo commits)
	Hash        string
	ParentHash  string
	Num         int // Internal monotonic counter unique to this package.

	User string
	Desc string `datastore:",noindex"`
	Time time.Time

	// ResultData is the Data string of each build Result for this Commit.
	// For non-Go commits, only the Results for the current Go tip, weekly,
	// and release Tags are stored here. This is purely de-normalized data.
	// The complete data set is stored in Result entities.
	ResultData []string `datastore:",noindex"`

	FailNotificationSent bool
}

func (com *Commit) Key(c appengine.Context) *datastore.Key {
	if com.Hash == "" {
		panic("tried Key on Commit with empty Hash")
	}
	p := Package{Path: com.PackagePath}
	key := com.PackagePath + "|" + com.Hash
	return datastore.NewKey(c, "Commit", key, 0, p.Key(c))
}

func (c *Commit) Valid() error {
	if !validHash(c.Hash) {
		return errors.New("invalid Hash")
	}
	if c.ParentHash != "" && !validHash(c.ParentHash) { // empty is OK
		return errors.New("invalid ParentHash")
	}
	return nil
}

// each result line is approx 105 bytes. This constant is a tradeoff between
// build history and the AppEngine datastore limit of 1mb.
const maxResults = 1000

// AddResult adds the denormalized Result data to the Commit's Result field.
// It must be called from inside a datastore transaction.
func (com *Commit) AddResult(c appengine.Context, r *Result) error {
	if err := datastore.Get(c, com.Key(c), com); err != nil {
		return fmt.Errorf("getting Commit: %v", err)
	}
	com.ResultData = trim(append(com.ResultData, r.Data()), maxResults)
	if _, err := datastore.Put(c, com.Key(c), com); err != nil {
		return fmt.Errorf("putting Commit: %v", err)
	}
	return nil
}

func trim(s []string, n int) []string {
	l := min(len(s), n)
	return s[len(s)-l:]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Result returns the build Result for this Commit for the given builder/goHash.
func (c *Commit) Result(builder, goHash string) *Result {
	for _, r := range c.ResultData {
		p := strings.SplitN(r, "|", 4)
		if len(p) != 4 || p[0] != builder || p[3] != goHash {
			continue
		}
		return partsToHash(c, p)
	}
	return nil
}

// Results returns the build Results for this Commit for the given goHash.
func (c *Commit) Results(goHash string) (results []*Result) {
	for _, r := range c.ResultData {
		p := strings.SplitN(r, "|", 4)
		if len(p) != 4 || p[3] != goHash {
			continue
		}
		results = append(results, partsToHash(c, p))
	}
	return
}

// partsToHash converts a Commit and ResultData substrings to a Result.
func partsToHash(c *Commit, p []string) *Result {
	return &Result{
		Builder:     p[0],
		Hash:        c.Hash,
		PackagePath: c.PackagePath,
		GoHash:      p[3],
		OK:          p[1] == "true",
		LogHash:     p[2],
	}
}

// A Result describes a build result for a Commit on an OS/architecture.
//
// Each Result entity is a descendant of its associated Package entity.
type Result struct {
	Builder     string // "os-arch[-note]"
	Hash        string
	PackagePath string // (empty for Go commits)

	// The Go Commit this was built against (empty for Go commits).
	GoHash string

	OK      bool
	Log     string `datastore:"-"`        // for JSON unmarshaling only
	LogHash string `datastore:",noindex"` // Key to the Log record.

	RunTime int64 // time to build+test in nanoseconds
}

func (r *Result) Key(c appengine.Context) *datastore.Key {
	p := Package{Path: r.PackagePath}
	key := r.Builder + "|" + r.PackagePath + "|" + r.Hash + "|" + r.GoHash
	return datastore.NewKey(c, "Result", key, 0, p.Key(c))
}

func (r *Result) Valid() error {
	if !validHash(r.Hash) {
		return errors.New("invalid Hash")
	}
	if r.PackagePath != "" && !validHash(r.GoHash) {
		return errors.New("invalid GoHash")
	}
	return nil
}

// Data returns the Result in string format
// to be stored in Commit's ResultData field.
func (r *Result) Data() string {
	return fmt.Sprintf("%v|%v|%v|%v", r.Builder, r.OK, r.LogHash, r.GoHash)
}

// A Log is a gzip-compressed log file stored under the SHA1 hash of the
// uncompressed log text.
type Log struct {
	CompressedLog []byte
}

func (l *Log) Text() ([]byte, error) {
	d, err := gzip.NewReader(bytes.NewBuffer(l.CompressedLog))
	if err != nil {
		return nil, fmt.Errorf("reading log data: %v", err)
	}
	b, err := ioutil.ReadAll(d)
	if err != nil {
		return nil, fmt.Errorf("reading log data: %v", err)
	}
	return b, nil
}

func PutLog(c appengine.Context, text string) (hash string, err error) {
	h := sha1.New()
	io.WriteString(h, text)
	b := new(bytes.Buffer)
	z, _ := gzip.NewWriterLevel(b, gzip.BestCompression)
	io.WriteString(z, text)
	z.Close()
	hash = fmt.Sprintf("%x", h.Sum(nil))
	key := datastore.NewKey(c, "Log", hash, 0, nil)
	_, err = datastore.Put(c, key, &Log{b.Bytes()})
	return
}

// A Tag is used to keep track of the most recent Go weekly and release tags.
// Typically there will be one Tag entity for each kind of hg tag.
type Tag struct {
	Kind string // "weekly", "release", or "tip"
	Name string // the tag itself (for example: "release.r60")
	Hash string
}

func (t *Tag) Key(c appengine.Context) *datastore.Key {
	p := &Package{}
	return datastore.NewKey(c, "Tag", t.Kind, 0, p.Key(c))
}

func (t *Tag) Valid() error {
	if t.Kind != "weekly" && t.Kind != "release" && t.Kind != "tip" {
		return errors.New("invalid Kind")
	}
	if !validHash(t.Hash) {
		return errors.New("invalid Hash")
	}
	return nil
}

// Commit returns the Commit that corresponds with this Tag.
func (t *Tag) Commit(c appengine.Context) (*Commit, error) {
	com := &Commit{Hash: t.Hash}
	err := datastore.Get(c, com.Key(c), com)
	return com, err
}

// GetTag fetches a Tag by name from the datastore.
func GetTag(c appengine.Context, tag string) (*Tag, error) {
	t := &Tag{Kind: tag}
	if err := datastore.Get(c, t.Key(c), t); err != nil {
		if err == datastore.ErrNoSuchEntity {
			return nil, errors.New("tag not found: " + tag)
		}
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, err
	}
	return t, nil
}

// Packages returns packages of the specified kind.
// Kind must be one of "external" or "subrepo".
func Packages(c appengine.Context, kind string) ([]*Package, error) {
	switch kind {
	case "external", "subrepo":
	default:
		return nil, errors.New(`kind must be one of "external" or "subrepo"`)
	}
	var pkgs []*Package
	q := datastore.NewQuery("Package").Filter("Kind=", kind)
	for t := q.Run(c); ; {
		pkg := new(Package)
		_, err := t.Next(pkg)
		if _, ok := err.(*datastore.ErrFieldMismatch); ok {
			// Some fields have been removed, so it's okay to ignore this error.
			err = nil
		}
		if err == datastore.Done {
			break
		} else if err != nil {
			return nil, err
		}
		if pkg.Path != "" {
			pkgs = append(pkgs, pkg)
		}
	}
	return pkgs, nil
}
