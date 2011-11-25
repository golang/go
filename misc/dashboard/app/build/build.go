// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"appengine"
	"appengine/datastore"
	"bytes"
	"compress/gzip"
	"crypto/sha1"
	"fmt"
	"http"
	"io"
	"json"
	"os"
	"strings"
)

const commitsPerPage = 20

// A Package describes a package that is listed on the dashboard.
type Package struct {
	Name    string
	Path    string // (empty for the main Go tree)
	NextNum int    // Num of the next head Commit
}

func (p *Package) Key(c appengine.Context) *datastore.Key {
	key := p.Path
	if key == "" {
		key = "go"
	}
	return datastore.NewKey(c, "Package", key, 0, nil)
}

func GetPackage(c appengine.Context, path string) (*Package, os.Error) {
	p := &Package{Path: path}
	err := datastore.Get(c, p.Key(c), p)
	if err == datastore.ErrNoSuchEntity {
		return nil, fmt.Errorf("package %q not found", path)
	}
	return p, err
}

// A Commit describes an individual commit in a package.
//
// Each Commit entity is a descendant of its associated Package entity.
// In other words, all Commits with the same PackagePath belong to the same
// datastore entity group.
type Commit struct {
	PackagePath string // (empty for Go commits)
	Hash        string
	ParentHash  string
	Num         int // Internal monotonic counter unique to this package.

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
	if com.Hash == "" {
		panic("tried Key on Commit with empty Hash")
	}
	p := Package{Path: com.PackagePath}
	key := com.PackagePath + "|" + com.Hash
	return datastore.NewKey(c, "Commit", key, 0, p.Key(c))
}

func (c *Commit) Valid() os.Error {
	if !validHash(c.Hash) {
		return os.NewError("invalid Hash")
	}
	if !validHash(c.ParentHash) {
		return os.NewError("invalid ParentHash")
	}
	return nil
}

// AddResult adds the denormalized Reuslt data to the Commit's Result field.
// It must be called from inside a datastore transaction.
func (com *Commit) AddResult(c appengine.Context, r *Result) os.Error {
	if err := datastore.Get(c, com.Key(c), com); err != nil {
		return err
	}
	com.Result = append(com.Result, r.Data())
	_, err := datastore.Put(c, com.Key(c), com)
	return err
}

func (com *Commit) HasResult(builder string) bool {
	for _, r := range com.Result {
		if strings.SplitN(r, "|", 2)[0] == builder {
			return true
		}
	}
	return false
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
	Log     []byte `datastore:"-"`        // for JSON unmarshaling
	LogHash string `datastore:",noindex"` // Key to the Log record.
}

func (r *Result) Key(c appengine.Context) *datastore.Key {
	p := Package{Path: r.PackagePath}
	key := r.Builder + "|" + r.PackagePath + "|" + r.Hash + "|" + r.GoHash
	return datastore.NewKey(c, "Result", key, 0, p.Key(c))
}

func (r *Result) Data() string {
	return fmt.Sprintf("%v|%v|%v|%v", r.Builder, r.OK, r.LogHash, r.GoHash)
}

func (r *Result) Valid() os.Error {
	if !validHash(r.Hash) {
		return os.NewError("invalid Hash")
	}
	if r.PackagePath != "" && !validHash(r.GoHash) {
		return os.NewError("invalid GoHash")
	}
	return nil
}

// A Log is a gzip-compressed log file stored under the SHA1 hash of the
// uncompressed log text.
type Log struct {
	CompressedLog []byte
}

func PutLog(c appengine.Context, text []byte) (hash string, err os.Error) {
	h := sha1.New()
	h.Write(text)
	b := new(bytes.Buffer)
	z, _ := gzip.NewWriterLevel(b, gzip.BestCompression)
	z.Write(text)
	z.Close()
	hash = fmt.Sprintf("%x", h.Sum())
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
	p := &Package{Path: ""}
	return datastore.NewKey(c, "Tag", t.Kind, 0, p.Key(c))
}

func (t *Tag) Valid() os.Error {
	if t.Kind != "weekly" || t.Kind != "release" || t.Kind != "tip" {
		return os.NewError("invalid Kind")
	}
	if !validHash(t.Hash) {
		return os.NewError("invalid Hash")
	}
	return nil
}

// commitHandler records a new commit. It reads a JSON-encoded Commit value
// from the request body and creates a new Commit entity.
// commitHandler also updates the "tip" Tag for each new commit at tip.
//
// This handler is used by a gobuilder process in -commit mode.
func commitHandler(w http.ResponseWriter, r *http.Request) {
	com := new(Commit)
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(com); err != nil {
		logErr(w, r, err)
		return
	}
	if err := com.Valid(); err != nil {
		logErr(w, r, err)
		return
	}
	tx := func(c appengine.Context) os.Error {
		return addCommit(c, com)
	}
	c := appengine.NewContext(r)
	if err := datastore.RunInTransaction(c, tx, nil); err != nil {
		logErr(w, r, err)
	}
}

// addCommit adds the Commit entity to the datastore and updates the tip Tag.
// It must be run inside a datastore transaction.
func addCommit(c appengine.Context, com *Commit) os.Error {
	// if this commit is already in the datastore, do nothing
	var tc Commit // temp value so we don't clobber com
	err := datastore.Get(c, com.Key(c), &tc)
	if err != datastore.ErrNoSuchEntity {
		return err
	}
	// get the next commit number
	p, err := GetPackage(c, com.PackagePath)
	if err != nil {
		return err
	}
	com.Num = p.NextNum
	p.NextNum++
	if _, err := datastore.Put(c, p.Key(c), p); err != nil {
		return err
	}
	// if this isn't the first Commit test the parent commit exists
	if com.Num > 0 {
		n, err := datastore.NewQuery("Commit").
			Filter("Hash =", com.ParentHash).
			Ancestor(p.Key(c)).
			Count(c)
		if err != nil {
			return err
		}
		if n == 0 {
			return os.NewError("parent commit not found")
		}
	}
	// update the tip Tag if this is the Go repo
	if p.Path == "" {
		t := &Tag{Kind: "tip", Hash: com.Hash}
		if _, err = datastore.Put(c, t.Key(c), t); err != nil {
			return err
		}
	}
	// put the Commit
	_, err = datastore.Put(c, com.Key(c), com)
	return err
}

// tagHandler records a new tag. It reads a JSON-encoded Tag value from the
// request body and updates the Tag entity for the Kind of tag provided.
//
// This handler is used by a gobuilder process in -commit mode.
func tagHandler(w http.ResponseWriter, r *http.Request) {
	t := new(Tag)
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(t); err != nil {
		logErr(w, r, err)
		return
	}
	if err := t.Valid(); err != nil {
		logErr(w, r, err)
		return
	}
	c := appengine.NewContext(r)
	if _, err := datastore.Put(c, t.Key(c), t); err != nil {
		logErr(w, r, err)
		return
	}
}

// todoHandler returns the string of the hash of the next Commit to be built.
// It expects a "builder" query parameter.
//
// By default it scans the first 20 Go Commits in Num-descending order and
// returns the first one it finds that doesn't have a Result for this builder.
//
// If provided with additional packagePath and goHash query parameters,
// and scans the first 20 Commits in Num-descending order for the specified
// packagePath and returns the first that doesn't have a Result for this builder
// and goHash combination.
func todoHandler(w http.ResponseWriter, r *http.Request) {
	builder := r.FormValue("builder")
	goHash := r.FormValue("goHash")

	c := appengine.NewContext(r)
	p, err := GetPackage(c, r.FormValue("packagePath"))
	if err != nil {
		logErr(w, r, err)
		return
	}

	q := datastore.NewQuery("Commit").
		Ancestor(p.Key(c)).
		Limit(commitsPerPage).
		Order("-Num")
	if goHash != "" && p.Path != "" {
		q.Filter("GoHash =", goHash)
	}
	var nextHash string
	for t := q.Run(c); ; {
		com := new(Commit)
		if _, err := t.Next(com); err == datastore.Done {
			break
		} else if err != nil {
			logErr(w, r, err)
			return
		}
		if !com.HasResult(builder) {
			nextHash = com.Hash
			break
		}
	}
	fmt.Fprint(w, nextHash)
}

// resultHandler records a build result.
// It reads a JSON-encoded Result value from the request body,
// creates a new Result entity, and updates the relevant Commit entity.
// If the Log field is not empty, resultHandler creates a new Log entity
// and updates the LogHash field before putting the Commit entity.
func resultHandler(w http.ResponseWriter, r *http.Request) {
	res := new(Result)
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(res); err != nil {
		logErr(w, r, err)
		return
	}
	if err := res.Valid(); err != nil {
		logErr(w, r, err)
		return
	}
	c := appengine.NewContext(r)
	// store the Log text if supplied
	if len(res.Log) > 0 {
		hash, err := PutLog(c, res.Log)
		if err != nil {
			logErr(w, r, err)
			return
		}
		res.LogHash = hash
	}
	tx := func(c appengine.Context) os.Error {
		// check Package exists
		if _, err := GetPackage(c, res.PackagePath); err != nil {
			return err
		}
		// put Result
		if _, err := datastore.Put(c, res.Key(c), res); err != nil {
			return err
		}
		// add Result to Commit
		com := &Commit{PackagePath: res.PackagePath, Hash: res.Hash}
		return com.AddResult(c, res)
	}
	if err := datastore.RunInTransaction(c, tx, nil); err != nil {
		logErr(w, r, err)
	}
}

func logHandler(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)
	h := r.URL.Path[len("/log/"):]
	k := datastore.NewKey(c, "Log", h, 0, nil)
	l := new(Log)
	if err := datastore.Get(c, k, l); err != nil {
		logErr(w, r, err)
		return
	}
	d, err := gzip.NewReader(bytes.NewBuffer(l.CompressedLog))
	if err != nil {
		logErr(w, r, err)
		return
	}
	if _, err := io.Copy(w, d); err != nil {
		logErr(w, r, err)
	}
}

// AuthHandler wraps a http.HandlerFunc with a handler that validates the
// supplied key and builder query parameters.
func AuthHandler(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Put the URL Query values into r.Form to avoid parsing the
		// request body when calling r.FormValue.
		r.Form = r.URL.Query()

		// Validate key query parameter.
		key := r.FormValue("key")
		if key != secretKey {
			h := sha1.New()
			h.Write([]byte(r.FormValue("builder") + secretKey))
			if key != fmt.Sprintf("%x", h.Sum()) {
				logErr(w, r, os.NewError("invalid key"))
				return
			}
		}

		h(w, r) // Call the original HandlerFunc.
	}
}

func init() {
	// authenticated handlers
	http.HandleFunc("/commit", AuthHandler(commitHandler))
	http.HandleFunc("/result", AuthHandler(resultHandler))
	http.HandleFunc("/tag", AuthHandler(tagHandler))
	http.HandleFunc("/todo", AuthHandler(todoHandler))

	// public handlers
	http.HandleFunc("/log/", logHandler)
}

func validHash(hash string) bool {
	// TODO(adg): correctly validate a hash
	return hash != ""
}

func logErr(w http.ResponseWriter, r *http.Request, err os.Error) {
	appengine.NewContext(r).Errorf("Error: %v", err)
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprint(w, "Error: ", err)
}
