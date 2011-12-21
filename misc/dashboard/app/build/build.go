// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"appengine"
	"appengine/datastore"
	"bytes"
	"compress/gzip"
	"crypto/hmac"
	"crypto/sha1"
	"fmt"
	"http"
	"io"
	"json"
	"os"
	"strings"
)

var defaultPackages = []*Package{
	&Package{Name: "Go"},
}

const (
	commitsPerPage        = 30
	maxDatastoreStringLen = 500
)

// A Package describes a package that is listed on the dashboard.
type Package struct {
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
func (p *Package) LastCommit(c appengine.Context) (*Commit, os.Error) {
	var commits []*Commit
	_, err := datastore.NewQuery("Commit").
		Ancestor(p.Key(c)).
		Order("-Time").
		Limit(1).
		GetAll(c, &commits)
	if err != nil {
		return nil, err
	}
	if len(commits) != 1 {
		return nil, datastore.ErrNoSuchEntity
	}
	return commits[0], nil
}

// GetPackage fetches a Package by path from the datastore.
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

	// ResultData is the Data string of each build Result for this Commit.
	// For non-Go commits, only the Results for the current Go tip, weekly,
	// and release Tags are stored here. This is purely de-normalized data.
	// The complete data set is stored in Result entities.
	ResultData []string `datastore:",noindex"`
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
	if c.ParentHash != "" && !validHash(c.ParentHash) { // empty is OK
		return os.NewError("invalid ParentHash")
	}
	return nil
}

// AddResult adds the denormalized Reuslt data to the Commit's Result field.
// It must be called from inside a datastore transaction.
func (com *Commit) AddResult(c appengine.Context, r *Result) os.Error {
	if err := datastore.Get(c, com.Key(c), com); err != nil {
		return fmt.Errorf("getting Commit: %v", err)
	}
	com.ResultData = append(com.ResultData, r.Data())
	if _, err := datastore.Put(c, com.Key(c), com); err != nil {
		return fmt.Errorf("putting Commit: %v", err)
	}
	return nil
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

func (r *Result) Valid() os.Error {
	if !validHash(r.Hash) {
		return os.NewError("invalid Hash")
	}
	if r.PackagePath != "" && !validHash(r.GoHash) {
		return os.NewError("invalid GoHash")
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
	p := &Package{}
	return datastore.NewKey(c, "Tag", t.Kind, 0, p.Key(c))
}

func (t *Tag) Valid() os.Error {
	if t.Kind != "weekly" && t.Kind != "release" && t.Kind != "tip" {
		return os.NewError("invalid Kind")
	}
	if !validHash(t.Hash) {
		return os.NewError("invalid Hash")
	}
	return nil
}

// GetTag fetches a Tag by name from the datastore.
func GetTag(c appengine.Context, tag string) (*Tag, os.Error) {
	t := &Tag{Kind: tag}
	if err := datastore.Get(c, t.Key(c), t); err != nil {
		if err == datastore.ErrNoSuchEntity {
			return nil, os.NewError("tag not found: " + tag)
		}
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, err
	}
	return t, nil
}

// commitHandler retrieves commit data or records a new commit.
//
// For GET requests it returns a Commit value for the specified
// packagePath and hash.
//
// For POST requests it reads a JSON-encoded Commit value from the request
// body and creates a new Commit entity. It also updates the "tip" Tag for
// each new commit at tip.
//
// This handler is used by a gobuilder process in -commit mode.
func commitHandler(r *http.Request) (interface{}, os.Error) {
	c := appengine.NewContext(r)
	com := new(Commit)

	if r.Method == "GET" {
		com.PackagePath = r.FormValue("packagePath")
		com.Hash = r.FormValue("hash")
		if err := datastore.Get(c, com.Key(c), com); err != nil {
			return nil, fmt.Errorf("getting Commit: %v", err)
		}
		return com, nil
	}
	if r.Method != "POST" {
		return nil, errBadMethod(r.Method)
	}

	// POST request
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(com); err != nil {
		return nil, fmt.Errorf("decoding Body: %v", err)
	}
	if len(com.Desc) > maxDatastoreStringLen {
		com.Desc = com.Desc[:maxDatastoreStringLen]
	}
	if err := com.Valid(); err != nil {
		return nil, fmt.Errorf("validating Commit: %v", err)
	}
	tx := func(c appengine.Context) os.Error {
		return addCommit(c, com)
	}
	return nil, datastore.RunInTransaction(c, tx, nil)
}

// addCommit adds the Commit entity to the datastore and updates the tip Tag.
// It must be run inside a datastore transaction.
func addCommit(c appengine.Context, com *Commit) os.Error {
	var tc Commit // temp value so we don't clobber com
	err := datastore.Get(c, com.Key(c), &tc)
	if err != datastore.ErrNoSuchEntity {
		// if this commit is already in the datastore, do nothing
		if err == nil {
			return nil
		}
		return fmt.Errorf("getting Commit: %v", err)
	}
	// get the next commit number
	p, err := GetPackage(c, com.PackagePath)
	if err != nil {
		return fmt.Errorf("GetPackage: %v", err)
	}
	com.Num = p.NextNum
	p.NextNum++
	if _, err := datastore.Put(c, p.Key(c), p); err != nil {
		return fmt.Errorf("putting Package: %v", err)
	}
	// if this isn't the first Commit test the parent commit exists
	if com.Num > 0 {
		n, err := datastore.NewQuery("Commit").
			Filter("Hash =", com.ParentHash).
			Ancestor(p.Key(c)).
			Count(c)
		if err != nil {
			return fmt.Errorf("testing for parent Commit: %v", err)
		}
		if n == 0 {
			return os.NewError("parent commit not found")
		}
	}
	// update the tip Tag if this is the Go repo
	if p.Path == "" {
		t := &Tag{Kind: "tip", Hash: com.Hash}
		if _, err = datastore.Put(c, t.Key(c), t); err != nil {
			return fmt.Errorf("putting Tag: %v", err)
		}
	}
	// put the Commit
	if _, err = datastore.Put(c, com.Key(c), com); err != nil {
		return fmt.Errorf("putting Commit: %v", err)
	}
	return nil
}

// tagHandler records a new tag. It reads a JSON-encoded Tag value from the
// request body and updates the Tag entity for the Kind of tag provided.
//
// This handler is used by a gobuilder process in -commit mode.
func tagHandler(r *http.Request) (interface{}, os.Error) {
	if r.Method != "POST" {
		return nil, errBadMethod(r.Method)
	}

	t := new(Tag)
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(t); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, err
	}
	c := appengine.NewContext(r)
	_, err := datastore.Put(c, t.Key(c), t)
	return nil, err
}

// Todo is a todoHandler response.
type Todo struct {
	Kind string // "build-go-commit" or "build-package"
	Data interface{}
}

// todoHandler returns the next action to be performed by a builder.
// It expects "builder" and "kind" query parameters and returns a *Todo value.
// Multiple "kind" parameters may be specified.
func todoHandler(r *http.Request) (todo interface{}, err os.Error) {
	c := appengine.NewContext(r)
	builder := r.FormValue("builder")
	for _, kind := range r.Form["kind"] {
		var data interface{}
		switch kind {
		case "build-go-commit":
			data, err = buildTodo(c, builder, "", "")
		case "build-package":
			data, err = buildTodo(
				c, builder,
				r.FormValue("packagePath"),
				r.FormValue("goHash"),
			)
		}
		if data != nil || err != nil {
			return &Todo{Kind: kind, Data: data}, err
		}
	}
	return nil, nil
}

// buildTodo returns the next Commit to be built (or nil if none available).
//
// If packagePath and goHash are empty, it scans the first 20 Go Commits in
// Num-descending order and returns the first one it finds that doesn't have a
// Result for this builder.
//
// If provided with non-empty packagePath and goHash args, it scans the first
// 20 Commits in Num-descending order for the specified packagePath and
// returns the first that doesn't have a Result for this builder and goHash.
func buildTodo(c appengine.Context, builder, packagePath, goHash string) (interface{}, os.Error) {
	p, err := GetPackage(c, packagePath)
	if err != nil {
		return nil, err
	}

	t := datastore.NewQuery("Commit").
		Ancestor(p.Key(c)).
		Limit(commitsPerPage).
		Order("-Num").
		Run(c)
	for {
		com := new(Commit)
		if _, err := t.Next(com); err != nil {
			if err == datastore.Done {
				err = nil
			}
			return nil, err
		}
		if com.Result(builder, goHash) == nil {
			return com, nil
		}
	}
	panic("unreachable")
}

// packagesHandler returns a list of the non-Go Packages monitored
// by the dashboard.
func packagesHandler(r *http.Request) (interface{}, os.Error) {
	return Packages(appengine.NewContext(r))
}

// Packages returns all non-Go packages.
func Packages(c appengine.Context) ([]*Package, os.Error) {
	var pkgs []*Package
	for t := datastore.NewQuery("Package").Run(c); ; {
		pkg := new(Package)
		if _, err := t.Next(pkg); err == datastore.Done {
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

// resultHandler records a build result.
// It reads a JSON-encoded Result value from the request body,
// creates a new Result entity, and updates the relevant Commit entity.
// If the Log field is not empty, resultHandler creates a new Log entity
// and updates the LogHash field before putting the Commit entity.
func resultHandler(r *http.Request) (interface{}, os.Error) {
	if r.Method != "POST" {
		return nil, errBadMethod(r.Method)
	}

	c := appengine.NewContext(r)
	res := new(Result)
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(res); err != nil {
		return nil, fmt.Errorf("decoding Body: %v", err)
	}
	if err := res.Valid(); err != nil {
		return nil, fmt.Errorf("validating Result: %v", err)
	}
	// store the Log text if supplied
	if len(res.Log) > 0 {
		hash, err := PutLog(c, res.Log)
		if err != nil {
			return nil, fmt.Errorf("putting Log: %v", err)
		}
		res.LogHash = hash
	}
	tx := func(c appengine.Context) os.Error {
		// check Package exists
		if _, err := GetPackage(c, res.PackagePath); err != nil {
			return fmt.Errorf("GetPackage: %v", err)
		}
		// put Result
		if _, err := datastore.Put(c, res.Key(c), res); err != nil {
			return fmt.Errorf("putting Result: %v", err)
		}
		// add Result to Commit
		com := &Commit{PackagePath: res.PackagePath, Hash: res.Hash}
		if err := com.AddResult(c, res); err != nil {
			return fmt.Errorf("AddResult: %v", err)
		}
		return nil
	}
	return nil, datastore.RunInTransaction(c, tx, nil)
}

// logHandler displays log text for a given hash.
// It handles paths like "/log/hash".
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

type dashHandler func(*http.Request) (interface{}, os.Error)

type dashResponse struct {
	Response interface{}
	Error    string
}

// errBadMethod is returned by a dashHandler when
// the request has an unsuitable method.
type errBadMethod string

func (e errBadMethod) String() string {
	return "bad method: " + string(e)
}

// AuthHandler wraps a http.HandlerFunc with a handler that validates the
// supplied key and builder query parameters.
func AuthHandler(h dashHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		c := appengine.NewContext(r)

		// Put the URL Query values into r.Form to avoid parsing the
		// request body when calling r.FormValue.
		r.Form = r.URL.Query()

		var err os.Error
		var resp interface{}

		// Validate key query parameter for POST requests only.
		key := r.FormValue("key")
		if r.Method == "POST" && key != secretKey && !appengine.IsDevAppServer() {
			h := hmac.NewMD5([]byte(secretKey))
			h.Write([]byte(r.FormValue("builder")))
			if key != fmt.Sprintf("%x", h.Sum()) {
				err = os.NewError("invalid key: " + key)
			}
		}

		// Call the original HandlerFunc and return the response.
		if err == nil {
			resp, err = h(r)
		}

		// Write JSON response.
		dashResp := &dashResponse{Response: resp}
		if err != nil {
			c.Errorf("%v", err)
			dashResp.Error = err.String()
		}
		w.Header().Set("Content-Type", "application/json")
		if err = json.NewEncoder(w).Encode(dashResp); err != nil {
			c.Criticalf("encoding response: %v", err)
		}
	}
}

func initHandler(w http.ResponseWriter, r *http.Request) {
	// TODO(adg): devise a better way of bootstrapping new packages
	c := appengine.NewContext(r)
	for _, p := range defaultPackages {
		if err := datastore.Get(c, p.Key(c), new(Package)); err == nil {
			continue
		} else if err != datastore.ErrNoSuchEntity {
			logErr(w, r, err)
			return
		}
		if _, err := datastore.Put(c, p.Key(c), p); err != nil {
			logErr(w, r, err)
			return
		}
	}
	fmt.Fprint(w, "OK")
}

func init() {
	// admin handlers
	http.HandleFunc("/init", initHandler)

	// authenticated handlers
	http.HandleFunc("/commit", AuthHandler(commitHandler))
	http.HandleFunc("/packages", AuthHandler(packagesHandler))
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
