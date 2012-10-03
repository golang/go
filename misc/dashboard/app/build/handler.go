// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"crypto/hmac"
	"crypto/md5"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"appengine"
	"appengine/datastore"
	"cache"
)

const commitsPerPage = 30

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
func commitHandler(r *http.Request) (interface{}, error) {
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
	defer cache.Tick(c)
	tx := func(c appengine.Context) error {
		return addCommit(c, com)
	}
	return nil, datastore.RunInTransaction(c, tx, nil)
}

// addCommit adds the Commit entity to the datastore and updates the tip Tag.
// It must be run inside a datastore transaction.
func addCommit(c appengine.Context, com *Commit) error {
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
			return errors.New("parent commit not found")
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
func tagHandler(r *http.Request) (interface{}, error) {
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
	defer cache.Tick(c)
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
func todoHandler(r *http.Request) (interface{}, error) {
	c := appengine.NewContext(r)
	now := cache.Now(c)
	key := "build-todo-" + r.Form.Encode()
	var todo *Todo
	if cache.Get(r, now, key, &todo) {
		return todo, nil
	}
	var err error
	builder := r.FormValue("builder")
	for _, kind := range r.Form["kind"] {
		var data interface{}
		switch kind {
		case "build-go-commit":
			data, err = buildTodo(c, builder, "", "")
		case "build-package":
			packagePath := r.FormValue("packagePath")
			goHash := r.FormValue("goHash")
			data, err = buildTodo(c, builder, packagePath, goHash)
		}
		if data != nil || err != nil {
			todo = &Todo{Kind: kind, Data: data}
			break
		}
	}
	if err == nil {
		cache.Set(r, now, key, todo)
	}
	return todo, err
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
func buildTodo(c appengine.Context, builder, packagePath, goHash string) (interface{}, error) {
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
		if _, err := t.Next(com); err == datastore.Done {
			break
		} else if err != nil {
			return nil, err
		}
		if com.Result(builder, goHash) == nil {
			return com, nil
		}
	}

	// Nothing left to do if this is a package (not the Go tree).
	if packagePath != "" {
		return nil, nil
	}

	// If there are no Go tree commits left to build,
	// see if there are any subrepo commits that need to be built at tip.
	// If so, ask the builder to build a go tree at the tip commit.
	// TODO(adg): do the same for "weekly" and "release" tags.

	tag, err := GetTag(c, "tip")
	if err != nil {
		return nil, err
	}

	// Check that this Go commit builds OK for this builder.
	// If not, don't re-build as the subrepos will never get built anyway.
	com, err := tag.Commit(c)
	if err != nil {
		return nil, err
	}
	if r := com.Result(builder, ""); r != nil && !r.OK {
		return nil, nil
	}

	pkgs, err := Packages(c, "subrepo")
	if err != nil {
		return nil, err
	}
	for _, pkg := range pkgs {
		com, err := pkg.LastCommit(c)
		if err != nil {
			c.Warningf("%v: no Commit found: %v", pkg, err)
			continue
		}
		if com.Result(builder, tag.Hash) == nil {
			return tag.Commit(c)
		}
	}

	return nil, nil
}

// packagesHandler returns a list of the non-Go Packages monitored
// by the dashboard.
func packagesHandler(r *http.Request) (interface{}, error) {
	kind := r.FormValue("kind")
	c := appengine.NewContext(r)
	now := cache.Now(c)
	key := "build-packages-" + kind
	var p []*Package
	if cache.Get(r, now, key, &p) {
		return p, nil
	}
	p, err := Packages(c, kind)
	if err != nil {
		return nil, err
	}
	cache.Set(r, now, key, p)
	return p, nil
}

// resultHandler records a build result.
// It reads a JSON-encoded Result value from the request body,
// creates a new Result entity, and updates the relevant Commit entity.
// If the Log field is not empty, resultHandler creates a new Log entity
// and updates the LogHash field before putting the Commit entity.
func resultHandler(r *http.Request) (interface{}, error) {
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
	defer cache.Tick(c)
	// store the Log text if supplied
	if len(res.Log) > 0 {
		hash, err := PutLog(c, res.Log)
		if err != nil {
			return nil, fmt.Errorf("putting Log: %v", err)
		}
		res.LogHash = hash
	}
	tx := func(c appengine.Context) error {
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
		// Send build failure notifications, if necessary.
		// Note this must run after the call AddResult, which
		// populates the Commit's ResultData field.
		return notifyOnFailure(c, com, res.Builder)
	}
	return nil, datastore.RunInTransaction(c, tx, nil)
}

// logHandler displays log text for a given hash.
// It handles paths like "/log/hash".
func logHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-type", "text/plain; charset=utf-8")
	c := appengine.NewContext(r)
	hash := r.URL.Path[len("/log/"):]
	key := datastore.NewKey(c, "Log", hash, 0, nil)
	l := new(Log)
	if err := datastore.Get(c, key, l); err != nil {
		logErr(w, r, err)
		return
	}
	b, err := l.Text()
	if err != nil {
		logErr(w, r, err)
		return
	}
	w.Write(b)
}

type dashHandler func(*http.Request) (interface{}, error)

type dashResponse struct {
	Response interface{}
	Error    string
}

// errBadMethod is returned by a dashHandler when
// the request has an unsuitable method.
type errBadMethod string

func (e errBadMethod) Error() string {
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

		var err error
		var resp interface{}

		// Validate key query parameter for POST requests only.
		key := r.FormValue("key")
		builder := r.FormValue("builder")
		if r.Method == "POST" && !validKey(c, key, builder) {
			err = errors.New("invalid key: " + key)
		}

		// Call the original HandlerFunc and return the response.
		if err == nil {
			resp, err = h(r)
		}

		// Write JSON response.
		dashResp := &dashResponse{Response: resp}
		if err != nil {
			c.Errorf("%v", err)
			dashResp.Error = err.Error()
		}
		w.Header().Set("Content-Type", "application/json")
		if err = json.NewEncoder(w).Encode(dashResp); err != nil {
			c.Criticalf("encoding response: %v", err)
		}
	}
}

func keyHandler(w http.ResponseWriter, r *http.Request) {
	builder := r.FormValue("builder")
	if builder == "" {
		logErr(w, r, errors.New("must supply builder in query string"))
		return
	}
	c := appengine.NewContext(r)
	fmt.Fprint(w, builderKey(c, builder))
}

func init() {
	// admin handlers
	http.HandleFunc("/init", initHandler)
	http.HandleFunc("/key", keyHandler)

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

func validKey(c appengine.Context, key, builder string) bool {
	if appengine.IsDevAppServer() {
		return true
	}
	if key == secretKey(c) {
		return true
	}
	return key == builderKey(c, builder)
}

func builderKey(c appengine.Context, builder string) string {
	h := hmac.New(md5.New, []byte(secretKey(c)))
	h.Write([]byte(builder))
	return fmt.Sprintf("%x", h.Sum(nil))
}

func logErr(w http.ResponseWriter, r *http.Request, err error) {
	appengine.NewContext(r).Errorf("Error: %v", err)
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprint(w, "Error: ", err)
}
