// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build golangorg

package dl

import (
	"context"
	"crypto/hmac"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"cloud.google.com/go/datastore"
	"golang.org/x/tools/godoc/env"
	"golang.org/x/tools/internal/memcache"
)

type server struct {
	datastore *datastore.Client
	memcache  *memcache.CodecClient
}

func RegisterHandlers(mux *http.ServeMux, dc *datastore.Client, mc *memcache.Client) {
	s := server{dc, mc.WithCodec(memcache.Gob)}
	mux.HandleFunc("/dl", s.getHandler)
	mux.HandleFunc("/dl/", s.getHandler) // also serves listHandler
	mux.HandleFunc("/dl/upload", s.uploadHandler)

	// NOTE(cbro): this only needs to be run once per project,
	// and should be behind an admin login.
	// TODO(cbro): move into a locally-run program? or remove?
	// mux.HandleFunc("/dl/init", initHandler)
}

// rootKey is the ancestor of all File entities.
var rootKey = datastore.NameKey("FileRoot", "root", nil)

func (h server) listHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	ctx := r.Context()
	var d listTemplateData

	if err := h.memcache.Get(ctx, cacheKey, &d); err != nil {
		if err != memcache.ErrCacheMiss {
			log.Printf("ERROR cache get error: %v", err)
			// NOTE(cbro): continue to hit datastore if the memcache is down.
		}

		var fs []File
		q := datastore.NewQuery("File").Ancestor(rootKey)
		if _, err := h.datastore.GetAll(ctx, q, &fs); err != nil {
			log.Printf("ERROR error listing: %v", err)
			http.Error(w, "Could not get download page. Try again in a few minutes.", 500)
			return
		}
		d.Stable, d.Unstable, d.Archive = filesToReleases(fs)
		if len(d.Stable) > 0 {
			d.Featured = filesToFeatured(d.Stable[0].Files)
		}

		item := &memcache.Item{Key: cacheKey, Object: &d, Expiration: cacheDuration}
		if err := h.memcache.Set(ctx, item); err != nil {
			log.Printf("ERROR cache set error: %v", err)
		}
	}

	if r.URL.Query().Get("mode") == "json" {
		w.Header().Set("Content-Type", "application/json")
		enc := json.NewEncoder(w)
		enc.SetIndent("", " ")
		if err := enc.Encode(d.Stable); err != nil {
			log.Printf("ERROR rendering JSON for releases: %v", err)
		}
		return
	}

	if err := listTemplate.ExecuteTemplate(w, "root", d); err != nil {
		log.Printf("ERROR executing template: %v", err)
	}
}

func (h server) uploadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	ctx := r.Context()

	// Authenticate using a user token (same as gomote).
	user := r.FormValue("user")
	if !validUser(user) {
		http.Error(w, "bad user", http.StatusForbidden)
		return
	}
	if r.FormValue("key") != h.userKey(ctx, user) {
		http.Error(w, "bad key", http.StatusForbidden)
		return
	}

	var f File
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(&f); err != nil {
		log.Printf("ERROR decoding upload JSON: %v", err)
		http.Error(w, "Something broke", http.StatusInternalServerError)
		return
	}
	if f.Filename == "" {
		http.Error(w, "Must provide Filename", http.StatusBadRequest)
		return
	}
	if f.Uploaded.IsZero() {
		f.Uploaded = time.Now()
	}
	k := datastore.NameKey("File", f.Filename, rootKey)
	if _, err := h.datastore.Put(ctx, k, &f); err != nil {
		log.Printf("ERROR File entity: %v", err)
		http.Error(w, "could not put File entity", http.StatusInternalServerError)
		return
	}
	if err := h.memcache.Delete(ctx, cacheKey); err != nil {
		log.Printf("ERROR delete error: %v", err)
	}
	io.WriteString(w, "OK")
}

func (h server) getHandler(w http.ResponseWriter, r *http.Request) {
	// For go get golang.org/dl/go1.x.y, we need to serve the
	// same meta tags at /dl for cmd/go to validate against /dl/go1.x.y:
	if r.URL.Path == "/dl" && (r.Method == "GET" || r.Method == "HEAD") && r.FormValue("go-get") == "1" {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprintf(w, `<!DOCTYPE html><html><head>
<meta name="go-import" content="golang.org/dl git https://go.googlesource.com/dl">
</head></html>`)
		return
	}
	if r.URL.Path == "/dl" {
		http.Redirect(w, r, "/dl/", http.StatusFound)
		return
	}

	name := strings.TrimPrefix(r.URL.Path, "/dl/")
	if name == "" {
		h.listHandler(w, r)
		return
	}
	if fileRe.MatchString(name) {
		http.Redirect(w, r, downloadBaseURL+name, http.StatusFound)
		return
	}
	if goGetRe.MatchString(name) {
		var isGoGet bool
		if r.Method == "GET" || r.Method == "HEAD" {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			isGoGet = r.FormValue("go-get") == "1"
		}
		if !isGoGet {
			w.Header().Set("Location", "https://golang.org/dl/#"+name)
			w.WriteHeader(http.StatusFound)
		}
		fmt.Fprintf(w, `<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
<meta name="go-import" content="golang.org/dl git https://go.googlesource.com/dl">
<meta http-equiv="refresh" content="0; url=https://golang.org/dl/#%s">
</head>
<body>
Nothing to see here; <a href="https://golang.org/dl/#%s">move along</a>.
</body>
</html>
`, html.EscapeString(name), html.EscapeString(name))
		return
	}
	http.NotFound(w, r)
}

func (h server) initHandler(w http.ResponseWriter, r *http.Request) {
	var fileRoot struct {
		Root string
	}
	ctx := r.Context()
	k := rootKey
	_, err := h.datastore.RunInTransaction(ctx, func(tx *datastore.Transaction) error {
		err := tx.Get(k, &fileRoot)
		if err != nil && err != datastore.ErrNoSuchEntity {
			return err
		}
		_, err = tx.Put(k, &fileRoot)
		return err
	}, nil)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	io.WriteString(w, "OK")
}

func (h server) userKey(c context.Context, user string) string {
	hash := hmac.New(md5.New, []byte(h.secret(c)))
	hash.Write([]byte("user-" + user))
	return fmt.Sprintf("%x", hash.Sum(nil))
}

// Code below copied from x/build/app/key

var theKey struct {
	sync.RWMutex
	builderKey
}

type builderKey struct {
	Secret string
}

func (k *builderKey) Key() *datastore.Key {
	return datastore.NameKey("BuilderKey", "root", nil)
}

func (h server) secret(ctx context.Context) string {
	// check with rlock
	theKey.RLock()
	k := theKey.Secret
	theKey.RUnlock()
	if k != "" {
		return k
	}

	// prepare to fill; check with lock and keep lock
	theKey.Lock()
	defer theKey.Unlock()
	if theKey.Secret != "" {
		return theKey.Secret
	}

	// fill
	if err := h.datastore.Get(ctx, theKey.Key(), &theKey.builderKey); err != nil {
		if err == datastore.ErrNoSuchEntity {
			// If the key is not stored in datastore, write it.
			// This only happens at the beginning of a new deployment.
			// The code is left here for SDK use and in case a fresh
			// deployment is ever needed.  "gophers rule" is not the
			// real key.
			if env.IsProd() {
				panic("lost key from datastore")
			}
			theKey.Secret = "gophers rule"
			h.datastore.Put(ctx, theKey.Key(), &theKey.builderKey)
			return theKey.Secret
		}
		panic("cannot load builder key: " + err.Error())
	}

	return theKey.Secret
}
