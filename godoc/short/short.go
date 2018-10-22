// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build golangorg

// Package short implements a simple URL shortener, serving an administrative
// interface at /s and shortened urls from /s/key.
// It is designed to run only on the instance of godoc that serves golang.org.
package short

// TODO(adg): collect statistics on URL visits

import (
	"context"
	"errors"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"net/url"
	"regexp"

	"cloud.google.com/go/datastore"
	"golang.org/x/tools/internal/memcache"
	"google.golang.org/appengine/user"
)

const (
	prefix  = "/s"
	kind    = "Link"
	baseURL = "https://golang.org" + prefix
)

// Link represents a short link.
type Link struct {
	Key, Target string
}

var validKey = regexp.MustCompile(`^[a-zA-Z0-9-_.]+$`)

type server struct {
	datastore *datastore.Client
	memcache  *memcache.CodecClient
}

func RegisterHandlers(mux *http.ServeMux, dc *datastore.Client, mc *memcache.Client) {
	s := server{dc, mc.WithCodec(memcache.JSON)}
	mux.HandleFunc(prefix+"/", s.linkHandler)

	// TODO(cbro): move storage of the links to a text file in Gerrit.
	// Disable the admin handler until that happens, since GAE Flex doesn't support
	// the "google.golang.org/appengine/user" package.
	// See golang.org/issue/27205#issuecomment-418673218
	// mux.HandleFunc(prefix, adminHandler)
	mux.HandleFunc(prefix, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		io.WriteString(w, "Link creation temporarily unavailable. See golang.org/issue/27205.")
	})
}

// linkHandler services requests to short URLs.
//   http://golang.org/s/key
// It consults memcache and datastore for the Link for key.
// It then sends a redirects or an error message.
func (h server) linkHandler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	key := r.URL.Path[len(prefix)+1:]
	if !validKey.MatchString(key) {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	var link Link
	if err := h.memcache.Get(ctx, cacheKey(key), &link); err != nil {
		k := datastore.NameKey(kind, key, nil)
		err = h.datastore.Get(ctx, k, &link)
		switch err {
		case datastore.ErrNoSuchEntity:
			http.Error(w, "not found", http.StatusNotFound)
			return
		default: // != nil
			log.Printf("ERROR %q: %v", key, err)
			http.Error(w, "internal server error", http.StatusInternalServerError)
			return
		case nil:
			item := &memcache.Item{
				Key:    cacheKey(key),
				Object: &link,
			}
			if err := h.memcache.Set(ctx, item); err != nil {
				log.Printf("WARNING %q: %v", key, err)
			}
		}
	}

	http.Redirect(w, r, link.Target, http.StatusFound)
}

var adminTemplate = template.Must(template.New("admin").Parse(templateHTML))

// adminHandler serves an administrative interface.
func (h server) adminHandler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	if !user.IsAdmin(ctx) {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}

	var newLink *Link
	var doErr error
	if r.Method == "POST" {
		key := r.FormValue("key")
		switch r.FormValue("do") {
		case "Add":
			newLink = &Link{key, r.FormValue("target")}
			doErr = h.putLink(ctx, newLink)
		case "Delete":
			k := datastore.NameKey(kind, key, nil)
			doErr = h.datastore.Delete(ctx, k)
		default:
			http.Error(w, "unknown action", http.StatusBadRequest)
		}
		err := h.memcache.Delete(ctx, cacheKey(key))
		if err != nil && err != memcache.ErrCacheMiss {
			log.Printf("WARNING %q: %v", key, err)
		}
	}

	var links []*Link
	q := datastore.NewQuery(kind).Order("Key")
	if _, err := h.datastore.GetAll(ctx, q, &links); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		log.Printf("ERROR %v", err)
		return
	}

	// Put the new link in the list if it's not there already.
	// (Eventual consistency means that it might not show up
	// immediately, which might be confusing for the user.)
	if newLink != nil && doErr == nil {
		found := false
		for i := range links {
			if links[i].Key == newLink.Key {
				found = true
				break
			}
		}
		if !found {
			links = append([]*Link{newLink}, links...)
		}
		newLink = nil
	}

	var data = struct {
		BaseURL string
		Prefix  string
		Links   []*Link
		New     *Link
		Error   error
	}{baseURL, prefix, links, newLink, doErr}
	if err := adminTemplate.Execute(w, &data); err != nil {
		log.Printf("ERROR adminTemplate: %v", err)
	}
}

// putLink validates the provided link and puts it into the datastore.
func (h server) putLink(ctx context.Context, link *Link) error {
	if !validKey.MatchString(link.Key) {
		return errors.New("invalid key; must match " + validKey.String())
	}
	if _, err := url.Parse(link.Target); err != nil {
		return fmt.Errorf("bad target: %v", err)
	}
	k := datastore.NameKey(kind, link.Key, nil)
	_, err := h.datastore.Put(ctx, k, link)
	return err
}

// cacheKey returns a short URL key as a memcache key.
func cacheKey(key string) string {
	return "link-" + key
}
