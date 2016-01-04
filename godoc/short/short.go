// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build appengine

// Package short implements a simple URL shortener, serving an administrative
// interface at /s and shortened urls from /s/key.
// It is designed to run only on the instance of godoc that serves golang.org.
package short

// TODO(adg): collect statistics on URL visits

import (
	"errors"
	"fmt"
	"html/template"
	"net/http"
	"net/url"
	"regexp"

	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/datastore"
	"google.golang.org/appengine/log"
	"google.golang.org/appengine/memcache"
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

func RegisterHandlers(mux *http.ServeMux) {
	mux.HandleFunc(prefix, adminHandler)
	mux.HandleFunc(prefix+"/", linkHandler)
}

// linkHandler services requests to short URLs.
//   http://golang.org/s/key
// It consults memcache and datastore for the Link for key.
// It then sends a redirects or an error message.
func linkHandler(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)

	key := r.URL.Path[len(prefix)+1:]
	if !validKey.MatchString(key) {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	var link Link
	_, err := memcache.JSON.Get(c, cacheKey(key), &link)
	if err != nil {
		k := datastore.NewKey(c, kind, key, 0, nil)
		err = datastore.Get(c, k, &link)
		switch err {
		case datastore.ErrNoSuchEntity:
			http.Error(w, "not found", http.StatusNotFound)
			return
		default: // != nil
			log.Errorf(c, "%q: %v", key, err)
			http.Error(w, "internal server error", http.StatusInternalServerError)
			return
		case nil:
			item := &memcache.Item{
				Key:    cacheKey(key),
				Object: &link,
			}
			if err := memcache.JSON.Set(c, item); err != nil {
				log.Warningf(c, "%q: %v", key, err)
			}
		}
	}

	http.Redirect(w, r, link.Target, http.StatusFound)
}

var adminTemplate = template.Must(template.New("admin").Parse(templateHTML))

// adminHandler serves an administrative interface.
func adminHandler(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)

	if !user.IsAdmin(c) {
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
			doErr = putLink(c, newLink)
		case "Delete":
			k := datastore.NewKey(c, kind, key, 0, nil)
			doErr = datastore.Delete(c, k)
		default:
			http.Error(w, "unknown action", http.StatusBadRequest)
		}
		err := memcache.Delete(c, cacheKey(key))
		if err != nil && err != memcache.ErrCacheMiss {
			log.Warningf(c, "%q: %v", key, err)
		}
	}

	var links []*Link
	_, err := datastore.NewQuery(kind).Order("Key").GetAll(c, &links)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		log.Errorf(c, "%v", err)
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
		log.Criticalf(c, "adminTemplate: %v", err)
	}
}

// putLink validates the provided link and puts it into the datastore.
func putLink(c context.Context, link *Link) error {
	if !validKey.MatchString(link.Key) {
		return errors.New("invalid key; must match " + validKey.String())
	}
	if _, err := url.Parse(link.Target); err != nil {
		return fmt.Errorf("bad target: %v", err)
	}
	k := datastore.NewKey(c, kind, link.Key, 0, nil)
	_, err := datastore.Put(c, k, link)
	return err
}

// cacheKey returns a short URL key as a memcache key.
func cacheKey(key string) string {
	return "link-" + key
}
