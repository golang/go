// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dashboard

// This file handles garbage collection of old CLs.

import (
	"net/http"
	"time"

	"appengine"
	"appengine/datastore"
)

func init() {
	http.HandleFunc("/gc", handleGC)
}

func handleGC(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)

	// Delete closed CLs that haven't been modified in 168 hours (7 days).
	cutoff := time.Now().Add(-168 * time.Hour)
	q := datastore.NewQuery("CL").
		Filter("Closed =", true).
		Filter("Modified <", cutoff).
		Limit(100).
		KeysOnly()
	keys, err := q.GetAll(c, nil)
	if err != nil {
		c.Errorf("GetAll failed for old CLs: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(keys) == 0 {
		return
	}

	if err := datastore.DeleteMulti(c, keys); err != nil {
		c.Errorf("DeleteMulti failed for old CLs: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	c.Infof("Deleted %d old CLs", len(keys))
}
