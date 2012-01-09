// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code snippets included in "Error Handling and Go."

package main

import (
	"net/http"
	"text/template"
)

func init() {
	http.HandleFunc("/view", viewRecord)
}

func viewRecord(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)
	key := datastore.NewKey(c, "Record", r.FormValue("id"), 0, nil)
	record := new(Record)
	if err := datastore.Get(c, key, record); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	if err := viewTemplate.Execute(w, record); err != nil {
		http.Error(w, err.Error(), 500)
	}
}

// STOP OMIT

type ap struct{}

func (ap) NewContext(*http.Request) *ctx { return nil }

type ctx struct{}

func (*ctx) Errorf(string, ...interface{}) {}

var appengine ap

type ds struct{}

func (ds) NewKey(*ctx, string, string, int, *int) string { return "" }
func (ds) Get(*ctx, string, *Record) error               { return nil }

var datastore ds

type Record struct{}

var viewTemplate *template.Template

func main() {}
