// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code snippets included in "Error Handling and Go."

package main

import (
	"net/http"
	"text/template"
)

type appError struct {
	Error   error
	Message string
	Code    int
}

// STOP OMIT

type appHandler func(http.ResponseWriter, *http.Request) *appError

func (fn appHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if e := fn(w, r); e != nil { // e is *appError, not error.
		c := appengine.NewContext(r)
		c.Errorf("%v", e.Error)
		http.Error(w, e.Message, e.Code)
	}
}

// STOP OMIT

func viewRecord(w http.ResponseWriter, r *http.Request) *appError {
	c := appengine.NewContext(r)
	key := datastore.NewKey(c, "Record", r.FormValue("id"), 0, nil)
	record := new(Record)
	if err := datastore.Get(c, key, record); err != nil {
		return &appError{err, "Record not found", 404}
	}
	if err := viewTemplate.Execute(w, record); err != nil {
		return &appError{err, "Can't display record", 500}
	}
	return nil
}

// STOP OMIT

func init() {
	http.Handle("/view", appHandler(viewRecord))
}

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
