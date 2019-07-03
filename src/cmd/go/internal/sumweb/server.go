// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sumweb implements the HTTP protocols for serving or accessing a go.sum database.
package sumweb

import (
	"context"
	"internal/lazyregexp"
	"net/http"
	"os"
	"strings"

	"cmd/go/internal/tlog"
)

// A Server provides the external operations
// (underlying database access and so on)
// needed to implement the HTTP server Handler.
type Server interface {
	// NewContext returns the context to use for the request r.
	NewContext(r *http.Request) (context.Context, error)

	// Signed returns the signed hash of the latest tree.
	Signed(ctx context.Context) ([]byte, error)

	// ReadRecords returns the content for the n records id through id+n-1.
	ReadRecords(ctx context.Context, id, n int64) ([][]byte, error)

	// Lookup looks up a record by its associated key ("module@version"),
	// returning the record ID.
	Lookup(ctx context.Context, key string) (int64, error)

	// ReadTileData reads the content of tile t.
	// It is only invoked for hash tiles (t.L ≥ 0).
	ReadTileData(ctx context.Context, t tlog.Tile) ([]byte, error)
}

// A Handler is the go.sum database server handler,
// which should be invoked to serve the paths listed in Paths.
// The calling code is responsible for initializing Server.
type Handler struct {
	Server Server
}

// Paths are the URL paths for which Handler should be invoked.
//
// Typically a server will do:
//
//	handler := &sumweb.Handler{Server: srv}
//	for _, path := range sumweb.Paths {
//		http.HandleFunc(path, handler)
//	}
//
var Paths = []string{
	"/lookup/",
	"/latest",
	"/tile/",
}

var modVerRE = lazyregexp.New(`^[^@]+@v[0-9]+\.[0-9]+\.[0-9]+(-[^@]*)?(\+incompatible)?$`)

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx, err := h.Server.NewContext(r)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	switch {
	default:
		http.NotFound(w, r)

	case strings.HasPrefix(r.URL.Path, "/lookup/"):
		mod := strings.TrimPrefix(r.URL.Path, "/lookup/")
		if !modVerRE.MatchString(mod) {
			http.Error(w, "invalid module@version syntax", http.StatusBadRequest)
			return
		}
		i := strings.Index(mod, "@")
		encPath, encVers := mod[:i], mod[i+1:]
		path, err := decodePath(encPath)
		if err != nil {
			reportError(w, r, err)
			return
		}
		vers, err := decodeVersion(encVers)
		if err != nil {
			reportError(w, r, err)
			return
		}
		id, err := h.Server.Lookup(ctx, path+"@"+vers)
		if err != nil {
			reportError(w, r, err)
			return
		}
		records, err := h.Server.ReadRecords(ctx, id, 1)
		if err != nil {
			// This should never happen - the lookup says the record exists.
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if len(records) != 1 {
			http.Error(w, "invalid record count returned by ReadRecords", http.StatusInternalServerError)
			return
		}
		msg, err := tlog.FormatRecord(id, records[0])
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		signed, err := h.Server.Signed(ctx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=UTF-8")
		w.Write(msg)
		w.Write(signed)

	case r.URL.Path == "/latest":
		data, err := h.Server.Signed(ctx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=UTF-8")
		w.Write(data)

	case strings.HasPrefix(r.URL.Path, "/tile/"):
		t, err := tlog.ParseTilePath(r.URL.Path[1:])
		if err != nil {
			http.Error(w, "invalid tile syntax", http.StatusBadRequest)
			return
		}
		if t.L == -1 {
			// Record data.
			start := t.N << uint(t.H)
			records, err := h.Server.ReadRecords(ctx, start, int64(t.W))
			if err != nil {
				reportError(w, r, err)
				return
			}
			if len(records) != t.W {
				http.Error(w, "invalid record count returned by ReadRecords", http.StatusInternalServerError)
				return
			}
			var data []byte
			for i, text := range records {
				msg, err := tlog.FormatRecord(start+int64(i), text)
				if err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
				}
				data = append(data, msg...)
			}
			w.Header().Set("Content-Type", "text/plain; charset=UTF-8")
			w.Write(data)
			return
		}

		data, err := h.Server.ReadTileData(ctx, t)
		if err != nil {
			reportError(w, r, err)
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Write(data)
	}
}

// reportError reports err to w.
// If it's a not-found, the reported error is 404.
// Otherwise it is an internal server error.
// The caller must only call reportError in contexts where
// a not-found err should be reported as 404.
func reportError(w http.ResponseWriter, r *http.Request, err error) {
	if os.IsNotExist(err) {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	http.Error(w, err.Error(), http.StatusInternalServerError)
}
