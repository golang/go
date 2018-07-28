// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"encoding/json"
	"net/http"
	"strings"
)

type proxyServer struct {
	list   []string
	info   *RevInfo
	mod    []byte
	zip    []byte
	latest *RevInfo
}

func (mp *proxyServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	p := r.URL.Path
	switch {
	case strings.HasSuffix(p, "list"):
		w.Write([]byte(strings.Join(mp.list, "\n")))
	case strings.HasSuffix(p, ".info"):
		json.NewEncoder(w).Encode(mp.info)
	case strings.HasSuffix(p, ".mod"):
		w.Write(mp.mod)
	case strings.HasSuffix(p, ".zip"):
		w.Write(mp.zip)
	case strings.HasSuffix(p, "@latest"):
		json.NewEncoder(w).Encode(mp.latest)
	default:
		w.WriteHeader(404)
	}
}
