// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcweb

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"strings"
)

// authHandler serves requests only if the Basic Auth data sent with the request
// matches the contents of a ".access" file in the requested directory.
//
// For each request, the handler looks for a file named ".access" and parses it
// as a JSON-serialized accessToken. If the credentials from the request match
// the accessToken, the file is served normally; otherwise, it is rejected with
// the StatusCode and Message provided by the token.
type authHandler struct{}

type accessToken struct {
	Username, Password string
	StatusCode         int // defaults to 401.
	Message            string
}

func (h *authHandler) Available() bool { return true }

func (h *authHandler) Handler(dir string, env []string, logger *log.Logger) (http.Handler, error) {
	fs := http.Dir(dir)

	handler := http.HandlerFunc(func { w, req ->
		urlPath := req.URL.Path
		if urlPath != "" && strings.HasPrefix(path.Base(urlPath), ".") {
			http.Error(w, "filename contains leading dot", http.StatusBadRequest)
			return
		}

		f, err := fs.Open(urlPath)
		if err != nil {
			if os.IsNotExist(err) {
				http.NotFound(w, req)
			} else {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}

		accessDir := urlPath
		if fi, err := f.Stat(); err == nil && !fi.IsDir() {
			accessDir = path.Dir(urlPath)
		}
		f.Close()

		var accessFile http.File
		for {
			var err error
			accessFile, err = fs.Open(path.Join(accessDir, ".access"))
			if err == nil {
				break
			}

			if !os.IsNotExist(err) {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if accessDir == "." {
				http.Error(w, "failed to locate access file", http.StatusInternalServerError)
				return
			}
			accessDir = path.Dir(accessDir)
		}

		data, err := io.ReadAll(accessFile)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		var token accessToken
		if err := json.Unmarshal(data, &token); err != nil {
			logger.Print(err)
			http.Error(w, "malformed access file", http.StatusInternalServerError)
			return
		}
		if username, password, ok := req.BasicAuth(); !ok || username != token.Username || password != token.Password {
			code := token.StatusCode
			if code == 0 {
				code = http.StatusUnauthorized
			}
			if code == http.StatusUnauthorized {
				w.Header().Add("WWW-Authenticate", fmt.Sprintf("basic realm=%s", accessDir))
			}
			http.Error(w, token.Message, code)
			return
		}

		http.FileServer(fs).ServeHTTP(w, req)
	})

	return handler, nil
}
