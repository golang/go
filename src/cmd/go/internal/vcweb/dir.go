// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vcweb

import (
	"log"
	"net/http"
)

// dirHandler is a vcsHandler that serves the raw contents of a directory.
type dirHandler struct{}

func (*dirHandler) Available() bool { return true }

func (*dirHandler) Handler(dir string, env []string, logger *log.Logger) (http.Handler, error) {
	return http.FileServer(http.Dir(dir)), nil
}
