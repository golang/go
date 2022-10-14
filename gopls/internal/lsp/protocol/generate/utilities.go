// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package main

import (
	"fmt"
	"log"
	"runtime"
	"strings"
	"time"
)

// goName returns the Go version of a name.
func goName(s string) string {
	if s == "" {
		return s // doesn't happen
	}
	s = strings.ToUpper(s[:1]) + s[1:]
	if rest := strings.TrimSuffix(s, "Uri"); rest != s {
		s = rest + "URI"
	}
	if rest := strings.TrimSuffix(s, "Id"); rest != s {
		s = rest + "ID"
	}
	return s
}

// the common header for all generated files
func (s *spec) createHeader() string {
	format := `// Copyright 2022 The Go Authors. All rights reserved.
	// Use of this source code is governed by a BSD-style
	// license that can be found in the LICENSE file.

	// Code generated for LSP. DO NOT EDIT.

	package protocol

	// Code generated from version %s of protocol/metaModel.json.
	// git hash %s (as of %s)

	`
	hdr := fmt.Sprintf(format, s.model.Version.Version, s.githash, s.modTime.Format(time.ANSIC))
	return hdr
}

// useful in debugging
func here() {
	_, f, l, _ := runtime.Caller(1)
	log.Printf("here: %s:%d", f, l)
}
