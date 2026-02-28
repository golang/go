// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modinfo

import (
	"cmd/go/internal/modfetch/codehost"
	"encoding/json"
	"time"
)

// Note that these structs are publicly visible (part of go list's API)
// and the fields are documented in the help text in ../list/list.go

type ModulePublic struct {
	Path       string           `json:",omitempty"` // module path
	Version    string           `json:",omitempty"` // module version
	Query      string           `json:",omitempty"` // version query corresponding to this version
	Versions   []string         `json:",omitempty"` // available module versions
	Replace    *ModulePublic    `json:",omitempty"` // replaced by this module
	Time       *time.Time       `json:",omitempty"` // time version was created
	Update     *ModulePublic    `json:",omitempty"` // available update (with -u)
	Main       bool             `json:",omitempty"` // is this the main module?
	Indirect   bool             `json:",omitempty"` // module is only indirectly needed by main module
	Dir        string           `json:",omitempty"` // directory holding local copy of files, if any
	GoMod      string           `json:",omitempty"` // path to go.mod file describing module, if any
	GoVersion  string           `json:",omitempty"` // go version used in module
	Retracted  []string         `json:",omitempty"` // retraction information, if any (with -retracted or -u)
	Deprecated string           `json:",omitempty"` // deprecation message, if any (with -u)
	Error      *ModuleError     `json:",omitempty"` // error loading module
	Sum        string           `json:",omitempty"` // checksum for path, version (as in go.sum)
	GoModSum   string           `json:",omitempty"` // checksum for go.mod (as in go.sum)
	Origin     *codehost.Origin `json:",omitempty"` // provenance of module
	Reuse      bool             `json:",omitempty"` // reuse of old module info is safe
}

type ModuleError struct {
	Err string // error text
}

type moduleErrorNoMethods ModuleError

// UnmarshalJSON accepts both {"Err":"text"} and "text",
// so that the output of go mod download -json can still
// be unmarshaled into a ModulePublic during -reuse processing.
func (e *ModuleError) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && data[0] == '"' {
		return json.Unmarshal(data, &e.Err)
	}
	return json.Unmarshal(data, (*moduleErrorNoMethods)(e))
}

func (m *ModulePublic) String() string {
	s := m.Path
	versionString := func(mm *ModulePublic) string {
		v := mm.Version
		if len(mm.Retracted) == 0 {
			return v
		}
		return v + " (retracted)"
	}

	if m.Version != "" {
		s += " " + versionString(m)
		if m.Update != nil {
			s += " [" + versionString(m.Update) + "]"
		}
	}
	if m.Deprecated != "" {
		s += " (deprecated)"
	}
	if m.Replace != nil {
		s += " => " + m.Replace.Path
		if m.Replace.Version != "" {
			s += " " + versionString(m.Replace)
			if m.Replace.Update != nil {
				s += " [" + versionString(m.Replace.Update) + "]"
			}
		}
		if m.Replace.Deprecated != "" {
			s += " (deprecated)"
		}
	}
	return s
}
