// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modinfo

import "time"

// Note that these structs are publicly visible (part of go list's API)
// and the fields are documented in the help text in ../list/list.go

type ModulePublic struct {
	Path      string        `json:",omitempty"` // module path
	Version   string        `json:",omitempty"` // module version
	Versions  []string      `json:",omitempty"` // available module versions
	Replace   *ModulePublic `json:",omitempty"` // replaced by this module
	Time      *time.Time    `json:",omitempty"` // time version was created
	Update    *ModulePublic `json:",omitempty"` // available update (with -u)
	Main      bool          `json:",omitempty"` // is this the main module?
	Indirect  bool          `json:",omitempty"` // module is only indirectly needed by main module
	Dir       string        `json:",omitempty"` // directory holding local copy of files, if any
	GoMod     string        `json:",omitempty"` // path to go.mod file describing module, if any
	GoVersion string        `json:",omitempty"` // go version used in module
	Retracted []string      `json:",omitempty"` // retraction information, if any (with -retracted or -u)
	Error     *ModuleError  `json:",omitempty"` // error loading module
}

type ModuleError struct {
	Err string // error text
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
	if m.Replace != nil {
		s += " => " + m.Replace.Path
		if m.Replace.Version != "" {
			s += " " + versionString(m.Replace)
			if m.Replace.Update != nil {
				s += " [" + versionString(m.Replace.Update) + "]"
			}
		}
	}
	return s
}
