// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package net

import (
	"errors"
	"internal/bytealg"
	"io"
	"os"
)

// nssConf represents the state of the machine's /etc/nsswitch.conf file.
type nssConf struct {
	err     error                  // any error encountered opening or parsing the file
	sources map[string][]nssSource // keyed by database (e.g. "hosts")
}

type nssSource struct {
	source   string // e.g. "compat", "files", "mdns4_minimal"
	criteria []nssCriterion
}

// standardCriteria reports all specified criteria have the default
// status actions.
func (s nssSource) standardCriteria() bool {
	for i, crit := range s.criteria {
		if !crit.standardStatusAction(i == len(s.criteria)-1) {
			return false
		}
	}
	return true
}

// nssCriterion is the parsed structure of one of the criteria in brackets
// after an NSS source name.
type nssCriterion struct {
	negate bool   // if "!" was present
	status string // e.g. "success", "unavail" (lowercase)
	action string // e.g. "return", "continue" (lowercase)
}

// standardStatusAction reports whether c is equivalent to not
// specifying the criterion at all. last is whether this criteria is the
// last in the list.
func (c nssCriterion) standardStatusAction(last bool) bool {
	if c.negate {
		return false
	}
	var def string
	switch c.status {
	case "success":
		def = "return"
	case "notfound", "unavail", "tryagain":
		def = "continue"
	default:
		// Unknown status
		return false
	}
	if last && c.action == "return" {
		return true
	}
	return c.action == def
}

func parseNSSConfFile(file string) *nssConf {
	f, err := os.Open(file)
	if err != nil {
		return &nssConf{err: err}
	}
	defer f.Close()
	return parseNSSConf(f)
}

func parseNSSConf(r io.Reader) *nssConf {
	slurp, err := readFull(r)
	if err != nil {
		return &nssConf{err: err}
	}
	conf := new(nssConf)
	conf.err = foreachLine(slurp, func(line []byte) error {
		line = trimSpace(removeComment(line))
		if len(line) == 0 {
			return nil
		}
		colon := bytealg.IndexByte(line, ':')
		if colon == -1 {
			return errors.New("no colon on line")
		}
		db := string(trimSpace(line[:colon]))
		srcs := line[colon+1:]
		for {
			srcs = trimSpace(srcs)
			if len(srcs) == 0 {
				break
			}
			sp := bytealg.IndexByte(srcs, ' ')
			var src string
			if sp == -1 {
				src = string(srcs)
				srcs = nil // done
			} else {
				src = string(srcs[:sp])
				srcs = trimSpace(srcs[sp+1:])
			}
			var criteria []nssCriterion
			// See if there's a criteria block in brackets.
			if len(srcs) > 0 && srcs[0] == '[' {
				bclose := bytealg.IndexByte(srcs, ']')
				if bclose == -1 {
					return errors.New("unclosed criterion bracket")
				}
				var err error
				criteria, err = parseCriteria(srcs[1:bclose])
				if err != nil {
					return errors.New("invalid criteria: " + string(srcs[1:bclose]))
				}
				srcs = srcs[bclose+1:]
			}
			if conf.sources == nil {
				conf.sources = make(map[string][]nssSource)
			}
			conf.sources[db] = append(conf.sources[db], nssSource{
				source:   src,
				criteria: criteria,
			})
		}
		return nil
	})
	return conf
}

// parses "foo=bar !foo=bar"
func parseCriteria(x []byte) (c []nssCriterion, err error) {
	err = foreachField(x, func(f []byte) error {
		not := false
		if len(f) > 0 && f[0] == '!' {
			not = true
			f = f[1:]
		}
		if len(f) < 3 {
			return errors.New("criterion too short")
		}
		eq := bytealg.IndexByte(f, '=')
		if eq == -1 {
			return errors.New("criterion lacks equal sign")
		}
		lowerASCIIBytes(f)
		c = append(c, nssCriterion{
			negate: not,
			status: string(f[:eq]),
			action: string(f[eq+1:]),
		})
		return nil
	})
	return
}
