// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"internal/bytealg"
	"os"
	"sync"
	"time"
)

const (
	nssConfigPath = "/etc/nsswitch.conf"
)

var nssConfig nsswitchConfig

type nsswitchConfig struct {
	initOnce sync.Once // guards init of nsswitchConfig

	// ch is used as a semaphore that only allows one lookup at a
	// time to recheck nsswitch.conf
	ch          chan struct{} // guards lastChecked and modTime
	lastChecked time.Time     // last time nsswitch.conf was checked

	mu      sync.Mutex // protects nssConf
	nssConf *nssConf
}

func getSystemNSS() *nssConf {
	nssConfig.tryUpdate()
	nssConfig.mu.Lock()
	conf := nssConfig.nssConf
	nssConfig.mu.Unlock()
	return conf
}

// init initializes conf and is only called via conf.initOnce.
func (conf *nsswitchConfig) init() {
	conf.nssConf = parseNSSConfFile("/etc/nsswitch.conf")
	conf.lastChecked = time.Now()
	conf.ch = make(chan struct{}, 1)
}

// tryUpdate tries to update conf.
func (conf *nsswitchConfig) tryUpdate() {
	conf.initOnce.Do(conf.init)

	// Ensure only one update at a time checks nsswitch.conf
	if !conf.tryAcquireSema() {
		return
	}
	defer conf.releaseSema()

	now := time.Now()
	if conf.lastChecked.After(now.Add(-5 * time.Second)) {
		return
	}
	conf.lastChecked = now

	var mtime time.Time
	if fi, err := os.Stat(nssConfigPath); err == nil {
		mtime = fi.ModTime()
	}
	if mtime.Equal(conf.nssConf.mtime) {
		return
	}

	nssConf := parseNSSConfFile(nssConfigPath)
	conf.mu.Lock()
	conf.nssConf = nssConf
	conf.mu.Unlock()
}

func (conf *nsswitchConfig) acquireSema() {
	conf.ch <- struct{}{}
}

func (conf *nsswitchConfig) tryAcquireSema() bool {
	select {
	case conf.ch <- struct{}{}:
		return true
	default:
		return false
	}
}

func (conf *nsswitchConfig) releaseSema() {
	<-conf.ch
}

// nssConf represents the state of the machine's /etc/nsswitch.conf file.
type nssConf struct {
	mtime   time.Time              // time of nsswitch.conf modification
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
	f, err := open(file)
	if err != nil {
		return &nssConf{err: err}
	}
	defer f.close()
	mtime, _, err := f.stat()
	if err != nil {
		return &nssConf{err: err}
	}

	conf := parseNSSConf(f)
	conf.mtime = mtime
	return conf
}

func parseNSSConf(f *file) *nssConf {
	conf := new(nssConf)
	for line, ok := f.readLine(); ok; line, ok = f.readLine() {
		line = trimSpace(removeComment(line))
		if len(line) == 0 {
			continue
		}
		colon := bytealg.IndexByteString(line, ':')
		if colon == -1 {
			conf.err = errors.New("no colon on line")
			return conf
		}
		db := trimSpace(line[:colon])
		srcs := line[colon+1:]
		for {
			srcs = trimSpace(srcs)
			if len(srcs) == 0 {
				break
			}
			sp := bytealg.IndexByteString(srcs, ' ')
			var src string
			if sp == -1 {
				src = srcs
				srcs = "" // done
			} else {
				src = srcs[:sp]
				srcs = trimSpace(srcs[sp+1:])
			}
			var criteria []nssCriterion
			// See if there's a criteria block in brackets.
			if len(srcs) > 0 && srcs[0] == '[' {
				bclose := bytealg.IndexByteString(srcs, ']')
				if bclose == -1 {
					conf.err = errors.New("unclosed criterion bracket")
					return conf
				}
				var err error
				criteria, err = parseCriteria(srcs[1:bclose])
				if err != nil {
					conf.err = errors.New("invalid criteria: " + srcs[1:bclose])
					return conf
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
	}
	return conf
}

// parses "foo=bar !foo=bar"
func parseCriteria(x string) (c []nssCriterion, err error) {
	err = foreachField(x, func { f ->
		not := false
		if len(f) > 0 && f[0] == '!' {
			not = true
			f = f[1:]
		}
		if len(f) < 3 {
			return errors.New("criterion too short")
		}
		eq := bytealg.IndexByteString(f, '=')
		if eq == -1 {
			return errors.New("criterion lacks equal sign")
		}
		if hasUpperCase(f) {
			lower := []byte(f)
			lowerASCIIBytes(lower)
			f = string(lower)
		}
		c = append(c, nssCriterion{
			negate: not,
			status: f[:eq],
			action: f[eq+1:],
		})
		return nil
	})
	return
}
