// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"bytes"
	"cmd/internal/notsha256"
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
)

type writeSyncer interface {
	io.Writer
	Sync() error
}

type hashAndMask struct {
	// a hash h matches if (h^hash)&mask == 0
	hash uint64
	mask uint64
	name string // base name, or base name + "0", "1", etc.
}

type HashDebug struct {
	mu   sync.Mutex // for logfile, posTmp, bytesTmp
	name string     // base name of the flag/variable.
	// what file (if any) receives the yes/no logging?
	// default is os.Stdout
	logfile  writeSyncer
	posTmp   []src.Pos
	bytesTmp bytes.Buffer
	matches  []hashAndMask // A hash matches if one of these matches.
	yes, no  bool
}

// The default compiler-debugging HashDebug, for "-d=gossahash=..."
var hashDebug *HashDebug
var FmaHash *HashDebug

// DebugHashMatch reports whether debug variable Gossahash
//
//  1. is empty (returns true; this is a special more-quickly implemented case of 4 below)
//
//  2. is "y" or "Y" (returns true)
//
//  3. is "n" or "N" (returns false)
//
//  4. is a suffix of the sha1 hash of pkgAndName (returns true)
//
//  5. OR
//     if the value is in the regular language "[01]+(;[01]+)+"
//     test the [01]+ substrings after in order returning true
//     for the first one that suffix-matches. The substrings AFTER
//     the first semicolon are numbered 0,1, etc and are named
//     fmt.Sprintf("%s%d", varname, number)
//     Clause 5 is not really intended for human use and only
//     matters for failures that require multiple triggers.
//
// Otherwise it returns false.
//
// Unless Flags.Gossahash is empty, when DebugHashMatch returns true the message
//
//	"%s triggered %s\n", varname, pkgAndName
//
// is printed on the file named in environment variable GSHS_LOGFILE,
// or standard out if that is empty.  "Varname" is either the name of
// the variable or the name of the substring, depending on which matched.
//
// Typical use:
//
//  1. you make a change to the compiler, say, adding a new phase
//
//  2. it is broken in some mystifying way, for example, make.bash builds a broken
//     compiler that almost works, but crashes compiling a test in run.bash.
//
//  3. add this guard to the code, which by default leaves it broken, but does not
//     run the broken new code if Flags.Gossahash is non-empty and non-matching:
//
//     if !base.DebugHashMatch(ir.PkgFuncName(fn)) {
//     return nil // early exit, do nothing
//     }
//
//  4. rebuild w/o the bad code,
//     GOCOMPILEDEBUG=gossahash=n ./all.bash
//     to verify that you put the guard in the right place with the right sense of the test.
//
//  5. use github.com/dr2chase/gossahash to search for the error:
//
//     go install github.com/dr2chase/gossahash@latest
//
//     gossahash -- <the thing that fails>
//
//     for example: GOMAXPROCS=1 gossahash -- ./all.bash
//
//  6. gossahash should return a single function whose miscompilation
//     causes the problem, and you can focus on that.
func DebugHashMatch(pkgAndName string) bool {
	return hashDebug.DebugHashMatch(pkgAndName)
}

// HasDebugHash returns true if Flags.Gossahash is non-empty, which
// results in hashDebug being not-nil.  I.e., if !HasDebugHash(),
// there is no need to create the string for hashing and testing.
func HasDebugHash() bool {
	return hashDebug != nil
}

func toHashAndMask(s, varname string) hashAndMask {
	l := len(s)
	if l > 64 {
		s = s[l-64:]
		l = 64
	}
	m := ^(^uint64(0) << l)
	h, err := strconv.ParseUint(s, 2, 64)
	if err != nil {
		Fatalf("Could not parse %s (=%s) as a binary number", varname, s)
	}

	return hashAndMask{name: varname, hash: h, mask: m}
}

// NewHashDebug returns a new hash-debug tester for the
// environment variable ev.  If ev is not set, it returns
// nil, allowing a lightweight check for normal-case behavior.
func NewHashDebug(ev, s string, file writeSyncer) *HashDebug {
	if s == "" {
		return nil
	}

	hd := &HashDebug{name: ev, logfile: file}
	switch s[0] {
	case 'y', 'Y':
		hd.yes = true
		return hd
	case 'n', 'N':
		hd.no = true
		return hd
	}
	ss := strings.Split(s, "/")
	hd.matches = append(hd.matches, toHashAndMask(ss[0], ev))
	// hash searches may use additional EVs with 0, 1, 2, ... suffixes.
	for i := 1; i < len(ss); i++ {
		evi := fmt.Sprintf("%s%d", ev, i-1) // convention is extras begin indexing at zero
		hd.matches = append(hd.matches, toHashAndMask(ss[i], evi))
	}
	return hd

}

func hashOf(pkgAndName string, param uint64) uint64 {
	return hashOfBytes([]byte(pkgAndName), param)
}

func hashOfBytes(sbytes []byte, param uint64) uint64 {
	hbytes := notsha256.Sum256(sbytes)
	hash := uint64(hbytes[7])<<56 + uint64(hbytes[6])<<48 +
		uint64(hbytes[5])<<40 + uint64(hbytes[4])<<32 +
		uint64(hbytes[3])<<24 + uint64(hbytes[2])<<16 +
		uint64(hbytes[1])<<8 + uint64(hbytes[0])

	if param != 0 {
		// Because param is probably a line number, probably near zero,
		// hash it up a little bit, but even so only the lower-order bits
		// likely matter because search focuses on those.
		p0 := param + uint64(hbytes[9]) + uint64(hbytes[10])<<8 +
			uint64(hbytes[11])<<16 + uint64(hbytes[12])<<24

		p1 := param + uint64(hbytes[13]) + uint64(hbytes[14])<<8 +
			uint64(hbytes[15])<<16 + uint64(hbytes[16])<<24

		param += p0 * p1
		param ^= param>>17 ^ param<<47
	}

	return hash ^ param
}

// DebugHashMatch returns true if either the variable used to create d is
// unset, or if its value is y, or if it is a suffix of the base-two
// representation of the hash of pkgAndName.  If the variable is not nil,
// then a true result is accompanied by stylized output to d.logfile, which
// is used for automated bug search.
func (d *HashDebug) DebugHashMatch(pkgAndName string) bool {
	return d.DebugHashMatchParam(pkgAndName, 0)
}

// DebugHashMatchParam returns true if either the variable used to create d is
// unset, or if its value is y, or if it is a suffix of the base-two
// representation of the hash of pkgAndName and param. If the variable is not
// nil, then a true result is accompanied by stylized output to d.logfile,
// which is used for automated bug search.
func (d *HashDebug) DebugHashMatchParam(pkgAndName string, param uint64) bool {
	if d == nil {
		return true
	}
	if d.no {
		return false
	}

	if d.yes {
		d.logDebugHashMatch(d.name, pkgAndName, "y", param)
		return true
	}

	hash := hashOf(pkgAndName, param)

	for _, m := range d.matches {
		if (m.hash^hash)&m.mask == 0 {
			hstr := ""
			if hash == 0 {
				hstr = "0"
			} else {
				for ; hash != 0; hash = hash >> 1 {
					hstr = string('0'+byte(hash&1)) + hstr
				}
			}
			d.logDebugHashMatch(m.name, pkgAndName, hstr, param)
			return true
		}
	}
	return false
}

// DebugHashMatchPos is similar to DebugHashMatchParam, but for hash computation
// it uses the source position including all inlining information instead of
// package name and path. The output trigger string is prefixed with "POS=" so
// that tools processing the output can reliably tell the difference. The mutex
// locking is also more frequent and more granular.
func (d *HashDebug) DebugHashMatchPos(ctxt *obj.Link, pos src.XPos) bool {
	if d == nil {
		return true
	}
	if d.no {
		return false
	}
	d.mu.Lock()
	defer d.mu.Unlock()

	b := d.bytesForPos(ctxt, pos)

	if d.yes {
		d.logDebugHashMatchLocked(d.name, string(b), "y", 0)
		return true
	}

	hash := hashOfBytes(b, 0)

	for _, m := range d.matches {
		if (m.hash^hash)&m.mask == 0 {
			hstr := ""
			if hash == 0 {
				hstr = "0"
			} else {
				for ; hash != 0; hash = hash >> 1 {
					hstr = string('0'+byte(hash&1)) + hstr
				}
			}
			d.logDebugHashMatchLocked(m.name, "POS="+string(b), hstr, 0)
			return true
		}
	}
	return false
}

// bytesForPos renders a position, including inlining, into d.bytesTmp
// and returns the byte array.  d.mu must be locked.
func (d *HashDebug) bytesForPos(ctxt *obj.Link, pos src.XPos) []byte {
	d.posTmp = ctxt.AllPos(pos, d.posTmp)
	// Reverse posTmp to put outermost first.
	b := &d.bytesTmp
	b.Reset()
	for i := len(d.posTmp) - 1; i >= 0; i-- {
		p := &d.posTmp[i]
		fmt.Fprintf(b, "%s:%d:%d", p.Filename(), p.Line(), p.Col())
		if i != 0 {
			b.WriteByte(';')
		}
	}
	return b.Bytes()
}

func (d *HashDebug) logDebugHashMatch(varname, name, hstr string, param uint64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.logDebugHashMatchLocked(varname, name, hstr, param)
}

func (d *HashDebug) logDebugHashMatchLocked(varname, name, hstr string, param uint64) {
	file := d.logfile
	if file == nil {
		if tmpfile := os.Getenv("GSHS_LOGFILE"); tmpfile != "" {
			var err error
			file, err = os.OpenFile(tmpfile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
			if err != nil {
				Fatalf("could not open hash-testing logfile %s", tmpfile)
				return
			}
		}
		if file == nil {
			file = os.Stdout
		}
		d.logfile = file
	}
	if len(hstr) > 24 {
		hstr = hstr[len(hstr)-24:]
	}
	// External tools depend on this string
	if param == 0 {
		fmt.Fprintf(file, "%s triggered %s %s\n", varname, name, hstr)
	} else {
		fmt.Fprintf(file, "%s triggered %s:%d %s\n", varname, name, param, hstr)
	}
	file.Sync()
}
