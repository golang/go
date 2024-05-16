// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package filepathlite implements a subset of path/filepath,
// only using packages which may be imported by "os".
//
// Tests for these functions are in path/filepath.
package filepathlite

import (
	"errors"
	"internal/stringslite"
	"io/fs"
	"slices"
)

var errInvalidPath = errors.New("invalid path")

// A lazybuf is a lazily constructed path buffer.
// It supports append, reading previously appended bytes,
// and retrieving the final string. It does not allocate a buffer
// to hold the output until that output diverges from s.
type lazybuf struct {
	path       string
	buf        []byte
	w          int
	volAndPath string
	volLen     int
}

func (b *lazybuf) index(i int) byte {
	if b.buf != nil {
		return b.buf[i]
	}
	return b.path[i]
}

func (b *lazybuf) append(c byte) {
	if b.buf == nil {
		if b.w < len(b.path) && b.path[b.w] == c {
			b.w++
			return
		}
		b.buf = make([]byte, len(b.path))
		copy(b.buf, b.path[:b.w])
	}
	b.buf[b.w] = c
	b.w++
}

func (b *lazybuf) prepend(prefix ...byte) {
	b.buf = slices.Insert(b.buf, 0, prefix...)
	b.w += len(prefix)
}

func (b *lazybuf) string() string {
	if b.buf == nil {
		return b.volAndPath[:b.volLen+b.w]
	}
	return b.volAndPath[:b.volLen] + string(b.buf[:b.w])
}

// Clean is filepath.Clean.
func Clean(path string) string {
	originalPath := path
	volLen := volumeNameLen(path)
	path = path[volLen:]
	if path == "" {
		if volLen > 1 && IsPathSeparator(originalPath[0]) && IsPathSeparator(originalPath[1]) {
			// should be UNC
			return FromSlash(originalPath)
		}
		return originalPath + "."
	}
	rooted := IsPathSeparator(path[0])

	// Invariants:
	//	reading from path; r is index of next byte to process.
	//	writing to buf; w is index of next byte to write.
	//	dotdot is index in buf where .. must stop, either because
	//		it is the leading slash or it is a leading ../../.. prefix.
	n := len(path)
	out := lazybuf{path: path, volAndPath: originalPath, volLen: volLen}
	r, dotdot := 0, 0
	if rooted {
		out.append(Separator)
		r, dotdot = 1, 1
	}

	for r < n {
		switch {
		case IsPathSeparator(path[r]):
			// empty path element
			r++
		case path[r] == '.' && (r+1 == n || IsPathSeparator(path[r+1])):
			// . element
			r++
		case path[r] == '.' && path[r+1] == '.' && (r+2 == n || IsPathSeparator(path[r+2])):
			// .. element: remove to last separator
			r += 2
			switch {
			case out.w > dotdot:
				// can backtrack
				out.w--
				for out.w > dotdot && !IsPathSeparator(out.index(out.w)) {
					out.w--
				}
			case !rooted:
				// cannot backtrack, but not rooted, so append .. element.
				if out.w > 0 {
					out.append(Separator)
				}
				out.append('.')
				out.append('.')
				dotdot = out.w
			}
		default:
			// real path element.
			// add slash if needed
			if rooted && out.w != 1 || !rooted && out.w != 0 {
				out.append(Separator)
			}
			// copy element
			for ; r < n && !IsPathSeparator(path[r]); r++ {
				out.append(path[r])
			}
		}
	}

	// Turn empty string into "."
	if out.w == 0 {
		out.append('.')
	}

	postClean(&out) // avoid creating absolute paths on Windows
	return FromSlash(out.string())
}

// IsLocal is filepath.IsLocal.
func IsLocal(path string) bool {
	return isLocal(path)
}

func unixIsLocal(path string) bool {
	if IsAbs(path) || path == "" {
		return false
	}
	hasDots := false
	for p := path; p != ""; {
		var part string
		part, p, _ = stringslite.Cut(p, "/")
		if part == "." || part == ".." {
			hasDots = true
			break
		}
	}
	if hasDots {
		path = Clean(path)
	}
	if path == ".." || stringslite.HasPrefix(path, "../") {
		return false
	}
	return true
}

// Localize is filepath.Localize.
func Localize(path string) (string, error) {
	if !fs.ValidPath(path) {
		return "", errInvalidPath
	}
	return localize(path)
}

// ToSlash is filepath.ToSlash.
func ToSlash(path string) string {
	if Separator == '/' {
		return path
	}
	return replaceStringByte(path, Separator, '/')
}

// FromSlash is filepath.ToSlash.
func FromSlash(path string) string {
	if Separator == '/' {
		return path
	}
	return replaceStringByte(path, '/', Separator)
}

func replaceStringByte(s string, old, new byte) string {
	if stringslite.IndexByte(s, old) == -1 {
		return s
	}
	n := []byte(s)
	for i := range n {
		if n[i] == old {
			n[i] = new
		}
	}
	return string(n)
}

// Split is filepath.Split.
func Split(path string) (dir, file string) {
	vol := VolumeName(path)
	i := len(path) - 1
	for i >= len(vol) && !IsPathSeparator(path[i]) {
		i--
	}
	return path[:i+1], path[i+1:]
}

// Ext is filepath.Ext.
func Ext(path string) string {
	for i := len(path) - 1; i >= 0 && !IsPathSeparator(path[i]); i-- {
		if path[i] == '.' {
			return path[i:]
		}
	}
	return ""
}

// Base is filepath.Base.
func Base(path string) string {
	if path == "" {
		return "."
	}
	// Strip trailing slashes.
	for len(path) > 0 && IsPathSeparator(path[len(path)-1]) {
		path = path[0 : len(path)-1]
	}
	// Throw away volume name
	path = path[len(VolumeName(path)):]
	// Find the last element
	i := len(path) - 1
	for i >= 0 && !IsPathSeparator(path[i]) {
		i--
	}
	if i >= 0 {
		path = path[i+1:]
	}
	// If empty now, it had only slashes.
	if path == "" {
		return string(Separator)
	}
	return path
}

// Dir is filepath.Dir.
func Dir(path string) string {
	vol := VolumeName(path)
	i := len(path) - 1
	for i >= len(vol) && !IsPathSeparator(path[i]) {
		i--
	}
	dir := Clean(path[len(vol) : i+1])
	if dir == "." && len(vol) > 2 {
		// must be UNC
		return vol
	}
	return vol + dir
}

// VolumeName is filepath.VolumeName.
func VolumeName(path string) string {
	return FromSlash(path[:volumeNameLen(path)])
}

// VolumeNameLen returns the length of the leading volume name on Windows.
// It returns 0 elsewhere.
func VolumeNameLen(path string) int {
	return volumeNameLen(path)
}
