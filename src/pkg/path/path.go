// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The path package implements utility routines for manipulating
// slash-separated filename paths.
package path

import (
	"io";
	"os";
	"strings";
)

// Clean returns the shortest path name equivalent to path
// by purely lexical processing.  It applies the following rules
// iteratively until no further processing can be done:
//
//	1. Replace multiple slashes with a single slash.
//	2. Eliminate each . path name element (the current directory).
//	3. Eliminate each inner .. path name element (the parent directory)
//	   along with the non-.. element that precedes it.
//	4. Eliminate .. elements that begin a rooted path:
//	   that is, replace "/.." by "/" at the beginning of a path.
//
// If the result of this process is an empty string, Clean
// returns the string ".".
//
// See also Rob Pike, ``Lexical File Names in Plan 9 or
// Getting Dot-Dot right,''
// http://plan9.bell-labs.com/sys/doc/lexnames.html
func Clean(path string) string {
	if path == "" {
		return "."
	}

	rooted := path[0] == '/';
	n := len(path);

	// Invariants:
	//	reading from path; r is index of next byte to process.
	//	writing to buf; w is index of next byte to write.
	//	dotdot is index in buf where .. must stop, either because
	//		it is the leading slash or it is a leading ../../.. prefix.
	buf := strings.Bytes(path);
	r, w, dotdot := 0, 0, 0;
	if rooted {
		r, w, dotdot = 1, 1, 1
	}

	for r < n {
		switch {
		case path[r] == '/':
			// empty path element
			r++
		case path[r] == '.' && (r+1 == n || path[r+1] == '/'):
			// . element
			r++
		case path[r] == '.' && path[r+1] == '.' && (r+2 == n || path[r+2] == '/'):
			// .. element: remove to last /
			r += 2;
			switch {
			case w > dotdot:
				// can backtrack
				w--;
				for w > dotdot && buf[w] != '/' {
					w--
				}
			case !rooted:
				// cannot backtrack, but not rooted, so append .. element.
				if w > 0 {
					buf[w] = '/';
					w++;
				}
				buf[w] = '.';
				w++;
				buf[w] = '.';
				w++;
				dotdot = w;
			}
		default:
			// real path element.
			// add slash if needed
			if rooted && w != 1 || !rooted && w != 0 {
				buf[w] = '/';
				w++;
			}
			// copy element
			for ; r < n && path[r] != '/'; r++ {
				buf[w] = path[r];
				w++;
			}
		}
	}

	// Turn empty string into "."
	if w == 0 {
		buf[w] = '.';
		w++;
	}

	return string(buf[0:w]);
}

// Split splits path immediately following the final slash,
// separating it into a directory and file name component.
// If there is no slash in path, DirFile returns an empty dir and
// file set to path.
func Split(path string) (dir, file string) {
	for i := len(path)-1; i >= 0; i-- {
		if path[i] == '/' {
			return path[0 : i+1], path[i+1 : len(path)]
		}
	}
	return "", path;
}

// Join joins dir and file into a single path, adding a separating
// slash if necessary.  If dir is empty, it returns file.
func Join(dir, file string) string {
	if dir == "" {
		return file
	}
	return Clean(dir+"/"+file);
}

// Ext returns the file name extension used by path.
// The extension is the suffix beginning at the final dot
// in the final slash-separated element of path;
// it is empty if there is no dot.
func Ext(path string) string {
	for i := len(path)-1; i >= 0 && path[i] != '/'; i-- {
		if path[i] == '.' {
			return path[i:len(path)]
		}
	}
	return "";
}

// Visitor methods are invoked for corresponding file tree entries
// visited by Walk. The parameter path is the full path of d relative
// to root.
type Visitor interface {
	VisitDir(path string, d *os.Dir) bool;
	VisitFile(path string, d *os.Dir);
}

func walk(path string, d *os.Dir, v Visitor, errors chan<- os.Error) {
	if !d.IsDirectory() {
		v.VisitFile(path, d);
		return;
	}

	if !v.VisitDir(path, d) {
		return	// skip directory entries
	}

	list, err := io.ReadDir(path);
	if err != nil {
		if errors != nil {
			errors <- err
		}
	}

	for _, e := range list {
		walk(Join(path, e.Name), e, v, errors)
	}
}

// Walk walks the file tree rooted at root, calling v.VisitDir or
// v.VisitFile for each directory or file in the tree, including root.
// If v.VisitDir returns false, Walk skips the directory's entries;
// otherwise it invokes itself for each directory entry in sorted order.
// An error reading a directory does not abort the Walk.
// If errors != nil, Walk sends each directory read error
// to the channel.  Otherwise Walk discards the error.
func Walk(root string, v Visitor, errors chan<- os.Error) {
	d, err := os.Lstat(root);
	if err != nil {
		if errors != nil {
			errors <- err
		}
		return;	// can't progress
	}
	walk(root, d, v, errors);
}
