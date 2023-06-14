// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fstest implements support for testing implementations and users of file systems.
package fstest

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"reflect"
	"sort"
	"strings"
	"testing/iotest"
)

// TestFS tests a file system implementation.
// It walks the entire tree of files in fsys,
// opening and checking that each file behaves correctly.
// It also checks that the file system contains at least the expected files.
// As a special case, if no expected files are listed, fsys must be empty.
// Otherwise, fsys must contain at least the listed files; it can also contain others.
// The contents of fsys must not change concurrently with TestFS.
//
// If TestFS finds any misbehaviors, it returns an error reporting all of them.
// The error text spans multiple lines, one per detected misbehavior.
//
// Typical usage inside a test is:
//
//	if err := fstest.TestFS(myFS, "file/that/should/be/present"); err != nil {
//		t.Fatal(err)
//	}
func TestFS(fsys fs.FS, expected ...string) error {
	if err := testFS(fsys, expected...); err != nil {
		return err
	}
	for _, name := range expected {
		if i := strings.Index(name, "/"); i >= 0 {
			dir, dirSlash := name[:i], name[:i+1]
			var subExpected []string
			for _, name := range expected {
				if strings.HasPrefix(name, dirSlash) {
					subExpected = append(subExpected, name[len(dirSlash):])
				}
			}
			sub, err := fs.Sub(fsys, dir)
			if err != nil {
				return err
			}
			if err := testFS(sub, subExpected...); err != nil {
				return fmt.Errorf("testing fs.Sub(fsys, %s): %v", dir, err)
			}
			break // one sub-test is enough
		}
	}
	return nil
}

func testFS(fsys fs.FS, expected ...string) error {
	t := fsTester{fsys: fsys}
	t.checkDir(".")
	t.checkOpen(".")
	found := make(map[string]bool)
	for _, dir := range t.dirs {
		found[dir] = true
	}
	for _, file := range t.files {
		found[file] = true
	}
	delete(found, ".")
	if len(expected) == 0 && len(found) > 0 {
		var list []string
		for k := range found {
			if k != "." {
				list = append(list, k)
			}
		}
		sort.Strings(list)
		if len(list) > 15 {
			list = append(list[:10], "...")
		}
		t.errorf("expected empty file system but found files:\n%s", strings.Join(list, "\n"))
	}
	for _, name := range expected {
		if !found[name] {
			t.errorf("expected but not found: %s", name)
		}
	}
	if len(t.errText) == 0 {
		return nil
	}
	return errors.New("TestFS found errors:\n" + string(t.errText))
}

// An fsTester holds state for running the test.
type fsTester struct {
	fsys    fs.FS
	errText []byte
	dirs    []string
	files   []string
}

// errorf adds an error line to errText.
func (t *fsTester) errorf(format string, args ...any) {
	if len(t.errText) > 0 {
		t.errText = append(t.errText, '\n')
	}
	t.errText = append(t.errText, fmt.Sprintf(format, args...)...)
}

func (t *fsTester) openDir(dir string) fs.ReadDirFile {
	f, err := t.fsys.Open(dir)
	if err != nil {
		t.errorf("%s: Open: %v", dir, err)
		return nil
	}
	d, ok := f.(fs.ReadDirFile)
	if !ok {
		f.Close()
		t.errorf("%s: Open returned File type %T, not a fs.ReadDirFile", dir, f)
		return nil
	}
	return d
}

// checkDir checks the directory dir, which is expected to exist
// (it is either the root or was found in a directory listing with IsDir true).
func (t *fsTester) checkDir(dir string) {
	// Read entire directory.
	t.dirs = append(t.dirs, dir)
	d := t.openDir(dir)
	if d == nil {
		return
	}
	list, err := d.ReadDir(-1)
	if err != nil {
		d.Close()
		t.errorf("%s: ReadDir(-1): %v", dir, err)
		return
	}

	// Check all children.
	var prefix string
	if dir == "." {
		prefix = ""
	} else {
		prefix = dir + "/"
	}
	for _, info := range list {
		name := info.Name()
		switch {
		case name == ".", name == "..", name == "":
			t.errorf("%s: ReadDir: child has invalid name: %#q", dir, name)
			continue
		case strings.Contains(name, "/"):
			t.errorf("%s: ReadDir: child name contains slash: %#q", dir, name)
			continue
		case strings.Contains(name, `\`):
			t.errorf("%s: ReadDir: child name contains backslash: %#q", dir, name)
			continue
		}
		path := prefix + name
		t.checkStat(path, info)
		t.checkOpen(path)
		if info.IsDir() {
			t.checkDir(path)
		} else {
			t.checkFile(path)
		}
	}

	// Check ReadDir(-1) at EOF.
	list2, err := d.ReadDir(-1)
	if len(list2) > 0 || err != nil {
		d.Close()
		t.errorf("%s: ReadDir(-1) at EOF = %d entries, %v, wanted 0 entries, nil", dir, len(list2), err)
		return
	}

	// Check ReadDir(1) at EOF (different results).
	list2, err = d.ReadDir(1)
	if len(list2) > 0 || err != io.EOF {
		d.Close()
		t.errorf("%s: ReadDir(1) at EOF = %d entries, %v, wanted 0 entries, EOF", dir, len(list2), err)
		return
	}

	// Check that close does not report an error.
	if err := d.Close(); err != nil {
		t.errorf("%s: Close: %v", dir, err)
	}

	// Check that closing twice doesn't crash.
	// The return value doesn't matter.
	d.Close()

	// Reopen directory, read a second time, make sure contents match.
	if d = t.openDir(dir); d == nil {
		return
	}
	defer d.Close()
	list2, err = d.ReadDir(-1)
	if err != nil {
		t.errorf("%s: second Open+ReadDir(-1): %v", dir, err)
		return
	}
	t.checkDirList(dir, "first Open+ReadDir(-1) vs second Open+ReadDir(-1)", list, list2)

	// Reopen directory, read a third time in pieces, make sure contents match.
	if d = t.openDir(dir); d == nil {
		return
	}
	defer d.Close()
	list2 = nil
	for {
		n := 1
		if len(list2) > 0 {
			n = 2
		}
		frag, err := d.ReadDir(n)
		if len(frag) > n {
			t.errorf("%s: third Open: ReadDir(%d) after %d: %d entries (too many)", dir, n, len(list2), len(frag))
			return
		}
		list2 = append(list2, frag...)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.errorf("%s: third Open: ReadDir(%d) after %d: %v", dir, n, len(list2), err)
			return
		}
		if n == 0 {
			t.errorf("%s: third Open: ReadDir(%d) after %d: 0 entries but nil error", dir, n, len(list2))
			return
		}
	}
	t.checkDirList(dir, "first Open+ReadDir(-1) vs third Open+ReadDir(1,2) loop", list, list2)

	// If fsys has ReadDir, check that it matches and is sorted.
	if fsys, ok := t.fsys.(fs.ReadDirFS); ok {
		list2, err := fsys.ReadDir(dir)
		if err != nil {
			t.errorf("%s: fsys.ReadDir: %v", dir, err)
			return
		}
		t.checkDirList(dir, "first Open+ReadDir(-1) vs fsys.ReadDir", list, list2)

		for i := 0; i+1 < len(list2); i++ {
			if list2[i].Name() >= list2[i+1].Name() {
				t.errorf("%s: fsys.ReadDir: list not sorted: %s before %s", dir, list2[i].Name(), list2[i+1].Name())
			}
		}
	}

	// Check fs.ReadDir as well.
	list2, err = fs.ReadDir(t.fsys, dir)
	if err != nil {
		t.errorf("%s: fs.ReadDir: %v", dir, err)
		return
	}
	t.checkDirList(dir, "first Open+ReadDir(-1) vs fs.ReadDir", list, list2)

	for i := 0; i+1 < len(list2); i++ {
		if list2[i].Name() >= list2[i+1].Name() {
			t.errorf("%s: fs.ReadDir: list not sorted: %s before %s", dir, list2[i].Name(), list2[i+1].Name())
		}
	}

	t.checkGlob(dir, list2)
}

// formatEntry formats an fs.DirEntry into a string for error messages and comparison.
func formatEntry(entry fs.DirEntry) string {
	return fmt.Sprintf("%s IsDir=%v Type=%v", entry.Name(), entry.IsDir(), entry.Type())
}

// formatInfoEntry formats an fs.FileInfo into a string like the result of formatEntry, for error messages and comparison.
func formatInfoEntry(info fs.FileInfo) string {
	return fmt.Sprintf("%s IsDir=%v Type=%v", info.Name(), info.IsDir(), info.Mode().Type())
}

// formatInfo formats an fs.FileInfo into a string for error messages and comparison.
func formatInfo(info fs.FileInfo) string {
	return fmt.Sprintf("%s IsDir=%v Mode=%v Size=%d ModTime=%v", info.Name(), info.IsDir(), info.Mode(), info.Size(), info.ModTime())
}

// checkGlob checks that various glob patterns work if the file system implements GlobFS.
func (t *fsTester) checkGlob(dir string, list []fs.DirEntry) {
	if _, ok := t.fsys.(fs.GlobFS); !ok {
		return
	}

	// Make a complex glob pattern prefix that only matches dir.
	var glob string
	if dir != "." {
		elem := strings.Split(dir, "/")
		for i, e := range elem {
			var pattern []rune
			for j, r := range e {
				if r == '*' || r == '?' || r == '\\' || r == '[' || r == '-' {
					pattern = append(pattern, '\\', r)
					continue
				}
				switch (i + j) % 5 {
				case 0:
					pattern = append(pattern, r)
				case 1:
					pattern = append(pattern, '[', r, ']')
				case 2:
					pattern = append(pattern, '[', r, '-', r, ']')
				case 3:
					pattern = append(pattern, '[', '\\', r, ']')
				case 4:
					pattern = append(pattern, '[', '\\', r, '-', '\\', r, ']')
				}
			}
			elem[i] = string(pattern)
		}
		glob = strings.Join(elem, "/") + "/"
	}

	// Test that malformed patterns are detected.
	// The error is likely path.ErrBadPattern but need not be.
	if _, err := t.fsys.(fs.GlobFS).Glob(glob + "nonexist/[]"); err == nil {
		t.errorf("%s: Glob(%#q): bad pattern not detected", dir, glob+"nonexist/[]")
	}

	// Try to find a letter that appears in only some of the final names.
	c := rune('a')
	for ; c <= 'z'; c++ {
		have, haveNot := false, false
		for _, d := range list {
			if strings.ContainsRune(d.Name(), c) {
				have = true
			} else {
				haveNot = true
			}
		}
		if have && haveNot {
			break
		}
	}
	if c > 'z' {
		c = 'a'
	}
	glob += "*" + string(c) + "*"

	var want []string
	for _, d := range list {
		if strings.ContainsRune(d.Name(), c) {
			want = append(want, path.Join(dir, d.Name()))
		}
	}

	names, err := t.fsys.(fs.GlobFS).Glob(glob)
	if err != nil {
		t.errorf("%s: Glob(%#q): %v", dir, glob, err)
		return
	}
	if reflect.DeepEqual(want, names) {
		return
	}

	if !sort.StringsAreSorted(names) {
		t.errorf("%s: Glob(%#q): unsorted output:\n%s", dir, glob, strings.Join(names, "\n"))
		sort.Strings(names)
	}

	var problems []string
	for len(want) > 0 || len(names) > 0 {
		switch {
		case len(want) > 0 && len(names) > 0 && want[0] == names[0]:
			want, names = want[1:], names[1:]
		case len(want) > 0 && (len(names) == 0 || want[0] < names[0]):
			problems = append(problems, "missing: "+want[0])
			want = want[1:]
		default:
			problems = append(problems, "extra: "+names[0])
			names = names[1:]
		}
	}
	t.errorf("%s: Glob(%#q): wrong output:\n%s", dir, glob, strings.Join(problems, "\n"))
}

// checkStat checks that a direct stat of path matches entry,
// which was found in the parent's directory listing.
func (t *fsTester) checkStat(path string, entry fs.DirEntry) {
	file, err := t.fsys.Open(path)
	if err != nil {
		t.errorf("%s: Open: %v", path, err)
		return
	}
	info, err := file.Stat()
	file.Close()
	if err != nil {
		t.errorf("%s: Stat: %v", path, err)
		return
	}
	fentry := formatEntry(entry)
	fientry := formatInfoEntry(info)
	// Note: mismatch here is OK for symlink, because Open dereferences symlink.
	if fentry != fientry && entry.Type()&fs.ModeSymlink == 0 {
		t.errorf("%s: mismatch:\n\tentry = %s\n\tfile.Stat() = %s", path, fentry, fientry)
	}

	einfo, err := entry.Info()
	if err != nil {
		t.errorf("%s: entry.Info: %v", path, err)
		return
	}
	finfo := formatInfo(info)
	if entry.Type()&fs.ModeSymlink != 0 {
		// For symlink, just check that entry.Info matches entry on common fields.
		// Open deferences symlink, so info itself may differ.
		feentry := formatInfoEntry(einfo)
		if fentry != feentry {
			t.errorf("%s: mismatch\n\tentry = %s\n\tentry.Info() = %s\n", path, fentry, feentry)
		}
	} else {
		feinfo := formatInfo(einfo)
		if feinfo != finfo {
			t.errorf("%s: mismatch:\n\tentry.Info() = %s\n\tfile.Stat() = %s\n", path, feinfo, finfo)
		}
	}

	// Stat should be the same as Open+Stat, even for symlinks.
	info2, err := fs.Stat(t.fsys, path)
	if err != nil {
		t.errorf("%s: fs.Stat: %v", path, err)
		return
	}
	finfo2 := formatInfo(info2)
	if finfo2 != finfo {
		t.errorf("%s: fs.Stat(...) = %s\n\twant %s", path, finfo2, finfo)
	}

	if fsys, ok := t.fsys.(fs.StatFS); ok {
		info2, err := fsys.Stat(path)
		if err != nil {
			t.errorf("%s: fsys.Stat: %v", path, err)
			return
		}
		finfo2 := formatInfo(info2)
		if finfo2 != finfo {
			t.errorf("%s: fsys.Stat(...) = %s\n\twant %s", path, finfo2, finfo)
		}
	}
}

// checkDirList checks that two directory lists contain the same files and file info.
// The order of the lists need not match.
func (t *fsTester) checkDirList(dir, desc string, list1, list2 []fs.DirEntry) {
	old := make(map[string]fs.DirEntry)
	checkMode := func(entry fs.DirEntry) {
		if entry.IsDir() != (entry.Type()&fs.ModeDir != 0) {
			if entry.IsDir() {
				t.errorf("%s: ReadDir returned %s with IsDir() = true, Type() & ModeDir = 0", dir, entry.Name())
			} else {
				t.errorf("%s: ReadDir returned %s with IsDir() = false, Type() & ModeDir = ModeDir", dir, entry.Name())
			}
		}
	}

	for _, entry1 := range list1 {
		old[entry1.Name()] = entry1
		checkMode(entry1)
	}

	var diffs []string
	for _, entry2 := range list2 {
		entry1 := old[entry2.Name()]
		if entry1 == nil {
			checkMode(entry2)
			diffs = append(diffs, "+ "+formatEntry(entry2))
			continue
		}
		if formatEntry(entry1) != formatEntry(entry2) {
			diffs = append(diffs, "- "+formatEntry(entry1), "+ "+formatEntry(entry2))
		}
		delete(old, entry2.Name())
	}
	for _, entry1 := range old {
		diffs = append(diffs, "- "+formatEntry(entry1))
	}

	if len(diffs) == 0 {
		return
	}

	sort.Slice(diffs, func(i, j int) bool {
		fi := strings.Fields(diffs[i])
		fj := strings.Fields(diffs[j])
		// sort by name (i < j) and then +/- (j < i, because + < -)
		return fi[1]+" "+fj[0] < fj[1]+" "+fi[0]
	})

	t.errorf("%s: diff %s:\n\t%s", dir, desc, strings.Join(diffs, "\n\t"))
}

// checkFile checks that basic file reading works correctly.
func (t *fsTester) checkFile(file string) {
	t.files = append(t.files, file)

	// Read entire file.
	f, err := t.fsys.Open(file)
	if err != nil {
		t.errorf("%s: Open: %v", file, err)
		return
	}

	data, err := io.ReadAll(f)
	if err != nil {
		f.Close()
		t.errorf("%s: Open+ReadAll: %v", file, err)
		return
	}

	if err := f.Close(); err != nil {
		t.errorf("%s: Close: %v", file, err)
	}

	// Check that closing twice doesn't crash.
	// The return value doesn't matter.
	f.Close()

	// Check that ReadFile works if present.
	if fsys, ok := t.fsys.(fs.ReadFileFS); ok {
		data2, err := fsys.ReadFile(file)
		if err != nil {
			t.errorf("%s: fsys.ReadFile: %v", file, err)
			return
		}
		t.checkFileRead(file, "ReadAll vs fsys.ReadFile", data, data2)

		// Modify the data and check it again. Modifying the
		// returned byte slice should not affect the next call.
		for i := range data2 {
			data2[i]++
		}
		data2, err = fsys.ReadFile(file)
		if err != nil {
			t.errorf("%s: second call to fsys.ReadFile: %v", file, err)
			return
		}
		t.checkFileRead(file, "Readall vs second fsys.ReadFile", data, data2)

		t.checkBadPath(file, "ReadFile",
			func(name string) error { _, err := fsys.ReadFile(name); return err })
	}

	// Check that fs.ReadFile works with t.fsys.
	data2, err := fs.ReadFile(t.fsys, file)
	if err != nil {
		t.errorf("%s: fs.ReadFile: %v", file, err)
		return
	}
	t.checkFileRead(file, "ReadAll vs fs.ReadFile", data, data2)

	// Use iotest.TestReader to check small reads, Seek, ReadAt.
	f, err = t.fsys.Open(file)
	if err != nil {
		t.errorf("%s: second Open: %v", file, err)
		return
	}
	defer f.Close()
	if err := iotest.TestReader(f, data); err != nil {
		t.errorf("%s: failed TestReader:\n\t%s", file, strings.ReplaceAll(err.Error(), "\n", "\n\t"))
	}
}

func (t *fsTester) checkFileRead(file, desc string, data1, data2 []byte) {
	if string(data1) != string(data2) {
		t.errorf("%s: %s: different data returned\n\t%q\n\t%q", file, desc, data1, data2)
		return
	}
}

// checkBadPath checks that various invalid forms of file's name cannot be opened using t.fsys.Open.
func (t *fsTester) checkOpen(file string) {
	t.checkBadPath(file, "Open", func(file string) error {
		f, err := t.fsys.Open(file)
		if err == nil {
			f.Close()
		}
		return err
	})
}

// checkBadPath checks that various invalid forms of file's name cannot be opened using open.
func (t *fsTester) checkBadPath(file string, desc string, open func(string) error) {
	bad := []string{
		"/" + file,
		file + "/.",
	}
	if file == "." {
		bad = append(bad, "/")
	}
	if i := strings.Index(file, "/"); i >= 0 {
		bad = append(bad,
			file[:i]+"//"+file[i+1:],
			file[:i]+"/./"+file[i+1:],
			file[:i]+`\`+file[i+1:],
			file[:i]+"/../"+file,
		)
	}
	if i := strings.LastIndex(file, "/"); i >= 0 {
		bad = append(bad,
			file[:i]+"//"+file[i+1:],
			file[:i]+"/./"+file[i+1:],
			file[:i]+`\`+file[i+1:],
			file+"/../"+file[i+1:],
		)
	}

	for _, b := range bad {
		if err := open(b); err == nil {
			t.errorf("%s: %s(%s) succeeded, want error", file, desc, b)
		}
	}
}
