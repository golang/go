// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fstest

import (
	"errors"
	"io"
	"io/fs"
	"path"
	"slices"
	"strings"
	"time"
)

// A MapFS is a simple in-memory file system for use in tests,
// represented as a map from path names (arguments to Open)
// to information about the files, directories, or symbolic links they represent.
//
// The map need not include parent directories for files contained
// in the map; those will be synthesized if needed.
// But a directory can still be included by setting the [MapFile.Mode]'s [fs.ModeDir] bit;
// this may be necessary for detailed control over the directory's [fs.FileInfo]
// or to create an empty directory.
//
// File system operations read directly from the map,
// so that the file system can be changed by editing the map as needed.
// An implication is that file system operations must not run concurrently
// with changes to the map, which would be a race.
// Another implication is that opening or reading a directory requires
// iterating over the entire map, so a MapFS should typically be used with not more
// than a few hundred entries or directory reads.
type MapFS map[string]*MapFile

// A MapFile describes a single file in a [MapFS].
type MapFile struct {
	Data    []byte      // file content or symlink destination
	Mode    fs.FileMode // fs.FileInfo.Mode
	ModTime time.Time   // fs.FileInfo.ModTime
	Sys     any         // fs.FileInfo.Sys
}

var (
	_ fs.FS           = MapFS(nil)
	_ fs.GlobFS       = MapFS(nil)
	_ fs.MkdirFS      = MapFS(nil)
	_ fs.OpenFileFS   = MapFS(nil)
	_ fs.PropertiesFS = MapFS(nil)
	_ fs.ReadDirFS    = MapFS(nil)
	_ fs.ReadFileFS   = MapFS(nil)
	_ fs.ReadLinkFS   = MapFS(nil)
	_ fs.RemoveFS     = MapFS(nil)
	_ fs.StatFS       = MapFS(nil)
	_ fs.SubFS        = MapFS(nil)
	_ fs.SymlinkFS    = MapFS(nil)

	_ fs.File          = (*openMapFile)(nil)
	_ fs.WriterFile    = (*openMapFile)(nil)
)

// Open opens the named file after following any symbolic links.
func (fsys MapFS) Open(name string) (fs.File, error) {
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}
	realName, ok := fsys.resolveSymlinks(name)
	if !ok {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}

	file := fsys[realName]
	if file != nil && file.Mode&fs.ModeDir == 0 {
		// Ordinary file
		return &openMapFile{path: name, mapFileInfo: mapFileInfo{path.Base(name), file}, offset: 0, fsys: fsys, realNameInMap: realName, flag: fs.O_RDONLY}, nil
	}

	// Directory, possibly synthesized.
	// Note that file can be nil here: the map need not contain explicit parent directories for all its files.
	// But file can also be non-nil, in case the user wants to set metadata for the directory explicitly.
	// Either way, we need to construct the list of children of this directory.
	var list []mapFileInfo
	var need = make(map[string]bool)
	if realName == "." {
		for fname, f := range fsys {
			i := strings.Index(fname, "/")
			if i < 0 {
				if fname != "." {
					list = append(list, mapFileInfo{fname, f})
				}
			} else {
				need[fname[:i]] = true
			}
		}
	} else {
		prefix := realName + "/"
		for fname, f := range fsys {
			if strings.HasPrefix(fname, prefix) {
				felem := fname[len(prefix):]
				i := strings.Index(felem, "/")
				if i < 0 {
					list = append(list, mapFileInfo{felem, f})
				} else {
					need[fname[len(prefix):len(prefix)+i]] = true
				}
			}
		}
		// If the directory name is not in the map,
		// and there are no children of the name in the map,
		// then the directory is treated as not existing.
		if file == nil && list == nil && len(need) == 0 {
			return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
		}
	}
	for _, fi := range list {
		delete(need, fi.name)
	}
	for name := range need {
		list = append(list, mapFileInfo{name, &MapFile{Mode: fs.ModeDir | 0555}})
	}
	slices.SortFunc(list, func(a, b mapFileInfo) int {
		return strings.Compare(a.name, b.name)
	})

	if file == nil {
		file = &MapFile{Mode: fs.ModeDir | 0555}
	}
	var elem string
	if name == "." {
		elem = "."
	} else {
		elem = name[strings.LastIndex(name, "/")+1:]
	}
	return &mapDir{name, mapFileInfo{elem, file}, list, 0}, nil
}

// OpenFile is the generalized open call; most users will use Open or Create instead.
// It opens the named file with specified flag (O_RDONLY etc.).
// If the file does not exist, and the O_CREATE flag is passed, it is created with mode perm (before umask).
// MapFS's OpenFile follows symlinks.
func (fsys MapFS) OpenFile(name string, flag int, perm fs.FileMode) (fs.WriterFile, error) {
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "openfile", Path: name, Err: fs.ErrNotExist}
	}

	realName, ok := fsys.resolveSymlinks(name)
	if !ok {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}

	if flag&fs.O_CREATE != 0 && fsys[realName] == nil {
		fsys[realName] = &MapFile{
			Mode:    perm,
			ModTime: time.Now(),
		}
	}

	if flag&(fs.O_RDWR|fs.O_APPEND|fs.O_CREATE|fs.O_TRUNC) != 0 {
		return &openMapFile{
			path:          name,
			offset:        0,
			fsys:          fsys,
			realNameInMap: realName,
			flag:          flag,
			mapFileInfo: mapFileInfo{
				name: path.Base(name),
				f:    fsys[realName],
			},
		}, nil
	}

	// Fallback to basic Open logic for O_RDONLY
	file, err := fsys.Open(name)
	if err != nil {
		return nil, err
	}
	return file.(fs.WriterFile), nil
}

func (fsys MapFS) Create(name string) (fs.WriterFile, error) {
	return fsys.OpenFile(name, fs.O_CREATE|fs.O_TRUNC|fs.O_WRONLY, 0666)
}

func (fsys MapFS) resolveSymlinks(name string) (_ string, ok bool) {
	// Fast path: if a symlink is in the map, resolve it.
	if file := fsys[name]; file != nil && file.Mode.Type() == fs.ModeSymlink {
		target := string(file.Data)
		if path.IsAbs(target) {
			return "", false
		}
		return fsys.resolveSymlinks(path.Join(path.Dir(name), target))
	}

	// Check if each parent directory (starting at root) is a symlink.
	for i := 0; i < len(name); {
		j := strings.Index(name[i:], "/")
		var dir string
		if j < 0 {
			dir = name
			i = len(name)
		} else {
			dir = name[:i+j]
			i += j
		}
		if file := fsys[dir]; file != nil && file.Mode.Type() == fs.ModeSymlink {
			target := string(file.Data)
			if path.IsAbs(target) {
				return "", false
			}
			return fsys.resolveSymlinks(path.Join(path.Dir(dir), target) + name[i:])
		}
		i += len("/")
	}
	return name, fs.ValidPath(name)
}

// ReadLink returns the destination of the named symbolic link.
func (fsys MapFS) ReadLink(name string) (string, error) {
	info, err := fsys.lstat(name)
	if err != nil {
		return "", &fs.PathError{Op: "readlink", Path: name, Err: err}
	}
	if info.f.Mode.Type() != fs.ModeSymlink {
		return "", &fs.PathError{Op: "readlink", Path: name, Err: fs.ErrInvalid}
	}
	return string(info.f.Data), nil
}

// Lstat returns a FileInfo describing the named file.
// If the file is a symbolic link, the returned FileInfo describes the symbolic link.
// Lstat makes no attempt to follow the link.
func (fsys MapFS) Lstat(name string) (fs.FileInfo, error) {
	info, err := fsys.lstat(name)
	if err != nil {
		return nil, &fs.PathError{Op: "lstat", Path: name, Err: err}
	}
	return info, nil
}

func (fsys MapFS) lstat(name string) (*mapFileInfo, error) {
	if !fs.ValidPath(name) {
		return nil, fs.ErrNotExist
	}
	realDir, ok := fsys.resolveSymlinks(path.Dir(name))
	if !ok {
		return nil, fs.ErrNotExist
	}
	elem := path.Base(name)
	realName := path.Join(realDir, elem)

	file := fsys[realName]
	if file != nil {
		return &mapFileInfo{elem, file}, nil
	}

	if realName == "." {
		return &mapFileInfo{elem, &MapFile{Mode: fs.ModeDir | 0555}}, nil
	}
	// Maybe a directory.
	prefix := realName + "/"
	for fname := range fsys {
		if strings.HasPrefix(fname, prefix) {
			return &mapFileInfo{elem, &MapFile{Mode: fs.ModeDir | 0555}}, nil
		}
	}
	// If the directory name is not in the map,
	// and there are no children of the name in the map,
	// then the directory is treated as not existing.
	return nil, fs.ErrNotExist
}

// fsOnly is a wrapper that hides all but the fs.FS methods,
// to avoid an infinite recursion when implementing special
// methods in terms of helpers that would use them.
// (In general, implementing these methods using the package fs helpers
// is redundant and unnecessary, but having the methods may make
// MapFS exercise more code paths when used in tests.)
type fsOnly struct{ fs.FS }

func (fsys MapFS) ReadFile(name string) ([]byte, error) {
	return fs.ReadFile(fsOnly{fsys}, name)
}

func (fsys MapFS) Stat(name string) (fs.FileInfo, error) {
	return fs.Stat(fsOnly{fsys}, name)
}

func (fsys MapFS) ReadDir(name string) ([]fs.DirEntry, error) {
	return fs.ReadDir(fsOnly{fsys}, name)
}

func (fsys MapFS) Glob(pattern string) ([]string, error) {
	return fs.Glob(fsOnly{fsys}, pattern)
}

type noSub struct {
	MapFS
}

func (noSub) Sub() {} // not the fs.SubFS signature

func (fsys MapFS) Sub(dir string) (fs.FS, error) {
	return fs.Sub(noSub{fsys}, dir)
}

// Chmod changes the mode of the named file to mode.
// If the file is a symbolic link, it changes the mode of the link's target.
func (fsys MapFS) Chmod(name string, mode fs.FileMode) error {
	if !fs.ValidPath(name) {
		return &fs.PathError{Op: "chmod", Path: name, Err: fs.ErrNotExist}
	}
	realName, ok := fsys.resolveSymlinks(name)
	if !ok {
		return &fs.PathError{Op: "chmod", Path: name, Err: fs.ErrNotExist}
	}

	entry := fsys[realName]
	if entry == nil {
		// Check if it's a synthesized directory that's not explicitly in the map
		isDir := false
		if realName == "." {
			isDir = true
		} else {
			prefix := realName + "/"
			for fname := range fsys {
				if strings.HasPrefix(fname, prefix) {
					isDir = true
					break
				}
			}
		}
		if !isDir {
			return &fs.PathError{Op: "chmod", Path: name, Err: fs.ErrNotExist}
		}
		// Cannot chmod a synthesized directory not in the map.
		// For consistency, one might add it to the map first, or disallow.
		// For now, let's say it needs to be in the map to be chmod-ed.
		return &fs.PathError{Op: "chmod", Path: name, Err: fs.ErrNotExist}
	}

	entry.Mode = (entry.Mode &^ fs.ModePerm) | (mode & fs.ModePerm) // Keep type bits
	entry.ModTime = time.Now()
	return nil
}

// Chown changes the numeric uid and gid of the named file.
// MapFS does not support ownership, so this returns fs.ErrPermission.
func (fsys MapFS) Chown(name string, uid, gid int) error {
	// To "succeed" silently:
	// _, err := fsys.Stat(name) // Check existence
	// return err
	return &fs.PathError{Op: "chown", Path: name, Err: fs.ErrPermission}
}

// Chtimes changes the access and modification times of the named file.
// MapFS only stores ModTime, so atime is ignored.
// If the file is a symbolic link, it changes the times of the link's target.
func (fsys MapFS) Chtimes(name string, atime time.Time, mtime time.Time) error {
	if !fs.ValidPath(name) {
		return &fs.PathError{Op: "chtimes", Path: name, Err: fs.ErrNotExist}
	}
	realName, ok := fsys.resolveSymlinks(name)
	if !ok {
		return &fs.PathError{Op: "chtimes", Path: name, Err: fs.ErrNotExist}
	}

	entry := fsys[realName]
	if entry == nil {
		// Similar to Chmod, check for synthesized directory
		isDir := false
		if realName == "." {
			isDir = true
		} else {
			prefix := realName + "/"
			for fname := range fsys {
				if strings.HasPrefix(fname, prefix) {
					isDir = true
					break
				}
			}
		}
		if !isDir {
			return &fs.PathError{Op: "chtimes", Path: name, Err: fs.ErrNotExist}
		}
		// Cannot chtimes a synthesized directory not in the map.
		return &fs.PathError{Op: "chtimes", Path: name, Err: fs.ErrNotExist}
	}

	entry.ModTime = mtime
	return nil
}

// Link creates newname as a hard link to the oldname file.
// MapFS does not support hard links, so this returns fs.ErrPermission.
func (fsys MapFS) Link(oldname, newname string) error {
	return &fs.PathError{Op: "link", Path: newname, Err: fs.ErrPermission}
}

// Mkdir creates a new directory with the specified name and permission bits.
func (fsys MapFS) Mkdir(name string, perm fs.FileMode) error {
	if !fs.ValidPath(name) || name == "." {
		return &fs.PathError{Op: "mkdir", Path: name, Err: fs.ErrInvalid}
	}

	dir := path.Dir(name)
	base := path.Base(name)

	resolvedDir, ok := fsys.resolveSymlinks(dir)
	if !ok {
		return &fs.PathError{Op: "mkdir", Path: name, Err: fs.ErrNotExist} // Parent path issue
	}

	parentEntry := fsys[resolvedDir]
	if resolvedDir != "." && (parentEntry == nil || !parentEntry.Mode.IsDir()) {
		// Check if parent is a synthesized directory
		isSynthesizedDir := false
		prefix := resolvedDir + "/"
		for fname := range fsys {
			if strings.HasPrefix(fname, prefix) {
				isSynthesizedDir = true
				break
			}
		}
		if !isSynthesizedDir {
			return &fs.PathError{Op: "mkdir", Path: name, Err: fs.ErrNotExist} // Parent does not exist or is not a dir
		}
	}

	targetPath := path.Join(resolvedDir, base)
	if _, err := fsys.lstat(targetPath); err == nil { // Use lstat to check direct existence
		return &fs.PathError{Op: "mkdir", Path: name, Err: fs.ErrExist}
	}

	fsys[targetPath] = &MapFile{
		Mode:    fs.ModeDir | (perm & fs.ModePerm),
		ModTime: time.Now(),
	}
	return nil
}

// MkdirAll creates a directory named path,
// along with any necessary parents, and returns nil,
// or else returns an error. The permission bits perm (before umask)
// are used for all directories that MkdirAll creates.
// If path is already a directory, MkdirAll does nothing and returns nil.
func (fsys MapFS) MkdirAll(name string, perm fs.FileMode) error {
	// This is a simplified version. A full version would be more careful
	// about existing non-directory paths.
	if !fs.ValidPath(name) {
		return &fs.PathError{Op: "mkdirall", Path: name, Err: fs.ErrInvalid}
	}

	currentPath := ""
	parts := strings.Split(name, "/")
	if name == "." {
		return nil
	}
	if len(parts) == 0 {
		return &fs.PathError{Op: "mkdirall", Path: name, Err: fs.ErrInvalid}
	}

	for i, part := range parts {
		if part == "" && i == 0 { // Absolute path, root always exists conceptually
			currentPath = "" // Keep it relative for MapFS keys unless MapFS handles absolute paths
			continue
		}
		if currentPath == "" {
			currentPath = part
		} else {
			currentPath = path.Join(currentPath, part)
		}

		fi, err := fsys.Lstat(currentPath) // Lstat to not follow last component if symlink
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) || (err.(*fs.PathError) != nil && errors.Is(err.(*fs.PathError).Err, fs.ErrNotExist)) {
				errMkdir := fsys.Mkdir(currentPath, perm)
				if errMkdir != nil {
					// Mkdir might fail if a parent component was just created as a file by a concurrent op (not an issue for typical MapFS use)
					// or if currentPath itself is invalid (e.g. ".." at root)
					return &fs.PathError{Op: "mkdirall", Path: currentPath, Err: errMkdir}
				}
			} else {
				return err // Other error from Lstat
			}
		} else if !fi.IsDir() {
			return &fs.PathError{Op: "mkdirall", Path: currentPath, Err: fs.ErrExist} // Exists but not a directory
		}
	}
	return nil
}

// A mapFileInfo implements fs.FileInfo and fs.DirEntry for a given map file.
type mapFileInfo struct {
	name string
	f    *MapFile
}

func (i *mapFileInfo) Name() string               { return path.Base(i.name) }
func (i *mapFileInfo) Size() int64                { return int64(len(i.f.Data)) }
func (i *mapFileInfo) Mode() fs.FileMode          { return i.f.Mode }
func (i *mapFileInfo) Type() fs.FileMode          { return i.f.Mode.Type() }
func (i *mapFileInfo) ModTime() time.Time         { return i.f.ModTime }
func (i *mapFileInfo) IsDir() bool                { return i.f.Mode&fs.ModeDir != 0 }
func (i *mapFileInfo) Sys() any                   { return i.f.Sys }
func (i *mapFileInfo) Info() (fs.FileInfo, error) { return i, nil }

func (i *mapFileInfo) String() string {
	return fs.FormatFileInfo(i)
}

// An openMapFile is a regular (non-directory) fs.File open for reading.
type openMapFile struct {
	mapFileInfo
	path          string
	offset        int64
	fsys          MapFS  // Reference to parent MapFS for modifications
	realNameInMap string // The actual key in the MapFS map
	flag          int    // Open flags
}

func (f *openMapFile) Stat() (fs.FileInfo, error) { return &f.mapFileInfo, nil }

func (f *openMapFile) Close() error { return nil }

func (f *openMapFile) Read(b []byte) (int, error) {
	if f.offset >= int64(len(f.f.Data)) {
		return 0, io.EOF
	}
	if f.offset < 0 {
		return 0, &fs.PathError{Op: "read", Path: f.path, Err: fs.ErrInvalid}
	}
	n := copy(b, f.f.Data[f.offset:])
	f.offset += int64(n)
	return n, nil
}

func (f *openMapFile) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case 0:
		// offset += 0
	case 1:
		offset += f.offset
	case 2:
		offset += int64(len(f.f.Data))
	}
	if offset < 0 || offset > int64(len(f.f.Data)) {
		return 0, &fs.PathError{Op: "seek", Path: f.path, Err: fs.ErrInvalid}
	}
	f.offset = offset
	return offset, nil
}

func (f *openMapFile) ReadAt(b []byte, offset int64) (int, error) {
	if offset < 0 || offset > int64(len(f.f.Data)) {
		return 0, &fs.PathError{Op: "read", Path: f.path, Err: fs.ErrInvalid}
	}
	n := copy(b, f.f.Data[offset:])
	if n < len(b) {
		return n, io.EOF
	}
	return n, nil
}

// Write writes len(b) bytes to the File.
// It returns the number of bytes written and an error, if any.
// Write returns a non-nil error when n != len(b).
func (f *openMapFile) Write(b []byte) (int, error) {
	if f.flag&fs.O_WRONLY == 0 && f.flag&fs.O_RDWR == 0 {
		return 0, &fs.PathError{Op: "write", Path: f.path, Err: fs.ErrPermission}
	}

	mapEntry := f.fsys[f.realNameInMap]
	if mapEntry == nil { // Should not happen if open succeeded and file wasn't deleted externally
		return 0, &fs.PathError{Op: "write", Path: f.path, Err: fs.ErrNotExist}
	} else if mapEntry.Mode.IsDir() {
		return 0, &fs.PathError{Op: "write", Path: f.path, Err: io.ErrUnexpectedEOF}
	}

	if f.flag&fs.O_APPEND != 0 {
		f.offset = int64(len(mapEntry.Data))
	}

	if f.offset < 0 {
		return 0, &fs.PathError{Op: "write", Path: f.path, Err: fs.ErrInvalid}
	}

	n := len(b)
	endOffset := f.offset + int64(n)
	if endOffset > int64(cap(mapEntry.Data)) {
		newData := make([]byte, len(mapEntry.Data), endOffset)
		copy(newData, mapEntry.Data)
		mapEntry.Data = newData
	}

	if endOffset > int64(len(mapEntry.Data)) {
		mapEntry.Data = mapEntry.Data[:endOffset]
	}

	copy(mapEntry.Data[f.offset:], b)
	f.offset = endOffset
	mapEntry.ModTime = time.Now()
	// f.mapFileInfo.f is already pointing to mapEntry due to how it's constructed
	// or should be if OpenFile is fully implemented.
	// For safety, ensure the embedded FileInfo's file pointer is the live one.
	f.mapFileInfo.f = mapEntry
	return n, nil
}

// WriteAt writes len(b) bytes to the File starting at byte offset off.
// It returns the number of bytes written and an error, if any.
// WriteAt returns a non-nil error when n != len(b).
func (f *openMapFile) WriteAt(b []byte, off int64) (int, error) {
	if f.flag&fs.O_WRONLY == 0 && f.flag&fs.O_RDWR == 0 {
		return 0, &fs.PathError{Op: "writeat", Path: f.path, Err: fs.ErrPermission}
	}
	if off < 0 {
		return 0, &fs.PathError{Op: "writeat", Path: f.path, Err: fs.ErrInvalid}
	}

	mapEntry := f.fsys[f.realNameInMap]
	if mapEntry == nil {
		return 0, &fs.PathError{Op: "writeat", Path: f.path, Err: fs.ErrNotExist}
	}
	if mapEntry.Mode.IsDir() {
		return 0, &fs.PathError{Op: "writeat", Path: f.path, Err: errors.New("is a directory")}
	}

	// Save current offset, write, then restore. WriteAt does not update current offset.
	// However, the implementation here will modify the underlying data.
	// For simplicity, we'll just use the provided offset 'off'.

	n := len(b)
	neededLen := off + int64(n)

	// Similar slice growth logic as Write
	if neededLen > int64(cap(mapEntry.Data)) {
		// Simplified growth, production code might be more nuanced
		newData := make([]byte, neededLen)
		copy(newData, mapEntry.Data)
		mapEntry.Data = newData
	} else if neededLen > int64(len(mapEntry.Data)) {
		mapEntry.Data = mapEntry.Data[:neededLen]
	}

	copy(mapEntry.Data[off:], b)
	mapEntry.ModTime = time.Now()
	f.mapFileInfo.f = mapEntry
	return n, nil
}

// A mapDir is a directory fs.File (so also an fs.ReadDirFile) open for reading.
type mapDir struct {
	path string
	mapFileInfo
	entry  []mapFileInfo
	offset int
}

func (d *mapDir) Stat() (fs.FileInfo, error) { return &d.mapFileInfo, nil }
func (d *mapDir) Close() error               { return nil }
func (d *mapDir) Read(b []byte) (int, error) {
	return 0, &fs.PathError{Op: "read", Path: d.path, Err: fs.ErrInvalid}
}

func (d *mapDir) ReadDir(count int) ([]fs.DirEntry, error) {
	n := len(d.entry) - d.offset
	if n == 0 && count > 0 {
		return nil, io.EOF
	}
	if count > 0 && n > count {
		n = count
	}
	list := make([]fs.DirEntry, n)
	for i := range list {
		list[i] = &d.entry[d.offset+i]
	}
	d.offset += n
	return list, nil
}

func (fsys MapFS) Rename(oldname, newname string) error {
	if !fs.ValidPath(oldname) || !fs.ValidPath(newname) {
		return &fs.LinkError{Op: "rename", Old: oldname, New: newname, Err: fs.ErrInvalid}
	} else if oldname == "." || oldname == newname {
		return nil
	} else if _, exists := fsys[newname]; exists {
		return &fs.LinkError{Op: "rename", Old: oldname, New: newname, Err: fs.ErrExist}
	} else if _, exists := fsys[oldname]; !exists {
		return &fs.LinkError{Op: "rename", Old: oldname, New: newname, Err: fs.ErrNotExist}
	}

	fsys[newname] = fsys[oldname]
	delete(fsys, oldname)
	return nil
}

// Remove removes the named file or (empty) directory.
// If name is a symlink, Remove removes the symlink itself, not its target.
func (fsys MapFS) Remove(name string) error {
	if !fs.ValidPath(name) {
		return &fs.PathError{Op: "remove", Path: name, Err: fs.ErrNotExist}
	}
	if name == "." {
		return &fs.PathError{Op: "remove", Path: name, Err: fs.ErrInvalid} // Cannot remove root
	}

	// Determine the actual map key without following the final component if it's a symlink.
	dir := path.Dir(name)
	base := path.Base(name)
	resolvedDir, ok := fsys.resolveSymlinks(dir)
	if !ok {
		return &fs.PathError{Op: "remove", Path: name, Err: fs.ErrNotExist} // Parent path issue
	}
	mapKey := path.Join(resolvedDir, base)

	entry, exists := fsys[mapKey]
	if !exists {
		// Check if it's a synthesized directory that's not explicitly in the map but has children
		isSynthesizedDirWithChildren := false
		prefix := mapKey + "/"
		for fname := range fsys {
			if strings.HasPrefix(fname, prefix) {
				isSynthesizedDirWithChildren = true
				break
			}
		}
		if isSynthesizedDirWithChildren { // Trying to remove a non-empty synthesized directory
			return &fs.PathError{Op: "remove", Path: name, Err: errors.New("directory not empty")} // Or fs.ErrInvalid
		}
		return &fs.PathError{Op: "remove", Path: name, Err: fs.ErrNotExist}
	}

	if entry.Mode.IsDir() && entry.Mode.Type() != fs.ModeSymlink { // Is a directory, not a symlink to a directory
		// Check if directory is empty
		prefix := mapKey + "/"
		if mapKey == "." { // Root directory special case
			prefix = "" // Check all non-rooted paths
		}
		for k := range fsys {
			if k != mapKey && ((mapKey == "." && strings.IndexByte(k, '/') == -1 && k != ".") || strings.HasPrefix(k, prefix)) {
				return &fs.PathError{Op: "remove", Path: name, Err: errors.New("directory not empty")} // Or fs.ErrInvalid
			}
		}
	}

	delete(fsys, mapKey)
	return nil
}

// RemoveAll removes path and any children it contains.
// It removes everything it can but returns the first error it encounters.
// If the path does not exist, RemoveAll returns nil (no error).
// If path is a symlink, it removes the symlink.
func (fsys MapFS) RemoveAll(name string) error {
	if !fs.ValidPath(name) {
		// fs.RemoveAll returns nil if path doesn't exist due to invalid chars.
		// To be safe, let's error here or ensure lstat handles it.
		return nil // Or &fs.PathError{Op: "removeall", Path: name, Err: fs.ErrInvalid}
	}

	dir := path.Dir(name)
	base := path.Base(name)
	resolvedDir, ok := fsys.resolveSymlinks(dir)
	if !ok { // Parent path issue, implies 'name' effectively doesn't exist in a valid tree
		return nil
	}
	mapKey := path.Join(resolvedDir, base)

	if _, exists := fsys[mapKey]; !exists && mapKey != "." { // If not in map and not root, check if it's a prefix for others
		// If it's not in map, it might be a prefix of other files.
		// fs.RemoveAll would succeed if 'name' doesn't exist.
	}

	delete(fsys, mapKey) // Delete the entry itself if it exists

	// Delete children if it was a directory or could have been a prefix
	prefix := mapKey + "/"
	if mapKey == "." { // Removing root means clearing the FS
		for k := range fsys {
			delete(fsys, k)
		}
		return nil
	}

	for k := range fsys {
		if strings.HasPrefix(k, prefix) {
			delete(fsys, k)
		}
	}
	return nil
}

// Symlink creates newname as a symbolic link to oldname.
func (fsys MapFS) Symlink(oldname, newname string) error {
	if !fs.ValidPath(newname) {
		return &fs.PathError{Op: "symlink", Path: newname, Err: fs.ErrInvalid}
	}

	dir := path.Dir(newname)
	base := path.Base(newname)

	resolvedDir, ok := fsys.resolveSymlinks(dir)
	if !ok {
		return &fs.PathError{Op: "symlink", Path: newname, Err: fs.ErrNotExist} // Parent path issue
	}

	parentEntry := fsys[resolvedDir]
	if resolvedDir != "." && (parentEntry == nil || !parentEntry.Mode.IsDir()) {
		// Check if parent is a synthesized directory
		isSynthesizedDir := false
		prefix := resolvedDir + "/"
		for fname := range fsys {
			if strings.HasPrefix(fname, prefix) {
				isSynthesizedDir = true
				break
			}
		}
		if !isSynthesizedDir {
			return &fs.PathError{Op: "symlink", Path: newname, Err: fs.ErrNotExist}
		}
	}

	targetPath := path.Join(resolvedDir, base)
	if _, err := fsys.lstat(targetPath); err == nil {
		return &fs.PathError{Op: "symlink", Path: newname, Err: fs.ErrExist}
	}

	fsys[targetPath] = &MapFile{
		Data:    []byte(oldname),
		Mode:    fs.ModeSymlink | 0777, // Typical perms for symlink
		ModTime: time.Now(),
	}
	return nil
}
