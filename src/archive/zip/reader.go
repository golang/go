// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bufio"
	"encoding/binary"
	"errors"
	"hash"
	"hash/crc32"
	"io"
	"io/fs"
	"os"
	"path"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	ErrFormat    = errors.New("zip: not a valid zip file")
	ErrAlgorithm = errors.New("zip: unsupported compression algorithm")
	ErrChecksum  = errors.New("zip: checksum error")
)

// A Reader serves content from a ZIP archive.
type Reader struct {
	r             io.ReaderAt
	File          []*File
	Comment       string
	decompressors map[uint16]Decompressor

	// fileList is a list of files sorted by ename,
	// for use by the Open method.
	fileListOnce sync.Once
	fileList     []fileListEntry
}

// A ReadCloser is a Reader that must be closed when no longer needed.
type ReadCloser struct {
	f *os.File
	Reader
}

// A File is a single file in a ZIP archive.
// The file information is in the embedded FileHeader.
// The file content can be accessed by calling Open.
type File struct {
	FileHeader
	zip          *Reader
	zipr         io.ReaderAt
	headerOffset int64
	zip64        bool  // zip64 extended information extra field presence
	descErr      error // error reading the data descriptor during init
}

// OpenReader will open the Zip file specified by name and return a ReadCloser.
func OpenReader(name string) (*ReadCloser, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	r := new(ReadCloser)
	if err := r.init(f, fi.Size()); err != nil {
		f.Close()
		return nil, err
	}
	r.f = f
	return r, nil
}

// NewReader returns a new Reader reading from r, which is assumed to
// have the given size in bytes.
func NewReader(r io.ReaderAt, size int64) (*Reader, error) {
	if size < 0 {
		return nil, errors.New("zip: size cannot be negative")
	}
	zr := new(Reader)
	if err := zr.init(r, size); err != nil {
		return nil, err
	}
	return zr, nil
}

func (z *Reader) init(r io.ReaderAt, size int64) error {
	end, err := readDirectoryEnd(r, size)
	if err != nil {
		return err
	}
	z.r = r
	// Since the number of directory records is not validated, it is not
	// safe to preallocate z.File without first checking that the specified
	// number of files is reasonable, since a malformed archive may
	// indicate it contains up to 1 << 128 - 1 files. Since each file has a
	// header which will be _at least_ 30 bytes we can safely preallocate
	// if (data size / 30) >= end.directoryRecords.
	if end.directorySize < uint64(size) && (uint64(size)-end.directorySize)/30 >= end.directoryRecords {
		z.File = make([]*File, 0, end.directoryRecords)
	}
	z.Comment = end.comment
	rs := io.NewSectionReader(r, 0, size)
	if _, err = rs.Seek(int64(end.directoryOffset), io.SeekStart); err != nil {
		return err
	}
	buf := bufio.NewReader(rs)

	// The count of files inside a zip is truncated to fit in a uint16.
	// Gloss over this by reading headers until we encounter
	// a bad one, and then only report an ErrFormat or UnexpectedEOF if
	// the file count modulo 65536 is incorrect.
	for {
		f := &File{zip: z, zipr: r}
		err = readDirectoryHeader(f, buf)
		if err == ErrFormat || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return err
		}
		z.File = append(z.File, f)
	}
	if uint16(len(z.File)) != uint16(end.directoryRecords) { // only compare 16 bits here
		// Return the readDirectoryHeader error if we read
		// the wrong number of directory entries.
		return err
	}
	return nil
}

// RegisterDecompressor registers or overrides a custom decompressor for a
// specific method ID. If a decompressor for a given method is not found,
// Reader will default to looking up the decompressor at the package level.
func (z *Reader) RegisterDecompressor(method uint16, dcomp Decompressor) {
	if z.decompressors == nil {
		z.decompressors = make(map[uint16]Decompressor)
	}
	z.decompressors[method] = dcomp
}

func (z *Reader) decompressor(method uint16) Decompressor {
	dcomp := z.decompressors[method]
	if dcomp == nil {
		dcomp = decompressor(method)
	}
	return dcomp
}

// Close closes the Zip file, rendering it unusable for I/O.
func (rc *ReadCloser) Close() error {
	return rc.f.Close()
}

// DataOffset returns the offset of the file's possibly-compressed
// data, relative to the beginning of the zip file.
//
// Most callers should instead use Open, which transparently
// decompresses data and verifies checksums.
func (f *File) DataOffset() (offset int64, err error) {
	bodyOffset, err := f.findBodyOffset()
	if err != nil {
		return
	}
	return f.headerOffset + bodyOffset, nil
}

// Open returns a ReadCloser that provides access to the File's contents.
// Multiple files may be read concurrently.
func (f *File) Open() (io.ReadCloser, error) {
	bodyOffset, err := f.findBodyOffset()
	if err != nil {
		return nil, err
	}
	size := int64(f.CompressedSize64)
	r := io.NewSectionReader(f.zipr, f.headerOffset+bodyOffset, size)
	dcomp := f.zip.decompressor(f.Method)
	if dcomp == nil {
		return nil, ErrAlgorithm
	}
	var rc io.ReadCloser = dcomp(r)
	var desr io.Reader
	if f.hasDataDescriptor() {
		desr = io.NewSectionReader(f.zipr, f.headerOffset+bodyOffset+size, dataDescriptorLen)
	}
	rc = &checksumReader{
		rc:   rc,
		hash: crc32.NewIEEE(),
		f:    f,
		desr: desr,
	}
	return rc, nil
}

// OpenRaw returns a Reader that provides access to the File's contents without
// decompression.
func (f *File) OpenRaw() (io.Reader, error) {
	bodyOffset, err := f.findBodyOffset()
	if err != nil {
		return nil, err
	}
	r := io.NewSectionReader(f.zipr, f.headerOffset+bodyOffset, int64(f.CompressedSize64))
	return r, nil
}

type checksumReader struct {
	rc    io.ReadCloser
	hash  hash.Hash32
	nread uint64 // number of bytes read so far
	f     *File
	desr  io.Reader // if non-nil, where to read the data descriptor
	err   error     // sticky error
}

func (r *checksumReader) Stat() (fs.FileInfo, error) {
	return headerFileInfo{&r.f.FileHeader}, nil
}

func (r *checksumReader) Read(b []byte) (n int, err error) {
	if r.err != nil {
		return 0, r.err
	}
	n, err = r.rc.Read(b)
	r.hash.Write(b[:n])
	r.nread += uint64(n)
	if err == nil {
		return
	}
	if err == io.EOF {
		if r.nread != r.f.UncompressedSize64 {
			return 0, io.ErrUnexpectedEOF
		}
		if r.desr != nil {
			if err1 := readDataDescriptor(r.desr, r.f); err1 != nil {
				if err1 == io.EOF {
					err = io.ErrUnexpectedEOF
				} else {
					err = err1
				}
			} else if r.hash.Sum32() != r.f.CRC32 {
				err = ErrChecksum
			}
		} else {
			// If there's not a data descriptor, we still compare
			// the CRC32 of what we've read against the file header
			// or TOC's CRC32, if it seems like it was set.
			if r.f.CRC32 != 0 && r.hash.Sum32() != r.f.CRC32 {
				err = ErrChecksum
			}
		}
	}
	r.err = err
	return
}

func (r *checksumReader) Close() error { return r.rc.Close() }

// findBodyOffset does the minimum work to verify the file has a header
// and returns the file body offset.
func (f *File) findBodyOffset() (int64, error) {
	var buf [fileHeaderLen]byte
	if _, err := f.zipr.ReadAt(buf[:], f.headerOffset); err != nil {
		return 0, err
	}
	b := readBuf(buf[:])
	if sig := b.uint32(); sig != fileHeaderSignature {
		return 0, ErrFormat
	}
	b = b[22:] // skip over most of the header
	filenameLen := int(b.uint16())
	extraLen := int(b.uint16())
	return int64(fileHeaderLen + filenameLen + extraLen), nil
}

// readDirectoryHeader attempts to read a directory header from r.
// It returns io.ErrUnexpectedEOF if it cannot read a complete header,
// and ErrFormat if it doesn't find a valid header signature.
func readDirectoryHeader(f *File, r io.Reader) error {
	var buf [directoryHeaderLen]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return err
	}
	b := readBuf(buf[:])
	if sig := b.uint32(); sig != directoryHeaderSignature {
		return ErrFormat
	}
	f.CreatorVersion = b.uint16()
	f.ReaderVersion = b.uint16()
	f.Flags = b.uint16()
	f.Method = b.uint16()
	f.ModifiedTime = b.uint16()
	f.ModifiedDate = b.uint16()
	f.CRC32 = b.uint32()
	f.CompressedSize = b.uint32()
	f.UncompressedSize = b.uint32()
	f.CompressedSize64 = uint64(f.CompressedSize)
	f.UncompressedSize64 = uint64(f.UncompressedSize)
	filenameLen := int(b.uint16())
	extraLen := int(b.uint16())
	commentLen := int(b.uint16())
	b = b[4:] // skipped start disk number and internal attributes (2x uint16)
	f.ExternalAttrs = b.uint32()
	f.headerOffset = int64(b.uint32())
	d := make([]byte, filenameLen+extraLen+commentLen)
	if _, err := io.ReadFull(r, d); err != nil {
		return err
	}
	f.Name = string(d[:filenameLen])
	f.Extra = d[filenameLen : filenameLen+extraLen]
	f.Comment = string(d[filenameLen+extraLen:])

	// Determine the character encoding.
	utf8Valid1, utf8Require1 := detectUTF8(f.Name)
	utf8Valid2, utf8Require2 := detectUTF8(f.Comment)
	switch {
	case !utf8Valid1 || !utf8Valid2:
		// Name and Comment definitely not UTF-8.
		f.NonUTF8 = true
	case !utf8Require1 && !utf8Require2:
		// Name and Comment use only single-byte runes that overlap with UTF-8.
		f.NonUTF8 = false
	default:
		// Might be UTF-8, might be some other encoding; preserve existing flag.
		// Some ZIP writers use UTF-8 encoding without setting the UTF-8 flag.
		// Since it is impossible to always distinguish valid UTF-8 from some
		// other encoding (e.g., GBK or Shift-JIS), we trust the flag.
		f.NonUTF8 = f.Flags&0x800 == 0
	}

	needUSize := f.UncompressedSize == ^uint32(0)
	needCSize := f.CompressedSize == ^uint32(0)
	needHeaderOffset := f.headerOffset == int64(^uint32(0))

	// Best effort to find what we need.
	// Other zip authors might not even follow the basic format,
	// and we'll just ignore the Extra content in that case.
	var modified time.Time
parseExtras:
	for extra := readBuf(f.Extra); len(extra) >= 4; { // need at least tag and size
		fieldTag := extra.uint16()
		fieldSize := int(extra.uint16())
		if len(extra) < fieldSize {
			break
		}
		fieldBuf := extra.sub(fieldSize)

		switch fieldTag {
		case zip64ExtraID:
			f.zip64 = true

			// update directory values from the zip64 extra block.
			// They should only be consulted if the sizes read earlier
			// are maxed out.
			// See golang.org/issue/13367.
			if needUSize {
				needUSize = false
				if len(fieldBuf) < 8 {
					return ErrFormat
				}
				f.UncompressedSize64 = fieldBuf.uint64()
			}
			if needCSize {
				needCSize = false
				if len(fieldBuf) < 8 {
					return ErrFormat
				}
				f.CompressedSize64 = fieldBuf.uint64()
			}
			if needHeaderOffset {
				needHeaderOffset = false
				if len(fieldBuf) < 8 {
					return ErrFormat
				}
				f.headerOffset = int64(fieldBuf.uint64())
			}
		case ntfsExtraID:
			if len(fieldBuf) < 4 {
				continue parseExtras
			}
			fieldBuf.uint32()        // reserved (ignored)
			for len(fieldBuf) >= 4 { // need at least tag and size
				attrTag := fieldBuf.uint16()
				attrSize := int(fieldBuf.uint16())
				if len(fieldBuf) < attrSize {
					continue parseExtras
				}
				attrBuf := fieldBuf.sub(attrSize)
				if attrTag != 1 || attrSize != 24 {
					continue // Ignore irrelevant attributes
				}

				const ticksPerSecond = 1e7    // Windows timestamp resolution
				ts := int64(attrBuf.uint64()) // ModTime since Windows epoch
				secs := int64(ts / ticksPerSecond)
				nsecs := (1e9 / ticksPerSecond) * int64(ts%ticksPerSecond)
				epoch := time.Date(1601, time.January, 1, 0, 0, 0, 0, time.UTC)
				modified = time.Unix(epoch.Unix()+secs, nsecs)
			}
		case unixExtraID, infoZipUnixExtraID:
			if len(fieldBuf) < 8 {
				continue parseExtras
			}
			fieldBuf.uint32()              // AcTime (ignored)
			ts := int64(fieldBuf.uint32()) // ModTime since Unix epoch
			modified = time.Unix(ts, 0)
		case extTimeExtraID:
			if len(fieldBuf) < 5 || fieldBuf.uint8()&1 == 0 {
				continue parseExtras
			}
			ts := int64(fieldBuf.uint32()) // ModTime since Unix epoch
			modified = time.Unix(ts, 0)
		}
	}

	msdosModified := msDosTimeToTime(f.ModifiedDate, f.ModifiedTime)
	f.Modified = msdosModified
	if !modified.IsZero() {
		f.Modified = modified.UTC()

		// If legacy MS-DOS timestamps are set, we can use the delta between
		// the legacy and extended versions to estimate timezone offset.
		//
		// A non-UTC timezone is always used (even if offset is zero).
		// Thus, FileHeader.Modified.Location() == time.UTC is useful for
		// determining whether extended timestamps are present.
		// This is necessary for users that need to do additional time
		// calculations when dealing with legacy ZIP formats.
		if f.ModifiedTime != 0 || f.ModifiedDate != 0 {
			f.Modified = modified.In(timeZone(msdosModified.Sub(modified)))
		}
	}

	// Assume that uncompressed size 2³²-1 could plausibly happen in
	// an old zip32 file that was sharding inputs into the largest chunks
	// possible (or is just malicious; search the web for 42.zip).
	// If needUSize is true still, it means we didn't see a zip64 extension.
	// As long as the compressed size is not also 2³²-1 (implausible)
	// and the header is not also 2³²-1 (equally implausible),
	// accept the uncompressed size 2³²-1 as valid.
	// If nothing else, this keeps archive/zip working with 42.zip.
	_ = needUSize

	if needCSize || needHeaderOffset {
		return ErrFormat
	}

	return nil
}

func readDataDescriptor(r io.Reader, f *File) error {
	var buf [dataDescriptorLen]byte
	// The spec says: "Although not originally assigned a
	// signature, the value 0x08074b50 has commonly been adopted
	// as a signature value for the data descriptor record.
	// Implementers should be aware that ZIP files may be
	// encountered with or without this signature marking data
	// descriptors and should account for either case when reading
	// ZIP files to ensure compatibility."
	//
	// dataDescriptorLen includes the size of the signature but
	// first read just those 4 bytes to see if it exists.
	if _, err := io.ReadFull(r, buf[:4]); err != nil {
		return err
	}
	off := 0
	maybeSig := readBuf(buf[:4])
	if maybeSig.uint32() != dataDescriptorSignature {
		// No data descriptor signature. Keep these four
		// bytes.
		off += 4
	}
	if _, err := io.ReadFull(r, buf[off:12]); err != nil {
		return err
	}
	b := readBuf(buf[:12])
	if b.uint32() != f.CRC32 {
		return ErrChecksum
	}

	// The two sizes that follow here can be either 32 bits or 64 bits
	// but the spec is not very clear on this and different
	// interpretations has been made causing incompatibilities. We
	// already have the sizes from the central directory so we can
	// just ignore these.

	return nil
}

func readDirectoryEnd(r io.ReaderAt, size int64) (dir *directoryEnd, err error) {
	// look for directoryEndSignature in the last 1k, then in the last 65k
	var buf []byte
	var directoryEndOffset int64
	for i, bLen := range []int64{1024, 65 * 1024} {
		if bLen > size {
			bLen = size
		}
		buf = make([]byte, int(bLen))
		if _, err := r.ReadAt(buf, size-bLen); err != nil && err != io.EOF {
			return nil, err
		}
		if p := findSignatureInBlock(buf); p >= 0 {
			buf = buf[p:]
			directoryEndOffset = size - bLen + int64(p)
			break
		}
		if i == 1 || bLen == size {
			return nil, ErrFormat
		}
	}

	// read header into struct
	b := readBuf(buf[4:]) // skip signature
	d := &directoryEnd{
		diskNbr:            uint32(b.uint16()),
		dirDiskNbr:         uint32(b.uint16()),
		dirRecordsThisDisk: uint64(b.uint16()),
		directoryRecords:   uint64(b.uint16()),
		directorySize:      uint64(b.uint32()),
		directoryOffset:    uint64(b.uint32()),
		commentLen:         b.uint16(),
	}
	l := int(d.commentLen)
	if l > len(b) {
		return nil, errors.New("zip: invalid comment length")
	}
	d.comment = string(b[:l])

	// These values mean that the file can be a zip64 file
	if d.directoryRecords == 0xffff || d.directorySize == 0xffff || d.directoryOffset == 0xffffffff {
		p, err := findDirectory64End(r, directoryEndOffset)
		if err == nil && p >= 0 {
			err = readDirectory64End(r, p, d)
		}
		if err != nil {
			return nil, err
		}
	}
	// Make sure directoryOffset points to somewhere in our file.
	if o := int64(d.directoryOffset); o < 0 || o >= size {
		return nil, ErrFormat
	}
	return d, nil
}

// findDirectory64End tries to read the zip64 locator just before the
// directory end and returns the offset of the zip64 directory end if
// found.
func findDirectory64End(r io.ReaderAt, directoryEndOffset int64) (int64, error) {
	locOffset := directoryEndOffset - directory64LocLen
	if locOffset < 0 {
		return -1, nil // no need to look for a header outside the file
	}
	buf := make([]byte, directory64LocLen)
	if _, err := r.ReadAt(buf, locOffset); err != nil {
		return -1, err
	}
	b := readBuf(buf)
	if sig := b.uint32(); sig != directory64LocSignature {
		return -1, nil
	}
	if b.uint32() != 0 { // number of the disk with the start of the zip64 end of central directory
		return -1, nil // the file is not a valid zip64-file
	}
	p := b.uint64()      // relative offset of the zip64 end of central directory record
	if b.uint32() != 1 { // total number of disks
		return -1, nil // the file is not a valid zip64-file
	}
	return int64(p), nil
}

// readDirectory64End reads the zip64 directory end and updates the
// directory end with the zip64 directory end values.
func readDirectory64End(r io.ReaderAt, offset int64, d *directoryEnd) (err error) {
	buf := make([]byte, directory64EndLen)
	if _, err := r.ReadAt(buf, offset); err != nil {
		return err
	}

	b := readBuf(buf)
	if sig := b.uint32(); sig != directory64EndSignature {
		return ErrFormat
	}

	b = b[12:]                        // skip dir size, version and version needed (uint64 + 2x uint16)
	d.diskNbr = b.uint32()            // number of this disk
	d.dirDiskNbr = b.uint32()         // number of the disk with the start of the central directory
	d.dirRecordsThisDisk = b.uint64() // total number of entries in the central directory on this disk
	d.directoryRecords = b.uint64()   // total number of entries in the central directory
	d.directorySize = b.uint64()      // size of the central directory
	d.directoryOffset = b.uint64()    // offset of start of central directory with respect to the starting disk number

	return nil
}

func findSignatureInBlock(b []byte) int {
	for i := len(b) - directoryEndLen; i >= 0; i-- {
		// defined from directoryEndSignature in struct.go
		if b[i] == 'P' && b[i+1] == 'K' && b[i+2] == 0x05 && b[i+3] == 0x06 {
			// n is length of comment
			n := int(b[i+directoryEndLen-2]) | int(b[i+directoryEndLen-1])<<8
			if n+directoryEndLen+i <= len(b) {
				return i
			}
		}
	}
	return -1
}

type readBuf []byte

func (b *readBuf) uint8() uint8 {
	v := (*b)[0]
	*b = (*b)[1:]
	return v
}

func (b *readBuf) uint16() uint16 {
	v := binary.LittleEndian.Uint16(*b)
	*b = (*b)[2:]
	return v
}

func (b *readBuf) uint32() uint32 {
	v := binary.LittleEndian.Uint32(*b)
	*b = (*b)[4:]
	return v
}

func (b *readBuf) uint64() uint64 {
	v := binary.LittleEndian.Uint64(*b)
	*b = (*b)[8:]
	return v
}

func (b *readBuf) sub(n int) readBuf {
	b2 := (*b)[:n]
	*b = (*b)[n:]
	return b2
}

// A fileListEntry is a File and its ename.
// If file == nil, the fileListEntry describes a directory without metadata.
type fileListEntry struct {
	name  string
	file  *File
	isDir bool
}

type fileInfoDirEntry interface {
	fs.FileInfo
	fs.DirEntry
}

func (e *fileListEntry) stat() fileInfoDirEntry {
	if !e.isDir {
		return headerFileInfo{&e.file.FileHeader}
	}
	return e
}

// Only used for directories.
func (f *fileListEntry) Name() string      { _, elem, _ := split(f.name); return elem }
func (f *fileListEntry) Size() int64       { return 0 }
func (f *fileListEntry) Mode() fs.FileMode { return fs.ModeDir | 0555 }
func (f *fileListEntry) Type() fs.FileMode { return fs.ModeDir }
func (f *fileListEntry) IsDir() bool       { return true }
func (f *fileListEntry) Sys() any          { return nil }

func (f *fileListEntry) ModTime() time.Time {
	if f.file == nil {
		return time.Time{}
	}
	return f.file.FileHeader.Modified.UTC()
}

func (f *fileListEntry) Info() (fs.FileInfo, error) { return f, nil }

// toValidName coerces name to be a valid name for fs.FS.Open.
func toValidName(name string) string {
	name = strings.ReplaceAll(name, `\`, `/`)
	p := path.Clean(name)
	if strings.HasPrefix(p, "/") {
		p = p[len("/"):]
	}
	for strings.HasPrefix(p, "../") {
		p = p[len("../"):]
	}
	return p
}

func (r *Reader) initFileList() {
	r.fileListOnce.Do(func() {
		dirs := make(map[string]bool)
		knownDirs := make(map[string]bool)
		for _, file := range r.File {
			isDir := len(file.Name) > 0 && file.Name[len(file.Name)-1] == '/'
			name := toValidName(file.Name)
			if name == "" {
				continue
			}
			for dir := path.Dir(name); dir != "."; dir = path.Dir(dir) {
				dirs[dir] = true
			}
			entry := fileListEntry{
				name:  name,
				file:  file,
				isDir: isDir,
			}
			r.fileList = append(r.fileList, entry)
			if isDir {
				knownDirs[name] = true
			}
		}
		for dir := range dirs {
			if !knownDirs[dir] {
				entry := fileListEntry{
					name:  dir,
					file:  nil,
					isDir: true,
				}
				r.fileList = append(r.fileList, entry)
			}
		}

		sort.Slice(r.fileList, func(i, j int) bool { return fileEntryLess(r.fileList[i].name, r.fileList[j].name) })
	})
}

func fileEntryLess(x, y string) bool {
	xdir, xelem, _ := split(x)
	ydir, yelem, _ := split(y)
	return xdir < ydir || xdir == ydir && xelem < yelem
}

// Open opens the named file in the ZIP archive,
// using the semantics of fs.FS.Open:
// paths are always slash separated, with no
// leading / or ../ elements.
func (r *Reader) Open(name string) (fs.File, error) {
	r.initFileList()

	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrInvalid}
	}
	e := r.openLookup(name)
	if e == nil {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	}
	if e.isDir {
		return &openDir{e, r.openReadDir(name), 0}, nil
	}
	rc, err := e.file.Open()
	if err != nil {
		return nil, err
	}
	return rc.(fs.File), nil
}

func split(name string) (dir, elem string, isDir bool) {
	if len(name) > 0 && name[len(name)-1] == '/' {
		isDir = true
		name = name[:len(name)-1]
	}
	i := len(name) - 1
	for i >= 0 && name[i] != '/' {
		i--
	}
	if i < 0 {
		return ".", name, isDir
	}
	return name[:i], name[i+1:], isDir
}

var dotFile = &fileListEntry{name: "./", isDir: true}

func (r *Reader) openLookup(name string) *fileListEntry {
	if name == "." {
		return dotFile
	}

	dir, elem, _ := split(name)
	files := r.fileList
	i := sort.Search(len(files), func(i int) bool {
		idir, ielem, _ := split(files[i].name)
		return idir > dir || idir == dir && ielem >= elem
	})
	if i < len(files) {
		fname := files[i].name
		if fname == name || len(fname) == len(name)+1 && fname[len(name)] == '/' && fname[:len(name)] == name {
			return &files[i]
		}
	}
	return nil
}

func (r *Reader) openReadDir(dir string) []fileListEntry {
	files := r.fileList
	i := sort.Search(len(files), func(i int) bool {
		idir, _, _ := split(files[i].name)
		return idir >= dir
	})
	j := sort.Search(len(files), func(j int) bool {
		jdir, _, _ := split(files[j].name)
		return jdir > dir
	})
	return files[i:j]
}

type openDir struct {
	e      *fileListEntry
	files  []fileListEntry
	offset int
}

func (d *openDir) Close() error               { return nil }
func (d *openDir) Stat() (fs.FileInfo, error) { return d.e.stat(), nil }

func (d *openDir) Read([]byte) (int, error) {
	return 0, &fs.PathError{Op: "read", Path: d.e.name, Err: errors.New("is a directory")}
}

func (d *openDir) ReadDir(count int) ([]fs.DirEntry, error) {
	n := len(d.files) - d.offset
	if count > 0 && n > count {
		n = count
	}
	if n == 0 {
		if count <= 0 {
			return nil, nil
		}
		return nil, io.EOF
	}
	list := make([]fs.DirEntry, n)
	for i := range list {
		list[i] = d.files[d.offset+i].stat()
	}
	d.offset += n
	return list, nil
}
