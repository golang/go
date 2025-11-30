// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestZip64WriterCDGoldens checks that the archive/zip Writer emits a Central
// Directory that matches the Zip64 conventions used by Info-ZIP, libarchive,
// and the pre-CL archive/zip writer (go126-*), for archives at or above 4 GiB,
// except where we intentionally diverged.
//
// For each golden in testdata/zip64/*.zsparse (see [sparseFile] for the
// committed format), the test:
//  1. Parses the golden's CD into a producer-independent snapshot — which
//     fields hold 0xFFFFFFFF placeholders, which Zip64 extra sub-fields are
//     present and in what order, and the EOCD/EOCD64 values.
//  2. Verifies the production [NewReader] parses the same archive.
//  3. Replays the same entries through a fresh [Writer] into a [sparseBuffer]
//     and parses our own CD.
//  4. Verifies the production [NewReader] parses our reproduced archive too.
//  5. Compares the two snapshots field-by-field, ignoring producer-specific
//     details (creator version, external attrs, non-Zip64 extras, absolute
//     byte offsets that depend on LFH/data-descriptor layout).
func TestZip64WriterCDGoldens(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode; each golden replays a multi-GiB write")
	}

	matches, err := filepath.Glob("testdata/zip64/*.zsparse")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("missing Zip64 goldens in testdata/zip64")
	}

	// Tail materialized for parseCD. Goldens have ≤ 2 entries; their CD
	// plus EOCD records fits in well under 1 MiB.
	const tailKeep = 1 << 20

	// archive/zip's writer takes the most defensive position on every
	// spec-fuzzy point: it always emits the Zip64 extra at the 0xFFFFFFFF
	// boundary (matching libarchive but more conservative than Info-ZIP) AND
	// emits EOCD64 whenever any entry has a Zip64 extra in its CD record
	// (matching Info-ZIP but more conservative than libarchive). The go126-
	// goldens are output of an older archive/zip writer, and the format
	// deliberately diverges; they are kept here so the reader-side check
	// enforces backwards compatibility with archives produced by our own past
	// writer, and to ensure we only diverge where intended.
	expectedDiff := map[string]bool{
		// Info-ZIP treats a CD size field of exactly 0xFFFFFFFF as a real
		// value and omits the Zip64 extra; archive/zip defensively emits
		// the Zip64 extra with USize64+CSize64.
		"infozip-store-4g-minus-1": true,

		// Info-ZIP treats a CD offset field of exactly 0xFFFFFFFF as a real
		// value and omits the Zip64 extra for offset; archive/zip defensively
		// emits the Zip64 extra with the offset sub-field.
		"infozip-offset-eq-4g": true,

		// libarchive's writer emits EOCD64 only on EOCD-level overflow (CD
		// size/offset > 4GiB, records > 0xFFFF); archive/zip also emits
		// EOCD64 when any per-entry CD record uses a Zip64 extra, even if
		// the EOCD fields fit in 32 bits.
		"libarchive-deflate-zeros-5g": true,

		// libarchive's LFH always carries a UT timestamp extra (~9 bytes),
		// so its dirOffset for a body of 4GiB-59 lands just past 0xFFFFFFFF
		// and it emits EOCD64. archive/zip's streaming LFH has no such
		// extras and stays under uint32max.
		"libarchive-store-just-under-4g": true,

		// The old archive/zip writer differs from the current writer on
		// every Zip64-using entry: it always wrote a fixed 24-byte Zip64
		// extra with all three sub-fields (usize, csize, offset) and set
		// both 32-bit size fields to 0xFFFFFFFF whenever the per-entry
		// trigger fired; it also set the EOCD records/size/offset to the
		// placeholder values whenever EOCD64 was present.
		"go126-store-5g":            true,
		"go126-deflate-zeros-5g":    true,
		"go126-store-4g-minus-1":    true,
		"go126-store-4g-minus-2":    true,
		"go126-store-exact-4g":      true,
		"go126-offset-past-4g":      true,
		"go126-offset-eq-4g":        true,
		"go126-store-just-under-4g": false,
	}

	for _, path := range matches {
		name := strings.TrimSuffix(filepath.Base(path), ".zsparse")
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			goldenSF, err := readSparseFile(path)
			if err != nil {
				t.Fatalf("read golden: %v", err)
			}
			goldenData, goldenBase := goldenSF.materializeTail(tailKeep)
			golden, err := parseCD(goldenData, goldenBase)
			if err != nil {
				t.Fatalf("parse golden CD: %v", err)
			}

			// Verify the production Reader can parse the full golden.
			checkReaderMatchesSnapshot(t, "golden", goldenSF, golden)

			oursSF := reproduceCD(t, golden)
			oursData, oursBase := oursSF.materializeTail(tailKeep)
			got, err := parseCD(oursData, oursBase)
			if err != nil {
				t.Fatalf("parse reproduced CD: %v\nbytes:\n%s", err, hexDump(oursData))
			}
			// Verify the production Reader can parse archive/zip's own
			// output and gets the same view of the entries.
			checkReaderMatchesSnapshot(t, "reproduced", oursSF, got)

			if expectedDiff[name] {
				var cap captureReporter
				compareCDSnapshots(&cap, golden, got)
				if !cap.failed {
					t.Errorf("expected this golden to fail equivalence, but it passed")
				} else {
					t.Logf("expected mismatch:\n%s", indent(cap.msg.String(), "  "))
				}
				return
			}
			compareCDSnapshots(t, golden, got)
		})
	}
}

// errReporter is the subset of [testing.TB] that [compareCDSnapshots] uses.
// The captureReporter implementation lets the test capture mismatches for
// expected-failure cases instead of propagating them to the outer t.
type errReporter interface {
	Errorf(format string, args ...any)
	Helper()
}

type captureReporter struct {
	failed bool
	msg    strings.Builder
}

func (c *captureReporter) Errorf(format string, args ...any) {
	c.failed = true
	fmt.Fprintf(&c.msg, format+"\n", args...)
}

func (c *captureReporter) Helper() {}

// checkReaderMatchesSnapshot opens the archive backed by the sparseFile
// using the production [NewReader] and asserts that the entry list it
// returns matches the [cdSnapshot] (entry count, names, resolved 64-bit
// sizes).
func checkReaderMatchesSnapshot(t *testing.T, label string, f *sparseFile, snap *cdSnapshot) {
	t.Helper()
	zr, err := NewReader(f, f.Size)
	if err != nil {
		t.Fatalf("%s: NewReader: %v", label, err)
	}
	if g, w := len(zr.File), len(snap.Entries); g != w {
		t.Errorf("%s: NewReader returned %d files, parseCD found %d", label, g, w)
		return
	}
	for i, f := range zr.File {
		want := &snap.Entries[i]
		if f.Name != want.Name {
			t.Errorf("%s entry %d: Name = %q, want %q", label, i, f.Name, want.Name)
		}
		if f.UncompressedSize64 != want.USize64 {
			t.Errorf("%s entry %d %q: UncompressedSize64 = %d, want %d", label, i, want.Name, f.UncompressedSize64, want.USize64)
		}
		if f.CompressedSize64 != want.CSize64 {
			t.Errorf("%s entry %d %q: CompressedSize64 = %d, want %d", label, i, want.Name, f.CompressedSize64, want.CSize64)
		}
	}
}

// indent prefixes every line of s with prefix.
func indent(s, prefix string) string {
	if s == "" {
		return s
	}
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	for i, l := range lines {
		lines[i] = prefix + l
	}
	return strings.Join(lines, "\n") + "\n"
}

// reproduceCD writes a zip archive with the same logical entries as golden
// into a [sparseBuffer] (which drops all-zero chunks, so pushing multi-GiB
// streams of zeros through the writer is essentially free) and returns the
// resulting [sparseFile].
//
// For entries where compressed == uncompressed (Store, or other 1:1 cases)
// we drive the Writer through [Writer.CreateHeader] so that the data
// descriptor, offset accounting, and Close-time CD emission all exercise
// the production streaming path. The CRC32 hasher is replaced with
// [fakeHash32] to avoid hashing many GiB of zeros.
//
// For entries where compressed ≪ uncompressed (Method=Deflate over zeros),
// actually deflating multi-GiB streams at test time is prohibitively slow,
// so we fall back to [Writer.CreateRaw] and declare the sizes directly.
// The Central Directory output is identical either way.
func reproduceCD(t *testing.T, golden *cdSnapshot) *sparseFile {
	t.Helper()
	sb := &sparseBuffer{}
	w := NewWriter(sb)
	for i, e := range golden.Entries {
		if e.CSize64 == e.USize64 {
			fh := &FileHeader{Name: e.Name, Method: e.Method}
			fw, err := w.CreateHeader(fh)
			if err != nil {
				t.Fatalf("CreateHeader[%d %q]: %v", i, e.Name, err)
			}
			fw.(*fileWriter).crc32 = fakeHash32{}
			if _, err := io.CopyN(fw, zeros{}, int64(e.USize64)); err != nil {
				t.Fatalf("CopyN[%d %q]: %v", i, e.Name, err)
			}
			continue
		}
		fh := &FileHeader{
			Name:               e.Name,
			Method:             e.Method,
			CompressedSize64:   e.CSize64,
			UncompressedSize64: e.USize64,
		}
		fw, err := w.CreateRaw(fh)
		if err != nil {
			t.Fatalf("CreateRaw[%d %q]: %v", i, e.Name, err)
		}
		if _, err := io.CopyN(fw, zeros{}, int64(e.CSize64)); err != nil {
			t.Fatalf("CopyN[%d %q]: %v", i, e.Name, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return &sb.f
}

// compareCDSnapshots asserts that got matches want on Zip64-relevant fields.
//
// Per-entry size fields (RawCSize, RawUSize, CSize64, USize64) are compared
// exactly — we feed them in from the golden when reproducing, so the writer
// has no excuse to disagree. Per-entry RawOffset and the EOCD records/size/
// offset fields are compared only as placeholder-or-not: their absolute
// values depend on producer-specific LFH layout (Info-ZIP packs sizes into
// the LFH; archive/zip's streaming path uses a data descriptor; libarchive
// adds UT extras) and that's not what this test is pinning down.
func compareCDSnapshots(t errReporter, want, got *cdSnapshot) {
	t.Helper()
	if g, w := len(got.Entries), len(want.Entries); g != w {
		t.Errorf("entry count = %d, want %d", g, w)
		return
	}
	for i := range want.Entries {
		we, ge := &want.Entries[i], &got.Entries[i]
		// csize and usize come from the declared FileHeader values, so the
		// raw 32-bit fields must match exactly (real value vs. placeholder
		// choice and, when not placeholder, the value itself).
		if we.RawCSize != ge.RawCSize {
			t.Errorf("entry %d %q: RawCSize = %#08x, want %#08x", i, we.Name, ge.RawCSize, we.RawCSize)
		}
		if we.RawUSize != ge.RawUSize {
			t.Errorf("entry %d %q: RawUSize = %#08x, want %#08x", i, we.Name, ge.RawUSize, we.RawUSize)
		}
		// Resolved csize/usize must match — we fed them in from the golden.
		if we.CSize64 != ge.CSize64 {
			t.Errorf("entry %d %q: CSize64 = %d, want %d", i, we.Name, ge.CSize64, we.CSize64)
		}
		if we.USize64 != ge.USize64 {
			t.Errorf("entry %d %q: USize64 = %d, want %d", i, we.Name, ge.USize64, we.USize64)
		}
		// Offset is layout-dependent. Compare placeholder-or-not, not value.
		if isPlaceholder32(we.RawOffset) != isPlaceholder32(ge.RawOffset) {
			t.Errorf("entry %d %q: RawOffset placeholder = %#08x, want %#08x", i, we.Name, ge.RawOffset, we.RawOffset)
		}

		// Zip64 sub-field presence/order, must match exactly.
		if !slices.Equal(we.Z64ExtraFields, ge.Z64ExtraFields) {
			t.Errorf("entry %d %q: Zip64 sub-field order = %v, want %v", i, we.Name, ge.Z64ExtraFields, we.Z64ExtraFields)
		}
		// ReaderVersion ≥ 45 whenever a Zip64 extra is present.
		if len(we.Z64ExtraFields) > 0 && ge.ReaderVersion < zipVersion45 {
			t.Errorf("entry %d %q: ReaderVersion = %d, want ≥ %d (Zip64 extra present)", i, we.Name, ge.ReaderVersion, zipVersion45)
		}
	}

	// EOCD: compare placeholder-or-not for each field. Exact values are
	// layout-dependent.
	if isPlaceholder16(want.EOCD.Records) != isPlaceholder16(got.EOCD.Records) {
		t.Errorf("EOCD records placeholder = %#x, want %#x", got.EOCD.Records, want.EOCD.Records)
	}
	if isPlaceholder32(want.EOCD.Size) != isPlaceholder32(got.EOCD.Size) {
		t.Errorf("EOCD size placeholder = %#x, want %#x", got.EOCD.Size, want.EOCD.Size)
	}
	if isPlaceholder32(want.EOCD.Offset) != isPlaceholder32(got.EOCD.Offset) {
		t.Errorf("EOCD offset placeholder = %#x, want %#x", got.EOCD.Offset, want.EOCD.Offset)
	}

	if got.HasEOCD64 != want.HasEOCD64 {
		t.Errorf("EOCD64 present = %v, want %v", got.HasEOCD64, want.HasEOCD64)
	}
	if want.HasEOCD64 && got.HasEOCD64 {
		if got.EOCD64.Records != want.EOCD64.Records {
			t.Errorf("EOCD64 records = %d, want %d", got.EOCD64.Records, want.EOCD64.Records)
		}
		// EOCD64.Size and EOCD64.Offset are layout-dependent.
	}
}

func isPlaceholder32(v uint32) bool { return v == uint32max }
func isPlaceholder16(v uint16) bool { return v == uint16max }

// CD snapshot types and parser

// zip64SubID identifies one of the three sub-fields that may appear in a
// Zip64 extended-information extra field, in the spec-defined order.
type zip64SubID int

const (
	z64USize zip64SubID = iota + 1
	z64CSize
	z64Offset
)

func (s zip64SubID) String() string {
	switch s {
	case z64USize:
		return "usize"
	case z64CSize:
		return "csize"
	case z64Offset:
		return "offset"
	}
	return fmt.Sprintf("zip64SubID(%d)", int(s))
}

type cdEntry struct {
	Name          string
	Method        uint16
	ReaderVersion uint16

	// Raw 32-bit fields from the CD record. A value of 0xFFFFFFFF indicates
	// the real value is in the Zip64 extended-information extra field.
	RawCSize  uint32
	RawUSize  uint32
	RawOffset uint32

	// Resolved 64-bit values (from the 32-bit field if not a placeholder,
	// otherwise from the Zip64 extra).
	CSize64  uint64
	USize64  uint64
	Offset64 uint64

	// Sub-fields present in the Zip64 extra, in the order they appear.
	Z64ExtraFields []zip64SubID
}

type eocdRec struct {
	Records uint16 // 0xFFFF if placeholder
	Size    uint32 // 0xFFFFFFFF if placeholder
	Offset  uint32 // 0xFFFFFFFF if placeholder
}

type eocd64Rec struct {
	Records uint64
	Size    uint64
	Offset  uint64
}

type cdSnapshot struct {
	Entries   []cdEntry
	EOCD      eocdRec
	HasEOCD64 bool
	EOCD64    eocd64Rec
}

var le = binary.LittleEndian

// parseCD parses the Central Directory and EOCD records of a zip archive
// from its raw bytes. data must be the tail of the archive, with baseOffset
// indicating where data[0] sits in the original archive (0 for whole-archive
// input).
func parseCD(data []byte, baseOffset uint64) (*cdSnapshot, error) {
	sigOff, err := findEOCD(data)
	if err != nil {
		return nil, err
	}
	snap := &cdSnapshot{}
	snap.EOCD.Records = le.Uint16(data[sigOff+10:])
	snap.EOCD.Size = le.Uint32(data[sigOff+12:])
	snap.EOCD.Offset = le.Uint32(data[sigOff+16:])

	dirOffset := uint64(snap.EOCD.Offset)
	nRecords := uint64(snap.EOCD.Records)

	// toData converts an absolute archive offset to a data slice offset,
	// returning false if it lies before our captured tail.
	toData := func(absOff uint64) (uint64, bool) {
		if absOff < baseOffset {
			return 0, false
		}
		return absOff - baseOffset, true
	}

	// Look for an EOCD64 locator immediately preceding the EOCD record.
	if sigOff >= directory64LocLen {
		locOff := sigOff - directory64LocLen
		if le.Uint32(data[locOff:]) == directory64LocSignature {
			eocd64Off := le.Uint64(data[locOff+8:])
			eocd64DataOff, ok := toData(eocd64Off)
			if !ok {
				return nil, fmt.Errorf("zip: EOCD64 at %#x before captured tail (base %#x)", eocd64Off, baseOffset)
			}
			if eocd64DataOff+directory64EndLen > uint64(len(data)) {
				return nil, errors.New("zip: EOCD64 offset out of range")
			}
			if le.Uint32(data[eocd64DataOff:]) != directory64EndSignature {
				return nil, errors.New("zip: EOCD64 signature mismatch")
			}
			snap.HasEOCD64 = true
			snap.EOCD64.Records = le.Uint64(data[eocd64DataOff+32:])
			snap.EOCD64.Size = le.Uint64(data[eocd64DataOff+40:])
			snap.EOCD64.Offset = le.Uint64(data[eocd64DataOff+48:])
			dirOffset = snap.EOCD64.Offset
			nRecords = snap.EOCD64.Records
		}
	}

	off, ok := toData(dirOffset)
	if !ok {
		return nil, fmt.Errorf("zip: CD at %#x before captured tail (base %#x)", dirOffset, baseOffset)
	}
	for i := uint64(0); i < nRecords; i++ {
		if off+directoryHeaderLen > uint64(len(data)) {
			return nil, fmt.Errorf("zip: CD entry %d out of range", i)
		}
		rec := data[off:]
		if le.Uint32(rec) != directoryHeaderSignature {
			return nil, fmt.Errorf("zip: bad CD signature at offset %d", off)
		}
		var e cdEntry
		e.ReaderVersion = le.Uint16(rec[6:])
		e.Method = le.Uint16(rec[10:])
		e.RawCSize = le.Uint32(rec[20:])
		e.RawUSize = le.Uint32(rec[24:])
		nameLen := uint64(le.Uint16(rec[28:]))
		extraLen := uint64(le.Uint16(rec[30:]))
		commLen := uint64(le.Uint16(rec[32:]))
		e.RawOffset = le.Uint32(rec[42:])

		recLen := uint64(directoryHeaderLen) + nameLen + extraLen + commLen
		if off+recLen > uint64(len(data)) {
			return nil, fmt.Errorf("zip: CD entry %d truncated", i)
		}
		nameOff := off + directoryHeaderLen
		extraOff := nameOff + nameLen
		e.Name = string(data[nameOff:extraOff])
		extra := data[extraOff : extraOff+extraLen]

		e.CSize64 = uint64(e.RawCSize)
		e.USize64 = uint64(e.RawUSize)
		e.Offset64 = uint64(e.RawOffset)

		// Walk extra fields; consume the Zip64 sub-field if present.
		// Per the spec and Info-ZIP convention, the Zip64 extra contains
		// 8-byte values for exactly the size/offset fields whose 32-bit
		// counterpart is 0xFFFFFFFF, in the order: USize, CSize, Offset.
		for len(extra) >= 4 {
			tag := le.Uint16(extra)
			size := uint64(le.Uint16(extra[2:]))
			if 4+size > uint64(len(extra)) {
				break
			}
			field := extra[4 : 4+size]
			extra = extra[4+size:]
			if tag != zip64ExtraID {
				continue
			}
			if e.RawUSize == uint32max && len(field) >= 8 {
				e.USize64 = le.Uint64(field)
				e.Z64ExtraFields = append(e.Z64ExtraFields, z64USize)
				field = field[8:]
			}
			if e.RawCSize == uint32max && len(field) >= 8 {
				e.CSize64 = le.Uint64(field)
				e.Z64ExtraFields = append(e.Z64ExtraFields, z64CSize)
				field = field[8:]
			}
			if e.RawOffset == uint32max && len(field) >= 8 {
				e.Offset64 = le.Uint64(field)
				e.Z64ExtraFields = append(e.Z64ExtraFields, z64Offset)
				field = field[8:]
			}
		}

		snap.Entries = append(snap.Entries, e)
		off += recLen
	}
	return snap, nil
}

// findEOCD locates the EOCD record by scanning back from the end of data,
// matching both the signature and the trailing comment-length field.
func findEOCD(data []byte) (uint64, error) {
	if len(data) < directoryEndLen {
		return 0, errors.New("zip: too short for EOCD")
	}
	maxComment := uint16max
	lo := len(data) - directoryEndLen
	hi := lo
	if hi > maxComment {
		lo = hi - maxComment
	} else {
		lo = 0
	}
	for i := hi; i >= lo; i-- {
		if le.Uint32(data[i:]) != directoryEndSignature {
			continue
		}
		cl := int(le.Uint16(data[i+20:]))
		if i+directoryEndLen+cl == len(data) {
			return uint64(i), nil
		}
	}
	return 0, errors.New("zip: EOCD not found")
}

// hexDump returns a short hex dump of data for failure messages.
func hexDump(data []byte) string {
	if len(data) > 4096 {
		data = data[len(data)-4096:]
	}
	var b strings.Builder
	for i := 0; i < len(data); i += 16 {
		end := min(i+16, len(data))
		fmt.Fprintf(&b, "%04x  % x\n", i, data[i:end])
	}
	return b.String()
}

// TestZip64LFHBothPlaceholders covers the [Writer.CreateRaw] + no-data-
// descriptor path where the entry's uncompressed or compressed size exceeds
// 4 GiB. The Local File Header carries a Zip64 extra with both 8-byte
// USize64 and CSize64 sub-fields (matching Info-ZIP), so per APPNOTE 4.5.3
// both 32-bit size fields in the LFH must be the 0xFFFFFFFF placeholder —
// even if only one of the sizes actually overflows.
func TestZip64LFHBothPlaceholders(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	fh := &FileHeader{
		Name:               "x",
		Method:             Deflate,
		CompressedSize64:   1024,
		UncompressedSize64: 5 << 30, // > 4 GiB
	}
	fw, err := w.CreateRaw(fh)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.CopyN(fw, zeros{}, int64(fh.CompressedSize64)); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	b := buf.Bytes()
	if got := le.Uint32(b[14:18]); got != fh.CRC32 {
		t.Errorf("LFH CRC32 = %#x, want %#x", got, fh.CRC32)
	}
	if got := le.Uint32(b[18:22]); got != uint32max {
		t.Errorf("LFH CompressedSize = %#x, want %#x (placeholder)", got, uint32(uint32max))
	}
	if got := le.Uint32(b[22:26]); got != uint32max {
		t.Errorf("LFH UncompressedSize = %#x, want %#x (placeholder)", got, uint32(uint32max))
	}

	// The Zip64 LFH extra should carry both 64-bit sub-fields in
	// USize64-then-CSize64 order.
	nameLen := uint64(le.Uint16(b[26:28]))
	extraLen := uint64(le.Uint16(b[28:30]))
	if want := uint64(20); extraLen != want {
		t.Fatalf("LFH extra length = %d, want %d", extraLen, want)
	}
	extra := b[30+nameLen : 30+nameLen+extraLen]
	if tag := le.Uint16(extra[:2]); tag != zip64ExtraID {
		t.Errorf("Zip64 extra tag = %#x, want %#x", tag, zip64ExtraID)
	}
	if dataLen := le.Uint16(extra[2:4]); dataLen != 16 {
		t.Errorf("Zip64 extra data length = %d, want 16", dataLen)
	}
	if got := le.Uint64(extra[4:12]); got != fh.UncompressedSize64 {
		t.Errorf("Zip64 USize64 = %d, want %d", got, fh.UncompressedSize64)
	}
	if got := le.Uint64(extra[12:20]); got != fh.CompressedSize64 {
		t.Errorf("Zip64 CSize64 = %d, want %d", got, fh.CompressedSize64)
	}
}
