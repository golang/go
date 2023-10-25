// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package coverage

import (
	"crypto/md5"
	"fmt"
	"internal/coverage"
	"internal/coverage/encodecounter"
	"internal/coverage/encodemeta"
	"internal/coverage/rtcov"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync/atomic"
	"time"
	"unsafe"
)

// This file contains functions that support the writing of data files
// emitted at the end of code coverage testing runs, from instrumented
// executables.

// getCovMetaList returns a list of meta-data blobs registered
// for the currently executing instrumented program. It is defined in the
// runtime.
func getCovMetaList() []rtcov.CovMetaBlob

// getCovCounterList returns a list of counter-data blobs registered
// for the currently executing instrumented program. It is defined in the
// runtime.
func getCovCounterList() []rtcov.CovCounterBlob

// getCovPkgMap returns a map storing the remapped package IDs for
// hard-coded runtime packages (see internal/coverage/pkgid.go for
// more on why hard-coded package IDs are needed). This function
// is defined in the runtime.
func getCovPkgMap() map[int]int

// emitState holds useful state information during the emit process.
//
// When an instrumented program finishes execution and starts the
// process of writing out coverage data, it's possible that an
// existing meta-data file already exists in the output directory. In
// this case openOutputFiles() below will leave the 'mf' field below
// as nil. If a new meta-data file is needed, field 'mfname' will be
// the final desired path of the meta file, 'mftmp' will be a
// temporary file, and 'mf' will be an open os.File pointer for
// 'mftmp'. The meta-data file payload will be written to 'mf', the
// temp file will be then closed and renamed (from 'mftmp' to
// 'mfname'), so as to insure that the meta-data file is created
// atomically; we want this so that things work smoothly in cases
// where there are several instances of a given instrumented program
// all terminating at the same time and trying to create meta-data
// files simultaneously.
//
// For counter data files there is less chance of a collision, hence
// the openOutputFiles() stores the counter data file in 'cfname' and
// then places the *io.File into 'cf'.
type emitState struct {
	mfname string   // path of final meta-data output file
	mftmp  string   // path to meta-data temp file (if needed)
	mf     *os.File // open os.File for meta-data temp file
	cfname string   // path of final counter data file
	cftmp  string   // path to counter data temp file
	cf     *os.File // open os.File for counter data file
	outdir string   // output directory

	// List of meta-data symbols obtained from the runtime
	metalist []rtcov.CovMetaBlob

	// List of counter-data symbols obtained from the runtime
	counterlist []rtcov.CovCounterBlob

	// Table to use for remapping hard-coded pkg ids.
	pkgmap map[int]int

	// emit debug trace output
	debug bool
}

var (
	// finalHash is computed at init time from the list of meta-data
	// symbols registered during init. It is used both for writing the
	// meta-data file and counter-data files.
	finalHash [16]byte
	// Set to true when we've computed finalHash + finalMetaLen.
	finalHashComputed bool
	// Total meta-data length.
	finalMetaLen uint64
	// Records whether we've already attempted to write meta-data.
	metaDataEmitAttempted bool
	// Counter mode for this instrumented program run.
	cmode coverage.CounterMode
	// Counter granularity for this instrumented program run.
	cgran coverage.CounterGranularity
	// Cached value of GOCOVERDIR environment variable.
	goCoverDir string
	// Copy of os.Args made at init time, converted into map format.
	capturedOsArgs map[string]string
	// Flag used in tests to signal that coverage data already written.
	covProfileAlreadyEmitted bool
)

// fileType is used to select between counter-data files and
// meta-data files.
type fileType int

const (
	noFile = 1 << iota
	metaDataFile
	counterDataFile
)

// emitMetaData emits the meta-data output file for this coverage run.
// This entry point is intended to be invoked by the compiler from
// an instrumented program's main package init func.
func emitMetaData() {
	if covProfileAlreadyEmitted {
		return
	}
	ml, err := prepareForMetaEmit()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: coverage meta-data prep failed: %v\n", err)
		if os.Getenv("GOCOVERDEBUG") != "" {
			panic("meta-data write failure")
		}
	}
	if len(ml) == 0 {
		fmt.Fprintf(os.Stderr, "program not built with -cover\n")
		return
	}

	goCoverDir = os.Getenv("GOCOVERDIR")
	if goCoverDir == "" {
		fmt.Fprintf(os.Stderr, "warning: GOCOVERDIR not set, no coverage data emitted\n")
		return
	}

	if err := emitMetaDataToDirectory(goCoverDir, ml); err != nil {
		fmt.Fprintf(os.Stderr, "error: coverage meta-data emit failed: %v\n", err)
		if os.Getenv("GOCOVERDEBUG") != "" {
			panic("meta-data write failure")
		}
	}
}

func modeClash(m coverage.CounterMode) bool {
	if m == coverage.CtrModeRegOnly || m == coverage.CtrModeTestMain {
		return false
	}
	if cmode == coverage.CtrModeInvalid {
		cmode = m
		return false
	}
	return cmode != m
}

func granClash(g coverage.CounterGranularity) bool {
	if cgran == coverage.CtrGranularityInvalid {
		cgran = g
		return false
	}
	return cgran != g
}

// prepareForMetaEmit performs preparatory steps needed prior to
// emitting a meta-data file, notably computing a final hash of
// all meta-data blobs and capturing os args.
func prepareForMetaEmit() ([]rtcov.CovMetaBlob, error) {
	// Ask the runtime for the list of coverage meta-data symbols.
	ml := getCovMetaList()

	// In the normal case (go build -o prog.exe ... ; ./prog.exe)
	// len(ml) will always be non-zero, but we check here since at
	// some point this function will be reachable via user-callable
	// APIs (for example, to write out coverage data from a server
	// program that doesn't ever call os.Exit).
	if len(ml) == 0 {
		return nil, nil
	}

	s := &emitState{
		metalist: ml,
		debug:    os.Getenv("GOCOVERDEBUG") != "",
	}

	// Capture os.Args() now so as to avoid issues if args
	// are rewritten during program execution.
	capturedOsArgs = captureOsArgs()

	if s.debug {
		fmt.Fprintf(os.Stderr, "=+= GOCOVERDIR is %s\n", os.Getenv("GOCOVERDIR"))
		fmt.Fprintf(os.Stderr, "=+= contents of covmetalist:\n")
		for k, b := range ml {
			fmt.Fprintf(os.Stderr, "=+= slot: %d path: %s ", k, b.PkgPath)
			if b.PkgID != -1 {
				fmt.Fprintf(os.Stderr, " hcid: %d", b.PkgID)
			}
			fmt.Fprintf(os.Stderr, "\n")
		}
		pm := getCovPkgMap()
		fmt.Fprintf(os.Stderr, "=+= remap table:\n")
		for from, to := range pm {
			fmt.Fprintf(os.Stderr, "=+= from %d to %d\n",
				uint32(from), uint32(to))
		}
	}

	h := md5.New()
	tlen := uint64(unsafe.Sizeof(coverage.MetaFileHeader{}))
	for _, entry := range ml {
		if _, err := h.Write(entry.Hash[:]); err != nil {
			return nil, err
		}
		tlen += uint64(entry.Len)
		ecm := coverage.CounterMode(entry.CounterMode)
		if modeClash(ecm) {
			return nil, fmt.Errorf("coverage counter mode clash: package %s uses mode=%d, but package %s uses mode=%s\n", ml[0].PkgPath, cmode, entry.PkgPath, ecm)
		}
		ecg := coverage.CounterGranularity(entry.CounterGranularity)
		if granClash(ecg) {
			return nil, fmt.Errorf("coverage counter granularity clash: package %s uses gran=%d, but package %s uses gran=%s\n", ml[0].PkgPath, cgran, entry.PkgPath, ecg)
		}
	}

	// Hash mode and granularity as well.
	h.Write([]byte(cmode.String()))
	h.Write([]byte(cgran.String()))

	// Compute final digest.
	fh := h.Sum(nil)
	copy(finalHash[:], fh)
	finalHashComputed = true
	finalMetaLen = tlen

	return ml, nil
}

// emitMetaDataToDirectory emits the meta-data output file to the specified
// directory, returning an error if something went wrong.
func emitMetaDataToDirectory(outdir string, ml []rtcov.CovMetaBlob) error {
	ml, err := prepareForMetaEmit()
	if err != nil {
		return err
	}
	if len(ml) == 0 {
		return nil
	}

	metaDataEmitAttempted = true

	s := &emitState{
		metalist: ml,
		debug:    os.Getenv("GOCOVERDEBUG") != "",
		outdir:   outdir,
	}

	// Open output files.
	if err := s.openOutputFiles(finalHash, finalMetaLen, metaDataFile); err != nil {
		return err
	}

	// Emit meta-data file only if needed (may already be present).
	if s.needMetaDataFile() {
		if err := s.emitMetaDataFile(finalHash, finalMetaLen); err != nil {
			return err
		}
	}
	return nil
}

// emitCounterData emits the counter data output file for this coverage run.
// This entry point is intended to be invoked by the runtime when an
// instrumented program is terminating or calling os.Exit().
func emitCounterData() {
	if goCoverDir == "" || !finalHashComputed || covProfileAlreadyEmitted {
		return
	}
	if err := emitCounterDataToDirectory(goCoverDir); err != nil {
		fmt.Fprintf(os.Stderr, "error: coverage counter data emit failed: %v\n", err)
		if os.Getenv("GOCOVERDEBUG") != "" {
			panic("counter-data write failure")
		}
	}
}

// emitCounterDataToDirectory emits the counter-data output file for this coverage run.
func emitCounterDataToDirectory(outdir string) error {
	// Ask the runtime for the list of coverage counter symbols.
	cl := getCovCounterList()
	if len(cl) == 0 {
		// no work to do here.
		return nil
	}

	if !finalHashComputed {
		return fmt.Errorf("error: meta-data not available (binary not built with -cover?)")
	}

	// Ask the runtime for the list of coverage counter symbols.
	pm := getCovPkgMap()
	s := &emitState{
		counterlist: cl,
		pkgmap:      pm,
		outdir:      outdir,
		debug:       os.Getenv("GOCOVERDEBUG") != "",
	}

	// Open output file.
	if err := s.openOutputFiles(finalHash, finalMetaLen, counterDataFile); err != nil {
		return err
	}
	if s.cf == nil {
		return fmt.Errorf("counter data output file open failed (no additional info")
	}

	// Emit counter data file.
	if err := s.emitCounterDataFile(finalHash, s.cf); err != nil {
		return err
	}
	if err := s.cf.Close(); err != nil {
		return fmt.Errorf("closing counter data file: %v", err)
	}

	// Counter file has now been closed. Rename the temp to the
	// final desired path.
	if err := os.Rename(s.cftmp, s.cfname); err != nil {
		return fmt.Errorf("writing %s: rename from %s failed: %v\n", s.cfname, s.cftmp, err)
	}

	return nil
}

// emitCounterDataToWriter emits counter data for this coverage run to an io.Writer.
func (s *emitState) emitCounterDataToWriter(w io.Writer) error {
	if err := s.emitCounterDataFile(finalHash, w); err != nil {
		return err
	}
	return nil
}

// openMetaFile determines whether we need to emit a meta-data output
// file, or whether we can reuse the existing file in the coverage out
// dir. It updates mfname/mftmp/mf fields in 's', returning an error
// if something went wrong. See the comment on the emitState type
// definition above for more on how file opening is managed.
func (s *emitState) openMetaFile(metaHash [16]byte, metaLen uint64) error {

	// Open meta-outfile for reading to see if it exists.
	fn := fmt.Sprintf("%s.%x", coverage.MetaFilePref, metaHash)
	s.mfname = filepath.Join(s.outdir, fn)
	fi, err := os.Stat(s.mfname)
	if err != nil || fi.Size() != int64(metaLen) {
		// We need a new meta-file.
		tname := "tmp." + fn + strconv.FormatInt(time.Now().UnixNano(), 10)
		s.mftmp = filepath.Join(s.outdir, tname)
		s.mf, err = os.Create(s.mftmp)
		if err != nil {
			return fmt.Errorf("creating meta-data file %s: %v", s.mftmp, err)
		}
	}
	return nil
}

// openCounterFile opens an output file for the counter data portion
// of a test coverage run. If updates the 'cfname' and 'cf' fields in
// 's', returning an error if something went wrong.
func (s *emitState) openCounterFile(metaHash [16]byte) error {
	processID := os.Getpid()
	fn := fmt.Sprintf(coverage.CounterFileTempl, coverage.CounterFilePref, metaHash, processID, time.Now().UnixNano())
	s.cfname = filepath.Join(s.outdir, fn)
	s.cftmp = filepath.Join(s.outdir, "tmp."+fn)
	var err error
	s.cf, err = os.Create(s.cftmp)
	if err != nil {
		return fmt.Errorf("creating counter data file %s: %v", s.cftmp, err)
	}
	return nil
}

// openOutputFiles opens output files in preparation for emitting
// coverage data. In the case of the meta-data file, openOutputFiles
// may determine that we can reuse an existing meta-data file in the
// outdir, in which case it will leave the 'mf' field in the state
// struct as nil. If a new meta-file is needed, the field 'mfname'
// will be the final desired path of the meta file, 'mftmp' will be a
// temporary file, and 'mf' will be an open os.File pointer for
// 'mftmp'. The idea is that the client/caller will write content into
// 'mf', close it, and then rename 'mftmp' to 'mfname'. This function
// also opens the counter data output file, setting 'cf' and 'cfname'
// in the state struct.
func (s *emitState) openOutputFiles(metaHash [16]byte, metaLen uint64, which fileType) error {
	fi, err := os.Stat(s.outdir)
	if err != nil {
		return fmt.Errorf("output directory %q inaccessible (err: %v); no coverage data written", s.outdir, err)
	}
	if !fi.IsDir() {
		return fmt.Errorf("output directory %q not a directory; no coverage data written", s.outdir)
	}

	if (which & metaDataFile) != 0 {
		if err := s.openMetaFile(metaHash, metaLen); err != nil {
			return err
		}
	}
	if (which & counterDataFile) != 0 {
		if err := s.openCounterFile(metaHash); err != nil {
			return err
		}
	}
	return nil
}

// emitMetaDataFile emits coverage meta-data to a previously opened
// temporary file (s.mftmp), then renames the generated file to the
// final path (s.mfname).
func (s *emitState) emitMetaDataFile(finalHash [16]byte, tlen uint64) error {
	if err := writeMetaData(s.mf, s.metalist, cmode, cgran, finalHash); err != nil {
		return fmt.Errorf("writing %s: %v\n", s.mftmp, err)
	}
	if err := s.mf.Close(); err != nil {
		return fmt.Errorf("closing meta data temp file: %v", err)
	}

	// Temp file has now been flushed and closed. Rename the temp to the
	// final desired path.
	if err := os.Rename(s.mftmp, s.mfname); err != nil {
		return fmt.Errorf("writing %s: rename from %s failed: %v\n", s.mfname, s.mftmp, err)
	}

	return nil
}

// needMetaDataFile returns TRUE if we need to emit a meta-data file
// for this program run. It should be used only after
// openOutputFiles() has been invoked.
func (s *emitState) needMetaDataFile() bool {
	return s.mf != nil
}

func writeMetaData(w io.Writer, metalist []rtcov.CovMetaBlob, cmode coverage.CounterMode, gran coverage.CounterGranularity, finalHash [16]byte) error {
	mfw := encodemeta.NewCoverageMetaFileWriter("<io.Writer>", w)

	var blobs [][]byte
	for _, e := range metalist {
		sd := unsafe.Slice(e.P, int(e.Len))
		blobs = append(blobs, sd)
	}
	return mfw.Write(finalHash, blobs, cmode, gran)
}

func (s *emitState) VisitFuncs(f encodecounter.CounterVisitorFn) error {
	var tcounters []uint32

	rdCounters := func(actrs []atomic.Uint32, ctrs []uint32) []uint32 {
		ctrs = ctrs[:0]
		for i := range actrs {
			ctrs = append(ctrs, actrs[i].Load())
		}
		return ctrs
	}

	dpkg := uint32(0)
	for _, c := range s.counterlist {
		sd := unsafe.Slice((*atomic.Uint32)(unsafe.Pointer(c.Counters)), int(c.Len))
		for i := 0; i < len(sd); i++ {
			// Skip ahead until the next non-zero value.
			sdi := sd[i].Load()
			if sdi == 0 {
				continue
			}

			// We found a function that was executed.
			nCtrs := sd[i+coverage.NumCtrsOffset].Load()
			pkgId := sd[i+coverage.PkgIdOffset].Load()
			funcId := sd[i+coverage.FuncIdOffset].Load()
			cst := i + coverage.FirstCtrOffset
			counters := sd[cst : cst+int(nCtrs)]

			// Check to make sure that we have at least one live
			// counter. See the implementation note in ClearCoverageCounters
			// for a description of why this is needed.
			isLive := false
			for i := 0; i < len(counters); i++ {
				if counters[i].Load() != 0 {
					isLive = true
					break
				}
			}
			if !isLive {
				// Skip this function.
				i += coverage.FirstCtrOffset + int(nCtrs) - 1
				continue
			}

			if s.debug {
				if pkgId != dpkg {
					dpkg = pkgId
					fmt.Fprintf(os.Stderr, "\n=+= %d: pk=%d visit live fcn",
						i, pkgId)
				}
				fmt.Fprintf(os.Stderr, " {i=%d F%d NC%d}", i, funcId, nCtrs)
			}

			// Vet and/or fix up package ID. A package ID of zero
			// indicates that there is some new package X that is a
			// runtime dependency, and this package has code that
			// executes before its corresponding init package runs.
			// This is a fatal error that we should only see during
			// Go development (e.g. tip).
			ipk := int32(pkgId)
			if ipk == 0 {
				fmt.Fprintf(os.Stderr, "\n")
				reportErrorInHardcodedList(int32(i), ipk, funcId, nCtrs)
			} else if ipk < 0 {
				if newId, ok := s.pkgmap[int(ipk)]; ok {
					pkgId = uint32(newId)
				} else {
					fmt.Fprintf(os.Stderr, "\n")
					reportErrorInHardcodedList(int32(i), ipk, funcId, nCtrs)
				}
			} else {
				// The package ID value stored in the counter array
				// has 1 added to it (so as to preclude the
				// possibility of a zero value ; see
				// runtime.addCovMeta), so subtract off 1 here to form
				// the real package ID.
				pkgId--
			}

			tcounters = rdCounters(counters, tcounters)
			if err := f(pkgId, funcId, tcounters); err != nil {
				return err
			}

			// Skip over this function.
			i += coverage.FirstCtrOffset + int(nCtrs) - 1
		}
		if s.debug {
			fmt.Fprintf(os.Stderr, "\n")
		}
	}
	return nil
}

// captureOsArgs converts os.Args() into the format we use to store
// this info in the counter data file (counter data file "args"
// section is a generic key-value collection). See the 'args' section
// in internal/coverage/defs.go for more info. The args map
// is also used to capture GOOS + GOARCH values as well.
func captureOsArgs() map[string]string {
	m := make(map[string]string)
	m["argc"] = strconv.Itoa(len(os.Args))
	for k, a := range os.Args {
		m[fmt.Sprintf("argv%d", k)] = a
	}
	m["GOOS"] = runtime.GOOS
	m["GOARCH"] = runtime.GOARCH
	return m
}

// emitCounterDataFile emits the counter data portion of a
// coverage output file (to the file 's.cf').
func (s *emitState) emitCounterDataFile(finalHash [16]byte, w io.Writer) error {
	cfw := encodecounter.NewCoverageDataWriter(w, coverage.CtrULeb128)
	if err := cfw.Write(finalHash, capturedOsArgs, s); err != nil {
		return err
	}
	return nil
}

// markProfileEmitted signals the runtime/coverage machinery that
// coverage data output files have already been written out, and there
// is no need to take any additional action at exit time. This
// function is called (via linknamed reference) from the
// coverage-related boilerplate code in _testmain.go emitted for go
// unit tests.
func markProfileEmitted(val bool) {
	covProfileAlreadyEmitted = val
}

func reportErrorInHardcodedList(slot, pkgID int32, fnID, nCtrs uint32) {
	metaList := getCovMetaList()
	pkgMap := getCovPkgMap()

	println("internal error in coverage meta-data tracking:")
	println("encountered bad pkgID:", pkgID, " at slot:", slot,
		" fnID:", fnID, " numCtrs:", nCtrs)
	println("list of hard-coded runtime package IDs needs revising.")
	println("[see the comment on the 'rtPkgs' var in ")
	println(" <goroot>/src/internal/coverage/pkid.go]")
	println("registered list:")
	for k, b := range metaList {
		print("slot: ", k, " path='", b.PkgPath, "' ")
		if b.PkgID != -1 {
			print(" hard-coded id: ", b.PkgID)
		}
		println("")
	}
	println("remap table:")
	for from, to := range pkgMap {
		println("from ", from, " to ", to)
	}
}
