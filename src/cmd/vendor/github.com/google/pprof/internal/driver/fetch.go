// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package driver

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/pprof/internal/measurement"
	"github.com/google/pprof/internal/plugin"
	"github.com/google/pprof/profile"
)

// fetchProfiles fetches and symbolizes the profiles specified by s.
// It will merge all the profiles it is able to retrieve, even if
// there are some failures. It will return an error if it is unable to
// fetch any profiles.
func fetchProfiles(s *source, o *plugin.Options) (*profile.Profile, error) {
	sources := make([]profileSource, 0, len(s.Sources))
	for _, src := range s.Sources {
		sources = append(sources, profileSource{
			addr:   src,
			source: s,
		})
	}

	bases := make([]profileSource, 0, len(s.Base))
	for _, src := range s.Base {
		bases = append(bases, profileSource{
			addr:   src,
			source: s,
		})
	}

	p, pbase, m, mbase, save, err := grabSourcesAndBases(sources, bases, o.Fetch, o.Obj, o.UI, o.HTTPTransport)
	if err != nil {
		return nil, err
	}

	if pbase != nil {
		if s.DiffBase {
			pbase.SetLabel("pprof::base", []string{"true"})
		}
		if s.Normalize {
			err := p.Normalize(pbase)
			if err != nil {
				return nil, err
			}
		}
		pbase.Scale(-1)
		p, m, err = combineProfiles([]*profile.Profile{p, pbase}, []plugin.MappingSources{m, mbase})
		if err != nil {
			return nil, err
		}
	}

	// Symbolize the merged profile.
	if err := o.Sym.Symbolize(s.Symbolize, m, p); err != nil {
		return nil, err
	}
	p.RemoveUninteresting()
	unsourceMappings(p)

	if s.Comment != "" {
		p.Comments = append(p.Comments, s.Comment)
	}

	// Save a copy of the merged profile if there is at least one remote source.
	if save {
		dir, err := setTmpDir(o.UI)
		if err != nil {
			return nil, err
		}

		prefix := "pprof."
		if len(p.Mapping) > 0 && p.Mapping[0].File != "" {
			prefix += filepath.Base(p.Mapping[0].File) + "."
		}
		for _, s := range p.SampleType {
			prefix += s.Type + "."
		}

		tempFile, err := newTempFile(dir, prefix, ".pb.gz")
		if err == nil {
			if err = p.Write(tempFile); err == nil {
				o.UI.PrintErr("Saved profile in ", tempFile.Name())
			}
		}
		if err != nil {
			o.UI.PrintErr("Could not save profile: ", err)
		}
	}

	if err := p.CheckValid(); err != nil {
		return nil, err
	}

	return p, nil
}

func grabSourcesAndBases(sources, bases []profileSource, fetch plugin.Fetcher, obj plugin.ObjTool, ui plugin.UI, tr http.RoundTripper) (*profile.Profile, *profile.Profile, plugin.MappingSources, plugin.MappingSources, bool, error) {
	wg := sync.WaitGroup{}
	wg.Add(2)
	var psrc, pbase *profile.Profile
	var msrc, mbase plugin.MappingSources
	var savesrc, savebase bool
	var errsrc, errbase error
	var countsrc, countbase int
	go func() {
		defer wg.Done()
		psrc, msrc, savesrc, countsrc, errsrc = chunkedGrab(sources, fetch, obj, ui, tr)
	}()
	go func() {
		defer wg.Done()
		pbase, mbase, savebase, countbase, errbase = chunkedGrab(bases, fetch, obj, ui, tr)
	}()
	wg.Wait()
	save := savesrc || savebase

	if errsrc != nil {
		return nil, nil, nil, nil, false, fmt.Errorf("problem fetching source profiles: %v", errsrc)
	}
	if errbase != nil {
		return nil, nil, nil, nil, false, fmt.Errorf("problem fetching base profiles: %v,", errbase)
	}
	if countsrc == 0 {
		return nil, nil, nil, nil, false, fmt.Errorf("failed to fetch any source profiles")
	}
	if countbase == 0 && len(bases) > 0 {
		return nil, nil, nil, nil, false, fmt.Errorf("failed to fetch any base profiles")
	}
	if want, got := len(sources), countsrc; want != got {
		ui.PrintErr(fmt.Sprintf("Fetched %d source profiles out of %d", got, want))
	}
	if want, got := len(bases), countbase; want != got {
		ui.PrintErr(fmt.Sprintf("Fetched %d base profiles out of %d", got, want))
	}

	return psrc, pbase, msrc, mbase, save, nil
}

// chunkedGrab fetches the profiles described in source and merges them into
// a single profile. It fetches a chunk of profiles concurrently, with a maximum
// chunk size to limit its memory usage.
func chunkedGrab(sources []profileSource, fetch plugin.Fetcher, obj plugin.ObjTool, ui plugin.UI, tr http.RoundTripper) (*profile.Profile, plugin.MappingSources, bool, int, error) {
	const chunkSize = 128

	var p *profile.Profile
	var msrc plugin.MappingSources
	var save bool
	var count int

	for start := 0; start < len(sources); start += chunkSize {
		end := min(start+chunkSize, len(sources))
		chunkP, chunkMsrc, chunkSave, chunkCount, chunkErr := concurrentGrab(sources[start:end], fetch, obj, ui, tr)
		switch {
		case chunkErr != nil:
			return nil, nil, false, 0, chunkErr
		case chunkP == nil:
			continue
		case p == nil:
			p, msrc, save, count = chunkP, chunkMsrc, chunkSave, chunkCount
		default:
			p, msrc, chunkErr = combineProfiles([]*profile.Profile{p, chunkP}, []plugin.MappingSources{msrc, chunkMsrc})
			if chunkErr != nil {
				return nil, nil, false, 0, chunkErr
			}
			if chunkSave {
				save = true
			}
			count += chunkCount
		}
	}

	return p, msrc, save, count, nil
}

// concurrentGrab fetches multiple profiles concurrently
func concurrentGrab(sources []profileSource, fetch plugin.Fetcher, obj plugin.ObjTool, ui plugin.UI, tr http.RoundTripper) (*profile.Profile, plugin.MappingSources, bool, int, error) {
	wg := sync.WaitGroup{}
	wg.Add(len(sources))
	for i := range sources {
		go func(s *profileSource) {
			defer wg.Done()
			s.p, s.msrc, s.remote, s.err = grabProfile(s.source, s.addr, fetch, obj, ui, tr)
		}(&sources[i])
	}
	wg.Wait()

	var save bool
	profiles := make([]*profile.Profile, 0, len(sources))
	msrcs := make([]plugin.MappingSources, 0, len(sources))
	for i := range sources {
		s := &sources[i]
		if err := s.err; err != nil {
			ui.PrintErr(s.addr + ": " + err.Error())
			continue
		}
		save = save || s.remote
		profiles = append(profiles, s.p)
		msrcs = append(msrcs, s.msrc)
		*s = profileSource{}
	}

	if len(profiles) == 0 {
		return nil, nil, false, 0, nil
	}

	p, msrc, err := combineProfiles(profiles, msrcs)
	if err != nil {
		return nil, nil, false, 0, err
	}
	return p, msrc, save, len(profiles), nil
}

func combineProfiles(profiles []*profile.Profile, msrcs []plugin.MappingSources) (*profile.Profile, plugin.MappingSources, error) {
	// Merge profiles.
	//
	// The merge call below only treats exactly matching sample type lists as
	// compatible and will fail otherwise. Make the profiles' sample types
	// compatible for the merge, see CompatibilizeSampleTypes() doc for details.
	if err := profile.CompatibilizeSampleTypes(profiles); err != nil {
		return nil, nil, err
	}
	if err := measurement.ScaleProfiles(profiles); err != nil {
		return nil, nil, err
	}

	// Avoid expensive work for the common case of a single profile/src.
	if len(profiles) == 1 && len(msrcs) == 1 {
		return profiles[0], msrcs[0], nil
	}

	p, err := profile.Merge(profiles)
	if err != nil {
		return nil, nil, err
	}

	// Combine mapping sources.
	msrc := make(plugin.MappingSources)
	for _, ms := range msrcs {
		for m, s := range ms {
			msrc[m] = append(msrc[m], s...)
		}
	}
	return p, msrc, nil
}

type profileSource struct {
	addr   string
	source *source

	p      *profile.Profile
	msrc   plugin.MappingSources
	remote bool
	err    error
}

func homeEnv() string {
	switch runtime.GOOS {
	case "windows":
		return "USERPROFILE"
	case "plan9":
		return "home"
	default:
		return "HOME"
	}
}

// setTmpDir prepares the directory to use to save profiles retrieved
// remotely. It is selected from PPROF_TMPDIR, defaults to $HOME/pprof, and, if
// $HOME is not set, falls back to os.TempDir().
func setTmpDir(ui plugin.UI) (string, error) {
	var dirs []string
	if profileDir := os.Getenv("PPROF_TMPDIR"); profileDir != "" {
		dirs = append(dirs, profileDir)
	}
	if homeDir := os.Getenv(homeEnv()); homeDir != "" {
		dirs = append(dirs, filepath.Join(homeDir, "pprof"))
	}
	dirs = append(dirs, os.TempDir())
	for _, tmpDir := range dirs {
		if err := os.MkdirAll(tmpDir, 0755); err != nil {
			ui.PrintErr("Could not use temp dir ", tmpDir, ": ", err.Error())
			continue
		}
		return tmpDir, nil
	}
	return "", fmt.Errorf("failed to identify temp dir")
}

const testSourceAddress = "pproftest.local"

// grabProfile fetches a profile. Returns the profile, sources for the
// profile mappings, a bool indicating if the profile was fetched
// remotely, and an error.
func grabProfile(s *source, source string, fetcher plugin.Fetcher, obj plugin.ObjTool, ui plugin.UI, tr http.RoundTripper) (p *profile.Profile, msrc plugin.MappingSources, remote bool, err error) {
	var src string
	duration, timeout := time.Duration(s.Seconds)*time.Second, time.Duration(s.Timeout)*time.Second
	if fetcher != nil {
		p, src, err = fetcher.Fetch(source, duration, timeout)
		if err != nil {
			return
		}
	}
	if err != nil || p == nil {
		// Fetch the profile over HTTP or from a file.
		p, src, err = fetch(source, duration, timeout, ui, tr)
		if err != nil {
			return
		}
	}

	if err = p.CheckValid(); err != nil {
		return
	}

	// Update the binary locations from command line and paths.
	locateBinaries(p, s, obj, ui)

	// Collect the source URL for all mappings.
	if src != "" {
		msrc = collectMappingSources(p, src)
		remote = true
		if strings.HasPrefix(src, "http://"+testSourceAddress) {
			// Treat test inputs as local to avoid saving
			// testcase profiles during driver testing.
			remote = false
		}
	}
	return
}

// collectMappingSources saves the mapping sources of a profile.
func collectMappingSources(p *profile.Profile, source string) plugin.MappingSources {
	ms := plugin.MappingSources{}
	for _, m := range p.Mapping {
		src := struct {
			Source string
			Start  uint64
		}{
			source, m.Start,
		}
		key := m.BuildID
		if key == "" {
			key = m.File
		}
		if key == "" {
			// If there is no build id or source file, use the source as the
			// mapping file. This will enable remote symbolization for this
			// mapping, in particular for Go profiles on the legacy format.
			// The source is reset back to empty string by unsourceMapping
			// which is called after symbolization is finished.
			m.File = source
			key = source
		}
		ms[key] = append(ms[key], src)
	}
	return ms
}

// unsourceMappings iterates over the mappings in a profile and replaces file
// set to the remote source URL by collectMappingSources back to empty string.
func unsourceMappings(p *profile.Profile) {
	for _, m := range p.Mapping {
		if m.BuildID == "" && filepath.VolumeName(m.File) == "" {
			if u, err := url.Parse(m.File); err == nil && u.IsAbs() {
				m.File = ""
			}
		}
	}
}

// locateBinaries searches for binary files listed in the profile and, if found,
// updates the profile accordingly.
func locateBinaries(p *profile.Profile, s *source, obj plugin.ObjTool, ui plugin.UI) {
	// Construct search path to examine
	searchPath := os.Getenv("PPROF_BINARY_PATH")
	if searchPath == "" {
		// Use $HOME/pprof/binaries as default directory for local symbolization binaries
		searchPath = filepath.Join(os.Getenv(homeEnv()), "pprof", "binaries")
	}
mapping:
	for _, m := range p.Mapping {
		var noVolumeFile string
		var baseName string
		var dirName string
		if m.File != "" {
			noVolumeFile = strings.TrimPrefix(m.File, filepath.VolumeName(m.File))
			baseName = filepath.Base(m.File)
			dirName = filepath.Dir(noVolumeFile)
		}

		for _, path := range filepath.SplitList(searchPath) {
			var fileNames []string
			if m.BuildID != "" {
				fileNames = []string{filepath.Join(path, m.BuildID, baseName)}
				if matches, err := filepath.Glob(filepath.Join(path, m.BuildID, "*")); err == nil {
					fileNames = append(fileNames, matches...)
				}
				fileNames = append(fileNames, filepath.Join(path, noVolumeFile, m.BuildID)) // perf path format
				// Llvm buildid protocol: the first two characters of the build id
				// are used as directory, and the remaining part is in the filename.
				// e.g. `/ab/cdef0123456.debug`
				fileNames = append(fileNames, filepath.Join(path, m.BuildID[:2], m.BuildID[2:]+".debug"))
			}
			if m.File != "" {
				// Try both the basename and the full path, to support the same directory
				// structure as the perf symfs option.
				fileNames = append(fileNames, filepath.Join(path, baseName))
				fileNames = append(fileNames, filepath.Join(path, noVolumeFile))
				// Other locations: use the same search paths as GDB, according to
				// https://sourceware.org/gdb/onlinedocs/gdb/Separate-Debug-Files.html
				fileNames = append(fileNames, filepath.Join(path, noVolumeFile+".debug"))
				fileNames = append(fileNames, filepath.Join(path, dirName, ".debug", baseName+".debug"))
				fileNames = append(fileNames, filepath.Join(path, "usr", "lib", "debug", dirName, baseName+".debug"))
			}
			for _, name := range fileNames {
				if f, err := obj.Open(name, m.Start, m.Limit, m.Offset, m.KernelRelocationSymbol); err == nil {
					defer f.Close()
					fileBuildID := f.BuildID()
					if m.BuildID != "" && m.BuildID != fileBuildID {
						ui.PrintErr("Ignoring local file " + name + ": build-id mismatch (" + m.BuildID + " != " + fileBuildID + ")")
					} else {
						// Explicitly do not update KernelRelocationSymbol --
						// the new local file name is most likely missing it.
						m.File = name
						continue mapping
					}
				}
			}
		}
	}
	if len(p.Mapping) == 0 {
		// If there are no mappings, add a fake mapping to attempt symbolization.
		// This is useful for some profiles generated by the golang runtime, which
		// do not include any mappings. Symbolization with a fake mapping will only
		// be successful against a non-PIE binary.
		m := &profile.Mapping{ID: 1}
		p.Mapping = []*profile.Mapping{m}
		for _, l := range p.Location {
			l.Mapping = m
		}
	}
	// If configured, apply executable filename override and (maybe, see below)
	// build ID override from source. Assume the executable is the first mapping.
	if execName, buildID := s.ExecName, s.BuildID; execName != "" || buildID != "" {
		m := p.Mapping[0]
		if execName != "" {
			// Explicitly do not update KernelRelocationSymbol --
			// the source override is most likely missing it.
			m.File = execName
		}
		// Only apply the build ID override if the build ID in the main mapping is
		// missing. Overwriting the build ID in case it's present is very likely a
		// wrong thing to do so we refuse to do that.
		if buildID != "" && m.BuildID == "" {
			m.BuildID = buildID
		}
	}
}

// fetch fetches a profile from source, within the timeout specified,
// producing messages through the ui. It returns the profile and the
// url of the actual source of the profile for remote profiles.
func fetch(source string, duration, timeout time.Duration, ui plugin.UI, tr http.RoundTripper) (p *profile.Profile, src string, err error) {
	var f io.ReadCloser

	// First determine whether the source is a file, if not, it will be treated as a URL.
	if _, err = os.Stat(source); err == nil {
		if isPerfFile(source) {
			f, err = convertPerfData(source, ui)
		} else {
			f, err = os.Open(source)
		}
	} else {
		sourceURL, timeout := adjustURL(source, duration, timeout)
		if sourceURL != "" {
			ui.Print("Fetching profile over HTTP from " + sourceURL)
			if duration > 0 {
				ui.Print(fmt.Sprintf("Please wait... (%v)", duration))
			}
			f, err = fetchURL(sourceURL, timeout, tr)
			src = sourceURL
		}
	}
	if err == nil {
		defer f.Close()
		p, err = profile.Parse(f)
	}
	return
}

// fetchURL fetches a profile from a URL using HTTP.
func fetchURL(source string, timeout time.Duration, tr http.RoundTripper) (io.ReadCloser, error) {
	client := &http.Client{
		Transport: tr,
		Timeout:   timeout + 5*time.Second,
	}
	resp, err := client.Get(source)
	if err != nil {
		return nil, fmt.Errorf("http fetch: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, statusCodeError(resp)
	}

	return resp.Body, nil
}

func statusCodeError(resp *http.Response) error {
	if resp.Header.Get("X-Go-Pprof") != "" && strings.Contains(resp.Header.Get("Content-Type"), "text/plain") {
		// error is from pprof endpoint
		if body, err := io.ReadAll(resp.Body); err == nil {
			return fmt.Errorf("server response: %s - %s", resp.Status, body)
		}
	}
	return fmt.Errorf("server response: %s", resp.Status)
}

// isPerfFile checks if a file is in perf.data format. It also returns false
// if it encounters an error during the check.
func isPerfFile(path string) bool {
	sourceFile, openErr := os.Open(path)
	if openErr != nil {
		return false
	}
	defer sourceFile.Close()

	// If the file is the output of a perf record command, it should begin
	// with the string PERFILE2.
	perfHeader := []byte("PERFILE2")
	actualHeader := make([]byte, len(perfHeader))
	if _, readErr := sourceFile.Read(actualHeader); readErr != nil {
		return false
	}
	return bytes.Equal(actualHeader, perfHeader)
}

// convertPerfData converts the file at path which should be in perf.data format
// using the perf_to_profile tool and returns the file containing the
// profile.proto formatted data.
func convertPerfData(perfPath string, ui plugin.UI) (*os.File, error) {
	ui.Print(fmt.Sprintf(
		"Converting %s to a profile.proto... (May take a few minutes)",
		perfPath))
	profile, err := newTempFile(os.TempDir(), "pprof_", ".pb.gz")
	if err != nil {
		return nil, err
	}
	deferDeleteTempFile(profile.Name())
	cmd := exec.Command("perf_to_profile", "-i", perfPath, "-o", profile.Name(), "-f")
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		profile.Close()
		return nil, fmt.Errorf("failed to convert perf.data file. Try github.com/google/perf_data_converter: %v", err)
	}
	return profile, nil
}

// adjustURL validates if a profile source is a URL and returns an
// cleaned up URL and the timeout to use for retrieval over HTTP.
// If the source cannot be recognized as a URL it returns an empty string.
func adjustURL(source string, duration, timeout time.Duration) (string, time.Duration) {
	u, err := url.Parse(source)
	if err != nil || (u.Host == "" && u.Scheme != "" && u.Scheme != "file") {
		// Try adding http:// to catch sources of the form hostname:port/path.
		// url.Parse treats "hostname" as the scheme.
		u, err = url.Parse("http://" + source)
	}
	if err != nil || u.Host == "" {
		return "", 0
	}

	// Apply duration/timeout overrides to URL.
	values := u.Query()
	if duration > 0 {
		values.Set("seconds", fmt.Sprint(int(duration.Seconds())))
	} else {
		if urlSeconds := values.Get("seconds"); urlSeconds != "" {
			if us, err := strconv.ParseInt(urlSeconds, 10, 32); err == nil {
				duration = time.Duration(us) * time.Second
			}
		}
	}
	if timeout <= 0 {
		if duration > 0 {
			timeout = duration + duration/2
		} else {
			timeout = 60 * time.Second
		}
	}
	u.RawQuery = values.Encode()
	return u.String(), timeout
}
