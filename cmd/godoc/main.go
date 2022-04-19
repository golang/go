// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// godoc: Go Documentation Server

// Web server tree:
//
//	http://godoc/		redirect to /pkg/
//	http://godoc/src/	serve files from $GOROOT/src; .go gets pretty-printed
//	http://godoc/cmd/	serve documentation about commands
//	http://godoc/pkg/	serve documentation about packages
//				(idea is if you say import "compress/zlib", you go to
//				http://godoc/pkg/compress/zlib)
//

package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	_ "expvar" // to serve /debug/vars
	"flag"
	"fmt"
	"go/build"
	"io"
	"log"
	"net/http"
	_ "net/http/pprof" // to serve /debug/pprof/*
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	exec "golang.org/x/sys/execabs"

	"golang.org/x/tools/godoc"
	"golang.org/x/tools/godoc/static"
	"golang.org/x/tools/godoc/vfs"
	"golang.org/x/tools/godoc/vfs/gatefs"
	"golang.org/x/tools/godoc/vfs/mapfs"
	"golang.org/x/tools/godoc/vfs/zipfs"
	"golang.org/x/tools/internal/gocommand"
)

const defaultAddr = "localhost:6060" // default webserver address

var (
	// file system to serve
	// (with e.g.: zip -r go.zip $GOROOT -i \*.go -i \*.html -i \*.css -i \*.js -i \*.txt -i \*.c -i \*.h -i \*.s -i \*.png -i \*.jpg -i \*.sh -i favicon.ico)
	zipfile = flag.String("zip", "", "zip file providing the file system to serve; disabled if empty")

	// file-based index
	writeIndex = flag.Bool("write_index", false, "write index to a file; the file name must be specified with -index_files")

	// network
	httpAddr = flag.String("http", defaultAddr, "HTTP service address")

	// layout control
	urlFlag = flag.String("url", "", "print HTML for named URL")

	verbose = flag.Bool("v", false, "verbose mode")

	// file system roots
	// TODO(gri) consider the invariant that goroot always end in '/'
	goroot = flag.String("goroot", findGOROOT(), "Go root directory")

	// layout control
	showTimestamps = flag.Bool("timestamps", false, "show timestamps with directory listings")
	templateDir    = flag.String("templates", "", "load templates/JS/CSS from disk in this directory")
	showPlayground = flag.Bool("play", false, "enable playground")
	declLinks      = flag.Bool("links", true, "link identifiers to their declarations")

	// search index
	indexEnabled  = flag.Bool("index", false, "enable search index")
	indexFiles    = flag.String("index_files", "", "glob pattern specifying index files; if not empty, the index is read from these files in sorted order")
	indexInterval = flag.Duration("index_interval", 0, "interval of indexing; 0 for default (5m), negative to only index once at startup")
	maxResults    = flag.Int("maxresults", 10000, "maximum number of full text search results shown")
	indexThrottle = flag.Float64("index_throttle", 0.75, "index throttle value; 0.0 = no time allocated, 1.0 = full throttle")

	// source code notes
	notesRx = flag.String("notes", "BUG", "regular expression matching note markers to show")
)

// An httpResponseRecorder is an http.ResponseWriter
type httpResponseRecorder struct {
	body   *bytes.Buffer
	header http.Header
	code   int
}

func (w *httpResponseRecorder) Header() http.Header         { return w.header }
func (w *httpResponseRecorder) Write(b []byte) (int, error) { return w.body.Write(b) }
func (w *httpResponseRecorder) WriteHeader(code int)        { w.code = code }

func usage() {
	fmt.Fprintf(os.Stderr, "usage: godoc -http="+defaultAddr+"\n")
	flag.PrintDefaults()
	os.Exit(2)
}

func loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		log.Printf("%s\t%s", req.RemoteAddr, req.URL)
		h.ServeHTTP(w, req)
	})
}

func handleURLFlag() {
	// Try up to 10 fetches, following redirects.
	urlstr := *urlFlag
	for i := 0; i < 10; i++ {
		// Prepare request.
		u, err := url.Parse(urlstr)
		if err != nil {
			log.Fatal(err)
		}
		req := &http.Request{
			URL: u,
		}

		// Invoke default HTTP handler to serve request
		// to our buffering httpWriter.
		w := &httpResponseRecorder{code: 200, header: make(http.Header), body: new(bytes.Buffer)}
		http.DefaultServeMux.ServeHTTP(w, req)

		// Return data, error, or follow redirect.
		switch w.code {
		case 200: // ok
			os.Stdout.Write(w.body.Bytes())
			return
		case 301, 302, 303, 307: // redirect
			redirect := w.header.Get("Location")
			if redirect == "" {
				log.Fatalf("HTTP %d without Location header", w.code)
			}
			urlstr = redirect
		default:
			log.Fatalf("HTTP error %d", w.code)
		}
	}
	log.Fatalf("too many redirects")
}

func initCorpus(corpus *godoc.Corpus) {
	err := corpus.Init()
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	flag.Usage = usage
	flag.Parse()

	// Check usage.
	if flag.NArg() > 0 {
		fmt.Fprintln(os.Stderr, `Unexpected arguments. Use "go doc" for command-line help output instead. For example, "go doc fmt.Printf".`)
		usage()
	}
	if *httpAddr == "" && *urlFlag == "" && !*writeIndex {
		fmt.Fprintln(os.Stderr, "At least one of -http, -url, or -write_index must be set to a non-zero value.")
		usage()
	}

	// Set the resolved goroot.
	vfs.GOROOT = *goroot

	fsGate := make(chan bool, 20)

	// Determine file system to use.
	if *zipfile == "" {
		// use file system of underlying OS
		rootfs := gatefs.New(vfs.OS(*goroot), fsGate)
		fs.Bind("/", rootfs, "/", vfs.BindReplace)
	} else {
		// use file system specified via .zip file (path separator must be '/')
		rc, err := zip.OpenReader(*zipfile)
		if err != nil {
			log.Fatalf("%s: %s\n", *zipfile, err)
		}
		defer rc.Close() // be nice (e.g., -writeIndex mode)
		fs.Bind("/", zipfs.New(rc, *zipfile), *goroot, vfs.BindReplace)
	}
	if *templateDir != "" {
		fs.Bind("/lib/godoc", vfs.OS(*templateDir), "/", vfs.BindBefore)
		fs.Bind("/favicon.ico", vfs.OS(*templateDir), "/favicon.ico", vfs.BindReplace)
	} else {
		fs.Bind("/lib/godoc", mapfs.New(static.Files), "/", vfs.BindReplace)
		fs.Bind("/favicon.ico", mapfs.New(static.Files), "/favicon.ico", vfs.BindReplace)
	}

	// Get the GOMOD value, use it to determine if godoc is being invoked in module mode.
	goModFile, err := goMod()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to determine go env GOMOD value: %v", err)
		goModFile = "" // Fall back to GOPATH mode.
	}

	if goModFile != "" {
		fmt.Printf("using module mode; GOMOD=%s\n", goModFile)

		// Detect whether to use vendor mode or not.
		vendorEnabled, mainModVendor, err := gocommand.VendorEnabled(context.Background(), gocommand.Invocation{}, &gocommand.Runner{})
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to determine if vendoring is enabled: %v", err)
			os.Exit(1)
		}
		if vendorEnabled {
			// Bind the root directory of the main module.
			fs.Bind(path.Join("/src", mainModVendor.Path), gatefs.New(vfs.OS(mainModVendor.Dir), fsGate), "/", vfs.BindAfter)

			// Bind the vendor directory.
			//
			// Note that in module mode, vendor directories in locations
			// other than the main module's root directory are ignored.
			// See https://golang.org/ref/mod#vendoring.
			vendorDir := filepath.Join(mainModVendor.Dir, "vendor")
			fs.Bind("/src", gatefs.New(vfs.OS(vendorDir), fsGate), "/", vfs.BindAfter)

		} else {
			// Try to download dependencies that are not in the module cache in order to
			// to show their documentation.
			// This may fail if module downloading is disallowed (GOPROXY=off) or due to
			// limited connectivity, in which case we print errors to stderr and show
			// documentation only for packages that are available.
			fillModuleCache(os.Stderr, goModFile)

			// Determine modules in the build list.
			mods, err := buildList(goModFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "failed to determine the build list of the main module: %v", err)
				os.Exit(1)
			}

			// Bind module trees into Go root.
			for _, m := range mods {
				if m.Dir == "" {
					// Module is not available in the module cache, skip it.
					continue
				}
				dst := path.Join("/src", m.Path)
				fs.Bind(dst, gatefs.New(vfs.OS(m.Dir), fsGate), "/", vfs.BindAfter)
			}
		}
	} else {
		fmt.Println("using GOPATH mode")

		// Bind $GOPATH trees into Go root.
		for _, p := range filepath.SplitList(build.Default.GOPATH) {
			fs.Bind("/src", gatefs.New(vfs.OS(p), fsGate), "/src", vfs.BindAfter)
		}
	}

	var corpus *godoc.Corpus
	if goModFile != "" {
		corpus = godoc.NewCorpus(moduleFS{fs})
	} else {
		corpus = godoc.NewCorpus(fs)
	}
	corpus.Verbose = *verbose
	corpus.MaxResults = *maxResults
	corpus.IndexEnabled = *indexEnabled
	if *maxResults == 0 {
		corpus.IndexFullText = false
	}
	corpus.IndexFiles = *indexFiles
	corpus.IndexDirectory = func(dir string) bool {
		return dir != "/pkg" && !strings.HasPrefix(dir, "/pkg/")
	}
	corpus.IndexThrottle = *indexThrottle
	corpus.IndexInterval = *indexInterval
	if *writeIndex || *urlFlag != "" {
		corpus.IndexThrottle = 1.0
		corpus.IndexEnabled = true
		initCorpus(corpus)
	} else {
		go initCorpus(corpus)
	}

	// Initialize the version info before readTemplates, which saves
	// the map value in a method value.
	corpus.InitVersionInfo()

	pres = godoc.NewPresentation(corpus)
	pres.ShowTimestamps = *showTimestamps
	pres.ShowPlayground = *showPlayground
	pres.DeclLinks = *declLinks
	if *notesRx != "" {
		pres.NotesRx = regexp.MustCompile(*notesRx)
	}

	readTemplates(pres)
	registerHandlers(pres)

	if *writeIndex {
		// Write search index and exit.
		if *indexFiles == "" {
			log.Fatal("no index file specified")
		}

		log.Println("initialize file systems")
		*verbose = true // want to see what happens

		corpus.UpdateIndex()

		log.Println("writing index file", *indexFiles)
		f, err := os.Create(*indexFiles)
		if err != nil {
			log.Fatal(err)
		}
		index, _ := corpus.CurrentIndex()
		_, err = index.WriteTo(f)
		if err != nil {
			log.Fatal(err)
		}

		log.Println("done")
		return
	}

	// Print content that would be served at the URL *urlFlag.
	if *urlFlag != "" {
		handleURLFlag()
		return
	}

	var handler http.Handler = http.DefaultServeMux
	if *verbose {
		log.Printf("Go Documentation Server")
		log.Printf("version = %s", runtime.Version())
		log.Printf("address = %s", *httpAddr)
		log.Printf("goroot = %s", *goroot)
		switch {
		case !*indexEnabled:
			log.Print("search index disabled")
		case *maxResults > 0:
			log.Printf("full text index enabled (maxresults = %d)", *maxResults)
		default:
			log.Print("identifier search index enabled")
		}
		fs.Fprint(os.Stderr)
		handler = loggingHandler(handler)
	}

	// Initialize search index.
	if *indexEnabled {
		go corpus.RunIndexer()
	}

	// Start http server.
	if *verbose {
		log.Println("starting HTTP server")
	}
	if err := http.ListenAndServe(*httpAddr, handler); err != nil {
		log.Fatalf("ListenAndServe %s: %v", *httpAddr, err)
	}
}

// goMod returns the go env GOMOD value in the current directory
// by invoking the go command.
//
// GOMOD is documented at https://golang.org/cmd/go/#hdr-Environment_variables:
//
//	The absolute path to the go.mod of the main module,
//	or the empty string if not using modules.
func goMod() (string, error) {
	out, err := exec.Command("go", "env", "-json", "GOMOD").Output()
	if ee := (*exec.ExitError)(nil); errors.As(err, &ee) {
		return "", fmt.Errorf("go command exited unsuccessfully: %v\n%s", ee.ProcessState.String(), ee.Stderr)
	} else if err != nil {
		return "", err
	}
	var env struct {
		GoMod string
	}
	err = json.Unmarshal(out, &env)
	if err != nil {
		return "", err
	}
	return env.GoMod, nil
}

// fillModuleCache does a best-effort attempt to fill the module cache
// with all dependencies of the main module in the current directory
// by invoking the go command. Module download logs are streamed to w.
// If there are any problems encountered, they are also written to w.
// It should only be used in module mode, when vendor mode isn't on.
//
// See https://golang.org/cmd/go/#hdr-Download_modules_to_local_cache.
func fillModuleCache(w io.Writer, goMod string) {
	if goMod == os.DevNull {
		// No module requirements, nothing to do.
		return
	}

	cmd := exec.Command("go", "mod", "download", "-json")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = w
	err := cmd.Run()
	if ee := (*exec.ExitError)(nil); errors.As(err, &ee) && ee.ExitCode() == 1 {
		// Exit code 1 from this command means there were some
		// non-empty Error values in the output. Print them to w.
		fmt.Fprintf(w, "documentation for some packages is not shown:\n")
		for dec := json.NewDecoder(&out); ; {
			var m struct {
				Path    string // Module path.
				Version string // Module version.
				Error   string // Error loading module.
			}
			err := dec.Decode(&m)
			if err == io.EOF {
				break
			} else if err != nil {
				fmt.Fprintf(w, "error decoding JSON object from go mod download -json: %v\n", err)
				continue
			}
			if m.Error == "" {
				continue
			}
			fmt.Fprintf(w, "\tmodule %s@%s is not in the module cache and there was a problem downloading it: %s\n", m.Path, m.Version, m.Error)
		}
	} else if err != nil {
		fmt.Fprintf(w, "there was a problem filling module cache: %v\n", err)
	}
}

type mod struct {
	Path string // Module path.
	Dir  string // Directory holding files for this module, if any.
}

// buildList determines the build list in the current directory
// by invoking the go command. It should only be used in module mode,
// when vendor mode isn't on.
//
// See https://golang.org/cmd/go/#hdr-The_main_module_and_the_build_list.
func buildList(goMod string) ([]mod, error) {
	if goMod == os.DevNull {
		// Empty build list.
		return nil, nil
	}

	out, err := exec.Command("go", "list", "-m", "-json", "all").Output()
	if ee := (*exec.ExitError)(nil); errors.As(err, &ee) {
		return nil, fmt.Errorf("go command exited unsuccessfully: %v\n%s", ee.ProcessState.String(), ee.Stderr)
	} else if err != nil {
		return nil, err
	}
	var mods []mod
	for dec := json.NewDecoder(bytes.NewReader(out)); ; {
		var m mod
		err := dec.Decode(&m)
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		mods = append(mods, m)
	}
	return mods, nil
}

// moduleFS is a vfs.FileSystem wrapper used when godoc is running
// in module mode. It's needed so that packages inside modules are
// considered to be third party.
//
// It overrides the RootType method of the underlying filesystem
// and implements it using a heuristic based on the import path.
// If the first element of the import path does not contain a dot,
// that package is considered to be inside GOROOT. If it contains
// a dot, then that package is considered to be third party.
//
// TODO(dmitshur): The RootType abstraction works well when GOPATH
// workspaces are bound at their roots, but scales poorly in the
// general case. It should be replaced by a more direct solution
// for determining whether a package is third party or not.
type moduleFS struct{ vfs.FileSystem }

func (moduleFS) RootType(path string) vfs.RootType {
	if !strings.HasPrefix(path, "/src/") {
		return ""
	}
	domain := path[len("/src/"):]
	if i := strings.Index(domain, "/"); i >= 0 {
		domain = domain[:i]
	}
	if !strings.Contains(domain, ".") {
		// No dot in the first element of import path
		// suggests this is a package in GOROOT.
		return vfs.RootTypeGoRoot
	} else {
		// A dot in the first element of import path
		// suggests this is a third party package.
		return vfs.RootTypeGoPath
	}
}
func (fs moduleFS) String() string { return "module(" + fs.FileSystem.String() + ")" }
