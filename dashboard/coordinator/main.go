// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build build_coordinator

// The coordinator runs on GCE and coordinates builds in Docker containers.
package main // import "golang.org/x/tools/dashboard/coordinator"

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/hmac"
	"crypto/md5"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"golang.org/x/tools/dashboard/types"
	"google.golang.org/api/compute/v1"
	"google.golang.org/cloud/compute/metadata"
)

var (
	masterKeyFile  = flag.String("masterkey", "", "Path to builder master key. Else fetched using GCE project attribute 'builder-master-key'.")
	maxLocalBuilds = flag.Int("maxbuilds", 6, "Max concurrent Docker builds (VM builds don't count)")

	cleanZones = flag.String("zones", "us-central1-a,us-central1-b,us-central1-f", "Comma-separated list of zones to periodically clean of stale build VMs (ones that failed to shut themselves down)")

	// Debug flags:
	addTemp = flag.Bool("temp", false, "Append -temp to all builders.")
	just    = flag.String("just", "", "If non-empty, run single build in the foreground. Requires rev.")
	rev     = flag.String("rev", "", "Revision to build.")
)

var (
	startTime = time.Now()
	builders  = map[string]buildConfig{} // populated at startup, keys like "openbsd-amd64-56"
	watchers  = map[string]watchConfig{} // populated at startup, keyed by repo, e.g. "https://go.googlesource.com/go"
	donec     = make(chan builderRev)    // reports of finished builders

	statusMu   sync.Mutex // guards both status (ongoing ones) and statusDone (just finished)
	status     = map[builderRev]*buildStatus{}
	statusDone []*buildStatus // finished recently, capped to maxStatusDone
)

const maxStatusDone = 30

// Initialized by initGCE:
var (
	projectID      string
	projectZone    string
	computeService *compute.Service
)

func initGCE() error {
	if !metadata.OnGCE() {
		return errors.New("not running on GCE; VM support disabled")
	}
	var err error
	projectID, err = metadata.ProjectID()
	if err != nil {
		return fmt.Errorf("failed to get current GCE ProjectID: %v", err)
	}
	projectZone, err = metadata.Get("instance/zone")
	if err != nil || projectZone == "" {
		return fmt.Errorf("failed to get current GCE zone: %v", err)
	}
	// Convert the zone from "projects/1234/zones/us-central1-a" to "us-central1-a".
	projectZone = path.Base(projectZone)
	if !hasComputeScope() {
		return errors.New("The coordinator is not running with access to read and write Compute resources. VM support disabled.")

	}
	ts := google.ComputeTokenSource("default")
	computeService, _ = compute.New(oauth2.NewClient(oauth2.NoContext, ts))
	return nil
}

type imageInfo struct {
	url string // of tar file

	mu      sync.Mutex
	lastMod string
}

var images = map[string]*imageInfo{
	"go-commit-watcher":          {url: "https://storage.googleapis.com/go-builder-data/docker-commit-watcher.tar.gz"},
	"gobuilders/linux-x86-base":  {url: "https://storage.googleapis.com/go-builder-data/docker-linux.base.tar.gz"},
	"gobuilders/linux-x86-clang": {url: "https://storage.googleapis.com/go-builder-data/docker-linux.clang.tar.gz"},
	"gobuilders/linux-x86-gccgo": {url: "https://storage.googleapis.com/go-builder-data/docker-linux.gccgo.tar.gz"},
	"gobuilders/linux-x86-nacl":  {url: "https://storage.googleapis.com/go-builder-data/docker-linux.nacl.tar.gz"},
	"gobuilders/linux-x86-sid":   {url: "https://storage.googleapis.com/go-builder-data/docker-linux.sid.tar.gz"},
}

// A buildConfig describes how to run either a Docker-based or VM-based build.
type buildConfig struct {
	name string // "linux-amd64-race"

	// VM-specific settings: (used if vmImage != "")
	vmImage     string // e.g. "openbsd-amd64-56"
	machineType string // optional GCE instance type

	// Docker-specific settings: (used if vmImage == "")
	image   string   // Docker image to use to build
	cmd     string   // optional -cmd flag (relative to go/src/)
	env     []string // extra environment ("key=value") pairs
	dashURL string   // url of the build dashboard
	tool    string   // the tool this configuration is for
}

func (c *buildConfig) usesDocker() bool { return c.vmImage == "" }
func (c *buildConfig) usesVM() bool     { return c.vmImage != "" }

func (c *buildConfig) MachineType() string {
	if v := c.machineType; v != "" {
		return v
	}
	return "n1-highcpu-4"
}

// recordResult sends build results to the dashboard
func (b *buildConfig) recordResult(ok bool, hash, buildLog string, runTime time.Duration) error {
	req := map[string]interface{}{
		"Builder":     b.name,
		"PackagePath": "",
		"Hash":        hash,
		"GoHash":      "",
		"OK":          ok,
		"Log":         buildLog,
		"RunTime":     runTime,
	}
	args := url.Values{"key": {builderKey(b.name)}, "builder": {b.name}}
	return dash("POST", "result", args, req, nil)
}

type watchConfig struct {
	repo     string        // "https://go.googlesource.com/go"
	dash     string        // "https://build.golang.org/" (must end in /)
	interval time.Duration // Polling interval
}

func main() {
	flag.Parse()

	if err := initGCE(); err != nil {
		log.Printf("VM support disabled due to error initializing GCE: %v", err)
	}

	addBuilder(buildConfig{name: "linux-386"})
	addBuilder(buildConfig{name: "linux-386-387", env: []string{"GO386=387"}})
	addBuilder(buildConfig{name: "linux-amd64"})
	addBuilder(buildConfig{name: "linux-amd64-nocgo", env: []string{"CGO_ENABLED=0", "USER=root"}})
	addBuilder(buildConfig{name: "linux-amd64-noopt", env: []string{"GO_GCFLAGS=-N -l"}})
	addBuilder(buildConfig{name: "linux-amd64-race"})
	addBuilder(buildConfig{name: "nacl-386"})
	addBuilder(buildConfig{name: "nacl-amd64p32"})
	addBuilder(buildConfig{
		name:    "linux-amd64-gccgo",
		image:   "gobuilders/linux-x86-gccgo",
		cmd:     "make RUNTESTFLAGS=\"--target_board=unix/-m64\" check-go -j16",
		dashURL: "https://build.golang.org/gccgo",
		tool:    "gccgo",
	})
	addBuilder(buildConfig{
		name:    "linux-386-gccgo",
		image:   "gobuilders/linux-x86-gccgo",
		cmd:     "make RUNTESTFLAGS=\"--target_board=unix/-m32\" check-go -j16",
		dashURL: "https://build.golang.org/gccgo",
		tool:    "gccgo",
	})
	addBuilder(buildConfig{name: "linux-386-sid", image: "gobuilders/linux-x86-sid"})
	addBuilder(buildConfig{name: "linux-amd64-sid", image: "gobuilders/linux-x86-sid"})
	addBuilder(buildConfig{name: "linux-386-clang", image: "gobuilders/linux-x86-clang"})
	addBuilder(buildConfig{name: "linux-amd64-clang", image: "gobuilders/linux-x86-clang"})

	// VMs:
	addBuilder(buildConfig{name: "openbsd-amd64-gce56", vmImage: "openbsd-amd64-56"})
	// addBuilder(buildConfig{name: "plan9-386-gce", vmImage: "plan9-386"})

	addWatcher(watchConfig{repo: "https://go.googlesource.com/go", dash: "https://build.golang.org/"})
	// TODO(adg,cmang): fix gccgo watcher
	// addWatcher(watchConfig{repo: "https://code.google.com/p/gofrontend", dash: "https://build.golang.org/gccgo/"})

	if (*just != "") != (*rev != "") {
		log.Fatalf("--just and --rev must be used together")
	}
	if *just != "" {
		conf, ok := builders[*just]
		if !ok {
			log.Fatalf("unknown builder %q", *just)
		}
		cmd := exec.Command("docker", append([]string{"run"}, conf.dockerRunArgs(*rev)...)...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Fatalf("Build failed: %v", err)
		}
		return
	}

	http.HandleFunc("/", handleStatus)
	http.HandleFunc("/logs", handleLogs)
	go http.ListenAndServe(":80", nil)

	go cleanUpOldContainers()
	go cleanUpOldVMs()

	stopWatchers() // clean up before we start new ones
	for _, watcher := range watchers {
		if err := startWatching(watchers[watcher.repo]); err != nil {
			log.Printf("Error starting watcher for %s: %v", watcher.repo, err)
		}
	}

	workc := make(chan builderRev)
	go findWorkLoop(workc)
	// TODO(cmang): gccgo will need its own findWorkLoop

	ticker := time.NewTicker(1 * time.Minute)
	for {
		select {
		case work := <-workc:
			log.Printf("workc received %+v; len(status) = %v, maxLocalBuilds = %v; cur = %p", work, len(status), *maxLocalBuilds, status[work])
			if mayBuildRev(work) {
				conf := builders[work.name]
				if st, err := startBuilding(conf, work.rev); err == nil {
					setStatus(work, st)
				} else {
					log.Printf("Error starting to build %v: %v", work, err)
				}
			}
		case done := <-donec:
			log.Printf("%v done", done)
			markDone(done)
		case <-ticker.C:
			if numCurrentBuilds() == 0 && time.Now().After(startTime.Add(10*time.Minute)) {
				// TODO: halt the whole machine to kill the VM or something
			}
		}
	}
}

func numCurrentBuilds() int {
	statusMu.Lock()
	defer statusMu.Unlock()
	return len(status)
}

func isBuilding(work builderRev) bool {
	statusMu.Lock()
	defer statusMu.Unlock()
	_, building := status[work]
	return building
}

// mayBuildRev reports whether the build type & revision should be started.
// It returns true if it's not already building, and there is capacity.
func mayBuildRev(work builderRev) bool {
	conf := builders[work.name]

	statusMu.Lock()
	_, building := status[work]
	statusMu.Unlock()

	if building {
		return false
	}
	if conf.usesVM() {
		// These don't count towards *maxLocalBuilds.
		return true
	}
	numDocker, err := numDockerBuilds()
	if err != nil {
		log.Printf("not starting %v due to docker ps failure: %v", work, err)
		return false
	}
	return numDocker < *maxLocalBuilds
}

func setStatus(work builderRev, st *buildStatus) {
	statusMu.Lock()
	defer statusMu.Unlock()
	status[work] = st
}

func markDone(work builderRev) {
	statusMu.Lock()
	defer statusMu.Unlock()
	st, ok := status[work]
	if !ok {
		return
	}
	delete(status, work)
	if len(statusDone) == maxStatusDone {
		copy(statusDone, statusDone[1:])
		statusDone = statusDone[:len(statusDone)-1]
	}
	statusDone = append(statusDone, st)
}

// statusPtrStr disambiguates which status to return if there are
// multiple in the history (e.g. recent failures where the build
// didn't finish for reasons outside of all.bash failing)
func getStatus(work builderRev, statusPtrStr string) *buildStatus {
	statusMu.Lock()
	defer statusMu.Unlock()
	match := func(st *buildStatus) bool {
		return statusPtrStr == "" || fmt.Sprintf("%p", st) == statusPtrStr
	}
	if st, ok := status[work]; ok && match(st) {
		return st
	}
	for _, st := range statusDone {
		if st.builderRev == work && match(st) {
			return st
		}
	}
	return nil
}

type byAge []*buildStatus

func (s byAge) Len() int           { return len(s) }
func (s byAge) Less(i, j int) bool { return s[i].start.Before(s[j].start) }
func (s byAge) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func handleStatus(w http.ResponseWriter, r *http.Request) {
	var active []*buildStatus
	var recent []*buildStatus
	statusMu.Lock()
	for _, st := range status {
		active = append(active, st)
	}
	recent = append(recent, statusDone...)
	numTotal := len(status)
	numDocker, err := numDockerBuilds()
	statusMu.Unlock()

	sort.Sort(byAge(active))
	sort.Sort(sort.Reverse(byAge(recent)))

	io.WriteString(w, "<html><body><h1>Go build coordinator</h1>")

	if err != nil {
		fmt.Fprintf(w, "<h2>Error</h2>Error fetching Docker build count: <i>%s</i>\n", html.EscapeString(err.Error()))
	}

	fmt.Fprintf(w, "<h2>running</h2><p>%d total builds active (Docker: %d/%d; VMs: %d/∞):",
		numTotal, numDocker, *maxLocalBuilds, numTotal-numDocker)

	io.WriteString(w, "<pre>")
	for _, st := range active {
		io.WriteString(w, st.htmlStatusLine())
	}
	io.WriteString(w, "</pre>")

	io.WriteString(w, "<h2>recently completed</h2><pre>")
	for _, st := range recent {
		io.WriteString(w, st.htmlStatusLine())
	}
	io.WriteString(w, "</pre>")

	fmt.Fprintf(w, "<h2>disk space</h2><pre>%s</pre></body></html>", html.EscapeString(diskFree()))
}

func diskFree() string {
	out, _ := exec.Command("df", "-h").Output()
	return string(out)
}

func handleLogs(w http.ResponseWriter, r *http.Request) {
	st := getStatus(builderRev{r.FormValue("name"), r.FormValue("rev")}, r.FormValue("st"))
	if st == nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	io.WriteString(w, st.logs())
	// TODO: if st is still building, stream them to the user with
	// http.Flusher.Flush and CloseNotifier and registering interest
	// of new writes with the buildStatus. Will require moving the
	// BUILDERKEY scrubbing into the Write method.
}

// findWorkLoop polls http://build.golang.org/?mode=json looking for new work
// for the main dashboard. It does not support gccgo.
// TODO(bradfitz): it also currently does not support subrepos.
func findWorkLoop(work chan<- builderRev) {
	ticker := time.NewTicker(15 * time.Second)
	for {
		if err := findWork(work); err != nil {
			log.Printf("failed to find new work: %v", err)
		}
		<-ticker.C
	}
}

func findWork(work chan<- builderRev) error {
	var bs types.BuildStatus
	res, err := http.Get("https://build.golang.org/?mode=json")
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if err := json.NewDecoder(res.Body).Decode(&bs); err != nil {
		return err
	}
	if res.StatusCode != 200 {
		return fmt.Errorf("unexpected http status %v", res.Status)
	}

	knownToDashboard := map[string]bool{} // keys are builder
	for _, b := range bs.Builders {
		knownToDashboard[b] = true
	}

	var goRevisions []string
	for _, br := range bs.Revisions {
		if br.Repo == "go" {
			goRevisions = append(goRevisions, br.Revision)
		} else {
			// TODO(bradfitz): support these: golang.org/issue/9506
			continue
		}
		if len(br.Results) != len(bs.Builders) {
			return errors.New("bogus JSON response from dashboard: results is too long.")
		}
		for i, res := range br.Results {
			if res != "" {
				// It's either "ok" or a failure URL.
				continue
			}
			builder := bs.Builders[i]
			if _, ok := builders[builder]; !ok {
				// Not managed by the coordinator.
				continue
			}
			br := builderRev{bs.Builders[i], br.Revision}
			if !isBuilding(br) {
				work <- br
			}
		}
	}

	// And to bootstrap new builders, see if we have any builders
	// that the dashboard doesn't know about.
	for b := range builders {
		if knownToDashboard[b] {
			continue
		}
		for _, rev := range goRevisions {
			br := builderRev{b, rev}
			if !isBuilding(br) {
				work <- br
			}
		}
	}
	return nil
}

// builderRev is a build configuration type and a revision.
type builderRev struct {
	name string // e.g. "linux-amd64-race"
	rev  string // lowercase hex git hash
}

// returns the part after "docker run"
func (conf buildConfig) dockerRunArgs(rev string) (args []string) {
	if key := builderKey(conf.name); key != "" {
		tmpKey := "/tmp/" + conf.name + ".buildkey"
		if _, err := os.Stat(tmpKey); err != nil {
			if err := ioutil.WriteFile(tmpKey, []byte(key), 0600); err != nil {
				log.Fatal(err)
			}
		}
		// Images may look for .gobuildkey in / or /root, so provide both.
		// TODO(adg): fix images that look in the wrong place.
		args = append(args, "-v", tmpKey+":/.gobuildkey")
		args = append(args, "-v", tmpKey+":/root/.gobuildkey")
	}
	for _, pair := range conf.env {
		args = append(args, "-e", pair)
	}
	args = append(args,
		conf.image,
		"/usr/local/bin/builder",
		"-rev="+rev,
		"-dashboard="+conf.dashURL,
		"-tool="+conf.tool,
		"-buildroot=/",
		"-v",
	)
	if conf.cmd != "" {
		args = append(args, "-cmd", conf.cmd)
	}
	args = append(args, conf.name)
	return
}

func addBuilder(c buildConfig) {
	if c.tool == "gccgo" {
		// TODO(cmang,bradfitz,adg): fix gccgo
		return
	}
	if c.name == "" {
		panic("empty name")
	}
	if *addTemp {
		c.name += "-temp"
	}
	if _, dup := builders[c.name]; dup {
		panic("dup name")
	}
	if c.dashURL == "" {
		c.dashURL = "https://build.golang.org"
	}
	if c.tool == "" {
		c.tool = "go"
	}

	if strings.HasPrefix(c.name, "nacl-") {
		if c.image == "" {
			c.image = "gobuilders/linux-x86-nacl"
		}
		if c.cmd == "" {
			c.cmd = "/usr/local/bin/build-command.pl"
		}
	}
	if strings.HasPrefix(c.name, "linux-") && c.image == "" {
		c.image = "gobuilders/linux-x86-base"
	}
	if c.image == "" && c.vmImage == "" {
		panic("empty image and vmImage")
	}
	if c.image != "" && c.vmImage != "" {
		panic("can't specify both image and vmImage")
	}
	builders[c.name] = c
}

// returns the part after "docker run"
func (conf watchConfig) dockerRunArgs() (args []string) {
	log.Printf("Running watcher with master key %q", masterKey())
	if key := masterKey(); len(key) > 0 {
		tmpKey := "/tmp/watcher.buildkey"
		if _, err := os.Stat(tmpKey); err != nil {
			if err := ioutil.WriteFile(tmpKey, key, 0600); err != nil {
				log.Fatal(err)
			}
		}
		// Images may look for .gobuildkey in / or /root, so provide both.
		// TODO(adg): fix images that look in the wrong place.
		args = append(args, "-v", tmpKey+":/.gobuildkey")
		args = append(args, "-v", tmpKey+":/root/.gobuildkey")
	}
	args = append(args,
		"go-commit-watcher",
		"/usr/local/bin/watcher",
		"-repo="+conf.repo,
		"-dash="+conf.dash,
		"-poll="+conf.interval.String(),
	)
	return
}

func addWatcher(c watchConfig) {
	if c.repo == "" {
		c.repo = "https://go.googlesource.com/go"
	}
	if c.dash == "" {
		c.dash = "https://build.golang.org/"
	}
	if c.interval == 0 {
		c.interval = 10 * time.Second
	}
	watchers[c.repo] = c
}

func condUpdateImage(img string) error {
	ii := images[img]
	if ii == nil {
		return fmt.Errorf("image %q doesn't exist", img)
	}
	ii.mu.Lock()
	defer ii.mu.Unlock()
	res, err := http.Head(ii.url)
	if err != nil {
		return fmt.Errorf("Error checking %s: %v", ii.url, err)
	}
	if res.StatusCode != 200 {
		return fmt.Errorf("Error checking %s: %v", ii.url, res.Status)
	}
	if res.Header.Get("Last-Modified") == ii.lastMod {
		return nil
	}

	res, err = http.Get(ii.url)
	if err != nil || res.StatusCode != 200 {
		return fmt.Errorf("Get after Head failed for %s: %v, %v", ii.url, err, res)
	}
	defer res.Body.Close()

	log.Printf("Running: docker load of %s\n", ii.url)
	cmd := exec.Command("docker", "load")
	cmd.Stdin = res.Body

	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	if cmd.Run(); err != nil {
		log.Printf("Failed to pull latest %s from %s and pipe into docker load: %v, %s", img, ii.url, err, out.Bytes())
		return err
	}
	ii.lastMod = res.Header.Get("Last-Modified")
	return nil
}

// numDockerBuilds finds the number of go builder instances currently running.
func numDockerBuilds() (n int, err error) {
	out, err := exec.Command("docker", "ps").Output()
	if err != nil {
		return 0, err
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.Contains(line, "gobuilders/") {
			n++
		}
	}
	return n, nil
}

func startBuilding(conf buildConfig, rev string) (*buildStatus, error) {
	if conf.usesVM() {
		return startBuildingInVM(conf, rev)
	} else {
		return startBuildingInDocker(conf, rev)
	}
}

func startBuildingInDocker(conf buildConfig, rev string) (*buildStatus, error) {
	if err := condUpdateImage(conf.image); err != nil {
		log.Printf("Failed to setup container for %v %v: %v", conf.name, rev, err)
		return nil, err
	}

	cmd := exec.Command("docker", append([]string{"run", "-d"}, conf.dockerRunArgs(rev)...)...)
	all, err := cmd.CombinedOutput()
	log.Printf("Docker run for %v %v = err:%v, output:%s", conf.name, rev, err, all)
	if err != nil {
		return nil, err
	}
	container := strings.TrimSpace(string(all))
	brev := builderRev{
		name: conf.name,
		rev:  rev,
	}
	st := &buildStatus{
		builderRev: brev,
		container:  container,
		start:      time.Now(),
	}
	log.Printf("%v now building in Docker container %v", brev, st.container)
	go func() {
		all, err := exec.Command("docker", "wait", container).CombinedOutput()
		output := strings.TrimSpace(string(all))
		var ok bool
		if err == nil {
			exit, err := strconv.Atoi(output)
			ok = (err == nil && exit == 0)
		}
		st.setDone(ok)
		log.Printf("docker wait %s/%s: %v, %s", container, rev, err, output)
		donec <- builderRev{conf.name, rev}
		exec.Command("docker", "rm", container).Run()
	}()
	go func() {
		cmd := exec.Command("docker", "logs", "-f", container)
		cmd.Stdout = st
		cmd.Stderr = st
		if err := cmd.Run(); err != nil {
			// The docker logs subcommand always returns
			// success, even if the underlying process
			// fails.
			log.Printf("failed to follow docker logs of %s: %v", container, err)
		}
	}()
	return st, nil
}

var osArchRx = regexp.MustCompile(`^(\w+-\w+)`)

// startBuildingInVM starts a VM on GCE running the buildlet binary to build rev.
func startBuildingInVM(conf buildConfig, rev string) (*buildStatus, error) {
	brev := builderRev{
		name: conf.name,
		rev:  rev,
	}
	st := &buildStatus{
		builderRev: brev,
		start:      time.Now(),
	}

	// name is the project-wide unique name of the GCE instance. It can't be longer
	// than 61 bytes, so we only use the first 8 bytes of the rev.
	name := "buildlet-" + conf.name + "-" + rev[:8]

	// buildletURL is the URL of the buildlet binary which the VMs
	// are configured to download at boot and run. This lets us
	// update the buildlet more easily than rebuilding the whole
	// VM image. We put this URL in a well-known GCE metadata attribute.
	// The value will be of the form:
	//  http://storage.googleapis.com/go-builder-data/buildlet.GOOS-GOARCH
	m := osArchRx.FindStringSubmatch(conf.name)
	if m == nil {
		return nil, fmt.Errorf("invalid builder name %q", conf.name)
	}
	buildletURL := "http://storage.googleapis.com/go-builder-data/buildlet." + m[1]

	prefix := "https://www.googleapis.com/compute/v1/projects/" + projectID
	machType := prefix + "/zones/" + projectZone + "/machineTypes/" + conf.MachineType()

	instance := &compute.Instance{
		Name:        name,
		Description: fmt.Sprintf("Go Builder building %s %s", conf.name, rev),
		MachineType: machType,
		Disks: []*compute.AttachedDisk{
			{
				AutoDelete: true,
				Boot:       true,
				Type:       "PERSISTENT",
				InitializeParams: &compute.AttachedDiskInitializeParams{
					DiskName:    name,
					SourceImage: "https://www.googleapis.com/compute/v1/projects/" + projectID + "/global/images/" + conf.vmImage,
					DiskType:    "https://www.googleapis.com/compute/v1/projects/" + projectID + "/zones/" + projectZone + "/diskTypes/pd-ssd",
				},
			},
		},
		Tags: &compute.Tags{
			// Warning: do NOT list "http-server" or "allow-ssh" (our
			// project's custom tag to allow ssh access) here; the
			// buildlet provides full remote code execution.
			Items: []string{},
		},
		Metadata: &compute.Metadata{
			Items: []*compute.MetadataItems{
				{
					Key:   "buildlet-binary-url",
					Value: buildletURL,
				},
				// In case the VM gets away from us (generally: if the
				// coordinator dies while a build is running), then we
				// set this attribute of when it should be killed so
				// we can kill it later when the coordinator is
				// restarted. The cleanUpOldVMs goroutine loop handles
				// that killing.
				{
					Key:   "delete-at",
					Value: fmt.Sprint(time.Now().Add(30 * time.Minute).Unix()),
				},
			},
		},
		NetworkInterfaces: []*compute.NetworkInterface{
			&compute.NetworkInterface{
				AccessConfigs: []*compute.AccessConfig{
					&compute.AccessConfig{
						Type: "ONE_TO_ONE_NAT",
						Name: "External NAT",
					},
				},
				Network: prefix + "/global/networks/default",
			},
		},
	}
	op, err := computeService.Instances.Insert(projectID, projectZone, instance).Do()
	if err != nil {
		return nil, fmt.Errorf("Failed to create instance: %v", err)
	}
	st.createOp = op.Name
	st.instName = name
	log.Printf("%v now building in VM %v", brev, st.instName)
	// Start the goroutine to monitor the VM now that it's booting. This might
	// take minutes for it to come up, and then even more time to do the build.
	go func() {
		err := watchVM(st)
		if st.hasEvent("instance_created") {
			deleteVM(projectZone, st.instName)
		}
		st.setDone(err == nil)
		if err != nil {
			fmt.Fprintf(st, "\n\nError: %v\n", err)
		}
		donec <- builderRev{conf.name, rev}
	}()
	return st, nil
}

// watchVM monitors a VM doing a build.
func watchVM(st *buildStatus) (err error) {
	goodRes := func(res *http.Response, err error, what string) bool {
		if err != nil {
			err = fmt.Errorf("%s: %v", what, err)
			return false
		}
		if res.StatusCode/100 != 2 {
			err = fmt.Errorf("%s: %v", what, res.Status)
			return false

		}
		return true
	}
	st.logEventTime("instance_create_requested")
	// Wait for instance create operation to succeed.
OpLoop:
	for {
		time.Sleep(2 * time.Second)
		op, err := computeService.ZoneOperations.Get(projectID, projectZone, st.createOp).Do()
		if err != nil {
			return fmt.Errorf("Failed to get op %s: %v", st.createOp, err)
		}
		switch op.Status {
		case "PENDING", "RUNNING":
			continue
		case "DONE":
			if op.Error != nil {
				for _, operr := range op.Error.Errors {
					return fmt.Errorf("Error creating instance: %+v", operr)
				}
				return errors.New("Failed to start.")
			}
			break OpLoop
		default:
			log.Fatalf("Unknown status %q: %+v", op.Status, op)
		}
	}
	st.logEventTime("instance_created")

	inst, err := computeService.Instances.Get(projectID, projectZone, st.instName).Do()
	if err != nil {
		return fmt.Errorf("Error getting instance %s details after creation: %v", st.instName, err)
	}
	st.logEventTime("got_instance_info")

	// Find its internal IP.
	var ip string
	for _, iface := range inst.NetworkInterfaces {
		if strings.HasPrefix(iface.NetworkIP, "10.") {
			ip = iface.NetworkIP
		}
	}
	if ip == "" {
		return errors.New("didn't find its internal IP address")
	}

	// Wait for it to boot and its buildlet to come up on port 80.
	st.logEventTime("waiting_for_buildlet")
	buildletURL := "http://" + ip
	const numTries = 60
	var alive bool
	for i := 1; i <= numTries; i++ {
		res, err := http.Get(buildletURL)
		if err != nil {
			time.Sleep(1 * time.Second)
			continue
		}
		res.Body.Close()
		if res.StatusCode != 200 {
			return fmt.Errorf("buildlet returned HTTP status code %d on try number %d", res.StatusCode, i)
		}
		st.logEventTime("buildlet_up")
		alive = true
		break
	}
	if !alive {
		return fmt.Errorf("buildlet didn't come up in %d seconds", numTries)
	}

	// Write the VERSION file.
	st.logEventTime("start_write_version_tar")
	verReq, err := http.NewRequest("PUT", buildletURL+"/writetgz", versionTgz(st.rev))
	if err != nil {
		return err
	}
	verRes, err := http.DefaultClient.Do(verReq)
	if !goodRes(verRes, err, "writing VERSION tgz") {
		return
	}

	// Feed the buildlet a tar file for it to extract.
	// TODO: cache these.
	st.logEventTime("start_fetch_gerrit_tgz")
	tarRes, err := http.Get("https://go.googlesource.com/go/+archive/" + st.rev + ".tar.gz")
	if !goodRes(tarRes, err, "fetching tarball from Gerrit") {
		return
	}

	st.logEventTime("start_write_tar")
	putReq, err := http.NewRequest("PUT", buildletURL+"/writetgz", tarRes.Body)
	if err != nil {
		tarRes.Body.Close()
		return err
	}
	putRes, err := http.DefaultClient.Do(putReq)
	st.logEventTime("end_write_tar")
	tarRes.Body.Close()
	if !goodRes(putRes, err, "writing tarball to buildlet") {
		return
	}

	// Run the builder
	cmd := "all.bash"
	if strings.HasPrefix(st.name, "windows-") {
		cmd = "all.bat"
	} else if strings.HasPrefix(st.name, "plan9-") {
		cmd = "all.rc"
	}
	execStartTime := time.Now()
	st.logEventTime("start_exec")
	res, err := http.PostForm(buildletURL+"/exec", url.Values{"cmd": {"src/" + cmd}})
	if !goodRes(res, err, "running "+cmd) {
		return
	}
	defer res.Body.Close()
	st.logEventTime("running_exec")
	// Stream the output:
	if _, err := io.Copy(st, res.Body); err != nil {
		return fmt.Errorf("error copying response: %v", err)
	}
	st.logEventTime("done")
	state := res.Trailer.Get("Process-State")

	// Don't record to the dashboard unless we heard the trailer from
	// the buildlet, otherwise it was probably some unrelated error
	// (like the VM being killed, or the buildlet crashing due to
	// e.g. https://golang.org/issue/9309, since we require a tip
	// build of the buildlet to get Trailers support)
	if state != "" {
		conf := builders[st.name]
		var log string
		if state != "ok" {
			log = st.logs()
		}
		if err := conf.recordResult(state == "ok", st.rev, log, time.Since(execStartTime)); err != nil {
			return fmt.Errorf("Status was %q but failed to report it to the dashboard: %v", state, err)
		}
	}
	if state != "ok" {

		return fmt.Errorf("got Trailer process state %q", state)
	}
	return nil
}

type eventAndTime struct {
	evt string
	t   time.Time
}

// buildStatus is the status of a build.
type buildStatus struct {
	// Immutable:
	builderRev
	start     time.Time
	container string // container ID for docker, else it's a VM

	// Immutable, used by VM only:
	createOp string // Instances.Insert operation name
	instName string

	mu        sync.Mutex   // guards following
	done      time.Time    // finished running
	succeeded bool         // set when done
	output    bytes.Buffer // stdout and stderr
	events    []eventAndTime
}

func (st *buildStatus) setDone(succeeded bool) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.succeeded = succeeded
	st.done = time.Now()
}

func (st *buildStatus) logEventTime(event string) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.events = append(st.events, eventAndTime{event, time.Now()})
}

func (st *buildStatus) hasEvent(event string) bool {
	st.mu.Lock()
	defer st.mu.Unlock()
	for _, e := range st.events {
		if e.evt == event {
			return true
		}
	}
	return false
}

// htmlStatusLine returns the HTML to show within the <pre> block on
// the main page's list of active builds.
func (st *buildStatus) htmlStatusLine() string {
	st.mu.Lock()
	defer st.mu.Unlock()

	urlPrefix := "https://go-review.googlesource.com/#/q/"
	if strings.Contains(st.name, "gccgo") {
		urlPrefix = "https://code.google.com/p/gofrontend/source/detail?r="
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "<a href='https://github.com/golang/go/wiki/DashboardBuilders'>%s</a> rev <a href='%s%s'>%s</a>",
		st.name, urlPrefix, st.rev, st.rev)

	if st.done.IsZero() {
		buf.WriteString(", running")
	} else if st.succeeded {
		buf.WriteString(", succeeded")
	} else {
		buf.WriteString(", failed")
	}

	logsURL := fmt.Sprintf("/logs?name=%s&rev=%s&st=%p", st.name, st.rev, st)
	if st.container != "" {
		fmt.Fprintf(&buf, " in container <a href='%s'>%s</a>", logsURL, st.container)
	} else {
		fmt.Fprintf(&buf, " in VM <a href='%s'>%s</a>", logsURL, st.instName)
	}

	t := st.done
	if t.IsZero() {
		t = st.start
	}
	fmt.Fprintf(&buf, ", %v ago\n", time.Since(t))
	for i, evt := range st.events {
		var elapsed string
		if i != 0 {
			elapsed = fmt.Sprintf("+%0.1fs", evt.t.Sub(st.events[i-1].t).Seconds())
		}
		msg := evt.evt
		if msg == "running_exec" {
			msg = fmt.Sprintf("<a href='%s'>%s</a>", logsURL, msg)
		}
		fmt.Fprintf(&buf, " %7s %v %s\n", elapsed, evt.t.Format(time.RFC3339), msg)
	}
	return buf.String()
}

func (st *buildStatus) logs() string {
	st.mu.Lock()
	logs := st.output.String()
	st.mu.Unlock()
	key := builderKey(st.name)
	return strings.Replace(string(logs), key, "BUILDERKEY", -1)
}

func (st *buildStatus) Write(p []byte) (n int, err error) {
	st.mu.Lock()
	defer st.mu.Unlock()
	const maxBufferSize = 2 << 20 // 2MB of output is way more than we expect.
	plen := len(p)
	if st.output.Len()+len(p) > maxBufferSize {
		p = p[:maxBufferSize-st.output.Len()]
	}
	st.output.Write(p) // bytes.Buffer can't fail
	return plen, nil
}

// Stop any previous go-commit-watcher Docker tasks, so they don't
// pile up upon restarts of the coordinator.
func stopWatchers() {
	out, err := exec.Command("docker", "ps").Output()
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(out), "\n") {
		if !strings.Contains(line, "go-commit-watcher:") {
			continue
		}
		f := strings.Fields(line)
		exec.Command("docker", "rm", "-f", "-v", f[0]).Run()
	}
}

func startWatching(conf watchConfig) (err error) {
	defer func() {
		if err != nil {
			restartWatcherSoon(conf)
		}
	}()
	log.Printf("Starting watcher for %v", conf.repo)
	if err := condUpdateImage("go-commit-watcher"); err != nil {
		log.Printf("Failed to setup container for commit watcher: %v", err)
		return err
	}

	cmd := exec.Command("docker", append([]string{"run", "-d"}, conf.dockerRunArgs()...)...)
	all, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Docker run for commit watcher = err:%v, output: %s", err, all)
		return err
	}
	container := strings.TrimSpace(string(all))
	// Start a goroutine to wait for the watcher to die.
	go func() {
		exec.Command("docker", "wait", container).Run()
		exec.Command("docker", "rm", "-v", container).Run()
		log.Printf("Watcher crashed. Restarting soon.")
		restartWatcherSoon(conf)
	}()
	return nil
}

func restartWatcherSoon(conf watchConfig) {
	time.AfterFunc(30*time.Second, func() {
		startWatching(conf)
	})
}

func builderKey(builder string) string {
	master := masterKey()
	if len(master) == 0 {
		return ""
	}
	h := hmac.New(md5.New, master)
	io.WriteString(h, builder)
	return fmt.Sprintf("%x", h.Sum(nil))
}

func masterKey() []byte {
	keyOnce.Do(loadKey)
	return masterKeyCache
}

var (
	keyOnce        sync.Once
	masterKeyCache []byte
)

func loadKey() {
	if *masterKeyFile != "" {
		b, err := ioutil.ReadFile(*masterKeyFile)
		if err != nil {
			log.Fatal(err)
		}
		masterKeyCache = bytes.TrimSpace(b)
		return
	}
	req, _ := http.NewRequest("GET", "http://metadata.google.internal/computeMetadata/v1/project/attributes/builder-master-key", nil)
	req.Header.Set("Metadata-Flavor", "Google")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Fatal("No builder master key available")
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		log.Fatalf("No builder-master-key project attribute available.")
	}
	slurp, err := ioutil.ReadAll(res.Body)
	if err != nil {
		log.Fatal(err)
	}
	masterKeyCache = bytes.TrimSpace(slurp)
}

func cleanUpOldContainers() {
	for {
		for _, cid := range oldContainers() {
			log.Printf("Cleaning old container %v", cid)
			exec.Command("docker", "rm", "-v", cid).Run()
		}
		time.Sleep(30 * time.Second)
	}
}

func oldContainers() []string {
	out, _ := exec.Command("docker", "ps", "-a", "--filter=status=exited", "--no-trunc", "-q").Output()
	return strings.Fields(string(out))
}

// cleanUpOldVMs loops forever and periodically enumerates virtual
// machines and deletes those which have expired.
//
// A VM is considered expired if it has a "delete-at" metadata
// attribute having a unix timestamp before the current time.
//
// This is the safety mechanism to delete VMs which stray from the
// normal deleting process. VMs are created to run a single build and
// should be shut down by a controlling process. Due to various types
// of failures, they might get stranded. To prevent them from getting
// stranded and wasting resources forever, we instead set the
// "delete-at" metadata attribute on them when created to some time
// that's well beyond their expected lifetime.
func cleanUpOldVMs() {
	if computeService == nil {
		return
	}
	for {
		for _, zone := range strings.Split(*cleanZones, ",") {
			zone = strings.TrimSpace(zone)
			if err := cleanZoneVMs(zone); err != nil {
				log.Printf("Error cleaning VMs in zone %q: %v", zone, err)
			}
		}
		time.Sleep(time.Minute)
	}
}

// cleanZoneVMs is part of cleanUpOldVMs, operating on a single zone.
func cleanZoneVMs(zone string) error {
	// Fetch the first 500 (default) running instances and clean
	// thoes. We expect that we'll be running many fewer than
	// that. Even if we have more, eventually the first 500 will
	// either end or be cleaned, and then the next call will get a
	// partially-different 500.
	// TODO(bradfitz): revist this code if we ever start running
	// thousands of VMs.
	list, err := computeService.Instances.List(projectID, zone).Do()
	if err != nil {
		return fmt.Errorf("listing instances: %v", err)
	}
	for _, inst := range list.Items {
		if inst.Metadata == nil {
			// Defensive. Not seen in practice.
			continue
		}
		for _, it := range inst.Metadata.Items {
			if it.Key == "delete-at" {
				unixDeadline, err := strconv.ParseInt(it.Value, 10, 64)
				if err != nil {
					log.Printf("invalid delete-at value %q seen; ignoring", it.Value)
				}
				if err == nil && time.Now().Unix() > unixDeadline {
					log.Printf("Deleting expired VM %q in zone %q ...", inst.Name, zone)
					deleteVM(zone, inst.Name)
				}
			}
		}
	}
	return nil
}

func deleteVM(zone, instName string) {
	op, err := computeService.Instances.Delete(projectID, zone, instName).Do()
	if err != nil {
		log.Printf("Failed to delete instance %q in zone %q: %v", instName, zone, err)
		return
	}
	log.Printf("Sent request to delete instance %q in zone %q. Operation ID == %v", instName, zone, op.Id)
}

func hasComputeScope() bool {
	if !metadata.OnGCE() {
		return false
	}
	scopes, err := metadata.Scopes("default")
	if err != nil {
		log.Printf("failed to query metadata default scopes: %v", err)
		return false
	}
	for _, v := range scopes {
		if v == compute.DevstorageFull_controlScope {
			return true
		}
	}
	return false
}

// dash is copied from the builder binary. It runs the given method and command on the dashboard.
//
// TODO(bradfitz,adg): unify this somewhere?
//
// If args is non-nil it is encoded as the URL query string.
// If req is non-nil it is JSON-encoded and passed as the body of the HTTP POST.
// If resp is non-nil the server's response is decoded into the value pointed
// to by resp (resp must be a pointer).
func dash(meth, cmd string, args url.Values, req, resp interface{}) error {
	const builderVersion = 1 // keep in sync with dashboard/app/build/handler.go
	argsCopy := url.Values{"version": {fmt.Sprint(builderVersion)}}
	for k, v := range args {
		if k == "version" {
			panic(`dash: reserved args key: "version"`)
		}
		argsCopy[k] = v
	}
	var r *http.Response
	var err error
	cmd = "https://build.golang.org/" + cmd + "?" + argsCopy.Encode()
	switch meth {
	case "GET":
		if req != nil {
			log.Panicf("%s to %s with req", meth, cmd)
		}
		r, err = http.Get(cmd)
	case "POST":
		var body io.Reader
		if req != nil {
			b, err := json.Marshal(req)
			if err != nil {
				return err
			}
			body = bytes.NewBuffer(b)
		}
		r, err = http.Post(cmd, "text/json", body)
	default:
		log.Panicf("%s: invalid method %q", cmd, meth)
		panic("invalid method: " + meth)
	}
	if err != nil {
		return err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		return fmt.Errorf("bad http response: %v", r.Status)
	}
	body := new(bytes.Buffer)
	if _, err := body.ReadFrom(r.Body); err != nil {
		return err
	}

	// Read JSON-encoded Response into provided resp
	// and return an error if present.
	var result = struct {
		Response interface{}
		Error    string
	}{
		// Put the provided resp in here as it can be a pointer to
		// some value we should unmarshal into.
		Response: resp,
	}
	if err = json.Unmarshal(body.Bytes(), &result); err != nil {
		log.Printf("json unmarshal %#q: %s\n", body.Bytes(), err)
		return err
	}
	if result.Error != "" {
		return errors.New(result.Error)
	}

	return nil
}

func versionTgz(rev string) io.Reader {
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(zw)

	contents := fmt.Sprintf("devel " + rev)
	check(tw.WriteHeader(&tar.Header{
		Name: "VERSION",
		Mode: 0644,
		Size: int64(len(contents)),
	}))
	_, err := io.WriteString(tw, contents)
	check(err)
	check(tw.Close())
	check(zw.Close())
	return bytes.NewReader(buf.Bytes())
}

// check is only for things which should be impossible (not even rare)
// to fail.
func check(err error) {
	if err != nil {
		panic("previously assumed to never fail: " + err.Error())
	}
}
