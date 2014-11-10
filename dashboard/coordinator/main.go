// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The coordinator runs on GCE and coordinates builds in Docker containers.
package main

import (
	"bytes"
	"crypto/hmac"
	"crypto/md5"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	masterKeyFile = flag.String("masterkey", "", "Path to builder master key. Else fetched using GCE project attribute 'builder-master-key'.")
	maxBuilds     = flag.Int("maxbuilds", 6, "Max concurrent builds")

	// Debug flags:
	addTemp = flag.Bool("temp", false, "Append -temp to all builders.")
	just    = flag.String("just", "", "If non-empty, run single build in the foreground. Requires rev.")
	rev     = flag.String("rev", "", "Revision to build.")
)

var (
	startTime = time.Now()
	builders  = map[string]buildConfig{} // populated once at startup
	donec     = make(chan builderRev)    // reports of finished builders

	statusMu sync.Mutex
	status   = map[builderRev]*buildStatus{}
)

type imageInfo struct {
	url string // of tar file

	mu      sync.Mutex
	lastMod string
}

var images = map[string]*imageInfo{
	"gobuilders/linux-x86-base":  {url: "https://storage.googleapis.com/go-builder-data/docker-linux.base.tar.gz"},
	"gobuilders/linux-x86-clang": {url: "https://storage.googleapis.com/go-builder-data/docker-linux.clang.tar.gz"},
	"gobuilders/linux-x86-gccgo": {url: "https://storage.googleapis.com/go-builder-data/docker-linux.gccgo.tar.gz"},
	"gobuilders/linux-x86-nacl":  {url: "https://storage.googleapis.com/go-builder-data/docker-linux.nacl.tar.gz"},
	"gobuilders/linux-x86-sid":   {url: "https://storage.googleapis.com/go-builder-data/docker-linux.sid.tar.gz"},
}

type buildConfig struct {
	name    string   // "linux-amd64-race"
	image   string   // Docker image to use to build
	cmd     string   // optional -cmd flag (relative to go/src/)
	env     []string // extra environment ("key=value") pairs
	dashURL string   // url of the build dashboard
	tool    string   // the tool this configuration is for
}

func main() {
	flag.Parse()
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

	workc := make(chan builderRev)
	for name, builder := range builders {
		go findWorkLoop(name, builder.dashURL, workc)
	}

	ticker := time.NewTicker(1 * time.Minute)
	for {
		select {
		case work := <-workc:
			log.Printf("workc received %+v; len(status) = %v, maxBuilds = %v; cur = %p", work, len(status), *maxBuilds, status[work])
			mayBuild := mayBuildRev(work)
			if mayBuild {
				out, _ := exec.Command("docker", "ps").Output()
				numBuilds := bytes.Count(out, []byte("\n")) - 1
				log.Printf("num current docker builds: %d", numBuilds)
				if numBuilds > *maxBuilds {
					mayBuild = false
				}
			}
			if mayBuild {
				if st, err := startBuilding(builders[work.name], work.rev); err == nil {
					setStatus(work, st)
					log.Printf("%v now building in %v", work, st.container)
				} else {
					log.Printf("Error starting to build %v: %v", work, err)
				}
			}
		case done := <-donec:
			log.Printf("%v done", done)
			setStatus(done, nil)
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

func mayBuildRev(work builderRev) bool {
	statusMu.Lock()
	defer statusMu.Unlock()
	return len(status) < *maxBuilds && status[work] == nil
}

func setStatus(work builderRev, st *buildStatus) {
	statusMu.Lock()
	defer statusMu.Unlock()
	if st == nil {
		delete(status, work)
	} else {
		status[work] = st
	}
}

func getStatus(work builderRev) *buildStatus {
	statusMu.Lock()
	defer statusMu.Unlock()
	return status[work]
}

type byAge []*buildStatus

func (s byAge) Len() int           { return len(s) }
func (s byAge) Less(i, j int) bool { return s[i].start.Before(s[j].start) }
func (s byAge) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func handleStatus(w http.ResponseWriter, r *http.Request) {
	var active []*buildStatus
	statusMu.Lock()
	for _, st := range status {
		active = append(active, st)
	}
	statusMu.Unlock()

	fmt.Fprintf(w, "<html><body><h1>Go build coordinator</h1>%d of max %d builds running:<p><pre>", len(status), *maxBuilds)
	sort.Sort(byAge(active))
	for _, st := range active {
		fmt.Fprintf(w, "%-22s hg %s in container <a href='/logs?name=%s&rev=%s'>%s</a>, %v ago\n", st.name, st.rev, st.name, st.rev,
			st.container, time.Now().Sub(st.start))
	}
	fmt.Fprintf(w, "</pre></body></html>")
}

func handleLogs(w http.ResponseWriter, r *http.Request) {
	st := getStatus(builderRev{r.FormValue("name"), r.FormValue("rev")})
	if st == nil {
		fmt.Fprintf(w, "<html><body><h1>not building</h1>")
		return
	}
	out, err := exec.Command("docker", "logs", st.container).CombinedOutput()
	if err != nil {
		log.Print(err)
		http.Error(w, "Error fetching logs. Already finished?", 500)
		return
	}
	key := builderKey(st.name)
	logs := strings.Replace(string(out), key, "BUILDERKEY", -1)
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	io.WriteString(w, logs)
}

func findWorkLoop(builderName, dashURL string, work chan<- builderRev) {
	// TODO: make this better
	for {
		rev, err := findWork(builderName, dashURL)
		if err != nil {
			log.Printf("Finding work for %s: %v", builderName, err)
		} else if rev != "" {
			work <- builderRev{builderName, rev}
		}
		time.Sleep(60 * time.Second)
	}
}

func findWork(builderName, dashURL string) (rev string, err error) {
	var jres struct {
		Response struct {
			Kind string
			Data struct {
				Hash        string
				PerfResults []string
			}
		}
	}
	res, err := http.Get(dashURL + "/todo?builder=" + builderName + "&kind=build-go-commit")
	if err != nil {
		return
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return "", fmt.Errorf("unexpected http status %d", res.StatusCode)
	}
	err = json.NewDecoder(res.Body).Decode(&jres)
	if jres.Response.Kind == "build-go-commit" {
		rev = jres.Response.Data.Hash
	}
	return rev, err
}

type builderRev struct {
	name, rev string
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
	if c.image == "" {
		panic("empty image")
	}
	builders[c.name] = c
}

func condUpdateImage(img string) error {
	ii := images[img]
	if ii == nil {
		log.Fatalf("Image %q not described.", img)
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

func startBuilding(conf buildConfig, rev string) (*buildStatus, error) {
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
	go func() {
		all, err := exec.Command("docker", "wait", container).CombinedOutput()
		log.Printf("docker wait %s/%s: %v, %s", container, rev, err, strings.TrimSpace(string(all)))
		donec <- builderRev{conf.name, rev}
		exec.Command("docker", "rm", container).Run()
	}()
	return &buildStatus{
		builderRev: builderRev{
			name: conf.name,
			rev:  rev,
		},
		container: container,
		start:     time.Now(),
	}, nil
}

type buildStatus struct {
	builderRev
	container string
	start     time.Time

	mu sync.Mutex
	// ...
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
