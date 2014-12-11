// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

import (
	"bytes"
	"crypto/hmac"
	"crypto/md5"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf8"

	"appengine"
	"appengine/datastore"

	"cache"
	"key"
)

const (
	commitsPerPage = 30
	watcherVersion = 3 // must match dashboard/watcher/watcher.go
	builderVersion = 1 // must match dashboard/builder/http.go
)

// commitHandler retrieves commit data or records a new commit.
//
// For GET requests it returns a Commit value for the specified
// packagePath and hash.
//
// For POST requests it reads a JSON-encoded Commit value from the request
// body and creates a new Commit entity. It also updates the "tip" Tag for
// each new commit at tip.
//
// This handler is used by a gobuilder process in -commit mode.
func commitHandler(r *http.Request) (interface{}, error) {
	c := contextForRequest(r)
	com := new(Commit)

	if r.Method == "GET" {
		com.PackagePath = r.FormValue("packagePath")
		com.Hash = r.FormValue("hash")
		err := datastore.Get(c, com.Key(c), com)
		if com.Num == 0 && com.Desc == "" {
			// Perf builder might have written an incomplete Commit.
			// Pretend it doesn't exist, so that we can get complete details.
			err = datastore.ErrNoSuchEntity
		}
		if err != nil {
			if err == datastore.ErrNoSuchEntity {
				// This error string is special.
				// The commit watcher expects it.
				// Do not change it.
				return nil, errors.New("Commit not found")
			}
			return nil, fmt.Errorf("getting Commit: %v", err)
		}
		if com.Num == 0 {
			// Corrupt state which shouldn't happen but does.
			// Return an error so builders' commit loops will
			// be willing to retry submitting this commit.
			return nil, errors.New("in datastore with zero Num")
		}
		if com.Desc == "" || com.User == "" {
			// Also shouldn't happen, but at least happened
			// once on a single commit when trying to fix data
			// in the datastore viewer UI?
			return nil, errors.New("missing field")
		}
		// Strip potentially large and unnecessary fields.
		com.ResultData = nil
		com.PerfResults = nil
		return com, nil
	}
	if r.Method != "POST" {
		return nil, errBadMethod(r.Method)
	}
	if !isMasterKey(c, r.FormValue("key")) {
		return nil, errors.New("can only POST commits with master key")
	}

	// For now, the commit watcher doesn't support gccgo.
	// TODO(adg,cmang): remove this exception when gccgo is supported.
	if dashboardForRequest(r) != gccgoDash {
		v, _ := strconv.Atoi(r.FormValue("version"))
		if v != watcherVersion {
			return nil, fmt.Errorf("rejecting POST from commit watcher; need version %v", watcherVersion)
		}
	}

	// POST request
	body, err := ioutil.ReadAll(r.Body)
	r.Body.Close()
	if err != nil {
		return nil, fmt.Errorf("reading Body: %v", err)
	}
	if !bytes.Contains(body, needsBenchmarkingBytes) {
		c.Warningf("old builder detected at %v", r.RemoteAddr)
		return nil, fmt.Errorf("rejecting old builder request, body does not contain %s: %q", needsBenchmarkingBytes, body)
	}
	if err := json.Unmarshal(body, com); err != nil {
		return nil, fmt.Errorf("unmarshaling body %q: %v", body, err)
	}
	com.Desc = limitStringLength(com.Desc, maxDatastoreStringLen)
	if err := com.Valid(); err != nil {
		return nil, fmt.Errorf("validating Commit: %v", err)
	}
	defer cache.Tick(c)
	tx := func(c appengine.Context) error {
		return addCommit(c, com)
	}
	return nil, datastore.RunInTransaction(c, tx, nil)
}

var needsBenchmarkingBytes = []byte(`"NeedsBenchmarking"`)

// addCommit adds the Commit entity to the datastore and updates the tip Tag.
// It must be run inside a datastore transaction.
func addCommit(c appengine.Context, com *Commit) error {
	var ec Commit // existing commit
	isUpdate := false
	err := datastore.Get(c, com.Key(c), &ec)
	if err != nil && err != datastore.ErrNoSuchEntity {
		return fmt.Errorf("getting Commit: %v", err)
	}
	if err == nil {
		// Commit already in the datastore. Any fields different?
		// If not, don't do anything.
		changes := (com.Num != 0 && com.Num != ec.Num) ||
			com.ParentHash != ec.ParentHash ||
			com.Desc != ec.Desc ||
			com.User != ec.User ||
			!com.Time.Equal(ec.Time)
		if !changes {
			return nil
		}
		ec.ParentHash = com.ParentHash
		ec.Desc = com.Desc
		ec.User = com.User
		if !com.Time.IsZero() {
			ec.Time = com.Time
		}
		if com.Num != 0 {
			ec.Num = com.Num
		}
		isUpdate = true
		com = &ec
	}
	p, err := GetPackage(c, com.PackagePath)
	if err != nil {
		return fmt.Errorf("GetPackage: %v", err)
	}
	if com.Num == 0 {
		// get the next commit number
		com.Num = p.NextNum
		p.NextNum++
		if _, err := datastore.Put(c, p.Key(c), p); err != nil {
			return fmt.Errorf("putting Package: %v", err)
		}
	} else if com.Num >= p.NextNum {
		p.NextNum = com.Num + 1
		if _, err := datastore.Put(c, p.Key(c), p); err != nil {
			return fmt.Errorf("putting Package: %v", err)
		}
	}
	// if this isn't the first Commit test the parent commit exists.
	// The all zeros are returned by hg's p1node template for parentless commits.
	if com.ParentHash != "" && com.ParentHash != "0000000000000000000000000000000000000000" && com.ParentHash != "0000" {
		n, err := datastore.NewQuery("Commit").
			Filter("Hash =", com.ParentHash).
			Ancestor(p.Key(c)).
			Count(c)
		if err != nil {
			return fmt.Errorf("testing for parent Commit: %v", err)
		}
		if n == 0 {
			return errors.New("parent commit not found")
		}
	} else if com.Num != 1 {
		// This is the first commit; fail if it is not number 1.
		// (This will happen if we try to upload a new/different repo
		// where there is already commit data. A bad thing to do.)
		return errors.New("this package already has a first commit; aborting")
	}
	// update the tip Tag if this is the Go repo and this isn't on a release branch
	if p.Path == "" && !strings.HasPrefix(com.Desc, "[") && !isUpdate {
		t := &Tag{Kind: "tip", Hash: com.Hash}
		if _, err = datastore.Put(c, t.Key(c), t); err != nil {
			return fmt.Errorf("putting Tag: %v", err)
		}
	}
	// put the Commit
	if err = putCommit(c, com); err != nil {
		return err
	}
	if com.NeedsBenchmarking {
		// add to CommitRun
		cr, err := GetCommitRun(c, com.Num)
		if err != nil {
			return err
		}
		if err = cr.AddCommit(c, com); err != nil {
			return err
		}
		// create PerfResult
		res := &PerfResult{CommitHash: com.Hash, CommitNum: com.Num}
		if _, err := datastore.Put(c, res.Key(c), res); err != nil {
			return fmt.Errorf("putting PerfResult: %v", err)
		}
		// Update perf todo if necessary.
		if err = AddCommitToPerfTodo(c, com); err != nil {
			return err
		}
	}
	return nil
}

// tagHandler records a new tag. It reads a JSON-encoded Tag value from the
// request body and updates the Tag entity for the Kind of tag provided.
//
// This handler is used by a gobuilder process in -commit mode.
func tagHandler(r *http.Request) (interface{}, error) {
	if r.Method != "POST" {
		return nil, errBadMethod(r.Method)
	}

	t := new(Tag)
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(t); err != nil {
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, err
	}
	c := contextForRequest(r)
	defer cache.Tick(c)
	_, err := datastore.Put(c, t.Key(c), t)
	return nil, err
}

// Todo is a todoHandler response.
type Todo struct {
	Kind string // "build-go-commit" or "build-package"
	Data interface{}
}

// todoHandler returns the next action to be performed by a builder.
// It expects "builder" and "kind" query parameters and returns a *Todo value.
// Multiple "kind" parameters may be specified.
func todoHandler(r *http.Request) (interface{}, error) {
	c := contextForRequest(r)
	now := cache.Now(c)
	key := "build-todo-" + r.Form.Encode()
	var todo *Todo
	if cache.Get(r, now, key, &todo) {
		return todo, nil
	}
	var err error
	builder := r.FormValue("builder")
	for _, kind := range r.Form["kind"] {
		var com *Commit
		switch kind {
		case "build-go-commit":
			com, err = buildTodo(c, builder, "", "")
			if com != nil {
				com.PerfResults = []string{}
			}
		case "build-package":
			packagePath := r.FormValue("packagePath")
			goHash := r.FormValue("goHash")
			com, err = buildTodo(c, builder, packagePath, goHash)
			if com != nil {
				com.PerfResults = []string{}
			}
		case "benchmark-go-commit":
			com, err = perfTodo(c, builder)
		}
		if com != nil || err != nil {
			if com != nil {
				// ResultData can be large and not needed on builder.
				com.ResultData = []string{}
			}
			todo = &Todo{Kind: kind, Data: com}
			break
		}
	}
	if err == nil {
		cache.Set(r, now, key, todo)
	}
	return todo, err
}

// buildTodo returns the next Commit to be built (or nil if none available).
//
// If packagePath and goHash are empty, it scans the first 20 Go Commits in
// Num-descending order and returns the first one it finds that doesn't have a
// Result for this builder.
//
// If provided with non-empty packagePath and goHash args, it scans the first
// 20 Commits in Num-descending order for the specified packagePath and
// returns the first that doesn't have a Result for this builder and goHash.
func buildTodo(c appengine.Context, builder, packagePath, goHash string) (*Commit, error) {
	p, err := GetPackage(c, packagePath)
	if err != nil {
		return nil, err
	}

	t := datastore.NewQuery("Commit").
		Ancestor(p.Key(c)).
		Limit(commitsPerPage).
		Order("-Num").
		Run(c)
	for {
		com := new(Commit)
		if _, err := t.Next(com); err == datastore.Done {
			break
		} else if err != nil {
			return nil, err
		}
		if com.Result(builder, goHash) == nil {
			return com, nil
		}
	}

	// Nothing left to do if this is a package (not the Go tree).
	if packagePath != "" {
		return nil, nil
	}

	// If there are no Go tree commits left to build,
	// see if there are any subrepo commits that need to be built at tip.
	// If so, ask the builder to build a go tree at the tip commit.
	// TODO(adg): do the same for "weekly" and "release" tags.

	tag, err := GetTag(c, "tip")
	if err != nil {
		return nil, err
	}

	// Check that this Go commit builds OK for this builder.
	// If not, don't re-build as the subrepos will never get built anyway.
	com, err := tag.Commit(c)
	if err != nil {
		return nil, err
	}
	if r := com.Result(builder, ""); r != nil && !r.OK {
		return nil, nil
	}

	pkgs, err := Packages(c, "subrepo")
	if err != nil {
		return nil, err
	}
	for _, pkg := range pkgs {
		com, err := pkg.LastCommit(c)
		if err != nil {
			c.Warningf("%v: no Commit found: %v", pkg, err)
			continue
		}
		if com.Result(builder, tag.Hash) == nil {
			return tag.Commit(c)
		}
	}

	return nil, nil
}

// perfTodo returns the next Commit to be benchmarked (or nil if none available).
func perfTodo(c appengine.Context, builder string) (*Commit, error) {
	p := &Package{}
	todo := &PerfTodo{Builder: builder}
	err := datastore.Get(c, todo.Key(c), todo)
	if err != nil && err != datastore.ErrNoSuchEntity {
		return nil, fmt.Errorf("fetching PerfTodo: %v", err)
	}
	if err == datastore.ErrNoSuchEntity {
		todo, err = buildPerfTodo(c, builder)
		if err != nil {
			return nil, err
		}
	}
	if len(todo.CommitNums) == 0 {
		return nil, nil
	}

	// Have commit to benchmark, fetch it.
	num := todo.CommitNums[len(todo.CommitNums)-1]
	t := datastore.NewQuery("Commit").
		Ancestor(p.Key(c)).
		Filter("Num =", num).
		Limit(1).
		Run(c)
	com := new(Commit)
	if _, err := t.Next(com); err != nil {
		return nil, err
	}
	if !com.NeedsBenchmarking {
		return nil, fmt.Errorf("commit from perf todo queue is not intended for benchmarking")
	}

	// Remove benchmarks from other builders.
	var benchs []string
	for _, b := range com.PerfResults {
		bb := strings.Split(b, "|")
		if bb[0] == builder && bb[1] != "meta-done" {
			benchs = append(benchs, bb[1])
		}
	}
	com.PerfResults = benchs

	return com, nil
}

// buildPerfTodo creates PerfTodo for the builder with all commits. In a transaction.
func buildPerfTodo(c appengine.Context, builder string) (*PerfTodo, error) {
	todo := &PerfTodo{Builder: builder}
	tx := func(c appengine.Context) error {
		err := datastore.Get(c, todo.Key(c), todo)
		if err != nil && err != datastore.ErrNoSuchEntity {
			return fmt.Errorf("fetching PerfTodo: %v", err)
		}
		if err == nil {
			return nil
		}
		t := datastore.NewQuery("CommitRun").
			Ancestor((&Package{}).Key(c)).
			Order("-StartCommitNum").
			Run(c)
		var nums []int
		var releaseNums []int
	loop:
		for {
			cr := new(CommitRun)
			if _, err := t.Next(cr); err == datastore.Done {
				break
			} else if err != nil {
				return fmt.Errorf("scanning commit runs for perf todo: %v", err)
			}
			for i := len(cr.Hash) - 1; i >= 0; i-- {
				if !cr.NeedsBenchmarking[i] || cr.Hash[i] == "" {
					continue // There's nothing to see here. Move along.
				}
				num := cr.StartCommitNum + i
				for k, v := range knownTags {
					// Releases are benchmarked first, because they are important (and there are few of them).
					if cr.Hash[i] == v {
						releaseNums = append(releaseNums, num)
						if k == "go1" {
							break loop // Point of no benchmark: test/bench/shootout: update timing.log to Go 1.
						}
					}
				}
				nums = append(nums, num)
			}
		}
		todo.CommitNums = orderPerfTodo(nums)
		todo.CommitNums = append(todo.CommitNums, releaseNums...)
		if _, err = datastore.Put(c, todo.Key(c), todo); err != nil {
			return fmt.Errorf("putting PerfTodo: %v", err)
		}
		return nil
	}
	return todo, datastore.RunInTransaction(c, tx, nil)
}

func removeCommitFromPerfTodo(c appengine.Context, builder string, num int) error {
	todo := &PerfTodo{Builder: builder}
	err := datastore.Get(c, todo.Key(c), todo)
	if err != nil && err != datastore.ErrNoSuchEntity {
		return fmt.Errorf("fetching PerfTodo: %v", err)
	}
	if err == datastore.ErrNoSuchEntity {
		return nil
	}
	for i := len(todo.CommitNums) - 1; i >= 0; i-- {
		if todo.CommitNums[i] == num {
			for ; i < len(todo.CommitNums)-1; i++ {
				todo.CommitNums[i] = todo.CommitNums[i+1]
			}
			todo.CommitNums = todo.CommitNums[:i]
			_, err = datastore.Put(c, todo.Key(c), todo)
			if err != nil {
				return fmt.Errorf("putting PerfTodo: %v", err)
			}
			break
		}
	}
	return nil
}

// packagesHandler returns a list of the non-Go Packages monitored
// by the dashboard.
func packagesHandler(r *http.Request) (interface{}, error) {
	kind := r.FormValue("kind")
	c := contextForRequest(r)
	now := cache.Now(c)
	key := "build-packages-" + kind
	var p []*Package
	if cache.Get(r, now, key, &p) {
		return p, nil
	}
	p, err := Packages(c, kind)
	if err != nil {
		return nil, err
	}
	cache.Set(r, now, key, p)
	return p, nil
}

// resultHandler records a build result.
// It reads a JSON-encoded Result value from the request body,
// creates a new Result entity, and updates the relevant Commit entity.
// If the Log field is not empty, resultHandler creates a new Log entity
// and updates the LogHash field before putting the Commit entity.
func resultHandler(r *http.Request) (interface{}, error) {
	if r.Method != "POST" {
		return nil, errBadMethod(r.Method)
	}

	// For now, the gccgo builders are using the old stuff.
	// TODO(adg,cmang): remove this exception when gccgo is updated.
	if dashboardForRequest(r) != gccgoDash {
		v, _ := strconv.Atoi(r.FormValue("version"))
		if v != builderVersion {
			return nil, fmt.Errorf("rejecting POST from builder; need version %v", builderVersion)
		}
	}

	c := contextForRequest(r)
	res := new(Result)
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(res); err != nil {
		return nil, fmt.Errorf("decoding Body: %v", err)
	}
	if err := res.Valid(); err != nil {
		return nil, fmt.Errorf("validating Result: %v", err)
	}
	defer cache.Tick(c)
	// store the Log text if supplied
	if len(res.Log) > 0 {
		hash, err := PutLog(c, res.Log)
		if err != nil {
			return nil, fmt.Errorf("putting Log: %v", err)
		}
		res.LogHash = hash
	}
	tx := func(c appengine.Context) error {
		// check Package exists
		if _, err := GetPackage(c, res.PackagePath); err != nil {
			return fmt.Errorf("GetPackage: %v", err)
		}
		// put Result
		if _, err := datastore.Put(c, res.Key(c), res); err != nil {
			return fmt.Errorf("putting Result: %v", err)
		}
		// add Result to Commit
		com := &Commit{PackagePath: res.PackagePath, Hash: res.Hash}
		if err := com.AddResult(c, res); err != nil {
			return fmt.Errorf("AddResult: %v", err)
		}
		// Send build failure notifications, if necessary.
		// Note this must run after the call AddResult, which
		// populates the Commit's ResultData field.
		return notifyOnFailure(c, com, res.Builder)
	}
	return nil, datastore.RunInTransaction(c, tx, nil)
}

// perf-result request payload
type PerfRequest struct {
	Builder   string
	Benchmark string
	Hash      string
	OK        bool
	Metrics   []PerfMetric
	Artifacts []PerfArtifact
}

type PerfMetric struct {
	Type string
	Val  uint64
}

type PerfArtifact struct {
	Type string
	Body string
}

// perfResultHandler records a becnhmarking result.
func perfResultHandler(r *http.Request) (interface{}, error) {
	defer r.Body.Close()
	if r.Method != "POST" {
		return nil, errBadMethod(r.Method)
	}

	req := new(PerfRequest)
	if err := json.NewDecoder(r.Body).Decode(req); err != nil {
		return nil, fmt.Errorf("decoding Body: %v", err)
	}

	c := contextForRequest(r)
	defer cache.Tick(c)

	// store the text files if supplied
	for i, a := range req.Artifacts {
		hash, err := PutLog(c, a.Body)
		if err != nil {
			return nil, fmt.Errorf("putting Log: %v", err)
		}
		req.Artifacts[i].Body = hash
	}
	tx := func(c appengine.Context) error {
		return addPerfResult(c, r, req)
	}
	return nil, datastore.RunInTransaction(c, tx, nil)
}

// addPerfResult creates PerfResult and updates Commit, PerfTodo,
// PerfMetricRun and PerfConfig.
// MUST be called from inside a transaction.
func addPerfResult(c appengine.Context, r *http.Request, req *PerfRequest) error {
	// check Package exists
	p, err := GetPackage(c, "")
	if err != nil {
		return fmt.Errorf("GetPackage: %v", err)
	}
	// add result to Commit
	com := &Commit{Hash: req.Hash}
	if err := com.AddPerfResult(c, req.Builder, req.Benchmark); err != nil {
		return fmt.Errorf("AddPerfResult: %v", err)
	}

	// add the result to PerfResult
	res := &PerfResult{CommitHash: req.Hash}
	if err := datastore.Get(c, res.Key(c), res); err != nil {
		return fmt.Errorf("getting PerfResult: %v", err)
	}
	present := res.AddResult(req)
	if _, err := datastore.Put(c, res.Key(c), res); err != nil {
		return fmt.Errorf("putting PerfResult: %v", err)
	}

	// Meta-done denotes that there are no benchmarks left.
	if req.Benchmark == "meta-done" {
		// Don't send duplicate emails for the same commit/builder.
		// And don't send emails about too old commits.
		if !present && com.Num >= p.NextNum-commitsPerPage {
			if err := checkPerfChanges(c, r, com, req.Builder, res); err != nil {
				return err
			}
		}
		if err := removeCommitFromPerfTodo(c, req.Builder, com.Num); err != nil {
			return nil
		}
		return nil
	}

	// update PerfConfig
	newBenchmark, err := UpdatePerfConfig(c, r, req)
	if err != nil {
		return fmt.Errorf("updating PerfConfig: %v", err)
	}
	if newBenchmark {
		// If this is a new benchmark on the builder, delete PerfTodo.
		// It will be recreated later with all commits again.
		todo := &PerfTodo{Builder: req.Builder}
		err = datastore.Delete(c, todo.Key(c))
		if err != nil && err != datastore.ErrNoSuchEntity {
			return fmt.Errorf("deleting PerfTodo: %v", err)
		}
	}

	// add perf metrics
	for _, metric := range req.Metrics {
		m, err := GetPerfMetricRun(c, req.Builder, req.Benchmark, metric.Type, com.Num)
		if err != nil {
			return fmt.Errorf("GetPerfMetrics: %v", err)
		}
		if err = m.AddMetric(c, com.Num, metric.Val); err != nil {
			return fmt.Errorf("AddMetric: %v", err)
		}
	}

	return nil
}

// MUST be called from inside a transaction.
func checkPerfChanges(c appengine.Context, r *http.Request, com *Commit, builder string, res *PerfResult) error {
	pc, err := GetPerfConfig(c, r)
	if err != nil {
		return err
	}

	results := res.ParseData()[builder]
	rcNewer := MakePerfResultCache(c, com, true)
	rcOlder := MakePerfResultCache(c, com, false)

	// Check whether we need to send failure notification email.
	if results["meta-done"].OK {
		// This one is successful, see if the next is failed.
		nextRes, err := rcNewer.Next(com.Num)
		if err != nil {
			return err
		}
		if nextRes != nil && isPerfFailed(nextRes, builder) {
			sendPerfFailMail(c, builder, nextRes)
		}
	} else {
		// This one is failed, see if the previous is successful.
		prevRes, err := rcOlder.Next(com.Num)
		if err != nil {
			return err
		}
		if prevRes != nil && !isPerfFailed(prevRes, builder) {
			sendPerfFailMail(c, builder, res)
		}
	}

	// Now see if there are any performance changes.
	// Find the previous and the next results for performance comparison.
	prevRes, err := rcOlder.NextForComparison(com.Num, builder)
	if err != nil {
		return err
	}
	nextRes, err := rcNewer.NextForComparison(com.Num, builder)
	if err != nil {
		return err
	}
	if results["meta-done"].OK {
		// This one is successful, compare with a previous one.
		if prevRes != nil {
			if err := comparePerfResults(c, pc, builder, prevRes, res); err != nil {
				return err
			}
		}
		// Compare a next one with the current.
		if nextRes != nil {
			if err := comparePerfResults(c, pc, builder, res, nextRes); err != nil {
				return err
			}
		}
	} else {
		// This one is failed, compare a previous one with a next one.
		if prevRes != nil && nextRes != nil {
			if err := comparePerfResults(c, pc, builder, prevRes, nextRes); err != nil {
				return err
			}
		}
	}

	return nil
}

func comparePerfResults(c appengine.Context, pc *PerfConfig, builder string, prevRes, res *PerfResult) error {
	changes := significantPerfChanges(pc, builder, prevRes, res)
	if len(changes) == 0 {
		return nil
	}
	com := &Commit{Hash: res.CommitHash}
	if err := datastore.Get(c, com.Key(c), com); err != nil {
		return fmt.Errorf("getting commit %v: %v", com.Hash, err)
	}
	sendPerfMailLater.Call(c, com, prevRes.CommitHash, builder, changes) // add task to queue
	return nil
}

// logHandler displays log text for a given hash.
// It handles paths like "/log/hash".
func logHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-type", "text/plain; charset=utf-8")
	c := contextForRequest(r)
	hash := r.URL.Path[strings.LastIndex(r.URL.Path, "/")+1:]
	key := datastore.NewKey(c, "Log", hash, 0, nil)
	l := new(Log)
	if err := datastore.Get(c, key, l); err != nil {
		logErr(w, r, err)
		return
	}
	b, err := l.Text()
	if err != nil {
		logErr(w, r, err)
		return
	}
	w.Write(b)
}

type dashHandler func(*http.Request) (interface{}, error)

type dashResponse struct {
	Response interface{}
	Error    string
}

// errBadMethod is returned by a dashHandler when
// the request has an unsuitable method.
type errBadMethod string

func (e errBadMethod) Error() string {
	return "bad method: " + string(e)
}

// AuthHandler wraps a http.HandlerFunc with a handler that validates the
// supplied key and builder query parameters.
func AuthHandler(h dashHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		c := contextForRequest(r)

		// Put the URL Query values into r.Form to avoid parsing the
		// request body when calling r.FormValue.
		r.Form = r.URL.Query()

		var err error
		var resp interface{}

		// Validate key query parameter for POST requests only.
		key := r.FormValue("key")
		builder := r.FormValue("builder")
		if r.Method == "POST" && !validKey(c, key, builder) {
			err = fmt.Errorf("invalid key %q for builder %q", key, builder)
		}

		// Call the original HandlerFunc and return the response.
		if err == nil {
			resp, err = h(r)
		}

		// Write JSON response.
		dashResp := &dashResponse{Response: resp}
		if err != nil {
			c.Errorf("%v", err)
			dashResp.Error = err.Error()
		}
		w.Header().Set("Content-Type", "application/json")
		if err = json.NewEncoder(w).Encode(dashResp); err != nil {
			c.Criticalf("encoding response: %v", err)
		}
	}
}

func keyHandler(w http.ResponseWriter, r *http.Request) {
	builder := r.FormValue("builder")
	if builder == "" {
		logErr(w, r, errors.New("must supply builder in query string"))
		return
	}
	c := contextForRequest(r)
	fmt.Fprint(w, builderKey(c, builder))
}

func init() {
	// admin handlers
	handleFunc("/init", initHandler)
	handleFunc("/key", keyHandler)

	// authenticated handlers
	handleFunc("/commit", AuthHandler(commitHandler))
	handleFunc("/packages", AuthHandler(packagesHandler))
	handleFunc("/result", AuthHandler(resultHandler))
	handleFunc("/perf-result", AuthHandler(perfResultHandler))
	handleFunc("/tag", AuthHandler(tagHandler))
	handleFunc("/todo", AuthHandler(todoHandler))

	// public handlers
	handleFunc("/log/", logHandler)
}

func validHash(hash string) bool {
	// TODO(adg): correctly validate a hash
	return hash != ""
}

func validKey(c appengine.Context, key, builder string) bool {
	return isMasterKey(c, key) || key == builderKey(c, builder)
}

func isMasterKey(c appengine.Context, k string) bool {
	return appengine.IsDevAppServer() || k == key.Secret(c)
}

func builderKey(c appengine.Context, builder string) string {
	h := hmac.New(md5.New, []byte(key.Secret(c)))
	h.Write([]byte(builder))
	return fmt.Sprintf("%x", h.Sum(nil))
}

func logErr(w http.ResponseWriter, r *http.Request, err error) {
	contextForRequest(r).Errorf("Error: %v", err)
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprint(w, "Error: ", err)
}

func contextForRequest(r *http.Request) appengine.Context {
	return dashboardForRequest(r).Context(appengine.NewContext(r))
}

// limitStringLength essentially does return s[:max],
// but it ensures that we dot not split UTF-8 rune in half.
// Otherwise appengine python scripts will break badly.
func limitStringLength(s string, max int) string {
	if len(s) <= max {
		return s
	}
	for {
		s = s[:max]
		r, size := utf8.DecodeLastRuneInString(s)
		if r != utf8.RuneError || size != 1 {
			return s
		}
		max--
	}
}
