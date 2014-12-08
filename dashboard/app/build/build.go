// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

import (
	"bytes"
	"compress/gzip"
	"crypto/sha1"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"appengine"
	"appengine/datastore"

	"cache"
)

const (
	maxDatastoreStringLen = 500
	PerfRunLength         = 1024
)

// A Package describes a package that is listed on the dashboard.
type Package struct {
	Kind    string // "subrepo", "external", or empty for the main Go tree
	Name    string
	Path    string // (empty for the main Go tree)
	NextNum int    // Num of the next head Commit
}

func (p *Package) String() string {
	return fmt.Sprintf("%s: %q", p.Path, p.Name)
}

func (p *Package) Key(c appengine.Context) *datastore.Key {
	key := p.Path
	if key == "" {
		key = "go"
	}
	return datastore.NewKey(c, "Package", key, 0, nil)
}

// LastCommit returns the most recent Commit for this Package.
func (p *Package) LastCommit(c appengine.Context) (*Commit, error) {
	var commits []*Commit
	_, err := datastore.NewQuery("Commit").
		Ancestor(p.Key(c)).
		Order("-Time").
		Limit(1).
		GetAll(c, &commits)
	if err != nil {
		return nil, err
	}
	if len(commits) != 1 {
		return nil, datastore.ErrNoSuchEntity
	}
	return commits[0], nil
}

// GetPackage fetches a Package by path from the datastore.
func GetPackage(c appengine.Context, path string) (*Package, error) {
	p := &Package{Path: path}
	err := datastore.Get(c, p.Key(c), p)
	if err == datastore.ErrNoSuchEntity {
		return nil, fmt.Errorf("package %q not found", path)
	}
	return p, err
}

// A Commit describes an individual commit in a package.
//
// Each Commit entity is a descendant of its associated Package entity.
// In other words, all Commits with the same PackagePath belong to the same
// datastore entity group.
type Commit struct {
	PackagePath string // (empty for main repo commits)
	Hash        string
	ParentHash  string
	Num         int // Internal monotonic counter unique to this package.

	User              string
	Desc              string `datastore:",noindex"`
	Time              time.Time
	NeedsBenchmarking bool
	TryPatch          bool
	Branch            string

	// ResultData is the Data string of each build Result for this Commit.
	// For non-Go commits, only the Results for the current Go tip, weekly,
	// and release Tags are stored here. This is purely de-normalized data.
	// The complete data set is stored in Result entities.
	ResultData []string `datastore:",noindex"`

	// PerfResults holds a set of “builder|benchmark” tuples denoting
	// what benchmarks have been executed on the commit.
	PerfResults []string `datastore:",noindex"`

	FailNotificationSent bool
}

func (com *Commit) Key(c appengine.Context) *datastore.Key {
	if com.Hash == "" {
		panic("tried Key on Commit with empty Hash")
	}
	p := Package{Path: com.PackagePath}
	key := com.PackagePath + "|" + com.Hash
	return datastore.NewKey(c, "Commit", key, 0, p.Key(c))
}

func (c *Commit) Valid() error {
	if !validHash(c.Hash) {
		return errors.New("invalid Hash")
	}
	if c.ParentHash != "" && !validHash(c.ParentHash) { // empty is OK
		return errors.New("invalid ParentHash")
	}
	return nil
}

func putCommit(c appengine.Context, com *Commit) error {
	if err := com.Valid(); err != nil {
		return fmt.Errorf("putting Commit: %v", err)
	}
	if com.Num == 0 && com.ParentHash != "0000" { // 0000 is used in tests
		return fmt.Errorf("putting Commit: invalid Num (must be > 0)")
	}
	if _, err := datastore.Put(c, com.Key(c), com); err != nil {
		return fmt.Errorf("putting Commit: %v", err)
	}
	return nil
}

// each result line is approx 105 bytes. This constant is a tradeoff between
// build history and the AppEngine datastore limit of 1mb.
const maxResults = 1000

// AddResult adds the denormalized Result data to the Commit's Result field.
// It must be called from inside a datastore transaction.
func (com *Commit) AddResult(c appengine.Context, r *Result) error {
	if err := datastore.Get(c, com.Key(c), com); err != nil {
		return fmt.Errorf("getting Commit: %v", err)
	}

	var resultExists bool
	for i, s := range com.ResultData {
		// if there already exists result data for this builder at com, overwrite it.
		if strings.HasPrefix(s, r.Builder+"|") && strings.HasSuffix(s, "|"+r.GoHash) {
			resultExists = true
			com.ResultData[i] = r.Data()
		}
	}
	if !resultExists {
		// otherwise, add the new result data for this builder.
		com.ResultData = trim(append(com.ResultData, r.Data()), maxResults)
	}
	return putCommit(c, com)
}

// AddPerfResult remembers that the builder has run the benchmark on the commit.
// It must be called from inside a datastore transaction.
func (com *Commit) AddPerfResult(c appengine.Context, builder, benchmark string) error {
	if err := datastore.Get(c, com.Key(c), com); err != nil {
		return fmt.Errorf("getting Commit: %v", err)
	}
	if !com.NeedsBenchmarking {
		return fmt.Errorf("trying to add perf result to Commit(%v) that does not require benchmarking", com.Hash)
	}
	s := builder + "|" + benchmark
	for _, v := range com.PerfResults {
		if v == s {
			return nil
		}
	}
	com.PerfResults = append(com.PerfResults, s)
	return putCommit(c, com)
}

func trim(s []string, n int) []string {
	l := min(len(s), n)
	return s[len(s)-l:]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Result returns the build Result for this Commit for the given builder/goHash.
func (c *Commit) Result(builder, goHash string) *Result {
	for _, r := range c.ResultData {
		p := strings.SplitN(r, "|", 4)
		if len(p) != 4 || p[0] != builder || p[3] != goHash {
			continue
		}
		return partsToHash(c, p)
	}
	return nil
}

// Results returns the build Results for this Commit.
func (c *Commit) Results() (results []*Result) {
	for _, r := range c.ResultData {
		p := strings.SplitN(r, "|", 4)
		if len(p) != 4 {
			continue
		}
		results = append(results, partsToHash(c, p))
	}
	return
}

func (c *Commit) ResultGoHashes() []string {
	// For the main repo, just return the empty string
	// (there's no corresponding main repo hash for a main repo Commit).
	// This function is only really useful for sub-repos.
	if c.PackagePath == "" {
		return []string{""}
	}
	var hashes []string
	for _, r := range c.ResultData {
		p := strings.SplitN(r, "|", 4)
		if len(p) != 4 {
			continue
		}
		// Append only new results (use linear scan to preserve order).
		if !contains(hashes, p[3]) {
			hashes = append(hashes, p[3])
		}
	}
	// Return results in reverse order (newest first).
	reverse(hashes)
	return hashes
}

func contains(t []string, s string) bool {
	for _, s2 := range t {
		if s2 == s {
			return true
		}
	}
	return false
}

func reverse(s []string) {
	for i := 0; i < len(s)/2; i++ {
		j := len(s) - i - 1
		s[i], s[j] = s[j], s[i]
	}
}

// A CommitRun provides summary information for commits [StartCommitNum, StartCommitNum + PerfRunLength).
// Descendant of Package.
type CommitRun struct {
	PackagePath       string // (empty for main repo commits)
	StartCommitNum    int
	Hash              []string    `datastore:",noindex"`
	User              []string    `datastore:",noindex"`
	Desc              []string    `datastore:",noindex"` // Only first line.
	Time              []time.Time `datastore:",noindex"`
	NeedsBenchmarking []bool      `datastore:",noindex"`
}

func (cr *CommitRun) Key(c appengine.Context) *datastore.Key {
	p := Package{Path: cr.PackagePath}
	key := strconv.Itoa(cr.StartCommitNum)
	return datastore.NewKey(c, "CommitRun", key, 0, p.Key(c))
}

// GetCommitRun loads and returns CommitRun that contains information
// for commit commitNum.
func GetCommitRun(c appengine.Context, commitNum int) (*CommitRun, error) {
	cr := &CommitRun{StartCommitNum: commitNum / PerfRunLength * PerfRunLength}
	err := datastore.Get(c, cr.Key(c), cr)
	if err != nil && err != datastore.ErrNoSuchEntity {
		return nil, fmt.Errorf("getting CommitRun: %v", err)
	}
	if len(cr.Hash) != PerfRunLength {
		cr.Hash = make([]string, PerfRunLength)
		cr.User = make([]string, PerfRunLength)
		cr.Desc = make([]string, PerfRunLength)
		cr.Time = make([]time.Time, PerfRunLength)
		cr.NeedsBenchmarking = make([]bool, PerfRunLength)
	}
	return cr, nil
}

func (cr *CommitRun) AddCommit(c appengine.Context, com *Commit) error {
	if com.Num < cr.StartCommitNum || com.Num >= cr.StartCommitNum+PerfRunLength {
		return fmt.Errorf("AddCommit: commit num %v out of range [%v, %v)",
			com.Num, cr.StartCommitNum, cr.StartCommitNum+PerfRunLength)
	}
	i := com.Num - cr.StartCommitNum
	// Be careful with string lengths,
	// we need to fit 1024 commits into 1 MB.
	cr.Hash[i] = com.Hash
	cr.User[i] = shortDesc(com.User)
	cr.Desc[i] = shortDesc(com.Desc)
	cr.Time[i] = com.Time
	cr.NeedsBenchmarking[i] = com.NeedsBenchmarking
	if _, err := datastore.Put(c, cr.Key(c), cr); err != nil {
		return fmt.Errorf("putting CommitRun: %v", err)
	}
	return nil
}

// GetCommits returns [startCommitNum, startCommitNum+n) commits.
// Commits information is partial (obtained from CommitRun),
// do not store them back into datastore.
func GetCommits(c appengine.Context, startCommitNum, n int) ([]*Commit, error) {
	if startCommitNum < 0 || n <= 0 {
		return nil, fmt.Errorf("GetCommits: invalid args (%v, %v)", startCommitNum, n)
	}

	p := &Package{}
	t := datastore.NewQuery("CommitRun").
		Ancestor(p.Key(c)).
		Filter("StartCommitNum >=", startCommitNum/PerfRunLength*PerfRunLength).
		Order("StartCommitNum").
		Limit(100).
		Run(c)

	res := make([]*Commit, n)
	for {
		cr := new(CommitRun)
		_, err := t.Next(cr)
		if err == datastore.Done {
			break
		}
		if err != nil {
			return nil, err
		}
		if cr.StartCommitNum >= startCommitNum+n {
			break
		}
		// Calculate start index for copying.
		i := 0
		if cr.StartCommitNum < startCommitNum {
			i = startCommitNum - cr.StartCommitNum
		}
		// Calculate end index for copying.
		e := PerfRunLength
		if cr.StartCommitNum+e > startCommitNum+n {
			e = startCommitNum + n - cr.StartCommitNum
		}
		for ; i < e; i++ {
			com := new(Commit)
			com.Hash = cr.Hash[i]
			com.User = cr.User[i]
			com.Desc = cr.Desc[i]
			com.Time = cr.Time[i]
			com.NeedsBenchmarking = cr.NeedsBenchmarking[i]
			res[cr.StartCommitNum-startCommitNum+i] = com
		}
		if e != PerfRunLength {
			break
		}
	}
	return res, nil
}

// partsToHash converts a Commit and ResultData substrings to a Result.
func partsToHash(c *Commit, p []string) *Result {
	return &Result{
		Builder:     p[0],
		Hash:        c.Hash,
		PackagePath: c.PackagePath,
		GoHash:      p[3],
		OK:          p[1] == "true",
		LogHash:     p[2],
	}
}

// A Result describes a build result for a Commit on an OS/architecture.
//
// Each Result entity is a descendant of its associated Package entity.
type Result struct {
	PackagePath string // (empty for Go commits)
	Builder     string // "os-arch[-note]"
	Hash        string

	// The Go Commit this was built against (empty for Go commits).
	GoHash string

	OK      bool
	Log     string `datastore:"-"`        // for JSON unmarshaling only
	LogHash string `datastore:",noindex"` // Key to the Log record.

	RunTime int64 // time to build+test in nanoseconds
}

func (r *Result) Key(c appengine.Context) *datastore.Key {
	p := Package{Path: r.PackagePath}
	key := r.Builder + "|" + r.PackagePath + "|" + r.Hash + "|" + r.GoHash
	return datastore.NewKey(c, "Result", key, 0, p.Key(c))
}

func (r *Result) Valid() error {
	if !validHash(r.Hash) {
		return errors.New("invalid Hash")
	}
	if r.PackagePath != "" && !validHash(r.GoHash) {
		return errors.New("invalid GoHash")
	}
	return nil
}

// Data returns the Result in string format
// to be stored in Commit's ResultData field.
func (r *Result) Data() string {
	return fmt.Sprintf("%v|%v|%v|%v", r.Builder, r.OK, r.LogHash, r.GoHash)
}

// A PerfResult describes all benchmarking result for a Commit.
// Descendant of Package.
type PerfResult struct {
	PackagePath string
	CommitHash  string
	CommitNum   int
	Data        []string `datastore:",noindex"` // "builder|benchmark|ok|metric1=val1|metric2=val2|file:log=hash|file:cpuprof=hash"

	// Local cache with parsed Data.
	// Maps builder->benchmark->ParsedPerfResult.
	parsedData map[string]map[string]*ParsedPerfResult
}

type ParsedPerfResult struct {
	OK        bool
	Metrics   map[string]uint64
	Artifacts map[string]string
}

func (r *PerfResult) Key(c appengine.Context) *datastore.Key {
	p := Package{Path: r.PackagePath}
	key := r.CommitHash
	return datastore.NewKey(c, "PerfResult", key, 0, p.Key(c))
}

// AddResult add the benchmarking result to r.
// Existing result for the same builder/benchmark is replaced if already exists.
// Returns whether the result was already present.
func (r *PerfResult) AddResult(req *PerfRequest) bool {
	present := false
	str := fmt.Sprintf("%v|%v|", req.Builder, req.Benchmark)
	for i, s := range r.Data {
		if strings.HasPrefix(s, str) {
			present = true
			last := len(r.Data) - 1
			r.Data[i] = r.Data[last]
			r.Data = r.Data[:last]
			break
		}
	}
	ok := "ok"
	if !req.OK {
		ok = "false"
	}
	str += ok
	for _, m := range req.Metrics {
		str += fmt.Sprintf("|%v=%v", m.Type, m.Val)
	}
	for _, a := range req.Artifacts {
		str += fmt.Sprintf("|file:%v=%v", a.Type, a.Body)
	}
	r.Data = append(r.Data, str)
	r.parsedData = nil
	return present
}

func (r *PerfResult) ParseData() map[string]map[string]*ParsedPerfResult {
	if r.parsedData != nil {
		return r.parsedData
	}
	res := make(map[string]map[string]*ParsedPerfResult)
	for _, str := range r.Data {
		ss := strings.Split(str, "|")
		builder := ss[0]
		bench := ss[1]
		ok := ss[2]
		m := res[builder]
		if m == nil {
			m = make(map[string]*ParsedPerfResult)
			res[builder] = m
		}
		var p ParsedPerfResult
		p.OK = ok == "ok"
		p.Metrics = make(map[string]uint64)
		p.Artifacts = make(map[string]string)
		for _, entry := range ss[3:] {
			if strings.HasPrefix(entry, "file:") {
				ss1 := strings.Split(entry[len("file:"):], "=")
				p.Artifacts[ss1[0]] = ss1[1]
			} else {
				ss1 := strings.Split(entry, "=")
				val, _ := strconv.ParseUint(ss1[1], 10, 64)
				p.Metrics[ss1[0]] = val
			}
		}
		m[bench] = &p
	}
	r.parsedData = res
	return res
}

// A PerfMetricRun entity holds a set of metric values for builder/benchmark/metric
// for commits [StartCommitNum, StartCommitNum + PerfRunLength).
// Descendant of Package.
type PerfMetricRun struct {
	PackagePath    string
	Builder        string
	Benchmark      string
	Metric         string // e.g. realtime, cputime, gc-pause
	StartCommitNum int
	Vals           []int64 `datastore:",noindex"`
}

func (m *PerfMetricRun) Key(c appengine.Context) *datastore.Key {
	p := Package{Path: m.PackagePath}
	key := m.Builder + "|" + m.Benchmark + "|" + m.Metric + "|" + strconv.Itoa(m.StartCommitNum)
	return datastore.NewKey(c, "PerfMetricRun", key, 0, p.Key(c))
}

// GetPerfMetricRun loads and returns PerfMetricRun that contains information
// for commit commitNum.
func GetPerfMetricRun(c appengine.Context, builder, benchmark, metric string, commitNum int) (*PerfMetricRun, error) {
	startCommitNum := commitNum / PerfRunLength * PerfRunLength
	m := &PerfMetricRun{Builder: builder, Benchmark: benchmark, Metric: metric, StartCommitNum: startCommitNum}
	err := datastore.Get(c, m.Key(c), m)
	if err != nil && err != datastore.ErrNoSuchEntity {
		return nil, fmt.Errorf("getting PerfMetricRun: %v", err)
	}
	if len(m.Vals) != PerfRunLength {
		m.Vals = make([]int64, PerfRunLength)
	}
	return m, nil
}

func (m *PerfMetricRun) AddMetric(c appengine.Context, commitNum int, v uint64) error {
	if commitNum < m.StartCommitNum || commitNum >= m.StartCommitNum+PerfRunLength {
		return fmt.Errorf("AddMetric: CommitNum %v out of range [%v, %v)",
			commitNum, m.StartCommitNum, m.StartCommitNum+PerfRunLength)
	}
	m.Vals[commitNum-m.StartCommitNum] = int64(v)
	if _, err := datastore.Put(c, m.Key(c), m); err != nil {
		return fmt.Errorf("putting PerfMetricRun: %v", err)
	}
	return nil
}

// GetPerfMetricsForCommits returns perf metrics for builder/benchmark/metric
// and commits [startCommitNum, startCommitNum+n).
func GetPerfMetricsForCommits(c appengine.Context, builder, benchmark, metric string, startCommitNum, n int) ([]uint64, error) {
	if startCommitNum < 0 || n <= 0 {
		return nil, fmt.Errorf("GetPerfMetricsForCommits: invalid args (%v, %v)", startCommitNum, n)
	}

	p := &Package{}
	t := datastore.NewQuery("PerfMetricRun").
		Ancestor(p.Key(c)).
		Filter("Builder =", builder).
		Filter("Benchmark =", benchmark).
		Filter("Metric =", metric).
		Filter("StartCommitNum >=", startCommitNum/PerfRunLength*PerfRunLength).
		Order("StartCommitNum").
		Limit(100).
		Run(c)

	res := make([]uint64, n)
	for {
		metrics := new(PerfMetricRun)
		_, err := t.Next(metrics)
		if err == datastore.Done {
			break
		}
		if err != nil {
			return nil, err
		}
		if metrics.StartCommitNum >= startCommitNum+n {
			break
		}
		// Calculate start index for copying.
		i := 0
		if metrics.StartCommitNum < startCommitNum {
			i = startCommitNum - metrics.StartCommitNum
		}
		// Calculate end index for copying.
		e := PerfRunLength
		if metrics.StartCommitNum+e > startCommitNum+n {
			e = startCommitNum + n - metrics.StartCommitNum
		}
		for ; i < e; i++ {
			res[metrics.StartCommitNum-startCommitNum+i] = uint64(metrics.Vals[i])
		}
		if e != PerfRunLength {
			break
		}
	}
	return res, nil
}

// PerfConfig holds read-mostly configuration related to benchmarking.
// There is only one PerfConfig entity.
type PerfConfig struct {
	BuilderBench []string `datastore:",noindex"` // "builder|benchmark" pairs
	BuilderProcs []string `datastore:",noindex"` // "builder|proc" pairs
	BenchMetric  []string `datastore:",noindex"` // "benchmark|metric" pairs
	NoiseLevels  []string `datastore:",noindex"` // "builder|benchmark|metric1=noise1|metric2=noise2"

	// Local cache of "builder|benchmark|metric" -> noise.
	noise map[string]float64
}

func PerfConfigKey(c appengine.Context) *datastore.Key {
	p := Package{}
	return datastore.NewKey(c, "PerfConfig", "PerfConfig", 0, p.Key(c))
}

const perfConfigCacheKey = "perf-config"

func GetPerfConfig(c appengine.Context, r *http.Request) (*PerfConfig, error) {
	pc := new(PerfConfig)
	now := cache.Now(c)
	if cache.Get(r, now, perfConfigCacheKey, pc) {
		return pc, nil
	}
	err := datastore.Get(c, PerfConfigKey(c), pc)
	if err != nil && err != datastore.ErrNoSuchEntity {
		return nil, fmt.Errorf("GetPerfConfig: %v", err)
	}
	cache.Set(r, now, perfConfigCacheKey, pc)
	return pc, nil
}

func (pc *PerfConfig) NoiseLevel(builder, benchmark, metric string) float64 {
	if pc.noise == nil {
		pc.noise = make(map[string]float64)
		for _, str := range pc.NoiseLevels {
			split := strings.Split(str, "|")
			builderBench := split[0] + "|" + split[1]
			for _, entry := range split[2:] {
				metricValue := strings.Split(entry, "=")
				noise, _ := strconv.ParseFloat(metricValue[1], 64)
				pc.noise[builderBench+"|"+metricValue[0]] = noise
			}
		}
	}
	me := fmt.Sprintf("%v|%v|%v", builder, benchmark, metric)
	n := pc.noise[me]
	if n == 0 {
		// Use a very conservative value
		// until we have learned the real noise level.
		n = 200
	}
	return n
}

// UpdatePerfConfig updates the PerfConfig entity with results of benchmarking.
// Returns whether it's a benchmark that we have not yet seem on the builder.
func UpdatePerfConfig(c appengine.Context, r *http.Request, req *PerfRequest) (newBenchmark bool, err error) {
	pc, err := GetPerfConfig(c, r)
	if err != nil {
		return false, err
	}

	modified := false
	add := func(arr *[]string, str string) {
		for _, s := range *arr {
			if s == str {
				return
			}
		}
		*arr = append(*arr, str)
		modified = true
		return
	}

	BenchProcs := strings.Split(req.Benchmark, "-")
	benchmark := BenchProcs[0]
	procs := "1"
	if len(BenchProcs) > 1 {
		procs = BenchProcs[1]
	}

	add(&pc.BuilderBench, req.Builder+"|"+benchmark)
	newBenchmark = modified
	add(&pc.BuilderProcs, req.Builder+"|"+procs)
	for _, m := range req.Metrics {
		add(&pc.BenchMetric, benchmark+"|"+m.Type)
	}

	if modified {
		if _, err := datastore.Put(c, PerfConfigKey(c), pc); err != nil {
			return false, fmt.Errorf("putting PerfConfig: %v", err)
		}
		cache.Tick(c)
	}
	return newBenchmark, nil
}

type MetricList []string

func (l MetricList) Len() int {
	return len(l)
}

func (l MetricList) Less(i, j int) bool {
	bi := strings.HasPrefix(l[i], "build-") || strings.HasPrefix(l[i], "binary-")
	bj := strings.HasPrefix(l[j], "build-") || strings.HasPrefix(l[j], "binary-")
	if bi == bj {
		return l[i] < l[j]
	}
	return !bi
}

func (l MetricList) Swap(i, j int) {
	l[i], l[j] = l[j], l[i]
}

func collectList(all []string, idx int, second string) (res []string) {
	m := make(map[string]bool)
	for _, str := range all {
		ss := strings.Split(str, "|")
		v := ss[idx]
		v2 := ss[1-idx]
		if (second == "" || second == v2) && !m[v] {
			m[v] = true
			res = append(res, v)
		}
	}
	sort.Sort(MetricList(res))
	return res
}

func (pc *PerfConfig) BuildersForBenchmark(bench string) []string {
	return collectList(pc.BuilderBench, 0, bench)
}

func (pc *PerfConfig) BenchmarksForBuilder(builder string) []string {
	return collectList(pc.BuilderBench, 1, builder)
}

func (pc *PerfConfig) MetricsForBenchmark(bench string) []string {
	return collectList(pc.BenchMetric, 1, bench)
}

func (pc *PerfConfig) BenchmarkProcList() (res []string) {
	bl := pc.BenchmarksForBuilder("")
	pl := pc.ProcList("")
	for _, b := range bl {
		for _, p := range pl {
			res = append(res, fmt.Sprintf("%v-%v", b, p))
		}
	}
	return res
}

func (pc *PerfConfig) ProcList(builder string) []int {
	ss := collectList(pc.BuilderProcs, 1, builder)
	var procs []int
	for _, s := range ss {
		p, _ := strconv.ParseInt(s, 10, 32)
		procs = append(procs, int(p))
	}
	sort.Ints(procs)
	return procs
}

// A PerfTodo contains outstanding commits for benchmarking for a builder.
// Descendant of Package.
type PerfTodo struct {
	PackagePath string // (empty for main repo commits)
	Builder     string
	CommitNums  []int `datastore:",noindex"` // LIFO queue of commits to benchmark.
}

func (todo *PerfTodo) Key(c appengine.Context) *datastore.Key {
	p := Package{Path: todo.PackagePath}
	key := todo.Builder
	return datastore.NewKey(c, "PerfTodo", key, 0, p.Key(c))
}

// AddCommitToPerfTodo adds the commit to all existing PerfTodo entities.
func AddCommitToPerfTodo(c appengine.Context, com *Commit) error {
	var todos []*PerfTodo
	_, err := datastore.NewQuery("PerfTodo").
		Ancestor((&Package{}).Key(c)).
		GetAll(c, &todos)
	if err != nil {
		return fmt.Errorf("fetching PerfTodo's: %v", err)
	}
	for _, todo := range todos {
		todo.CommitNums = append(todo.CommitNums, com.Num)
		_, err = datastore.Put(c, todo.Key(c), todo)
		if err != nil {
			return fmt.Errorf("updating PerfTodo: %v", err)
		}
	}
	return nil
}

// A Log is a gzip-compressed log file stored under the SHA1 hash of the
// uncompressed log text.
type Log struct {
	CompressedLog []byte
}

func (l *Log) Text() ([]byte, error) {
	d, err := gzip.NewReader(bytes.NewBuffer(l.CompressedLog))
	if err != nil {
		return nil, fmt.Errorf("reading log data: %v", err)
	}
	b, err := ioutil.ReadAll(d)
	if err != nil {
		return nil, fmt.Errorf("reading log data: %v", err)
	}
	return b, nil
}

func PutLog(c appengine.Context, text string) (hash string, err error) {
	h := sha1.New()
	io.WriteString(h, text)
	b := new(bytes.Buffer)
	z, _ := gzip.NewWriterLevel(b, gzip.BestCompression)
	io.WriteString(z, text)
	z.Close()
	hash = fmt.Sprintf("%x", h.Sum(nil))
	key := datastore.NewKey(c, "Log", hash, 0, nil)
	_, err = datastore.Put(c, key, &Log{b.Bytes()})
	return
}

// A Tag is used to keep track of the most recent Go weekly and release tags.
// Typically there will be one Tag entity for each kind of hg tag.
type Tag struct {
	Kind string // "weekly", "release", or "tip"
	Name string // the tag itself (for example: "release.r60")
	Hash string
}

func (t *Tag) Key(c appengine.Context) *datastore.Key {
	p := &Package{}
	return datastore.NewKey(c, "Tag", t.Kind, 0, p.Key(c))
}

func (t *Tag) Valid() error {
	if t.Kind != "weekly" && t.Kind != "release" && t.Kind != "tip" {
		return errors.New("invalid Kind")
	}
	if !validHash(t.Hash) {
		return errors.New("invalid Hash")
	}
	return nil
}

// Commit returns the Commit that corresponds with this Tag.
func (t *Tag) Commit(c appengine.Context) (*Commit, error) {
	com := &Commit{Hash: t.Hash}
	err := datastore.Get(c, com.Key(c), com)
	return com, err
}

// GetTag fetches a Tag by name from the datastore.
func GetTag(c appengine.Context, tag string) (*Tag, error) {
	t := &Tag{Kind: tag}
	if err := datastore.Get(c, t.Key(c), t); err != nil {
		if err == datastore.ErrNoSuchEntity {
			return nil, errors.New("tag not found: " + tag)
		}
		return nil, err
	}
	if err := t.Valid(); err != nil {
		return nil, err
	}
	return t, nil
}

// Packages returns packages of the specified kind.
// Kind must be one of "external" or "subrepo".
func Packages(c appengine.Context, kind string) ([]*Package, error) {
	switch kind {
	case "external", "subrepo":
	default:
		return nil, errors.New(`kind must be one of "external" or "subrepo"`)
	}
	var pkgs []*Package
	q := datastore.NewQuery("Package").Filter("Kind=", kind)
	for t := q.Run(c); ; {
		pkg := new(Package)
		_, err := t.Next(pkg)
		if err == datastore.Done {
			break
		} else if err != nil {
			return nil, err
		}
		if pkg.Path != "" {
			pkgs = append(pkgs, pkg)
		}
	}
	return pkgs, nil
}
