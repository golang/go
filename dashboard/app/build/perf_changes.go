// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

import (
	"bytes"
	"fmt"
	"html/template"
	"net/http"
	"sort"
	"strconv"

	"appengine"
	"appengine/datastore"
)

func init() {
	handleFunc("/perf", perfChangesHandler)
}

// perfSummaryHandler draws the main benchmarking page.
func perfChangesHandler(w http.ResponseWriter, r *http.Request) {
	d := dashboardForRequest(r)
	c := d.Context(appengine.NewContext(r))

	page, _ := strconv.Atoi(r.FormValue("page"))
	if page < 0 {
		page = 0
	}

	pc, err := GetPerfConfig(c, r)
	if err != nil {
		logErr(w, r, err)
		return
	}

	commits, err := dashPerfCommits(c, page)
	if err != nil {
		logErr(w, r, err)
		return
	}

	// Fetch PerfResult's for the commits.
	var uiCommits []*perfChangesCommit
	rc := MakePerfResultCache(c, commits[0], false)

	// But first compare tip with the last release.
	if page == 0 {
		res0 := &PerfResult{CommitHash: knownTags[lastRelease]}
		if err := datastore.Get(c, res0.Key(c), res0); err != nil && err != datastore.ErrNoSuchEntity {
			logErr(w, r, fmt.Errorf("getting PerfResult: %v", err))
			return
		}
		if err != datastore.ErrNoSuchEntity {
			uiCom, err := handleOneCommit(pc, commits[0], rc, res0)
			if err != nil {
				logErr(w, r, err)
				return
			}
			uiCom.IsSummary = true
			uiCom.ParentHash = lastRelease
			uiCommits = append(uiCommits, uiCom)
		}
	}

	for _, com := range commits {
		uiCom, err := handleOneCommit(pc, com, rc, nil)
		if err != nil {
			logErr(w, r, err)
			return
		}
		uiCommits = append(uiCommits, uiCom)
	}

	p := &Pagination{}
	if len(commits) == commitsPerPage {
		p.Next = page + 1
	}
	if page > 0 {
		p.Prev = page - 1
		p.HasPrev = true
	}

	data := &perfChangesData{d, p, uiCommits}

	var buf bytes.Buffer
	if err := perfChangesTemplate.Execute(&buf, data); err != nil {
		logErr(w, r, err)
		return
	}

	buf.WriteTo(w)
}

func handleOneCommit(pc *PerfConfig, com *Commit, rc *PerfResultCache, baseRes *PerfResult) (*perfChangesCommit, error) {
	uiCom := new(perfChangesCommit)
	uiCom.Commit = com
	res1 := rc.Get(com.Num)
	for builder, benchmarks1 := range res1.ParseData() {
		for benchmark, data1 := range benchmarks1 {
			if benchmark != "meta-done" || !data1.OK {
				uiCom.NumResults++
			}
			if !data1.OK {
				v := new(perfChangesChange)
				v.diff = 10000
				v.Style = "fail"
				v.Builder = builder
				v.Link = fmt.Sprintf("log/%v", data1.Artifacts["log"])
				v.Val = builder
				v.Hint = builder
				if benchmark != "meta-done" {
					v.Hint += "/" + benchmark
				}
				m := findMetric(uiCom, "failure")
				m.BadChanges = append(m.BadChanges, v)
			}
		}
		res0 := baseRes
		if res0 == nil {
			var err error
			res0, err = rc.NextForComparison(com.Num, builder)
			if err != nil {
				return nil, err
			}
			if res0 == nil {
				continue
			}
		}
		changes := significantPerfChanges(pc, builder, res0, res1)
		changes = dedupPerfChanges(changes)
		for _, ch := range changes {
			v := new(perfChangesChange)
			v.Builder = builder
			v.Benchmark, v.Procs = splitBench(ch.Bench)
			v.diff = ch.Diff
			v.Val = fmt.Sprintf("%+.2f%%", ch.Diff)
			v.Hint = fmt.Sprintf("%v/%v", builder, ch.Bench)
			v.Link = fmt.Sprintf("perfdetail?commit=%v&commit0=%v&builder=%v&benchmark=%v", com.Hash, res0.CommitHash, builder, v.Benchmark)
			m := findMetric(uiCom, ch.Metric)
			if v.diff > 0 {
				v.Style = "bad"
				m.BadChanges = append(m.BadChanges, v)
			} else {
				v.Style = "good"
				m.GoodChanges = append(m.GoodChanges, v)
			}
		}
	}

	// Sort metrics and changes.
	for _, m := range uiCom.Metrics {
		sort.Sort(m.GoodChanges)
		sort.Sort(m.BadChanges)
	}
	sort.Sort(uiCom.Metrics)
	// Need at least one metric for UI.
	if len(uiCom.Metrics) == 0 {
		uiCom.Metrics = append(uiCom.Metrics, &perfChangesMetric{})
	}
	uiCom.Metrics[0].First = true
	return uiCom, nil
}

// Find builder-procs with the maximum absolute diff for every benchmark-metric, drop the rest.
func dedupPerfChanges(changes []*PerfChange) (deduped []*PerfChange) {
	maxDiff := make(map[string]float64)
	maxBench := make(map[string]string)
	// First, find the maximum.
	for _, ch := range changes {
		bench, _ := splitBench(ch.Bench)
		k := bench + "|" + ch.Metric
		v := ch.Diff
		if v < 0 {
			v = -v
		}
		if maxDiff[k] < v {
			maxDiff[k] = v
			maxBench[k] = ch.Builder + "|" + ch.Bench
		}
	}
	// Then, remove the rest.
	for _, ch := range changes {
		bench, _ := splitBench(ch.Bench)
		k := bench + "|" + ch.Metric
		if maxBench[k] == ch.Builder+"|"+ch.Bench {
			deduped = append(deduped, ch)
		}
	}
	return
}

func findMetric(c *perfChangesCommit, metric string) *perfChangesMetric {
	for _, m := range c.Metrics {
		if m.Name == metric {
			return m
		}
	}
	m := new(perfChangesMetric)
	m.Name = metric
	c.Metrics = append(c.Metrics, m)
	return m
}

type uiPerfConfig struct {
	Builders    []uiPerfConfigElem
	Benchmarks  []uiPerfConfigElem
	Metrics     []uiPerfConfigElem
	Procs       []uiPerfConfigElem
	CommitsFrom []uiPerfConfigElem
	CommitsTo   []uiPerfConfigElem
}

type uiPerfConfigElem struct {
	Name     string
	Selected bool
}

var perfChangesTemplate = template.Must(
	template.New("perf_changes.html").Funcs(tmplFuncs).ParseFiles("build/perf_changes.html"),
)

type perfChangesData struct {
	Dashboard  *Dashboard
	Pagination *Pagination
	Commits    []*perfChangesCommit
}

type perfChangesCommit struct {
	*Commit
	IsSummary  bool
	NumResults int
	Metrics    perfChangesMetricSlice
}

type perfChangesMetric struct {
	Name        string
	First       bool
	BadChanges  perfChangesChangeSlice
	GoodChanges perfChangesChangeSlice
}

type perfChangesChange struct {
	Builder   string
	Benchmark string
	Link      string
	Hint      string
	Style     string
	Val       string
	Procs     int
	diff      float64
}

type perfChangesMetricSlice []*perfChangesMetric

func (l perfChangesMetricSlice) Len() int      { return len(l) }
func (l perfChangesMetricSlice) Swap(i, j int) { l[i], l[j] = l[j], l[i] }
func (l perfChangesMetricSlice) Less(i, j int) bool {
	if l[i].Name == "failure" || l[j].Name == "failure" {
		return l[i].Name == "failure"
	}
	return l[i].Name < l[j].Name
}

type perfChangesChangeSlice []*perfChangesChange

func (l perfChangesChangeSlice) Len() int      { return len(l) }
func (l perfChangesChangeSlice) Swap(i, j int) { l[i], l[j] = l[j], l[i] }
func (l perfChangesChangeSlice) Less(i, j int) bool {
	vi, vj := l[i].diff, l[j].diff
	if vi > 0 && vj > 0 {
		return vi > vj
	} else if vi < 0 && vj < 0 {
		return vi < vj
	} else {
		panic("comparing positive and negative diff")
	}
}
