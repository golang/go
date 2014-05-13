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
	"strconv"

	"appengine"
	"appengine/datastore"
)

func init() {
	for _, d := range dashboards {
		http.HandleFunc(d.RelPath+"perfgraph", perfGraphHandler)
	}
}

func perfGraphHandler(w http.ResponseWriter, r *http.Request) {
	d := dashboardForRequest(r)
	c := d.Context(appengine.NewContext(r))
	pc, err := GetPerfConfig(c, r)
	if err != nil {
		logErr(w, r, err)
		return
	}
	allBuilders := pc.BuildersForBenchmark("")
	allBenchmarks := pc.BenchmarksForBuilder("")
	allMetrics := pc.MetricsForBenchmark("")
	allProcs := pc.ProcList("")
	r.ParseForm()
	absolute := r.FormValue("absolute") != ""
	selBuilders := r.Form["builder"]
	selBenchmarks := r.Form["benchmark"]
	selMetrics := r.Form["metric"]
	selProcs := r.Form["procs"]
	if len(selBuilders) == 0 {
		selBuilders = append(selBuilders, allBuilders[0])
	}
	if len(selBenchmarks) == 0 {
		selBenchmarks = append(selBenchmarks, "json")
	}
	if len(selMetrics) == 0 {
		selMetrics = append(selMetrics, "time")
	}
	if len(selProcs) == 0 {
		selProcs = append(selProcs, "1")
	}
	// TODO(dvyukov): validate input

	present := func(set []string, s string) bool {
		for _, s1 := range set {
			if s1 == s {
				return true
			}
		}
		return false
	}

	cfg := &uiPerfConfig{}
	for _, v := range allBuilders {
		cfg.Builders = append(cfg.Builders, uiPerfConfigElem{v, present(selBuilders, v)})
	}
	for _, v := range allBenchmarks {
		cfg.Benchmarks = append(cfg.Benchmarks, uiPerfConfigElem{v, present(selBenchmarks, v)})
	}
	for _, v := range allMetrics {
		cfg.Metrics = append(cfg.Metrics, uiPerfConfigElem{v, present(selMetrics, v)})
	}
	for _, v := range allProcs {
		cfg.Procs = append(cfg.Procs, uiPerfConfigElem{strconv.Itoa(v), present(selProcs, strconv.Itoa(v))})
	}

	// Select last commit.
	startCommit := 0
	commitsToDisplay := 100
	if r.FormValue("startcommit") != "" {
		startCommit, _ = strconv.Atoi(r.FormValue("startcommit"))
		commitsToDisplay, _ = strconv.Atoi(r.FormValue("commitnum"))
	} else {
		var commits1 []*Commit
		_, err = datastore.NewQuery("Commit").
			Ancestor((&Package{}).Key(c)).
			Order("-Num").
			Filter("NeedsBenchmarking =", true).
			Limit(1).
			GetAll(c, &commits1)
		if err != nil || len(commits1) != 1 {
			logErr(w, r, err)
			return
		}
		startCommit = commits1[0].Num
	}

	if r.FormValue("zoomin") != "" {
		commitsToDisplay /= 2
	} else if r.FormValue("zoomout") != "" {
		commitsToDisplay *= 2
	} else if r.FormValue("older") != "" {
		startCommit -= commitsToDisplay / 2
	} else if r.FormValue("newer") != "" {
		startCommit += commitsToDisplay / 2
	}

	// TODO(dvyukov): limit number of lines on the graph?
	startCommitNum := startCommit - commitsToDisplay + 1
	if startCommitNum < 0 {
		startCommitNum = 0
	}
	var vals [][]float64
	var hints [][]string
	var certainty [][]bool
	var headers []string
	commits2, err := GetCommits(c, startCommitNum, commitsToDisplay)
	if err != nil {
		logErr(w, r, err)
		return
	}
	for _, builder := range selBuilders {
		for _, metric := range selMetrics {
			for _, benchmark := range selBenchmarks {
				for _, procs := range selProcs {
					benchProcs := fmt.Sprintf("%v-%v", benchmark, procs)
					vv, err := GetPerfMetricsForCommits(c, builder, benchProcs, metric, startCommitNum, commitsToDisplay)
					if err != nil {
						logErr(w, r, err)
						return
					}
					nonzero := false
					min := ^uint64(0)
					max := uint64(0)
					for _, v := range vv {
						if v == 0 {
							continue
						}
						if max < v {
							max = v
						}
						if min > v {
							min = v
						}
						nonzero = true
					}
					if nonzero {
						noise := pc.NoiseLevel(builder, benchProcs, metric)
						diff := (float64(max) - float64(min)) / float64(max) * 100
						// Scale graph passes through 2 points: (noise, minScale) and (growthFactor*noise, 100).
						// Plus it's bottom capped at minScale and top capped at 100.
						// Intention:
						// Diffs below noise are scaled to minScale.
						// Diffs above growthFactor*noise are scaled to 100.
						// Between noise and growthFactor*noise scale growths linearly.
						const minScale = 5
						const growthFactor = 4
						scale := diff*(100-minScale)/(noise*(growthFactor-1)) + (minScale*growthFactor-100)/(growthFactor-1)
						if scale < minScale {
							scale = minScale
						}
						if scale > 100 {
							scale = 100
						}
						descBuilder := "/" + builder
						descBenchmark := "/" + benchProcs
						descMetric := "/" + metric
						if len(selBuilders) == 1 {
							descBuilder = ""
						}
						if len(selBenchmarks) == 1 && len(selProcs) == 1 {
							descBenchmark = ""
						}
						if len(selMetrics) == 1 && (len(selBuilders) > 1 || len(selBenchmarks) > 1 || len(selProcs) > 1) {
							descMetric = ""
						}
						desc := fmt.Sprintf("%v%v%v", descBuilder, descBenchmark, descMetric)[1:]
						hh := make([]string, commitsToDisplay)
						valf := make([]float64, commitsToDisplay)
						cert := make([]bool, commitsToDisplay)
						lastval := uint64(0)
						lastval0 := uint64(0)
						for i, v := range vv {
							cert[i] = true
							if v == 0 {
								if lastval == 0 {
									continue
								}
								nextval := uint64(0)
								nextidx := 0
								for i2, v2 := range vv[i+1:] {
									if v2 != 0 {
										nextval = v2
										nextidx = i + i2 + 1
										break
									}
								}
								if nextval == 0 {
									continue
								}
								cert[i] = false
								v = lastval + uint64(int64(nextval-lastval)/int64(nextidx-i+1))
								_, _ = nextval, nextidx
							}
							f := float64(v)
							if !absolute {
								f = (float64(v) - float64(min)) * 100 / (float64(max) - float64(min))
								f = f*scale/100 + (100-scale)/2
								f += 0.000001
							}
							valf[i] = f
							com := commits2[i]
							comLink := "https://code.google.com/p/go/source/detail?r=" + com.Hash
							if cert[i] {
								d := ""
								if lastval0 != 0 {
									d = fmt.Sprintf(" (%.02f%%)", perfDiff(lastval0, v))
								}
								cmpLink := fmt.Sprintf("/perfdetail?commit=%v&builder=%v&benchmark=%v", com.Hash, builder, benchmark)
								hh[i] = fmt.Sprintf("%v: <a href='%v'>%v%v</a><br><a href='%v'>%v</a><br>%v", desc, cmpLink, v, d, comLink, com.Desc, com.Time.Format("Jan 2, 2006 1:04"))
							} else {
								hh[i] = fmt.Sprintf("%v: NO DATA<br><a href='%v'>%v</a><br>%v", desc, comLink, com.Desc, com.Time.Format("Jan 2, 2006 1:04"))
							}
							lastval = v
							if cert[i] {
								lastval0 = v
							}
						}
						vals = append(vals, valf)
						hints = append(hints, hh)
						certainty = append(certainty, cert)
						headers = append(headers, fmt.Sprintf("%s (%.2f%% [%.2f%%])", desc, diff, noise))
					}
				}
			}
		}
	}

	var commits []perfGraphCommit
	if len(vals) != 0 && len(vals[0]) != 0 {
		for i := range vals[0] {
			if !commits2[i].NeedsBenchmarking {
				continue
			}
			var c perfGraphCommit
			for j := range vals {
				c.Vals = append(c.Vals, perfGraphValue{float64(vals[j][i]), certainty[j][i], hints[j][i]})
			}
			commits = append(commits, c)
		}
	}

	data := &perfGraphData{d, cfg, startCommit, commitsToDisplay, absolute, headers, commits}

	var buf bytes.Buffer
	if err := perfGraphTemplate.Execute(&buf, data); err != nil {
		logErr(w, r, err)
		return
	}

	buf.WriteTo(w)
}

var perfGraphTemplate = template.Must(
	template.New("perf_graph.html").ParseFiles("build/perf_graph.html"),
)

type perfGraphData struct {
	Dashboard   *Dashboard
	Config      *uiPerfConfig
	StartCommit int
	CommitNum   int
	Absolute    bool
	Headers     []string
	Commits     []perfGraphCommit
}

type perfGraphCommit struct {
	Name string
	Vals []perfGraphValue
}

type perfGraphValue struct {
	Val       float64
	Certainty bool
	Hint      string
}
