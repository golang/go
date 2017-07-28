// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Minimum mutator utilization (MMU) graphing.

package main

import (
	"encoding/json"
	"fmt"
	trace "internal/traceparser"
	"log"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

func init() {
	http.HandleFunc("/mmu", httpMMU)
	http.HandleFunc("/mmuPlot", httpMMUPlot)
	http.HandleFunc("/mmuDetails", httpMMUDetails)
}

var mmuCache struct {
	init     sync.Once
	util     []trace.MutatorUtil
	mmuCurve *trace.MMUCurve
	err      error
}

func getMMUCurve() ([]trace.MutatorUtil, *trace.MMUCurve, error) {
	mmuCache.init.Do(func() {
		tr, err := parseTrace()
		if err != nil {
			mmuCache.err = err
		} else {
			mmuCache.util = tr.MutatorUtilization(trace.UtilSTW | trace.UtilBackground | trace.UtilAssist)
			mmuCache.mmuCurve = trace.NewMMUCurve(mmuCache.util)
		}
	})
	return mmuCache.util, mmuCache.mmuCurve, mmuCache.err
}

// httpMMU serves the MMU plot page.
func httpMMU(w http.ResponseWriter, r *http.Request) {
	http.ServeContent(w, r, "", time.Time{}, strings.NewReader(templMMU))
}

// httpMMUPlot serves the JSON data for the MMU plot.
func httpMMUPlot(w http.ResponseWriter, r *http.Request) {
	mu, mmuCurve, err := getMMUCurve()
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to parse events: %v", err), http.StatusInternalServerError)
		return
	}

	// Find a nice starting point for the plot.
	xMin := time.Second
	for xMin > 1 {
		if mmu := mmuCurve.MMU(xMin); mmu < 0.0001 {
			break
		}
		xMin /= 1000
	}
	// Cover six orders of magnitude.
	xMax := xMin * 1e6
	// But no more than the length of the trace.
	if maxMax := time.Duration(mu[len(mu)-1].Time - mu[0].Time); xMax > maxMax {
		xMax = maxMax
	}
	// Compute MMU curve.
	logMin, logMax := math.Log(float64(xMin)), math.Log(float64(xMax))
	const samples = 100
	plot := make([][2]float64, samples)
	for i := 0; i < samples; i++ {
		window := time.Duration(math.Exp(float64(i)/(samples-1)*(logMax-logMin) + logMin))
		y := mmuCurve.MMU(window)
		plot[i] = [2]float64{float64(window), y}
	}

	// Create JSON response.
	err = json.NewEncoder(w).Encode(map[string]interface{}{"xMin": int64(xMin), "xMax": int64(xMax), "curve": plot})
	if err != nil {
		log.Printf("failed to serialize response: %v", err)
		return
	}
}

var templMMU = `<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {'packages':['corechart']});
      google.charts.setOnLoadCallback(refreshChart);

      var chart;
      var curve;

      function niceDuration(ns) {
          if (ns < 1e3) { return ns + 'ns'; }
          else if (ns < 1e6) { return ns / 1e3 + 'µs'; }
          else if (ns < 1e9) { return ns / 1e6 + 'ms'; }
          else { return ns / 1e9 + 's'; }
      }

      function refreshChart() {
        $.getJSON('/mmuPlot')
         .fail(function(xhr, status, error) {
           alert('failed to load plot: ' + status);
         })
         .done(drawChart);
      }

      function drawChart(plotData) {
        curve = plotData.curve;
        var data = new google.visualization.DataTable();
        data.addColumn('number', 'Window duration');
        data.addColumn('number', 'Minimum mutator utilization');
        data.addRows(curve);
        for (var i = 0; i < curve.length; i++) {
          data.setFormattedValue(i, 0, niceDuration(curve[i][0]));
        }

        var options = {
          chart: {
            title: 'Minimum mutator utilization',
          },
          hAxis: {
            title: 'Window duration',
            scaleType: 'log',
            ticks: [],
          },
          vAxis: {
            title: 'Minimum mutator utilization',
            minValue: 0.0,
            maxValue: 1.0,
          },
          legend: { position: 'none' },
          width: 900,
          height: 500,
          chartArea: { width: '80%', height: '80%' },
        };
        for (var v = plotData.xMin; v <= plotData.xMax; v *= 10) {
          options.hAxis.ticks.push({v:v, f:niceDuration(v)});
        }

        var container = $('#mmu_chart');
        container.empty();
        chart = new google.visualization.LineChart(container[0]);
        chart = new google.visualization.LineChart(document.getElementById('mmu_chart'));
        chart.draw(data, options);

        google.visualization.events.addListener(chart, 'select', selectHandler);
      }

      function selectHandler() {
        var items = chart.getSelection();
        if (items.length === 0) {
          return;
        }
        var details = $('#details');
        details.empty();
        var windowNS = curve[items[0].row][0];
        var url = '/mmuDetails?window=' + windowNS;
        $.getJSON(url)
         .fail(function(xhr, status, error) {
            details.text(status + ': ' + url + ' could not be loaded');
         })
         .done(function(worst) {
            details.text('Lowest mutator utilization in ' + niceDuration(windowNS) + ' windows:');
            for (var i = 0; i < worst.length; i++) {
              details.append($('<br/>'));
              var text = worst[i].MutatorUtil.toFixed(3) + ' at time ' + niceDuration(worst[i].Time);
              details.append($('<a/>').text(text).attr('href', worst[i].URL));
            }
         });
      }
    </script>
  </head>
  <body>
    <div id="mmu_chart" style="width: 900px; height: 500px">Loading plot...</div>
    <div id="details">Select a point for details.</div>
  </body>
</html>
`

// httpMMUDetails serves details of an MMU graph at a particular window.
func httpMMUDetails(w http.ResponseWriter, r *http.Request) {
	_, mmuCurve, err := getMMUCurve()
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to parse events: %v", err), http.StatusInternalServerError)
		return
	}

	windowStr := r.FormValue("window")
	window, err := strconv.ParseUint(windowStr, 10, 64)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to parse window parameter %q: %v", windowStr, err), http.StatusBadRequest)
		return
	}
	worst := mmuCurve.Examples(time.Duration(window), 10)

	// Construct a link for each window.
	var links []linkedUtilWindow
	for _, ui := range worst {
		links = append(links, newLinkedUtilWindow(ui, time.Duration(window)))
	}

	err = json.NewEncoder(w).Encode(links)
	if err != nil {
		log.Printf("failed to serialize trace: %v", err)
		return
	}
}

type linkedUtilWindow struct {
	trace.UtilWindow
	URL string
}

func newLinkedUtilWindow(ui trace.UtilWindow, window time.Duration) linkedUtilWindow {
	// Find the range containing this window.
	var r Range
	for _, r = range ranges {
		if r.EndTime > ui.Time {
			break
		}
	}
	return linkedUtilWindow{ui, fmt.Sprintf("%s#%v:%v", r.URL(), float64(ui.Time)/1e6, float64(ui.Time+int64(window))/1e6)}
}
