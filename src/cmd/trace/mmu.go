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
	"strings"
	"sync"
	"time"
)

func init() {
	http.HandleFunc("/mmu", httpMMU)
	http.HandleFunc("/mmuPlot", httpMMUPlot)
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
			mmuCache.util = tr.MutatorUtilization()
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
        var curve = plotData.curve;
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
        var chart = new google.visualization.LineChart(container[0]);
        chart.draw(data, options);
      }
    </script>
  </head>
  <body>
    <div id="mmu_chart" style="width: 900px; height: 500px">Loading plot...</div>
  </body>
</html>
`
