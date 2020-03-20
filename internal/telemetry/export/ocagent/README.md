# Exporting Metrics with OpenCensus and Prometheus

This tutorial provides a minimum example to verify that metrics can be exported to OpenCensus from Go tools.

## Setting up oragent

1. Ensure you have [docker](https://www.docker.com/get-started) and [docker-compose](https://docs.docker.com/compose/install/).
2. Clone [oragent](https://github.com/orijtech/oragent).
3. In the oragent directory, start the services:
```bash
docker-compose up
```
If everything goes well, you should see output resembling the following:
```
Starting oragent_zipkin_1 ... done
Starting oragent_oragent_1 ... done
Starting oragent_prometheus_1 ... done
...
```
* You can check the status of the OpenCensus agent using zPages at http://localhost:55679/debug/tracez.
* You can now access the Prometheus UI at http://localhost:9445.
* You can now access the Zipkin UI at http://localhost:9444.
4. To shut down oragent, hit Ctrl+C in the terminal.
5. You can also start oragent in detached mode by running `docker-compose up -d`. To stop oragent while detached, run `docker-compose down`.

## Exporting Metrics
1. Clone the [tools](https://golang.org/x/tools) subrepository.
1. Inside `internal`, create a file named `main.go` with the following contents:
```go
package main

import (
	"context"
	"fmt"
	"math/rand"
	"net/http"
	"time"

	"golang.org/x/tools/internal/telemetry/export"
	"golang.org/x/tools/internal/telemetry/export/ocagent"
	"golang.org/x/tools/internal/telemetry/metric"
	"golang.org/x/tools/internal/telemetry/stats"
)

func main() {

	exporter := ocagent.Connect(&ocagent.Config{
		Start:   time.Now(),
		Address: "http://127.0.0.1:55678",
		Service: "go-tools-test",
		Rate:    5 * time.Second,
		Client:  &http.Client{},
	})
	export.SetExporter(exporter)

	ctx := context.TODO()
	mLatency := stats.Float64("latency", "the latency in milliseconds", "ms")
	distribution := metric.HistogramFloat64Data{
		Info: &metric.HistogramFloat64{
			Name:        "latencyDistribution",
			Description: "the various latencies",
			Buckets:     []float64{0, 10, 50, 100, 200, 400, 800, 1000, 1400, 2000, 5000, 10000, 15000},
		},
	}

	distribution.Info.Record(mLatency)

	for {
		sleep := randomSleep()
		time.Sleep(time.Duration(sleep) * time.Millisecond)
		mLatency.Record(ctx, float64(sleep))

		fmt.Println("Latency: ", float64(sleep))
	}
}

func randomSleep() int64 {
	var max int64
	switch modulus := time.Now().Unix() % 5; modulus {
	case 0:
		max = 17001
	case 1:
		max = 8007
	case 2:
		max = 917
	case 3:
		max = 87
	case 4:
		max = 1173
	}
	return rand.Int63n(max)
}

```
3. Run the new file from within the tools repository:
```bash
go run internal/main.go
```
4. After about 5 seconds, OpenCensus should start receiving your new metrics, which you can see at http://localhost:8844/metrics. This page will look similar to the following:
```
# HELP promdemo_latencyDistribution the various latencies
# TYPE promdemo_latencyDistribution histogram
promdemo_latencyDistribution_bucket{vendor="otc",le="0"} 0
promdemo_latencyDistribution_bucket{vendor="otc",le="10"} 2
promdemo_latencyDistribution_bucket{vendor="otc",le="50"} 9
promdemo_latencyDistribution_bucket{vendor="otc",le="100"} 22
promdemo_latencyDistribution_bucket{vendor="otc",le="200"} 35
promdemo_latencyDistribution_bucket{vendor="otc",le="400"} 49
promdemo_latencyDistribution_bucket{vendor="otc",le="800"} 63
promdemo_latencyDistribution_bucket{vendor="otc",le="1000"} 78
promdemo_latencyDistribution_bucket{vendor="otc",le="1400"} 93
promdemo_latencyDistribution_bucket{vendor="otc",le="2000"} 108
promdemo_latencyDistribution_bucket{vendor="otc",le="5000"} 123
promdemo_latencyDistribution_bucket{vendor="otc",le="10000"} 138
promdemo_latencyDistribution_bucket{vendor="otc",le="15000"} 153
promdemo_latencyDistribution_bucket{vendor="otc",le="+Inf"} 15
promdemo_latencyDistribution_sum{vendor="otc"} 1641
promdemo_latencyDistribution_count{vendor="otc"} 15
```
5. After a few more seconds, Prometheus should start displaying your new metrics. You can view the distribution at http://localhost:9445/graph?g0.range_input=5m&g0.stacked=1&g0.expr=rate(oragent_latencyDistribution_bucket%5B5m%5D)&g0.tab=0.
