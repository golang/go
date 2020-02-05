# Exporting Metrics with OpenCensus and Prometheus

This tutorial provides a minimum example to verify that metrics can be exported to OpenCensus from Go tools.

## Setting up the OpenCensus Agent
1. Follow the instructions for setting up the [OpenCensus Service](https://opencensus.io/agent). You will need to be able to run the agent.
2. Create a file named `config.yaml` with the following contents:
```yaml
exporters:
  prometheus:
    namespace: "promdemo"
    address: "localhost:8888"
    const_labels: {
        "vendor": "gotools"
    }
receivers:
  opencensus:
    address: "localhost:55678"

```
3. Run the OpenCensus Agent with the configuration file. The following command assumes that you are running from binary:
```bash
bin/ocagent_linux --config config.yaml
```
4. If you see output similar to the following, the OpenCensus Agent is now running:
```bash
{"level":"info","ts":1574381470.1922305,"caller":"config/config.go:497","msg":"Metrics Exporter enabled","exporter":"prometheus"}
2019/11/21 18:11:11 Running OpenCensus Trace and Metrics receivers as a gRPC service at "localhost:55678"
2019/11/21 18:11:11 Running zPages on port 55679
```
5. You can check the status of the agent using zPages at http://localhost:55679/debug/tracez.

## Setting up Prometheus
1. Follow the instructions for setting up [Prometheus](https://prometheus.io/docs/prometheus/latest/installation/).
2. Create a file named `prom.yaml` with the following contents:
```yaml
scrape_configs:
  - job_name: 'agent1'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:8888']
```
3. Run Prometheus with the new configuration file. The following command assumes that you are running from pre-compiled binaries:
```bash
./prometheus --config.file=prom.yaml
```
4. If you see output similar to the following, Prometheus is now running:
```bash
level=info ts=2019-11-22T00:27:13.772Z caller=main.go:626 msg="Server is ready to receive web requests."
```
5. You can now access the Prometheus UI at http://localhost:9090.
6. Check to make sure Prometheus is able to scrape metrics from OpenCensus at http://localhost:9090/targets. If the state for the endpoint `http://localhost:8888/metrics` is not `UP`, make sure the OpenCensus agent is running. If you are running Prometheus using Docker, you may have to add `--net="host"` to your run command so that `localhost` resolves correctly.

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
4. After about 5 seconds, OpenCensus should start receiving your new metrics, which you can see at http://localhost:8888/metrics. This page will look similar to the following:
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
5. After a few more seconds, Prometheus should start displaying your new metrics. You can view the distribution at http://localhost:9090/graph?g0.range_input=5m&g0.stacked=1&g0.expr=rate(promdemo_latencyDistribution_bucket%5B5m%5D)&g0.tab=0.
