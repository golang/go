// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"internal/testenv"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

var modulesForTest = []struct {
	name                string
	testDataFolder      string
	url                 string
	dependenciesToStrip []string
}{
	{
		name:           "Empty",
		testDataFolder: "empty",
	},
	{
		name:           "Cmd",
		testDataFolder: "cmd",
	},
	{
		name:           "K8S",
		testDataFolder: "k8s",
		url:            "https://raw.githubusercontent.com/kubernetes/kubernetes/b33ef18bdfa62d3975f2c72743373f1d69740b99",
		dependenciesToStrip: []string{
			"k8s.io/api",
			"k8s.io/apiextensions-apiserver",
			"k8s.io/apimachinery",
			"k8s.io/apiserver",
			"k8s.io/cli-runtime",
			"k8s.io/client-go",
			"k8s.io/cloud-provider",
			"k8s.io/cluster-bootstrap",
			"k8s.io/code-generator",
			"k8s.io/component-base",
			"k8s.io/component-helpers",
			"k8s.io/controller-manager",
			"k8s.io/cri-api",
			"k8s.io/cri-client",
			"k8s.io/csi-translation-lib",
			"k8s.io/dynamic-resource-allocation",
			"k8s.io/endpointslice",
			"k8s.io/kms",
			"k8s.io/kube-aggregator",
			"k8s.io/kube-controller-manager",
			"k8s.io/kube-proxy",
			"k8s.io/kube-scheduler",
			"k8s.io/kubectl",
			"k8s.io/kubelet",
			"k8s.io/metrics",
			"k8s.io/mount-utils",
			"k8s.io/pod-security-admission",
			"k8s.io/sample-apiserver",
			"k8s.io/sample-cli-plugin",
			"k8s.io/sample-controller",
		},
	},
}

func BenchmarkListModules(b *testing.B) {
	testenv.MustHaveExec(b)
	gotool, err := testenv.GoTool()
	if err != nil {
		b.Fatal(err)
	}

	tempDir := b.TempDir()
	if err := prepareTestModules(tempDir); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for _, m := range modulesForTest {
		// Do not run in parallel. GOPROXY rate limits may affect parallel executions.
		b.Run(m.name, func(b *testing.B) {
			// We collect extra metrics.
			var n, userTime, systemTime int64
			modDir := filepath.Join(tempDir, m.testDataFolder, "go.mod")
			for i := 0; i < b.N; i++ {
				cmd := testenv.Command(b, gotool, "list", "-m", "-modfile="+modDir, "-mod=readonly", "all")

				// Guarantees clean module cache for every execution.
				gopath := b.TempDir()
				cmd.Env = append(cmd.Env, "GOPATH="+gopath)

				if err := cmd.Run(); err != nil {
					b.Fatal(err)
				}
				atomic.AddInt64(&n, 1)
				atomic.AddInt64(&userTime, int64(cmd.ProcessState.UserTime()))
				atomic.AddInt64(&systemTime, int64(cmd.ProcessState.SystemTime()))

			}
			b.ReportMetric(float64(userTime)/float64(n), "user-ns/op")
			b.ReportMetric(float64(systemTime)/float64(n), "sys-ns/op")
		})
	}
}

func prepareTestModules(tempDir string) error {
	err := os.CopyFS(tempDir, os.DirFS(filepath.Join("testdata", "list")))
	if err != nil {
		return err
	}
	for _, m := range modulesForTest {
		if m.url != "" {
			modulePath := filepath.Join(tempDir, m.testDataFolder)
			if err := downloadModule(modulePath, m.url, m.dependenciesToStrip); err != nil {
				return err
			}
		}
	}
	return nil
}

func downloadModule(modulePath string, url string, dependenciesToStrip []string) error {

	if err := os.Mkdir(modulePath, 0755); err != nil {
		return err
	}
	sumText, err := fetchText(url + "/go.sum")
	if err != nil {
		return err
	}

	if err := writeFile(filepath.Join(modulePath, "go.sum"), sumText); err != nil {
		return err
	}

	modText, err := fetchText(url + "/go.mod")
	if err != nil {
		return err
	}

	strippedModText := stripDependencies(modText, dependenciesToStrip)

	return writeFile(filepath.Join(modulePath, "go.mod"), strippedModText)
}

func stripDependencies(goModText string, dependenciesToStrip []string) string {
	var filteredLines []string
	lines := strings.Split(goModText, "\n")
	for _, line := range lines {
		if !containsAny(line, dependenciesToStrip) {
			filteredLines = append(filteredLines, line)
		}
	}
	filteredText := strings.Join(filteredLines, "\n")
	return filteredText
}

func containsAny(text string, options []string) bool {
	for _, option := range options {
		if strings.Contains(text, option) {
			return true
		}
	}
	return false
}

func writeFile(filePath, content string) error {
	out, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.WriteString(out, content)
	return err
}

func fetchText(url string) (string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	return string(body), err
}
