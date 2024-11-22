// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// package config provides methods for loading and querying a
// telemetry upload config file.
package config

import (
	"encoding/json"
	"os"
	"strings"

	"golang.org/x/telemetry/internal/telemetry"
)

// Config is a wrapper around telemetry.UploadConfig that provides some
// convenience methods for checking the contents of a report.
type Config struct {
	*telemetry.UploadConfig
	program         map[string]bool
	goos            map[string]bool
	goarch          map[string]bool
	goversion       map[string]bool
	pgversion       map[pgkey]bool
	pgcounter       map[pgkey]bool
	pgcounterprefix map[pgkey]bool
	pgstack         map[pgkey]bool
	rate            map[pgkey]float64
}

type pgkey struct {
	program, key string
}

func ReadConfig(file string) (*Config, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	var cfg telemetry.UploadConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return NewConfig(&cfg), nil
}

func NewConfig(cfg *telemetry.UploadConfig) *Config {
	ucfg := Config{UploadConfig: cfg}
	ucfg.goos = set(ucfg.GOOS)
	ucfg.goarch = set(ucfg.GOARCH)
	ucfg.goversion = set(ucfg.GoVersion)
	ucfg.program = make(map[string]bool, len(ucfg.Programs))
	ucfg.pgversion = make(map[pgkey]bool, len(ucfg.Programs))
	ucfg.pgcounter = make(map[pgkey]bool, len(ucfg.Programs))
	ucfg.pgcounterprefix = make(map[pgkey]bool, len(ucfg.Programs))
	ucfg.pgstack = make(map[pgkey]bool, len(ucfg.Programs))
	ucfg.rate = make(map[pgkey]float64)
	for _, p := range ucfg.Programs {
		ucfg.program[p.Name] = true
		for _, v := range p.Versions {
			ucfg.pgversion[pgkey{p.Name, v}] = true
		}
		for _, c := range p.Counters {
			for _, e := range Expand(c.Name) {
				ucfg.pgcounter[pgkey{p.Name, e}] = true
				ucfg.rate[pgkey{p.Name, e}] = c.Rate
			}
			prefix, _, found := strings.Cut(c.Name, ":")
			if found {
				ucfg.pgcounterprefix[pgkey{p.Name, prefix}] = true
			}
		}
		for _, s := range p.Stacks {
			ucfg.pgstack[pgkey{p.Name, s.Name}] = true
			ucfg.rate[pgkey{p.Name, s.Name}] = s.Rate
		}
	}
	return &ucfg
}

func (r *Config) HasProgram(s string) bool {
	return r.program[s]
}

func (r *Config) HasGOOS(s string) bool {
	return r.goos[s]
}

func (r *Config) HasGOARCH(s string) bool {
	return r.goarch[s]
}

func (r *Config) HasGoVersion(s string) bool {
	return r.goversion[s]
}

func (r *Config) HasVersion(program, version string) bool {
	return r.pgversion[pgkey{program, version}]
}

func (r *Config) HasCounter(program, counter string) bool {
	return r.pgcounter[pgkey{program, counter}]
}

func (r *Config) HasCounterPrefix(program, prefix string) bool {
	return r.pgcounterprefix[pgkey{program, prefix}]
}

func (r *Config) HasStack(program, stack string) bool {
	return r.pgstack[pgkey{program, stack}]
}

func (r *Config) Rate(program, name string) float64 {
	return r.rate[pgkey{program, name}]
}

func set(slice []string) map[string]bool {
	s := make(map[string]bool, len(slice))
	for _, v := range slice {
		s[v] = true
	}
	return s
}

// Expand takes a counter defined with buckets and expands it into distinct
// strings for each bucket.
func Expand(counter string) []string {
	prefix, rest, hasBuckets := strings.Cut(counter, "{")
	var counters []string
	if hasBuckets {
		buckets := strings.Split(strings.TrimSuffix(rest, "}"), ",")
		for _, b := range buckets {
			counters = append(counters, prefix+b)
		}
	} else {
		counters = append(counters, prefix)
	}
	return counters
}
