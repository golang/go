// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"path/filepath"
	"testing"
)

var t *testing.T

type TypeMeta struct {
	Kind       string
	APIVersion string
}

type ObjectMeta struct {
	Name         string `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`
	GenerateName string `json:"generateName,omitempty" protobuf:"bytes,2,opt,name=generateName"`
	Namespace    string `json:"namespace,omitempty" protobuf:"bytes,3,opt,name=namespace"`
	SelfLink     string `json:"selfLink,omitempty" protobuf:"bytes,4,opt,name=selfLink"`
}

type ConfigSpec struct {
	Disks        []DiskSpec
	StorageClass string
}

type DiskSpec struct {
	Name         string
	Size         string
	StorageClass string
	Annotations  map[string]string
	VolumeName   string
}

// Config is the Schema for the configs API.
type Config struct {
	TypeMeta
	ObjectMeta

	Spec ConfigSpec
}

func findDiskSize(diskSpec *DiskSpec, configSpec *ConfigSpec) string {
	t.Log(fmt.Sprintf("Hello World"))
	return diskSpec.Size
}

func findStorageClassName(diskSpec *DiskSpec, configSpec *ConfigSpec) *string {
	if diskSpec.StorageClass != "" {
		return &diskSpec.StorageClass
	}

	if configSpec != nil {
		for _, d := range configSpec.Disks {
			if d.Name == diskSpec.Name {
				if d.StorageClass != "" {
					return &d.StorageClass
				}
				break
			}
		}

		if configSpec.StorageClass != "" {
			return &configSpec.StorageClass
		}
	}
	return nil
}

func Bar(config *Config) *ConfigSpec {
	var configSpec *ConfigSpec
	if config != nil {
		configSpec = &config.Spec
	}
	return configSpec
}

func Foo(diskSpec DiskSpec, config *Config) {
	cs := Bar(config)
	_ = findDiskSize(&diskSpec, cs)
	cs = Bar(config)
	_ = findStorageClassName(&diskSpec, cs)

}

func TestPanic(tt *testing.T) {
	t = tt
	myarray := []string{filepath.Join("..", "config", "crd", "bases")}

	for i := 0; i < 1000; i++ {
		Foo(DiskSpec{
			Name: "DataDisk",
			Size: "1Gi",
		}, nil)
	}

	t.Log(myarray)
}

// Hack to run tests in a playground
func matchString(a, b string) (bool, error) {
	return a == b, nil
}
func main() {
	testSuite := []testing.InternalTest{
		{
			Name: "TestPanic",
			F:    TestPanic,
		},
	}
	testing.Main(matchString, testSuite, nil, nil)
}
