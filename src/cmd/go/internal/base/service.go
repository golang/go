// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"os"
	"os/exec"
	"path/filepath"
)

// CreateServiceScaffold creates the scaffolding of a minimal backend service.
// creates api/, cmd/, internal/, pkg/ directories and a main.go file in cmd
// This will allow deveopers to have a common structure for backend services.
func CreateServiceScaffold(serviceName string) {
	//get current directory
	currentDir, err := os.Getwd()
	if err != nil {
		Fatalf("could not get current directory: %v", err)
	}

	//create service directory
	serviceDir := filepath.Join(currentDir, serviceName)
	err = os.Mkdir(serviceDir, 0755)
	if err != nil {
		Fatalf("could not create service directory: %v", err)
	}

	//create a directory cmd and a main.go file inside it
	cmdDir := filepath.Join(serviceDir, "cmd")
	err = os.Mkdir(cmdDir, 0755)
	if err != nil {
		Fatalf("could not create cmd directory: %v", err)
	}
	//create main.go file
	mainFilePath := filepath.Join(cmdDir, "main.go")
	mainFileContent := `package main
import "net/http"
func main() {
	http.ListenAndServe(":8080", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("Go is great!"))
	}))
}`
	err = os.WriteFile(mainFilePath, []byte(mainFileContent), 0644)
	if err != nil {
		Fatalf("could not create main.go file: %v", err)
	}

	// run go mod init
	cmd := exec.Command("go", "mod", "init", serviceName)
	cmd.Dir = serviceDir
	output, err := cmd.CombinedOutput()
	if err != nil {
		Fatalf("could not initialize go module: %v\n%s", err, output)
	}
	// create a pkg directory and create a sample package file
	pkgDir := filepath.Join(serviceDir, "pkg")
	err = os.Mkdir(pkgDir, 0755)
	if err != nil {
		Fatalf("could not create pkg directory: %v", err)
	}
	samplePkgFilePath := filepath.Join(pkgDir, "sample.go")
	samplePkgContent := `//package pkg is for all shared packages
package pkg 
	`
	err = os.WriteFile(samplePkgFilePath, []byte(samplePkgContent), 0644)
	if err != nil {
		Fatalf("could not create sample package file: %v", err)
	}
	//create api directory and create api.go file
	apiDir := filepath.Join(serviceDir, "api")
	err = os.Mkdir(apiDir, 0755)
	if err != nil {
		Fatalf("could not create api directory: %v", err)
	}
	apiFilePath := filepath.Join(apiDir, "api.go")
	apiFileContent := `// package api contains API definitions
package api 
	`
	err = os.WriteFile(apiFilePath, []byte(apiFileContent), 0644)
	if err != nil {
		Fatalf("could not create api.go file: %v", err)
	}
	//create an internal directory and create a service.go file
	internalDir := filepath.Join(serviceDir, "internal")
	err = os.Mkdir(internalDir, 0755)
	if err != nil {
		Fatalf("could not create internal directory: %v", err)
	}
	serviceFilePath := filepath.Join(internalDir, "service.go")
	serviceFileContent := `// package internal contains internal service logic
package internal 
	`
	err = os.WriteFile(serviceFilePath, []byte(serviceFileContent), 0644)
	if err != nil {
		Fatalf("could not create service.go file: %v", err)
	}
	//create model directory and create a model.go file inside inyternal
	modelDir := filepath.Join(internalDir, "model")
	err = os.Mkdir(modelDir, 0755)
	if err != nil {
		Fatalf("could not create model directory: %v", err)
	}
	modelFilePath := filepath.Join(modelDir, "model.go")
	modelFileContent := `// package model contains data models
package model 
	`
	err = os.WriteFile(modelFilePath, []byte(modelFileContent), 0644)
	if err != nil {
		Fatalf("could not create model.go file: %v", err)
	}

	println("Service scaffolding created successfully at", serviceDir)
}
