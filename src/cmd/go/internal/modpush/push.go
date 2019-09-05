// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package modpush

import (
	"fmt"
	"net/http"
	"os"
)

func Push(endpoint string, project string, file string, username string, password string) {
	content, err := os.Open(file)
	if err != nil {
		fmt.Printf("Unable to find %s to push to endpoint %s\n", file, endpoint)
	}

	defer content.Close()

	client := http.Client{}
	request, _ := http.NewRequest("PUT", endpoint+project+"/@v/"+file, content)
	maybeAddCredentials(username, password, request, file, project)

	response, err := client.Do(request)
	if err != nil || response.StatusCode != http.StatusCreated {
		fmt.Printf("Unable to post %s to %s\n", file, endpoint)
		return
	}

	fmt.Println("Successfully posted " + file + " to " + endpoint)
}

func maybeAddCredentials(username string, password string, request *http.Request, file string, project string) {
	if username != "" && password != "" {
		fmt.Println("Using basic authentication arguments provided")
		request.SetBasicAuth(username, password)
	} else {
		fmt.Printf("Using no credentials to push package %s to %s\n", file, project)
	}
}
