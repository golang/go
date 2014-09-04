// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"code.google.com/p/goauth2/oauth"
	compute "code.google.com/p/google-api-go-client/compute/v1"
	storage "code.google.com/p/google-api-go-client/storage/v1"
)

var (
	proj     = flag.String("project", "symbolic-datum-552", "name of Project")
	zone     = flag.String("zone", "us-central1-a", "GCE zone")
	mach     = flag.String("machinetype", "n1-standard-16", "Machine type")
	instName = flag.String("instance_name", "go-builder-1", "Name of VM instance.")
	sshPub   = flag.String("ssh_public_key", "", "ssh public key file to authorize. Can modify later in Google's web UI anyway.")
	staticIP = flag.String("static_ip", "", "Static IP to use. If empty, automatic.")
)

func readFile(v string) string {
	slurp, err := ioutil.ReadFile(v)
	if err != nil {
		log.Fatalf("Error reading %s: %v", v, err)
	}
	return strings.TrimSpace(string(slurp))
}

var config = &oauth.Config{
	// The client-id and secret should be for an "Installed Application" when using
	// the CLI. Later we'll use a web application with a callback.
	ClientId:     readFile("client-id.dat"),
	ClientSecret: readFile("client-secret.dat"),
	Scope: strings.Join([]string{
		compute.DevstorageFull_controlScope,
		compute.ComputeScope,
		"https://www.googleapis.com/auth/sqlservice",
		"https://www.googleapis.com/auth/sqlservice.admin",
	}, " "),
	AuthURL:     "https://accounts.google.com/o/oauth2/auth",
	TokenURL:    "https://accounts.google.com/o/oauth2/token",
	RedirectURL: "urn:ietf:wg:oauth:2.0:oob",
}

const baseConfig = `#cloud-config
coreos:
  units:
    - name: gobuild.service
      command: start
      content: |
        [Unit]
        Description=Go Builders
        After=docker.service
        Requires=docker.service
        
        [Service]
        ExecStartPre=/bin/bash -c 'mkdir -p /opt/bin && curl -s -o /opt/bin/coordinator http://storage.googleapis.com/go-builder-data/coordinator && chmod +x /opt/bin/coordinator'
        ExecStart=/opt/bin/coordinator
        RestartSec=10s
        Restart=always
        Type=simple
        
        [Install]
        WantedBy=multi-user.target
`

func main() {
	flag.Parse()
	if *proj == "" {
		log.Fatalf("Missing --project flag")
	}
	prefix := "https://www.googleapis.com/compute/v1/projects/" + *proj
	imageURL := "https://www.googleapis.com/compute/v1/projects/coreos-cloud/global/images/coreos-alpha-402-2-0-v20140807"
	machType := prefix + "/zones/" + *zone + "/machineTypes/" + *mach

	tr := &oauth.Transport{
		Config: config,
	}

	tokenCache := oauth.CacheFile("token.dat")
	token, err := tokenCache.Token()
	if err != nil {
		log.Printf("Error getting token from %s: %v", string(tokenCache), err)
		log.Printf("Get auth code from %v", config.AuthCodeURL("my-state"))
		fmt.Print("\nEnter auth code: ")
		sc := bufio.NewScanner(os.Stdin)
		sc.Scan()
		authCode := strings.TrimSpace(sc.Text())
		token, err = tr.Exchange(authCode)
		if err != nil {
			log.Fatalf("Error exchanging auth code for a token: %v", err)
		}
		tokenCache.PutToken(token)
	}

	tr.Token = token
	oauthClient := &http.Client{Transport: tr}
	computeService, _ := compute.New(oauthClient)
	storageService, _ := storage.New(oauthClient)
	_ = storageService // TODO?

	natIP := *staticIP
	if natIP == "" {
		// Try to find it by name.
		aggAddrList, err := computeService.Addresses.AggregatedList(*proj).Do()
		if err != nil {
			log.Fatal(err)
		}
		// http://godoc.org/code.google.com/p/google-api-go-client/compute/v1#AddressAggregatedList
	IPLoop:
		for _, asl := range aggAddrList.Items {
			for _, addr := range asl.Addresses {
				if addr.Name == *instName+"-ip" && addr.Status == "RESERVED" {
					natIP = addr.Address
					break IPLoop
				}
			}
		}
	}

	cloudConfig := baseConfig
	if *sshPub != "" {
		key := strings.TrimSpace(readFile(*sshPub))
		cloudConfig += fmt.Sprintf("\nssh_authorized_keys:\n    - %s\n", key)
	}
	if os.Getenv("USER") == "bradfitz" {
		cloudConfig += fmt.Sprintf("\nssh_authorized_keys:\n    - %s\n", "ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAIEAwks9dwWKlRC+73gRbvYtVg0vdCwDSuIlyt4z6xa/YU/jTDynM4R4W10hm2tPjy8iR1k8XhDv4/qdxe6m07NjG/By1tkmGpm1mGwho4Pr5kbAAy/Qg+NLCSdAYnnE00FQEcFOC15GFVMOW2AzDGKisReohwH9eIzHPzdYQNPRWXE= bradfitz@papag.bradfitz.com")
	}
	const maxCloudConfig = 32 << 10 // per compute API docs
	if len(cloudConfig) > maxCloudConfig {
		log.Fatalf("cloud config length of %d bytes is over %d byte limit", len(cloudConfig), maxCloudConfig)
	}

	instance := &compute.Instance{
		Name:        *instName,
		Description: "Go Builder",
		MachineType: machType,
		Disks: []*compute.AttachedDisk{
			{
				AutoDelete: true,
				Boot:       true,
				Type:       "PERSISTENT",
				InitializeParams: &compute.AttachedDiskInitializeParams{
					DiskName:    *instName + "-coreos-stateless-pd",
					SourceImage: imageURL,
					DiskSizeGb:  50,
				},
			},
		},
		Tags: &compute.Tags{
			Items: []string{"http-server", "https-server"},
		},
		Metadata: &compute.Metadata{
			Items: []*compute.MetadataItems{
				{
					Key:   "user-data",
					Value: cloudConfig,
				},
			},
		},
		NetworkInterfaces: []*compute.NetworkInterface{
			&compute.NetworkInterface{
				AccessConfigs: []*compute.AccessConfig{
					&compute.AccessConfig{
						Type:  "ONE_TO_ONE_NAT",
						Name:  "External NAT",
						NatIP: natIP,
					},
				},
				Network: prefix + "/global/networks/default",
			},
		},
		ServiceAccounts: []*compute.ServiceAccount{
			{
				Email: "default",
				Scopes: []string{
					compute.DevstorageFull_controlScope,
					compute.ComputeScope,
				},
			},
		},
	}

	log.Printf("Creating instance...")
	op, err := computeService.Instances.Insert(*proj, *zone, instance).Do()
	if err != nil {
		log.Fatalf("Failed to create instance: %v", err)
	}
	opName := op.Name
	log.Printf("Created. Waiting on operation %v", opName)
OpLoop:
	for {
		time.Sleep(2 * time.Second)
		op, err := computeService.ZoneOperations.Get(*proj, *zone, opName).Do()
		if err != nil {
			log.Fatalf("Failed to get op %s: %v", opName, err)
		}
		switch op.Status {
		case "PENDING", "RUNNING":
			log.Printf("Waiting on operation %v", opName)
			continue
		case "DONE":
			if op.Error != nil {
				for _, operr := range op.Error.Errors {
					log.Printf("Error: %+v", operr)
				}
				log.Fatalf("Failed to start.")
			}
			log.Printf("Success. %+v", op)
			break OpLoop
		default:
			log.Fatalf("Unknown status %q: %+v", op.Status, op)
		}
	}

	inst, err := computeService.Instances.Get(*proj, *zone, *instName).Do()
	if err != nil {
		log.Fatalf("Error getting instance after creation: %v", err)
	}
	ij, _ := json.MarshalIndent(inst, "", "    ")
	log.Printf("Instance: %s", ij)
}
