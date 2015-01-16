// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

package buildlet

import (
	"crypto/tls"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/tools/dashboard"
	"google.golang.org/api/compute/v1"
)

// VMOpts control how new VMs are started.
type VMOpts struct {
	// Zone is the GCE zone to create the VM in. Required.
	Zone string

	// ProjectID is the GCE project ID. Required.
	ProjectID string

	// TLS optionally specifies the TLS keypair to use.
	// If zero, http without auth is used.
	TLS KeyPair

	// Optional description of the VM.
	Description string

	// Optional metadata to put on the instance.
	Meta map[string]string

	// DeleteIn optionally specifies a duration at which
	// to delete the VM.
	DeleteIn time.Duration

	// OnInstanceRequested optionally specifies a hook to run synchronously
	// after the computeService.Instances.Insert call, but before
	// waiting for its operation to proceed.
	OnInstanceRequested func()

	// OnInstanceCreated optionally specifies a hook to run synchronously
	// after the instance operation succeeds.
	OnInstanceCreated func()

	// OnInstanceCreated optionally specifies a hook to run synchronously
	// after the computeService.Instances.Get call.
	OnGotInstanceInfo func()
}

// StartNewVM boots a new VM on GCE and returns a buildlet client
// configured to speak to it.
func StartNewVM(ts oauth2.TokenSource, instName, builderType string, opts VMOpts) (*Client, error) {
	computeService, _ := compute.New(oauth2.NewClient(oauth2.NoContext, ts))

	conf, ok := dashboard.Builders[builderType]
	if !ok {
		return nil, fmt.Errorf("invalid builder type %q", builderType)
	}

	zone := opts.Zone
	if zone == "" {
		// TODO: automatic? maybe that's not useful.
		// For now just return an error.
		return nil, errors.New("buildlet: missing required Zone option")
	}
	projectID := opts.ProjectID
	if projectID == "" {
		return nil, errors.New("buildlet: missing required ProjectID option")
	}

	prefix := "https://www.googleapis.com/compute/v1/projects/" + projectID
	machType := prefix + "/zones/" + zone + "/machineTypes/" + conf.MachineType()

	instance := &compute.Instance{
		Name:        instName,
		Description: opts.Description,
		MachineType: machType,
		Disks: []*compute.AttachedDisk{
			{
				AutoDelete: true,
				Boot:       true,
				Type:       "PERSISTENT",
				InitializeParams: &compute.AttachedDiskInitializeParams{
					DiskName:    instName,
					SourceImage: "https://www.googleapis.com/compute/v1/projects/" + projectID + "/global/images/" + conf.VMImage,
					DiskType:    "https://www.googleapis.com/compute/v1/projects/" + projectID + "/zones/" + zone + "/diskTypes/pd-ssd",
				},
			},
		},
		Tags: &compute.Tags{
			// Warning: do NOT list "http-server" or "allow-ssh" (our
			// project's custom tag to allow ssh access) here; the
			// buildlet provides full remote code execution.
			// The https-server is authenticated, though.
			Items: []string{"https-server"},
		},
		Metadata: &compute.Metadata{
			Items: []*compute.MetadataItems{
				// The buildlet-binary-url is the URL of the buildlet binary
				// which the VMs are configured to download at boot and run.
				// This lets us/ update the buildlet more easily than
				// rebuilding the whole VM image.
				{
					Key:   "buildlet-binary-url",
					Value: "http://storage.googleapis.com/go-builder-data/buildlet." + conf.GOOS() + "-" + conf.GOARCH(),
				},
			},
		},
		NetworkInterfaces: []*compute.NetworkInterface{
			&compute.NetworkInterface{
				AccessConfigs: []*compute.AccessConfig{
					&compute.AccessConfig{
						Type: "ONE_TO_ONE_NAT",
						Name: "External NAT",
					},
				},
				Network: prefix + "/global/networks/default",
			},
		},
	}

	if opts.DeleteIn != 0 {
		// In case the VM gets away from us (generally: if the
		// coordinator dies while a build is running), then we
		// set this attribute of when it should be killed so
		// we can kill it later when the coordinator is
		// restarted. The cleanUpOldVMs goroutine loop handles
		// that killing.
		instance.Metadata.Items = append(instance.Metadata.Items, &compute.MetadataItems{
			Key:   "delete-at",
			Value: fmt.Sprint(time.Now().Add(opts.DeleteIn).Unix()),
		})
	}
	for k, v := range opts.Meta {
		instance.Metadata.Items = append(instance.Metadata.Items, &compute.MetadataItems{
			Key:   k,
			Value: v,
		})
	}

	op, err := computeService.Instances.Insert(projectID, zone, instance).Do()
	if err != nil {
		return nil, fmt.Errorf("Failed to create instance: %v", err)
	}
	condRun(opts.OnInstanceRequested)
	createOp := op.Name

	// Wait for instance create operation to succeed.
OpLoop:
	for {
		time.Sleep(2 * time.Second)
		op, err := computeService.ZoneOperations.Get(projectID, zone, createOp).Do()
		if err != nil {
			return nil, fmt.Errorf("Failed to get op %s: %v", createOp, err)
		}
		switch op.Status {
		case "PENDING", "RUNNING":
			continue
		case "DONE":
			if op.Error != nil {
				for _, operr := range op.Error.Errors {
					return nil, fmt.Errorf("Error creating instance: %+v", operr)
				}
				return nil, errors.New("Failed to start.")
			}
			break OpLoop
		default:
			return nil, fmt.Errorf("Unknown create status %q: %+v", op.Status, op)
		}
	}
	condRun(opts.OnInstanceCreated)

	inst, err := computeService.Instances.Get(projectID, zone, instName).Do()
	if err != nil {
		return nil, fmt.Errorf("Error getting instance %s details after creation: %v", instName, err)
	}

	// Find its internal IP.
	var ip string
	for _, iface := range inst.NetworkInterfaces {
		if strings.HasPrefix(iface.NetworkIP, "10.") {
			ip = iface.NetworkIP
		}
	}
	if ip == "" {
		return nil, errors.New("didn't find its internal IP address")
	}

	// Wait for it to boot and its buildlet to come up.
	var buildletURL string
	var ipPort string
	if opts.TLS != NoKeyPair {
		buildletURL = "https://" + ip
		ipPort = ip + ":443"
	} else {
		buildletURL = "http://" + ip
		ipPort = ip + ":80"
	}
	condRun(opts.OnGotInstanceInfo)

	const timeout = 90 * time.Second
	var alive bool
	impatientClient := &http.Client{
		Timeout: 5 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true,
			},
		},
	}
	deadline := time.Now().Add(timeout)
	try := 0
	for time.Now().Before(deadline) {
		try++
		res, err := impatientClient.Get(buildletURL)
		if err != nil {
			time.Sleep(1 * time.Second)
			continue
		}
		res.Body.Close()
		if res.StatusCode != 200 {
			return nil, fmt.Errorf("buildlet returned HTTP status code %d on try number %d", res.StatusCode, try)
		}
		alive = true
		break
	}
	if !alive {
		return nil, fmt.Errorf("buildlet didn't come up in %v", timeout)
	}

	return NewClient(ipPort, opts.TLS), nil
}
