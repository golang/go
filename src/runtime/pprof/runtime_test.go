// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"context"
	"fmt"
	"reflect"
	"testing"
)

func TestSetGoroutineLabels(t *testing.T) {
	sync := make(chan struct{})

	wantLabels := map[string]string{}
	if gotLabels := getProfLabel(); !reflect.DeepEqual(gotLabels, wantLabels) {
		t.Errorf("Expected parent goroutine's profile labels to be empty before test, got %v", gotLabels)
	}
	go func() {
		if gotLabels := getProfLabel(); !reflect.DeepEqual(gotLabels, wantLabels) {
			t.Errorf("Expected child goroutine's profile labels to be empty before test, got %v", gotLabels)
		}
		sync <- struct{}{}
	}()
	<-sync

	wantLabels = map[string]string{"key": "value"}
	ctx := WithLabels(context.Background(), Labels("key", "value"))
	SetGoroutineLabels(ctx)
	if gotLabels := getProfLabel(); !reflect.DeepEqual(gotLabels, wantLabels) {
		t.Errorf("parent goroutine's profile labels: got %v, want %v", gotLabels, wantLabels)
	}
	go func() {
		if gotLabels := getProfLabel(); !reflect.DeepEqual(gotLabels, wantLabels) {
			t.Errorf("child goroutine's profile labels: got %v, want %v", gotLabels, wantLabels)
		}
		sync <- struct{}{}
	}()
	<-sync

	wantLabels = map[string]string{}
	ctx = context.Background()
	SetGoroutineLabels(ctx)
	if gotLabels := getProfLabel(); !reflect.DeepEqual(gotLabels, wantLabels) {
		t.Errorf("Expected parent goroutine's profile labels to be empty, got %v", gotLabels)
	}
	go func() {
		if gotLabels := getProfLabel(); !reflect.DeepEqual(gotLabels, wantLabels) {
			t.Errorf("Expected child goroutine's profile labels to be empty, got %v", gotLabels)
		}
		sync <- struct{}{}
	}()
	<-sync
}

func TestDo(t *testing.T) {
	wantLabels := map[string]string{}
	if gotLabels := getProfLabel(); !reflect.DeepEqual(gotLabels, wantLabels) {
		t.Errorf("Expected parent goroutine's profile labels to be empty before Do, got %v", gotLabels)
	}

	Do(context.Background(), Labels("key1", "value1", "key2", "value2"), func(ctx context.Context) {
		wantLabels := map[string]string{"key1": "value1", "key2": "value2"}
		if gotLabels := getProfLabel(); !reflect.DeepEqual(gotLabels, wantLabels) {
			t.Errorf("parent goroutine's profile labels: got %v, want %v", gotLabels, wantLabels)
		}

		sync := make(chan struct{})
		go func() {
			wantLabels := map[string]string{"key1": "value1", "key2": "value2"}
			if gotLabels := getProfLabel(); !reflect.DeepEqual(gotLabels, wantLabels) {
				t.Errorf("child goroutine's profile labels: got %v, want %v", gotLabels, wantLabels)
			}
			sync <- struct{}{}
		}()
		<-sync

	})

	wantLabels = map[string]string{}
	if gotLabels := getProfLabel(); !reflect.DeepEqual(gotLabels, wantLabels) {
		fmt.Printf("%#v", gotLabels)
		fmt.Printf("%#v", wantLabels)
		t.Errorf("Expected parent goroutine's profile labels to be empty after Do, got %v", gotLabels)
	}
}

func getProfLabel() map[string]string {
	l := (*labelMap)(runtime_getProfLabel())
	if l == nil {
		return map[string]string{}
	}
	return *l
}
