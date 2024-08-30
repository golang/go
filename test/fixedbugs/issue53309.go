// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type TaskInput interface {
	deps() []*taskDefinition
}

type Value[T any] interface {
	metaValue
}

type metaValue interface {
	TaskInput
}

type taskDefinition struct {
}

type taskResult struct {
	task *taskDefinition
}

func (tr *taskResult) deps() []*taskDefinition {
	return nil
}

func use[T any](v Value[T]) {
	_, ok := v.(*taskResult)
	if !ok {
		panic("output must be *taskResult")
	}
}

func main() {
	tr := &taskResult{&taskDefinition{}}
	use[string](Value[string](tr))
}
